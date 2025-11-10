import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from grounded_sam2_vggt_pointcloud import (
    aggregate_point_clouds,
    determine_seg_device,
    label_to_color,
    run_grounded_sam2_on_frame,
    save_ply_ascii,
    subsample_points,
)
from loop_utils.segmentation_postprocess import (
    aggregate_instance_cloud,
    run_segmentation_clustering,
    save_instance_metadata,
)
from loop_utils.wire_fitting import fit_electric_pole_wires
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SegmentationManager:
    """Encapsulates the SAM + GroundingDINO segmentation pipeline."""

    def __init__(self, config: dict, output_dir: str, aligned_dir: str, overlap: int) -> None:
        self.config = config
        self.seg_cfg = config.get("Segmentation", {}) or {}
        self.enabled = bool(self.seg_cfg.get("enable", False))
        self.output_dir = output_dir
        self.result_aligned_dir = aligned_dir
        self.overlap = overlap
        self.segmentation_output_dir = os.path.join(output_dir, "segmentation")
        self.segmentation_models = None
        self.pointcloud_cfg = self.config.get("Model", {}).get("Pointcloud_Save", {})

    def validate_config(self) -> None:
        if not self.enabled:
            return
        required = ["prompt", "sam2_checkpoint", "sam2_config", "grounding_dino_model_id"]
        missing = [key for key in required if not self.seg_cfg.get(key)]
        if missing:
            raise ValueError(f"[Segmentation] Missing required config keys: {missing}")

    def run(self, img_list: Sequence[str], chunk_indices: Sequence[Tuple[int, int]]) -> None:
        if not self.enabled:
            return
        prompt = self.seg_cfg.get("prompt")
        if not prompt:
            print("[Segmentation] Prompt not provided; skipping segmentation.")
            return
        allowed_labels = {item.strip().lower() for item in prompt.split(".") if item.strip()}
        if not allowed_labels:
            print("[Segmentation] Prompt produced no valid labels; skipping segmentation.")
            return

        try:
            predictor, grounding_model, processor, seg_device = self._get_segmentation_models()
        except Exception as exc:
            print(f"[Segmentation] Failed to initialize models: {exc}")
            return

        base_dir, per_label_dir = self._prepare_segmentation_dirs()
        per_label_points: Dict[str, List[np.ndarray]] = {}
        per_label_stats: Dict[str, Dict[str, float]] = {}
        total_frames = 0
        max_points_per_label = int(self.seg_cfg.get("max_points_per_label", 150_000))
        print("[Segmentation] Starting Grounded SAM 2 over aligned chunks...")

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_indices):
            chunk_file = Path(self.result_aligned_dir) / f"chunk_{chunk_idx}.npy"
            if not chunk_file.exists():
                print(f"[Segmentation] Missing aligned chunk {chunk_file}, skipping.")
                continue
            chunk_data = np.load(chunk_file, allow_pickle=True).item()
            chunk_points = chunk_data.get("points")
            chunk_conf = chunk_data.get("conf")
            chunk_images = chunk_data.get("images")
            if chunk_points is None or chunk_conf is None or chunk_images is None:
                print(f"[Segmentation] Chunk {chunk_idx} lacks required tensors; skipping.")
                continue

            unique_start = self._unique_chunk_start(chunk_idx, chunk_indices)
            chunk_len = chunk_points.shape[0]
            if unique_start >= chunk_len:
                continue

            frame_paths = [Path(p) for p in img_list[chunk_start + unique_start : chunk_end]]
            if not frame_paths:
                continue

            points_slice = np.asarray(chunk_points[unique_start:])
            conf_slice = self._prepare_conf_map(chunk_conf[unique_start:])
            depth_conf_threshold = self._compute_segmentation_conf_threshold(conf_slice)
            target_h, target_w = chunk_images.shape[2], chunk_images.shape[3]
            segments_per_frame = []

            for frame_path in frame_paths:
                segs = run_grounded_sam2_on_frame(
                    frame_path=frame_path,
                    prompt=prompt,
                    predictor=predictor,
                    grounding_model=grounding_model,
                    device=seg_device,
                    box_threshold=self.seg_cfg.get("box_threshold", 0.35),
                    text_threshold=self.seg_cfg.get("text_threshold", 0.25),
                    box_shrink_ratio=self.seg_cfg.get("box_shrink_ratio", 1.0),
                    morph_kernel=self.seg_cfg.get("morph_kernel", 0),
                    target_size=(target_h, target_w),
                    processor=processor,
                    debug=self.seg_cfg.get("debug", False),
                )
                filtered = [seg for seg in segs if seg.label.strip().lower() in allowed_labels]
                segments_per_frame.append(filtered)

            label_points, label_stats = aggregate_point_clouds(
                frame_paths=frame_paths,
                segments_per_frame=segments_per_frame,
                depth_conf=conf_slice,
                depth_conf_threshold=depth_conf_threshold,
                min_mask_area=self.seg_cfg.get("min_mask_area", 200),
                max_points_per_label=max_points_per_label,
                world_points=points_slice,
                frame_offset=chunk_start + unique_start,
            )

            for label, pts in label_points.items():
                per_label_points.setdefault(label, []).append(pts)
                src_stats = label_stats.get(label, {})
                dst_stats = per_label_stats.setdefault(
                    label, {"mask_pixels": 0.0, "instances": 0.0, "unique_frames": 0.0}
                )
                dst_stats["mask_pixels"] += src_stats.get("mask_pixels", 0.0)
                dst_stats["instances"] += src_stats.get("instances", 0.0)
                dst_stats["unique_frames"] += src_stats.get("unique_frames", 0.0)

            total_frames += len(frame_paths)
            print(
                f"[Segmentation] Chunk {chunk_idx}: processed {len(frame_paths)} frames, "
                f"labels found: {list(label_points.keys())}"
            )

        if not per_label_points:
            print("[Segmentation] No segmented points were produced. Check prompts or thresholds.")
            self._release_models()
            return

        merged_points = []
        merged_colors = []
        metadata = {}

        for label, point_lists in per_label_points.items():
            concatenated = np.concatenate(point_lists, axis=0)
            if max_points_per_label > 0 and len(concatenated) > max_points_per_label:
                concatenated = subsample_points(concatenated, max_points_per_label)
            color = label_to_color(label)
            colors = np.repeat(color[None, :], len(concatenated), axis=0)
            label_filename = f"{label.replace(' ', '_')}.ply"
            save_ply_ascii(per_label_dir / label_filename, concatenated, colors)
            merged_points.append(concatenated)
            merged_colors.append(colors)
            stats = per_label_stats.get(label, {})
            metadata[label] = {
                "points": len(concatenated),
                "mask_pixels": float(stats.get("mask_pixels", 0.0)),
                "unique_frames": float(stats.get("unique_frames", 0.0)),
                "instances": float(stats.get("instances", 0.0)),
            }
            print(f"[Segmentation] Saved {len(concatenated)} points for label '{label}' to {label_filename}")

        if merged_points and not self.seg_cfg.get("no_merged_cloud", False):
            all_points = np.concatenate(merged_points, axis=0)
            all_colors = np.concatenate(merged_colors, axis=0)
            save_ply_ascii(base_dir / "merged_point_cloud.ply", all_points, all_colors)
            print(f"[Segmentation] Merged point cloud written to {(base_dir / 'merged_point_cloud.ply')}")

        metadata_path = base_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Segmentation] Metadata saved to {metadata_path}")
        print(f"[Segmentation] Completed segmentation over {total_frames} unique frames.")

        self._run_optional_clustering(base_dir, per_label_dir)
        self._release_models()

    def _unique_chunk_start(self, chunk_idx: int, chunk_indices: Sequence[Tuple[int, int]]) -> int:
        if chunk_idx == 0:
            return 0
        chunk_start, chunk_end = chunk_indices[chunk_idx]
        return min(self.overlap, chunk_end - chunk_start)

    def _prepare_segmentation_dirs(self):
        base_dir = Path(self.segmentation_output_dir)
        per_label_dir = base_dir / "per_label"
        per_label_dir.mkdir(parents=True, exist_ok=True)
        return base_dir, per_label_dir

    @staticmethod
    def _prepare_conf_map(conf_array):
        if conf_array is None:
            return None
        conf_np = np.asarray(conf_array)
        if conf_np.ndim == 4 and conf_np.shape[-1] == 1:
            conf_np = np.squeeze(conf_np, axis=-1)
        return conf_np

    def _compute_segmentation_conf_threshold(self, conf_slice):
        explicit = self.seg_cfg.get("confidence_threshold")
        if explicit is not None:
            return float(explicit)
        if conf_slice is None:
            return None
        valid = conf_slice[np.isfinite(conf_slice)]
        valid = valid[valid > 1e-5]
        if valid.size == 0:
            return None
        coef = float(self.pointcloud_cfg.get("conf_threshold_coef", 0.75))
        return float(valid.mean() * coef)

    def _get_segmentation_models(self):
        if self.segmentation_models is not None:
            return self.segmentation_models

        seg_device = determine_seg_device(self.seg_cfg.get("device"))
        print(f"[Segmentation] Loading Grounded SAM 2 components on {seg_device}...")
        sam2_model = build_sam2(self.seg_cfg["sam2_config"], self.seg_cfg["sam2_checkpoint"], device=str(seg_device))
        predictor = SAM2ImagePredictor(sam2_model)
        dino_kwargs = {"trust_remote_code": True}
        processor = AutoProcessor.from_pretrained(self.seg_cfg["grounding_dino_model_id"], **dino_kwargs)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.seg_cfg["grounding_dino_model_id"], **dino_kwargs
        ).to(seg_device)
        grounding_model.eval()
        self.segmentation_models = (predictor, grounding_model, processor, seg_device)
        return self.segmentation_models

    def _run_optional_clustering(self, base_dir: Path, per_label_dir: Path) -> None:
        clustering_cfg = self.seg_cfg.get("clustering", {})
        if not clustering_cfg.get("enable"):
            return
        try:
            cluster_summary = run_segmentation_clustering(
                per_label_dir=per_label_dir,
                instance_root=base_dir / "instances",
                cfg_mapping=clustering_cfg,
            )
        except ImportError as exc:
            print(f"[Segmentation] Clustering skipped (scikit-learn missing): {exc}")
            return

        if not cluster_summary:
            print("[Segmentation] Clustering produced no instances.")
            return

        summary_path = save_instance_metadata(base_dir, cluster_summary)
        print(f"[Segmentation] Instance metadata saved to {summary_path}")

        aggregated = aggregate_instance_cloud(base_dir / "instances")
        if aggregated is not None:
            inst_points, inst_colors = aggregated
            save_ply_ascii(base_dir / "merged_point_cloud.ply", inst_points, inst_colors)
            print("[Segmentation] Merged point cloud updated with clustered instances.")
        else:
            print("[Segmentation] Clustering produced metadata but no points to merge.")

        wire_cfg = self.seg_cfg.get("wire_fitting", {})
        if not wire_cfg.get("enable", False):
            return
        spacing_cfg = wire_cfg.get("spacing_factor", 2.5)
        spacing_factor = float(spacing_cfg) if spacing_cfg is not None else None
        try:
            wire_result = fit_electric_pole_wires(
                instance_root=base_dir / "instances",
                label_name=wire_cfg.get("label", "electric pole"),
                min_points=int(wire_cfg.get("min_points", 150)),
                outlier_z_thresh=float(wire_cfg.get("outlier_z_thresh", 2.5)),
                samples_per_segment=int(wire_cfg.get("samples_per_segment", 32)),
                sag_fraction=float(wire_cfg.get("sag_fraction", 0.025)),
                spacing_factor=spacing_factor,
            )
        except Exception as exc:
            print(f"[Segmentation] Wire fitting failed: {exc}")
        else:
            if wire_result:
                print(
                    "[Segmentation] Electric pole wires saved to "
                    f"{wire_result.wire_path} ({wire_result.num_connections} connections)."
                )
                print(f"[Segmentation] Wire metadata saved to {wire_result.metadata_path}.")
            else:
                print("[Segmentation] Wire fitting skipped (insufficient electric pole instances).")

    def _release_models(self) -> None:
        self.segmentation_models = None
        torch.cuda.empty_cache()

