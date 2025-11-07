import argparse
import glob
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass  # Sky segmentation is optional; maintain parity with pi_long.py

REPO_ROOT = Path(__file__).resolve().parent
MAP_ANYTHING_DIR = REPO_ROOT / "map-anything"
if MAP_ANYTHING_DIR.exists() and str(MAP_ANYTHING_DIR) not in sys.path:
    sys.path.insert(0, str(MAP_ANYTHING_DIR))

from LoopModels.LoopModel import LoopDetector  # noqa: E402
from LoopModelDBoW.retrieval.retrieval_dbow import RetrievalDBOW  # noqa: E402
from loop_utils.sim3loop import Sim3LoopOptimizer  # noqa: E402
from loop_utils.sim3utils import *  # noqa: E402,F401,F403
from loop_utils.config_utils import load_config  # noqa: E402

from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402


def remove_duplicates(data_list):
    """
        data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    seen = {}
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = (item[0], item[2])

        if key not in seen.keys():
            seen[key] = True
            result.append(item)

    return result


class LongSeqResult:
    def __init__(self):
        self.combined_extrinsics = []
        self.combined_intrinsics = []
        self.combined_depth_maps = []
        self.combined_depth_confs = []
        self.combined_world_points = []
        self.combined_world_points_confs = []
        self.all_camera_poses = []


class MapAnything_Long:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sky_mask = False
        self.useDBoW = self.config["Model"]["useDBoW"]

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)

        self.all_camera_poses = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        mapanything_cfg = self.config.get("MapAnything", {})
        self.mapanything_model_name = self._resolve_model_name(
            mapanything_cfg.get("model_name", "facebook/map-anything")
        )
        self.memory_efficient_inference = mapanything_cfg.get(
            "memory_efficient_inference", False
        )
        self.use_amp = mapanything_cfg.get("use_amp", True)
        self.amp_dtype = mapanything_cfg.get("amp_dtype", "bf16")
        self.apply_mask = mapanything_cfg.get("apply_mask", True)
        self.mask_edges = mapanything_cfg.get("mask_edges", True)
        self.apply_confidence_mask = mapanything_cfg.get("apply_confidence_mask", False)
        self.confidence_percentile = mapanything_cfg.get("confidence_percentile", 10.0)
        self.norm_type = mapanything_cfg.get("norm_type", "dinov2")
        self.resize_mode = mapanything_cfg.get("resize_mode", "fixed_mapping")
        self.resize_size = mapanything_cfg.get("size", None)
        self.patch_size = mapanything_cfg.get("patch_size", 14)
        self.resolution_set = mapanything_cfg.get("resolution_set", 518)
        self.load_stride = mapanything_cfg.get("stride", 1)
        self.verbose_loading = mapanything_cfg.get("verbose", False)
        if self.load_stride != 1:
            raise ValueError("MapAnything-Long currently requires stride=1 for consistent chunk processing.")

        print("Loading MapAnything model...")
        try:
            self.model = (
                MapAnything.from_pretrained(self.mapanything_model_name)
                .to(self.device)
                .eval()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MapAnything weights '{self.mapanything_model_name}': {exc}"
            ) from exc

        self.skyseg_session = None

        self.chunk_indices = None  # [(begin_idx, end_idx), ...]

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config)

        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            if self.useDBoW:
                self.retrieval = RetrievalDBOW(config=self.config)
            else:
                loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
                self.loop_detector = LoopDetector(
                    image_dir=image_dir,
                    output=loop_info_save_path,
                    config=self.config,
                )

        print("init done.")

    def _resolve_model_name(self, model_name):
        """
        Allow weights specified relative to repository root.
        """
        if not model_name:
            return "facebook/map-anything"

        candidate = Path(model_name)
        if candidate.exists():
            return str(candidate.resolve())

        repo_relative = (REPO_ROOT / model_name).resolve()
        if repo_relative.exists():
            return str(repo_relative)

        return model_name

    def _load_views(self, image_paths):
        return load_images(
            image_paths,
            resize_mode=self.resize_mode,
            size=self.resize_size,
            norm_type=self.norm_type,
            patch_size=self.patch_size,
            resolution_set=self.resolution_set,
            stride=self.load_stride,
            verbose=self.verbose_loading,
        )

    def _aggregate_predictions(self, outputs):
        points = []
        confs = []
        images = []
        camera_poses = []
        intrinsics = []
        depth_z = []
        masks = []
        metric_scale = []

        for pred in outputs:
            pts_np = (
                pred["pts3d"][0].detach().cpu().numpy().astype(np.float32)
            )  # (H, W, 3)
            conf_np = (
                pred["conf"][0].detach().cpu().numpy().astype(np.float32)
            )  # (H, W)
            img_np = pred["img_no_norm"][0].detach().cpu().numpy().astype(np.float32)
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
            cam_pose = (
                pred["camera_poses"][0].detach().cpu().numpy().astype(np.float32)
            )
            intrinsic = (
                pred["intrinsics"][0].detach().cpu().numpy().astype(np.float32)
            )
            points.append(pts_np)
            confs.append(conf_np)
            images.append(img_np)
            camera_poses.append(cam_pose)
            intrinsics.append(intrinsic)

            if "depth_z" in pred:
                depth_np = (
                    pred["depth_z"][0].detach().cpu().numpy().squeeze(-1).astype(np.float32)
                )
                depth_z.append(depth_np)
            if "mask" in pred:
                mask_np = (
                    pred["mask"][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze(-1)
                    .astype(np.float32)
                )
                masks.append(mask_np)
            if "metric_scaling_factor" in pred:
                metric_np = (
                    pred["metric_scaling_factor"][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                metric_scale.append(metric_np)

        result = {
            "points": np.stack(points),
            "conf": np.stack(confs),
            "images": np.stack(images),
            "camera_poses": np.stack(camera_poses),
            "intrinsics": np.stack(intrinsics),
        }
        if depth_z:
            result["depth_z"] = np.stack(depth_z)
        if masks:
            result["mask"] = np.stack(masks)
        if metric_scale:
            result["metric_scaling_factor"] = np.stack(metric_scale)

        return result

    def process_single_chunk(self, range_1, chunk_idx=None, range_2=None, is_loop=False):
        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        views = self._load_views(chunk_image_paths)
        print(f"Loaded {len(views)} images")

        with torch.no_grad():
            outputs = self.model.infer(
                views,
                memory_efficient_inference=self.memory_efficient_inference,
                use_amp=self.use_amp,
                amp_dtype=self.amp_dtype,
                apply_mask=self.apply_mask,
                mask_edges=self.mask_edges,
                apply_confidence_mask=self.apply_confidence_mask,
                confidence_percentile=self.confidence_percentile,
            )

        predictions = self._aggregate_predictions(outputs)

        if is_loop:
            save_dir = self.result_loop_dir
            filename = (
                f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
                if range_2 is not None
                else f"loop_{range_1[0]}_{range_1[1]}.npy"
            )
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions["camera_poses"]
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))

        np.save(save_path, predictions)

        return predictions if is_loop or range_2 is not None else None

    def get_loop_pairs(self):
        if self.useDBoW:  # DBoW2
            for frame_id, img_path in tqdm(enumerate(self.img_list)):
                image_ori = np.array(Image.open(img_path))
                if len(image_ori.shape) == 2:
                    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2RGB)

                frame = image_ori
                frame = cv2.resize(
                    frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                )
                self.retrieval(frame, frame_id)
                cands = self.retrieval.detect_loop(
                    thresh=self.config["Loop"]["DBoW"]["thresh"],
                    num_repeat=self.config["Loop"]["DBoW"]["num_repeat"],
                )

                if cands is not None:
                    (i, j) = cands
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                    self.loop_list.append(cands)

                self.retrieval.save_up_to(frame_id)

        else:  # DNIO v2
            self.loop_detector.run()
            self.loop_list = self.loop_detector.get_loop_list()

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) must be less than chunk size ({self.chunk_size})"
            )
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            self.chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                self.chunk_indices.append((start_idx, end_idx))

        if self.loop_enable:
            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            loop_results = remove_duplicates(loop_results)
            print(loop_results)
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True
                )

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

        print(
            f"Processing {len(self.img_list)} images in {len(self.chunk_indices)} chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")
            self.process_single_chunk(
                self.chunk_indices[chunk_idx], chunk_idx=chunk_idx
            )
            torch.cuda.empty_cache()

        del self.model
        torch.cuda.empty_cache()

        print("Aligning all the chunks...")
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(
                f"Aligning {chunk_idx} and {chunk_idx+1} (Total {len(self.chunk_indices)-1})"
            )
            chunk_data1 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"),
                allow_pickle=True,
            ).item()
            chunk_data2 = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            point_map1 = chunk_data1["points"][-self.overlap :]
            point_map2 = chunk_data2["points"][: self.overlap]
            conf1 = chunk_data1["conf"][-self.overlap :]
            conf2 = chunk_data2["conf"][: self.overlap]

            conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
            s, R, t = weighted_align_point_maps(
                point_map1,
                conf1,
                point_map2,
                conf2,
                conf_threshold=conf_threshold,
                config=self.config,
            )
            print("Estimated Scale:", s)
            print("Estimated Rotation:\n", R)
            print("Estimated Translation:", t)

            self.sim3_list.append((s, R, t))

        if self.loop_enable:
            for item in self.loop_predict_list:
                chunk_idx_a = item[0][0]
                chunk_idx_b = item[0][2]
                chunk_a_range = item[0][1]
                chunk_b_range = item[0][3]

                print("chunk_a align")
                point_map_loop = item[1]["points"][
                    : chunk_a_range[1] - chunk_a_range[0]
                ]
                conf_loop = item[1]["conf"][: chunk_a_range[1] - chunk_a_range[0]]
                chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
                chunk_a_rela_end = (
                    chunk_a_rela_begin + chunk_a_range[1] - chunk_a_range[0]
                )
                print(self.chunk_indices[chunk_idx_a])
                print(chunk_a_range)
                print(chunk_a_rela_begin, chunk_a_rela_end)
                chunk_data_a = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                    allow_pickle=True,
                ).item()

                point_map_a = chunk_data_a["points"][chunk_a_rela_begin:chunk_a_rela_end]
                conf_a = chunk_data_a["conf"][chunk_a_rela_begin:chunk_a_rela_end]

                conf_threshold = min(np.median(conf_a), np.median(conf_loop)) * 0.1
                s_a, R_a, t_a = weighted_align_point_maps(
                    point_map_a,
                    conf_a,
                    point_map_loop,
                    conf_loop,
                    conf_threshold=conf_threshold,
                    config=self.config,
                )
                print("Estimated Scale:", s_a)
                print("Estimated Rotation:\n", R_a)
                print("Estimated Translation:", t_a)

                print("chunk_b align")
                chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
                chunk_b_rela_end = (
                    chunk_b_rela_begin + chunk_b_range[1] - chunk_b_range[0]
                )
                print(self.chunk_indices[chunk_idx_b])
                print(chunk_b_range)
                print(chunk_b_rela_begin, chunk_b_rela_end)
                chunk_data_b = np.load(
                    os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                    allow_pickle=True,
                ).item()

                point_map_b = chunk_data_b["points"][chunk_b_rela_begin:chunk_b_rela_end]
                conf_b = chunk_data_b["conf"][chunk_b_rela_begin:chunk_b_rela_end]

                conf_threshold = min(np.median(conf_b), np.median(conf_loop)) * 0.1
                s_b, R_b, t_b = weighted_align_point_maps(
                    point_map_b,
                    conf_b,
                    point_map_loop,
                    conf_loop,
                    conf_threshold=conf_threshold,
                    config=self.config,
                )
                print("Estimated Scale:", s_b)
                print("Estimated Rotation:\n", R_b)
                print("Estimated Translation:", t_b)

                print("a -> b SIM 3")
                s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
                print("Estimated Scale:", s_ab)
                print("Estimated Rotation:\n", R_ab)
                print("Estimated Translation:", t_ab)

                self.loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        if self.loop_enable:
            input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )
            self.sim3_list = self.loop_optimizer.optimize(
                self.sim3_list, self.loop_sim3_list
            )
            optimized_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                self.sim3_list
            )

            def extract_xyz(pose_tensor):
                poses = pose_tensor.cpu().numpy()
                return poses[:, 0], poses[:, 1], poses[:, 2]

            x0, _, y0 = extract_xyz(input_abs_poses)
            x1, _, y1 = extract_xyz(optimized_abs_poses)

            plt.figure(figsize=(8, 8))
            plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
            plt.plot(x1, y1, "o-", label="After Optimization")
            for i, j, _ in self.loop_sim3_list:
                plt.plot(
                    [x0[i], x0[j]],
                    [y0[i], y0[j]],
                    "r--",
                    alpha=0.25,
                    label="Loop (Before)" if i == 5 else "",
                )
                plt.plot(
                    [x1[i], x1[j]],
                    [y1[i], y1[j]],
                    "g-",
                    alpha=0.35,
                    label="Loop (After)" if i == 5 else "",
                )
            plt.gca().set_aspect("equal")
            plt.title("Sim3 Loop Closure Optimization")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.grid(True)
            plt.axis("equal")
            save_path = os.path.join(self.output_dir, "sim3_opt_result.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            chunk_data["points"] = apply_sim3_direct(chunk_data["points"], s, R, t)

            aligned_path = os.path.join(
                self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy"
            )
            np.save(aligned_path, chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"),
                    allow_pickle=True,
                ).item()
                np.save(
                    os.path.join(self.result_aligned_dir, "chunk_0.npy"), chunk_data_first
                )

            aligned_chunk_data = (
                np.load(
                    os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx}.npy"),
                    allow_pickle=True,
                ).item()
                if chunk_idx > 0
                else chunk_data_first
            )

            points = aligned_chunk_data["points"].reshape(-1, 3)
            colors = (
                aligned_chunk_data["images"]
                .transpose(0, 2, 3, 1)
                .reshape(-1, 3)
                .clip(min=0.0, max=1.0)
                * 255
            ).astype(np.uint8)
            confs = aligned_chunk_data["conf"].reshape(-1)
            ply_path = os.path.join(self.pcd_dir, f"{chunk_idx}_pcd.ply")
            save_confident_pointcloud_batch(
                points=points,
                colors=colors,
                confs=confs,
                output_path=ply_path,
                conf_threshold=np.mean(confs)
                * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"],
                sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
            )

        self.save_camera_poses()

        print("Done.")

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        if self.loop_enable:
            self.get_loop_pairs()

            if self.useDBoW:
                self.retrieval.close()
                gc.collect()
            else:
                del self.loop_detector
        torch.cuda.empty_cache()

        self.process_long_sequence()

    def save_camera_poses(self):
        chunk_colors = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [128, 0, 0],
            [0, 128, 0],
            [0, 0, 128],
            [128, 128, 0],
        ]
        print("Saving all camera poses to txt file...")

        all_poses = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = self.all_camera_poses[0]
        for i, idx in enumerate(range(first_chunk_range[0], first_chunk_range[1])):
            c2w = first_chunk_extrinsics[i]
            all_poses[idx] = c2w

        for chunk_idx in range(1, len(self.all_camera_poses)):
            chunk_range, chunk_extrinsics = self.all_camera_poses[chunk_idx]
            s, R, t = self.sim3_list[
                chunk_idx - 1
            ]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            for i, idx in enumerate(range(chunk_range[0], chunk_range[1])):
                c2w = chunk_extrinsics[i]
                transformed_c2w = S @ c2w
                all_poses[idx] = transformed_c2w

        poses_path = os.path.join(self.output_dir, "camera_poses.txt")
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        print(f"Camera poses saved to {poses_path}")

        ply_path = os.path.join(self.output_dir, "camera_poses.ply")
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(
                    f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n"
                )

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        if not self.delete_temp_files:
            return

        total_space = 0

        print(f"Deleting the temp files under {self.result_unaligned_dir}")
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_aligned_dir}")
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_loop_dir}")
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print("Deleting temp files done.")

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


import shutil


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapAnything-Long")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/base_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default=None,
        help="Optional HuggingFace model identifier to override config",
    )
    parser.add_argument(
        "--apache",
        action="store_true",
        help="Use Apache 2.0 licensed MapAnything model (facebook/map-anything-apache)",
    )
    parser.add_argument(
        "--memory_efficient_inference",
        action="store_true",
        help="Enable memory efficient inference (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    mapanything_cfg = config.setdefault("MapAnything", {})

    if args.apache:
        mapanything_cfg["model_name"] = "facebook/map-anything-apache"
    elif args.model_name:
        mapanything_cfg["model_name"] = args.model_name

    if args.memory_efficient_inference:
        mapanything_cfg["memory_efficient_inference"] = True

    image_dir = args.image_dir
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = "./exps"

    save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(args.config, save_dir)

    if config["Model"]["align_method"] == "numba":
        warmup_numba()

    mapanything_long = MapAnything_Long(image_dir, save_dir, config)
    mapanything_long.run()
    mapanything_long.close()

    del mapanything_long
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print("MapAnything Long done.")
    sys.exit()
