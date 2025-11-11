#!/usr/bin/env python3
"""
Bridge VGGT depth maps with Grounded SAM 2 segmentations to create segmented 3D point clouds.

This script expects a directory of frames and will:
  1. Run VGGT to predict per-frame depth maps and camera parameters.
  2. Run Grounding DINO + SAM 2 on the same frames to obtain segmentation masks.
  3. Unproject depth values inside each mask to a labelled 3D point cloud.
  4. Export per-label PLY files (and an optional merged PLY) alongside JSON metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from more_utils.video_utils import create_video_from_images
from vggt.models.vggt import VGGT
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# --------------------------------------------------------------------------------------
# Utility dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class SegmentMask:
    label: str
    score: float
    mask: np.ndarray  # bool array aligned with VGGT resolution (H, W)
    bbox_xyxy: np.ndarray  # (4,)


# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate segmented 3D point clouds using VGGT + Grounded SAM 2.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory containing ordered RGB frames.")
    parser.add_argument("--video-path", type=Path, default=None, help="Optional MP4/MOV file to extract frames from.")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="When using --video-path, downsample frames to this FPS (omit to keep the source rate).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for Grounding DINO (use lowercase, end each object with a dot).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/vggt_gsam2_pointcloud"),
        help="Folder where point clouds and metadata will be written.",
    )
    # VGGT options
    parser.add_argument(
        "--vggt-checkpoint",
        type=str,
        default="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        help="VGGT checkpoint path or URL.",
    )
    parser.add_argument(
        "--vggt-resolution",
        type=int,
        default=518,
        help="Resolution used when running VGGT (must be divisible by the 14px patch size).",
    )
    parser.add_argument(
        "--img-load-resolution",
        type=int,
        default=518,
        help="Resolution used when loading images before VGGT (square padded).",
    )
    parser.add_argument(
        "--vggt-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference for VGGT inference.",
    )
    parser.add_argument(
        "--depth-confidence-threshold",
        type=float,
        default=5.0,
        help="Minimum depth confidence (VGGT) required to keep a 3D point.",
    )

    # Grounded SAM 2 options
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="./checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM 2 checkpoint.",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM 2 config YAML.",
    )
    parser.add_argument(
        "--grounding-dino-model-id",
        type=str,
        default="rziga/mm_grounding_dino_large_all",
        help="Hugging Face repo id for Grounding DINO.",
    )
    parser.add_argument("--seg-device", type=str, default=None, help="Device for Grounded SAM 2 (defaults to CUDA if available).")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Grounding DINO score threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO text threshold.")
    parser.add_argument(
        "--box-shrink-ratio",
        type=float,
        default=1.0,
        help="Factor (0,1] to shrink boxes before SAM 2 prompting. Use <1.0 to tighten masks.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Odd kernel size for morphological opening on masks (set to 1 to disable).",
    )

    # Output control
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=200,
        help="Minimum number of pixels (after resizing to VGGT resolution) for a mask to contribute points.",
    )
    parser.add_argument(
        "--max-points-per-label",
        type=int,
        default=150_000,
        help="Randomly subsample points per label to this maximum (set <=0 to keep all).",
    )
    parser.add_argument(
        "--no-merged-cloud",
        action="store_true",
        help="Skip writing the merged point cloud (only per-label clouds will be stored).",
    )
    parser.add_argument(
        "--export-full-reconstruction",
        action="store_true",
        help="Also export a full-scene VGGT point cloud using all confident pixels.",
    )
    parser.add_argument(
        "--max-full-reconstruction-points",
        type=int,
        default=1_000_000,
        help="Cap the number of points kept in the full reconstruction export (set <=0 to keep all).",
    )
    parser.add_argument(
        "--save-vggt-debug",
        action="store_true",
        help="Persist raw VGGT outputs (depth, confidence, intrinsics/extrinsics) for debugging.",
    )

    return parser.parse_args()


def natural_key(path: Path) -> Tuple:
    """
    Sort helper that splits digits to keep frame0001 < frame0002 ordering even with mixed names.
    """
    import re

    return tuple(int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name))


def list_frame_paths(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"Frames directory must be a directory: {frames_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in image_extensions]
    frame_paths.sort(key=natural_key)
    if not frame_paths:
        raise ValueError(f"No image frames found in {frames_dir} (looked for {sorted(image_extensions)})")
    return frame_paths


def extract_video_frames(
    video_path: Path, output_dir: Path, target_fps: float | None = None
) -> Tuple[List[Path], List[float]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    if target_fps is not None and target_fps <= 0:
        cap.release()
        raise ValueError(f"Target FPS must be positive, got {target_fps}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    has_native_fps = native_fps is not None and native_fps > 0
    frame_interval = None
    if target_fps is not None and has_native_fps:
        frame_interval = native_fps / target_fps
    next_frame_to_save = 0.0
    target_period = 1.0 / target_fps if target_fps else None
    next_timestamp = 0.0

    paths: List[Path] = []
    timestamps: List[float] = []
    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        should_save = target_fps is None
        if target_fps is not None:
            if frame_interval is not None:
                if frame_idx >= next_frame_to_save - 1e-6:
                    should_save = True
                    while next_frame_to_save <= frame_idx + 1e-6:
                        next_frame_to_save += frame_interval
            else:
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if timestamp_ms > 0:
                    current_time = timestamp_ms / 1000.0
                else:
                    current_time = frame_idx * target_period
                if current_time >= next_timestamp - 1e-6:
                    should_save = True
                    while next_timestamp <= current_time + 1e-6:
                        next_timestamp += target_period

        if should_save:
            frame_path = output_dir / f"{saved_idx:05d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                cap.release()
                raise RuntimeError(f"Failed to write frame {frame_path}")
            paths.append(frame_path)
            if has_native_fps and native_fps > 0:
                timestamp = frame_idx / native_fps
            elif target_fps:
                timestamp = saved_idx / target_fps
            else:
                timestamp = float(saved_idx)
            timestamps.append(float(timestamp))
            saved_idx += 1
        frame_idx += 1

    cap.release()
    if not paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return paths, timestamps


def square_mask_to_original(mask_square: np.ndarray, coord_row: np.ndarray) -> np.ndarray:
    """
    Convert a VGGT-square mask back to the original image resolution using stored padding metadata.
    """
    x1, y1, x2, y2, width, height = coord_row
    width_i = int(round(width))
    height_i = int(round(height))
    if width_i <= 0 or height_i <= 0:
        return np.zeros((0, 0), dtype=bool)

    x1_i = int(np.floor(x1))
    y1_i = int(np.floor(y1))
    x2_i = int(np.ceil(x2))
    y2_i = int(np.ceil(y2))
    x1_i = max(0, x1_i)
    y1_i = max(0, y1_i)
    x2_i = min(mask_square.shape[1], x2_i)
    y2_i = min(mask_square.shape[0], y2_i)
    if x2_i <= x1_i or y2_i <= y1_i:
        return np.zeros((height_i, width_i), dtype=bool)

    cropped = mask_square[y1_i:y2_i, x1_i:x2_i]
    if cropped.size == 0:
        return np.zeros((height_i, width_i), dtype=bool)
    resized = cv2.resize(cropped.astype(np.uint8), (width_i, height_i), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def select_device_dtype(preferred_device: str = "auto") -> Tuple[torch.device, torch.dtype, callable]:
    """
    Choose device and dtype following demo_colmap._select_device_dtype, but kept local to avoid extra deps.
    Returns (device, dtype, autocast_context_fn).
    """

    from contextlib import nullcontext

    def cuda_choice() -> Tuple[torch.device, torch.dtype, callable]:
        device = torch.device("cuda")
        capability_major = torch.cuda.get_device_capability(device=device)[0]
        dtype = torch.bfloat16 if capability_major >= 8 else torch.float16

        def autocast_ctx():
            return torch.cuda.amp.autocast(dtype=dtype)

        return device, dtype, autocast_ctx

    preferred_device = preferred_device.lower()

    if preferred_device in ("auto", "cuda"):
        if torch.cuda.is_available():
            return cuda_choice()
        if preferred_device == "cuda":
            print("CUDA requested but unavailable; falling back to CPU.")

    if preferred_device in ("auto", "mps"):
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if has_mps:
            device = torch.device("mps")
            dtype = torch.float16
            return device, dtype, nullcontext
        if preferred_device == "mps":
            print("MPS requested but unavailable; falling back to CPU.")

    device = torch.device("cpu")
    dtype = torch.float32
    return device, dtype, nullcontext


def run_vggt(model: VGGT, images: torch.Tensor, autocast_ctx, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Minimal VGGT forward pass copied from demo_colmap.run_VGGT with minor tweaks.
    Args:
        model: VGGT model on the proper device.
        images: Tensor [S, 3, H, W] on same device.
        autocast_ctx: callable returning autocast context manager.
        resolution: target square resolution for VGGT (divisible by 14).
    Returns:
        extrinsic [S, 3, 4], intrinsic [S, 3, 3], depth_map [S, H, W, 1], depth_conf [S, H, W]
    """
    assert images.ndim == 4 and images.shape[1] == 3, f"Expected [S,3,H,W] images, got {images.shape}"

    images_resized = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with autocast_ctx():
            seq = images_resized[None]  # [1, S, 3, H, W]
            aggregated_tokens_list, ps_idx = model.aggregator(seq)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, seq.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, seq, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def load_vggt_model(checkpoint: str, device: torch.device) -> VGGT:
    model = VGGT()
    if checkpoint.startswith("http://") or checkpoint.startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu")
    else:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def determine_seg_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def shrink_boxes_xyxy(boxes: np.ndarray, ratio: float, img_width: int, img_height: int) -> np.ndarray:
    if boxes.size == 0 or not (0 < ratio <= 1):
        return boxes
    shrunk = boxes.copy()
    widths = shrunk[:, 2] - shrunk[:, 0]
    heights = shrunk[:, 3] - shrunk[:, 1]
    centers_x = shrunk[:, 0] + widths * 0.5
    centers_y = shrunk[:, 1] + heights * 0.5
    half_widths = widths * ratio * 0.5
    half_heights = heights * ratio * 0.5
    shrunk[:, 0] = centers_x - half_widths
    shrunk[:, 2] = centers_x + half_widths
    shrunk[:, 1] = centers_y - half_heights
    shrunk[:, 3] = centers_y + half_heights
    shrunk[:, [0, 2]] = np.clip(shrunk[:, [0, 2]], 0, img_width)
    shrunk[:, [1, 3]] = np.clip(shrunk[:, [1, 3]], 0, img_height)
    return shrunk


def apply_morphological_opening(masks: np.ndarray, kernel_size: int) -> np.ndarray:
    if masks.size == 0 or kernel_size <= 1:
        return masks
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size for morphological opening must be odd.")
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    processed_masks = []
    for mask in masks:
        mask_uint8 = (mask > 0).astype(np.uint8)
        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        processed_masks.append(opened.astype(bool))
    return np.stack(processed_masks, axis=0)


def ensure_masks_batch_first(masks: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unsupported mask shape {arr.shape}")
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Cannot normalize mask array with shape {arr.shape}")
    return arr.astype(bool)


def mask_to_vggt_resolution(mask: np.ndarray, target_size: int | Tuple[int, int]) -> np.ndarray:
    """
    Resize binary masks so they align with the tensors consumed by downstream models.

    If target_size is an int, the mask is first center-padded to a square before resizing, matching
    the behaviour of load_and_preprocess_images_square (VGGT pipeline). When target_size is a tuple
    (target_height, target_width), the mask is resized directly to that spatial resolution. This
    makes the helper usable for both VGGT (square inputs) and Pi-Long (rectangular inputs).
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    use_square_padding = False
    if isinstance(target_size, np.ndarray):
        if target_size.size != 2:
            raise ValueError(f"target_size ndarray must contain exactly two values, got {target_size}")
        target_height, target_width = int(target_size[0]), int(target_size[1])
    elif isinstance(target_size, (tuple, list)):
        if len(target_size) != 2:
            raise ValueError(f"target_size sequence must have length 2, got {target_size}")
        target_height, target_width = int(target_size[0]), int(target_size[1])
    else:
        use_square_padding = True
        target_height = target_width = int(target_size)

    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Invalid target resolution ({target_height}, {target_width})")

    if use_square_padding:
        # Square resize path (VGGT-style padding).
        height, width = mask.shape
        max_dim = max(width, height)
        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        square[top : top + height, left : left + width] = mask.astype(np.uint8)
        resized = cv2.resize(square, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        return resized.astype(bool)

    # Generic rectangular resize (Pi-Long).
    resized = cv2.resize(mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def label_to_color(label: str) -> np.ndarray:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return np.frombuffer(digest[:3], dtype=np.uint8)


def save_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
    return points[idx]


def subsample_points_with_colors(points: np.ndarray, colors: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    idx = np.random.default_rng(0).choice(len(points), size=max_points, replace=False)
    return points[idx], colors[idx]


def run_grounded_sam2_on_frame(
    frame_path: Path,
    prompt: str,
    predictor: SAM2ImagePredictor,
    grounding_model,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
    box_shrink_ratio: float,
    morph_kernel: int,
    target_size: int,
    processor: AutoProcessor,
    debug: bool = False,
) -> List[SegmentMask]:
    image_bgr = cv2.imread(str(frame_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read frame: {frame_path}")
    image_source = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_source)
    raw_inputs = processor(
        images=image_source,
        text=prompt,
        return_tensors="pt",
    )
    inputs = {}
    for key, value in raw_inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
        else:
            inputs[key] = value
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_source.shape[:2]],
    )
    if not results or results[0]["boxes"].numel() == 0:
        return []
    boxes_xyxy = results[0]["boxes"].cpu().numpy()
    confidences_np = results[0].get("scores")
    if confidences_np is None:
        confidences_np = torch.ones(len(boxes_xyxy))
    confidences_np = confidences_np.cpu().numpy()
    labels = results[0]["labels"]
    labels = [label if isinstance(label, str) else str(label) for label in labels]
    h, w = image_source.shape[:2]

    boxes_xyxy = shrink_boxes_xyxy(boxes_xyxy, box_shrink_ratio, w, h)

    use_autocast = device.type == "cuda"
    autocast_device = "cuda" if use_autocast else "cpu"
    autocast_dtype = torch.bfloat16 if use_autocast else torch.float32
    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_autocast):
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=False,
        )

    masks = ensure_masks_batch_first(masks)
    masks = apply_morphological_opening(masks, morph_kernel)

    results: List[SegmentMask] = []
    for mask_np, score, label, bbox in zip(masks, confidences_np, labels, boxes_xyxy):
        mask_vggt = mask_to_vggt_resolution(mask_np, target_size)
        if mask_vggt.sum() == 0:
            if debug:
                print(f"[Debug] Mask '{label}' skipped after resizing (zero area).")
            continue
        results.append(SegmentMask(label=label.strip(), score=float(score), mask=mask_vggt, bbox_xyxy=bbox))
    return results


def aggregate_point_clouds(
    frame_paths: Sequence[Path],
    segments_per_frame: Sequence[List[SegmentMask]],
    extrinsic: np.ndarray | None = None,
    intrinsic: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
    depth_conf: np.ndarray | None = None,
    depth_conf_threshold: float | None = 5.0,
    min_mask_area: int = 200,
    max_points_per_label: int = 150_000,
    *,
    world_points: np.ndarray | None = None,
    frame_offset: int = 0,
    frame_numbers: Sequence[int] | None = None,
    full_reconstruction_accumulator: List[np.ndarray] | None = None,
    full_reconstruction_colors: List[np.ndarray] | None = None,
    debug_log: bool = False,
    frame_debug_stats: List[Dict[str, Any]] | None = None,
    square_images: np.ndarray | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    num_frames = len(frame_paths)
    if len(segments_per_frame) != num_frames:
        raise ValueError("segments_per_frame length must match frame_paths")

    use_precomputed_points = world_points is not None
    if use_precomputed_points:
        if world_points.shape[0] != num_frames:
            raise ValueError("world_points must have the same number of frames as frame_paths")
    else:
        if depth_map is None or extrinsic is None or intrinsic is None:
            raise ValueError("extrinsic, intrinsic, and depth_map are required when world_points is None")
        if depth_map.shape[0] != num_frames:
            raise ValueError("depth_map must have the same number of frames as frame_paths")
        if extrinsic.shape[0] != num_frames or intrinsic.shape[0] != num_frames:
            raise ValueError("extrinsic/intrinsic must have the same number of frames as frame_paths")
    if depth_conf is not None and depth_conf.shape[0] != num_frames:
        raise ValueError("depth_conf must have the same number of frames as frame_paths")

    label_to_points: Dict[str, List[np.ndarray]] = {}
    label_stats: Dict[str, Dict[str, float]] = {}
    if frame_numbers is not None and len(frame_numbers) != num_frames:
        raise ValueError("frame_numbers must have the same length as frame_paths")

    for frame_idx, frame_path in enumerate(frame_paths):
        frame_path = Path(frame_path)
        segments = segments_per_frame[frame_idx]
        if frame_numbers is not None:
            global_frame_idx = int(frame_numbers[frame_idx])
        else:
            global_frame_idx = frame_offset + frame_idx
        if use_precomputed_points:
            world_points_frame = world_points[frame_idx]
            if world_points_frame.ndim != 3 or world_points_frame.shape[-1] != 3:
                raise ValueError(f"world_points frame must have shape (H, W, 3); got {world_points_frame.shape}")
            valid_depth_mask = np.all(np.isfinite(world_points_frame), axis=-1)
            depth_frame = None
        else:
            depth_frame = depth_map[frame_idx, ..., 0] if depth_map.shape[-1] == 1 else depth_map[frame_idx]
            world_points_frame, _, valid_depth_mask = depth_to_world_coords_points(
                depth_frame, extrinsic[frame_idx], intrinsic[frame_idx]
            )

        conf_frame = None
        if depth_conf is not None:
            conf_frame = depth_conf[frame_idx]
            if conf_frame.ndim == 3 and conf_frame.shape[-1] == 1:
                conf_frame = conf_frame[..., 0]

        if depth_conf_threshold is not None and conf_frame is not None:
            conf_mask = conf_frame >= depth_conf_threshold
        else:
            conf_mask = np.ones_like(valid_depth_mask, dtype=bool)

        valid_mask = valid_depth_mask & conf_mask

        total_pixels = int(valid_mask.size)
        valid_pixels = int(valid_mask.sum())

        mean_conf_all = float(conf_frame.mean()) if conf_frame is not None else None
        mean_conf_valid = float(conf_frame[valid_mask].mean()) if (conf_frame is not None and valid_pixels > 0) else None
        if depth_frame is not None and valid_pixels > 0:
            depth_values_valid = depth_frame[valid_mask]
            depth_min = float(depth_values_valid.min())
            depth_max = float(depth_values_valid.max())
            depth_mean = float(depth_values_valid.mean())
        else:
            depth_min = depth_max = depth_mean = None

        if frame_debug_stats is not None:
            frame_debug_stats.append(
                {
                    "frame_index": global_frame_idx,
                    "frame_name": frame_path.name,
                    "detected_masks": len(segments),
                    "valid_pixels": valid_pixels,
                    "total_pixels": total_pixels,
                    "mean_conf_all": mean_conf_all,
                    "mean_conf_valid": mean_conf_valid,
                    "depth_min": depth_min,
                    "depth_max": depth_max,
                    "depth_mean": depth_mean,
                }
            )

        if debug_log:
            valid_ratio = valid_pixels / total_pixels if total_pixels else 0.0
            debug_msg = (
                f"[Debug] Frame {global_frame_idx:04d} ({frame_path.name}): "
                f"{len(segments)} masks, valid depth pixels {valid_pixels}/{total_pixels} ({valid_ratio:.2%}), "
            )
            if mean_conf_all is not None:
                debug_msg += f"mean conf {mean_conf_all:.3f}"
            else:
                debug_msg += "mean conf N/A"
            if mean_conf_valid is not None:
                debug_msg += f", valid conf {mean_conf_valid:.3f}"
            else:
                debug_msg += ", no pixels above confidence threshold" if depth_conf_threshold is not None else ", no confidence map"
            print(debug_msg)
            if depth_min is not None:
                print(
                    f"[Debug] Frame {global_frame_idx:04d}: depth range [{depth_min:.3f}, {depth_max:.3f}], mean {depth_mean:.3f}"
                )
            else:
                print(f"[Debug] Frame {global_frame_idx:04d}: no valid depth samples after thresholding.")

        if full_reconstruction_accumulator is not None:
            if valid_pixels > 0:
                full_reconstruction_accumulator.append(world_points_frame[valid_mask].astype(np.float32, copy=False))
                if full_reconstruction_colors is not None and square_images is not None:
                    img = square_images[frame_idx]
                    if depth_frame is not None and img.shape[:2] != depth_frame.shape[:2]:
                        raise ValueError("Square image resolution does not match VGGT target size.")
                    full_reconstruction_colors.append(img[valid_mask].astype(np.float32, copy=False))
            elif debug_log:
                print(
                    f"[Debug] Frame {global_frame_idx:04d}: skipping full reconstruction contribution (no valid points)."
                )

        segment_list = segments or []
        if segment_list:
            label_indices: Dict[str, int] = {}
            pixel_labels = np.full(valid_mask.shape, -1, dtype=np.int32)
            pixel_scores = np.full(valid_mask.shape, -np.inf, dtype=np.float32)

            for seg in segment_list:
                seg_mask = seg.mask
                if seg_mask.shape != valid_mask.shape:
                    raise ValueError(
                        f"Segment mask shape {seg_mask.shape} does not match VGGT grid {valid_mask.shape}"
                    )
                mask = seg_mask & valid_mask
                mask_sum = int(mask.sum())
                if mask_sum < max(min_mask_area, 1):
                    if debug_log:
                        print(
                            f"[Debug]   Mask '{seg.label}' skipped: area {mask_sum} < min_mask_area {min_mask_area}."
                        )
                    continue
                better_mask = mask & (seg.score > pixel_scores)
                if not np.any(better_mask):
                    if debug_log:
                        print(
                            f"[Debug]   Mask '{seg.label}' skipped: no pixels beat existing assignments."
                        )
                    continue
                label_key = (seg.label or "unknown").strip()
                label_idx = label_indices.setdefault(label_key, len(label_indices))
                pixel_scores[better_mask] = seg.score
                pixel_labels[better_mask] = label_idx

            for label_key, label_idx in label_indices.items():
                assignment_mask = (pixel_labels == label_idx)
                assigned_pixels = int(assignment_mask.sum())
                if assigned_pixels == 0:
                    continue
                pts = world_points_frame[assignment_mask]
                if pts.size == 0:
                    continue
                label_to_points.setdefault(label_key, []).append(pts)
                stat = label_stats.setdefault(
                    label_key,
                    {"total_pixels": 0, "instances": 0, "frames": set()},
                )
                stat["total_pixels"] += assigned_pixels
                stat["instances"] += 1
                stat["frames"].add(global_frame_idx)

    concatenated: Dict[str, np.ndarray] = {}
    normalized_stats: Dict[str, Dict[str, float]] = {}
    for label, chunks in label_to_points.items():
        pts = np.concatenate(chunks, axis=0)
        if max_points_per_label > 0 and len(pts) > max_points_per_label:
            if debug_log:
                print(
                    f"[Debug]   Label '{label}' exceeds max_points_per_label ({len(pts)} > {max_points_per_label}); subsampling."
                )
            pts = subsample_points(pts, max_points_per_label)
        concatenated[label] = pts
        stats = label_stats[label]
        normalized_stats[label] = {
            "points": len(pts),
            "mask_pixels": float(stats["total_pixels"]),
            "unique_frames": len(stats["frames"]),
            "instances": float(stats["instances"]),
        }

    return concatenated, normalized_stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir: Path | None = None
    if args.save_vggt_debug or args.export_full_reconstruction:
        debug_dir = args.output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
    annotated_frames_dir: Path | None = None
    annotated_video_path: Path | None = None
    annotated_any = False
    annotated_frame_rate = args.video_fps if args.video_fps is not None and args.video_fps > 0 else 25.0
    if debug_dir is not None:
        annotated_frames_dir = debug_dir / "detections"
        annotated_video_path = debug_dir / "detections.mp4"
        annotated_frames_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.frames_dir) == bool(args.video_path):
        raise ValueError("Provide exactly one of --frames-dir or --video-path.")

    temp_dir: Path | None = None
    try:
        if args.video_path is not None:
            temp_dir = Path(tempfile.mkdtemp(prefix="vggt_pointcloud_frames_"))
            frame_paths, _ = extract_video_frames(args.video_path, temp_dir, args.video_fps)
            source_desc = str(args.video_path)
        else:
            frame_paths = list_frame_paths(args.frames_dir)  # type: ignore[arg-type]
            source_desc = str(args.frames_dir)

        print(f"Found {len(frame_paths)} frames from {source_desc}")

        # VGGT setup
        vggt_device, dtype, autocast_ctx = select_device_dtype(args.vggt_device)
        print(f"Running VGGT on device {vggt_device} with dtype {dtype}")
        vggt_model = load_vggt_model(args.vggt_checkpoint, vggt_device)

        images_tensor, original_coords_tensor = load_and_preprocess_images_square(
            [str(p) for p in frame_paths], target_size=args.img_load_resolution
        )
        images_tensor = images_tensor.to(device=vggt_device, dtype=dtype)
        original_coords_np = original_coords_tensor.cpu().numpy()

        extrinsic, intrinsic, depth_map, depth_conf = run_vggt(vggt_model, images_tensor, autocast_ctx, args.vggt_resolution)
        print("VGGT inference complete.")

        # Grounded SAM 2 setup
        seg_device = determine_seg_device(args.seg_device)
        print(f"Running Grounded SAM 2 on device {seg_device}")
        sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=str(seg_device))
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        dino_kwargs = {"trust_remote_code": True}
        dino_processor = AutoProcessor.from_pretrained(args.grounding_dino_model_id, **dino_kwargs)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.grounding_dino_model_id, **dino_kwargs
        ).to(seg_device)
        grounding_model.eval()

        square_images_np = images_tensor.detach().to("cpu", dtype=torch.float32).numpy()
        if square_images_np.ndim != 4:
            raise ValueError(f"Unexpected preprocessed image tensor shape {square_images_np.shape}")
        if square_images_np.shape[1] == 3:  # NCHW -> NHWC
            square_images_np = np.transpose(square_images_np, (0, 2, 3, 1))
        elif square_images_np.shape[-1] != 3:
            raise ValueError(f"Preprocessed image tensor must have 3 channels; got {square_images_np.shape}")

        debug_enabled = args.save_vggt_debug or args.export_full_reconstruction
        target_mask_size = args.vggt_resolution
        mask_annotator = sv.MaskAnnotator() if annotated_frames_dir is not None else None
        box_annotator = sv.BoxAnnotator() if annotated_frames_dir is not None else None
        label_annotator = sv.LabelAnnotator() if annotated_frames_dir is not None else None
        segments_per_frame: List[List[SegmentMask]] = []
        for idx, frame_path in enumerate(frame_paths):
            segs = run_grounded_sam2_on_frame(
                frame_path=frame_path,
                prompt=args.prompt,
                predictor=sam2_predictor,
                grounding_model=grounding_model,
                device=seg_device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                box_shrink_ratio=args.box_shrink_ratio,
                morph_kernel=args.morph_kernel,
                target_size=target_mask_size,
                processor=dino_processor,
                debug=debug_enabled,
            )
            segments_per_frame.append(segs)
            print(f"[{idx+1:04d}/{len(frame_paths):04d}] {frame_path.name}: {len(segs)} masks")
            if annotated_frames_dir is not None:
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    raise RuntimeError(f"Failed to read frame for annotation: {frame_path}")
                mask_original_list: List[np.ndarray] = []
                label_list: List[str] = []
                for seg in segs:
                    mask_original = square_mask_to_original(seg.mask, original_coords_np[idx])
                    if mask_original.size == 0:
                        continue
                    mask_original_list.append(mask_original)
                    label_list.append(seg.label or "object")
                annotated_frame = frame_bgr.copy()
                if mask_original_list:
                    mask_stack = np.stack(mask_original_list).astype(bool)
                    detections = sv.Detections(
                        xyxy=sv.mask_to_xyxy(mask_stack),
                        mask=mask_stack,
                        class_id=np.arange(len(mask_original_list), dtype=np.int32),
                    )
                    if mask_annotator is not None:
                        annotated_frame = mask_annotator.annotate(annotated_frame, detections)
                    if box_annotator is not None:
                        annotated_frame = box_annotator.annotate(annotated_frame, detections)
                    if label_annotator is not None:
                        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=label_list)
                annotated_path = annotated_frames_dir / f"{idx:05d}.jpg"
                if not cv2.imwrite(str(annotated_path), annotated_frame):
                    raise RuntimeError(f"Failed to write annotated frame {annotated_path}")
                annotated_any = True

        frame_debug_stats: List[Dict[str, Any]] = []
        full_recon_chunks: List[np.ndarray] | None = [] if args.export_full_reconstruction else None
        full_recon_color_chunks: List[np.ndarray] | None = [] if args.export_full_reconstruction else None

        label_to_points, label_stats = aggregate_point_clouds(
            frame_paths=frame_paths,
            segments_per_frame=segments_per_frame,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            depth_map=depth_map,
            depth_conf=depth_conf,
            depth_conf_threshold=args.depth_confidence_threshold,
            min_mask_area=args.min_mask_area,
            max_points_per_label=args.max_points_per_label,
            full_reconstruction_accumulator=full_recon_chunks,
            full_reconstruction_colors=full_recon_color_chunks,
            debug_log=debug_enabled,
            frame_debug_stats=frame_debug_stats,
            square_images=square_images_np,
        )

        if not label_to_points:
            print("No segmented points produced. Check prompts and thresholds.")
            return

        merged_points = []
        merged_colors = []
        for label, points in label_to_points.items():
            color = label_to_color(label)
            colors = np.repeat(color[None, :], len(points), axis=0)
            label_filename = f"{label.replace(' ', '_')}.ply"
            save_ply_ascii(args.output_dir / "per_label" / label_filename, points, colors)
            merged_points.append(points)
            merged_colors.append(colors)
            print(f"Saved {len(points)} points for label '{label}' to {label_filename}")

        if not args.no_merged_cloud:
            all_points = np.concatenate(merged_points, axis=0)
            all_colors = np.concatenate(merged_colors, axis=0)
            save_ply_ascii(args.output_dir / "merged_point_cloud.ply", all_points, all_colors)
            print(f"Merged point cloud written with {len(all_points)} points.")

        # Persist metadata
        label_stats_serializable = {
            label: {k: (list(v) if isinstance(v, set) else v) for k, v in stats.items()}
            for label, stats in label_stats.items()
        }
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(label_stats_serializable, f, indent=2)
        print(f"Metadata written to {args.output_dir / 'metadata.json'}")

        if args.save_vggt_debug and debug_dir is not None:
            debug_npz_path = debug_dir / "vggt_outputs.npz"
            np.savez_compressed(
                debug_npz_path,
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                depth_map=depth_map,
                depth_conf=depth_conf,
            )
            with (debug_dir / "vggt_frame_stats.json").open("w", encoding="utf-8") as f:
                json.dump(frame_debug_stats, f, indent=2)
            print(f"[Debug] Saved VGGT raw outputs to {debug_npz_path}")

        if args.export_full_reconstruction and full_recon_chunks is not None and full_recon_color_chunks is not None:
            if full_recon_chunks:
                full_points = np.concatenate(full_recon_chunks, axis=0)
                full_colors = np.concatenate(full_recon_color_chunks, axis=0)
                print(f"[Debug] Full reconstruction accumulated {len(full_points)} points before subsampling.")
                if args.max_full_reconstruction_points > 0 and len(full_points) > args.max_full_reconstruction_points:
                    print(
                        f"[Debug] Subsampling full reconstruction from {len(full_points)} to {args.max_full_reconstruction_points} points."
                    )
                    full_points, full_colors = subsample_points_with_colors(
                        full_points, full_colors, args.max_full_reconstruction_points
                    )
                full_colors_uint8 = np.clip(full_colors * 255.0, 0, 255).astype(np.uint8)
                full_recon_path = (debug_dir or args.output_dir) / "full_vggt_point_cloud.ply"
                save_ply_ascii(full_recon_path, full_points, full_colors_uint8)
                print(f"[Debug] Full VGGT point cloud written to {full_recon_path} ({len(full_points)} points).")
            else:
                print("[Debug] Full VGGT reconstruction requested but produced no valid points.")

        if annotated_frames_dir is not None and annotated_any and annotated_video_path is not None:
            create_video_from_images(str(annotated_frames_dir), str(annotated_video_path), frame_rate=annotated_frame_rate)
            print(f"[Debug] Detection video written to {annotated_video_path}")
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
