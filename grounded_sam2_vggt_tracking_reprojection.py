#!/usr/bin/env python3
"""
VGGT-assisted Grounded SAM 2 tracking via geometric reprojection.

Instead of prompting SAM 2 on every frame, this script:
  1. Runs VGGT once to recover per-frame depth + camera parameters (with FP16-aware device selection).
  2. Periodically calls Grounding DINO + SAM 2 to detect/segment objects on key frames.
  3. Unprojects detections into world-space point sets and reprojects them onto the remaining frames.

The result is a lightweight tracking pipeline that reuses VGGT geometry to propagate masks,
reducing repeated SAM 2 calls while retaining the fp16 / autocast logic from the VGGT pointcloud workflow.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.ops import box_convert
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from utils.video_utils import create_video_from_images
from vggt.models.vggt import VGGT
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# --------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class SegmentMask:
    label: str
    score: float
    mask: np.ndarray  # bool array aligned with VGGT resolution (H, W)
    bbox_xyxy: np.ndarray  # (4,)


@dataclass
class WorldObject:
    object_id: int
    label: str
    score: float
    world_points: np.ndarray
    first_frame: int
    last_update_frame: int


# --------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track objects by combining VGGT geometry with periodic Grounded SAM 2 detections."
    )
    parser.add_argument("--frames-dir", type=Path, default=None, help="Directory containing ordered RGB frames.")
    parser.add_argument("--video-path", type=Path, default=None, help="Optional MP4/MOV file to extract frames from.")
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="When using --video-path, downsample frames to this FPS for VGGT geometry (omit to keep the source rate).",
    )
    parser.add_argument(
        "--tracking-video-fps",
        type=float,
        default=None,
        help="When using --video-path, optionally extract additional frames for tracking at this FPS (>= --video-fps).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Grounding prompt (lowercase, each object ending with a dot).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/vggt_gsam2_tracking"),
        help="Folder where annotated frames, videos, and metadata will be stored.",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Optional path for the output MP4. Defaults to <output-dir>/vggt_gsam2_tracking.mp4.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=25.0,
        help="Frame rate (fps) for the optional output video.",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=10,
        help="Run Grounded SAM 2 every N frames to refresh detections / discover new objects.",
    )
    parser.add_argument(
        "--track-iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold to associate new detections with existing world objects.",
    )
    parser.add_argument(
        "--min-project-pixels",
        type=int,
        default=64,
        help="Discard reprojected masks with fewer pixels than this value (after dilation).",
    )
    parser.add_argument(
        "--projection-dilation",
        type=int,
        default=3,
        help="Odd kernel size for dilating reprojected masks (>=1).",
    )
    parser.add_argument(
        "--max-points-per-object",
        type=int,
        default=50_000,
        help="Randomly subsample each object's world point set to this cap. Set <=0 to keep all points.",
    )
    parser.add_argument(
        "--depth-confidence-threshold",
        type=float,
        default=5.0,
        help="Minimum VGGT depth confidence required when collecting world points.",
    )
    parser.add_argument(
        "--save-mask-cache",
        action="store_true",
        help="If set, persist per-frame square masks as compressed NPZ files for downstream use.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for point subsampling / reprojection randomness.",
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
        help="Resolution used when running VGGT (must be divisible by 14).",
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
    parser.add_argument("--seg-device", type=str, default=None, help="Device for Grounded SAM 2 (defaults to CUDA if available).")

    # Grounding DINO options
    parser.add_argument(
        "--grounding-dino-model-id",
        type=str,
        default="rziga/mm_grounding_dino_large_all",
        help="Hugging Face repo id for Grounding DINO.",
    )
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
        help="Odd kernel size for morphological opening on SAM 2 masks (set to 1 to disable).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------


def list_frame_paths(frames_dir: Path) -> List[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"Frames directory must be a directory: {frames_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in image_extensions]
    frame_paths.sort(key=numeric_sort_key)
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


def numeric_sort_key(path: Path) -> Tuple:
    import re

    return tuple(int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name))


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

    images_resized = torch.nn.functional.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

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


def determine_seg_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


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


def mask_to_vggt_resolution(mask: np.ndarray, target_size: int) -> np.ndarray:
    """
    Reproduce load_and_preprocess_images_square padding/resizing so masks align with VGGT depth maps.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    height, width = mask.shape
    max_dim = max(width, height)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    square[top : top + height, left : left + width] = mask.astype(np.uint8)
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to an [x, y, z, w] quaternion.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {matrix.shape}")
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert an [x, y, z, w] quaternion to a 3x3 rotation matrix.
    """
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion of shape (4,), got {quat.shape}")
    x, y, z, w = quat / np.linalg.norm(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def slerp_quaternion(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical linear interpolation between two [x, y, z, w] quaternions.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Interpolation factor must be in [0,1], got {alpha}")

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        blended = q0 + alpha * (q1 - q0)
        return blended / np.linalg.norm(blended)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - alpha) * theta) / sin_theta
    w1 = np.sin(alpha * theta) / sin_theta
    return w0 * q0 + w1 * q1


def interpolate_extrinsic(ex0: np.ndarray, ex1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolate between two 3x4 camera extrinsics using quaternion slerp for rotation
    and linear interpolation for translation.
    """
    if ex0.shape != (3, 4) or ex1.shape != (3, 4):
        raise ValueError(f"Extrinsics must be 3x4, got {ex0.shape} and {ex1.shape}")
    if alpha <= 0.0:
        return ex0.astype(np.float32)
    if alpha >= 1.0:
        return ex1.astype(np.float32)
    r0 = ex0[:, :3]
    r1 = ex1[:, :3]
    t0 = ex0[:, 3]
    t1 = ex1[:, 3]
    q0 = rotation_matrix_to_quaternion(r0)
    q1 = rotation_matrix_to_quaternion(r1)
    q_interp = slerp_quaternion(q0, q1, alpha)
    r_interp = quaternion_to_rotation_matrix(q_interp)
    t_interp = (1.0 - alpha) * t0 + alpha * t1
    extrinsic = np.concatenate([r_interp.astype(np.float32), t_interp.astype(np.float32)[:, None]], axis=1)
    return extrinsic


def compute_original_coords(path: Path, target_size: int) -> np.ndarray:
    """
    Compute the padding metadata matching load_and_preprocess_images_square for a single image.
    """
    with Image.open(path) as img:
        width, height = img.size
    max_dim = max(width, height)
    if max_dim == 0:
        return np.zeros(6, dtype=np.float32)
    left = (max_dim - width) // 2
    top = (max_dim - height) // 2
    scale = target_size / max_dim
    x1 = left * scale
    y1 = top * scale
    x2 = (left + width) * scale
    y2 = (top + height) * scale
    return np.array([x1, y1, x2, y2, float(width), float(height)], dtype=np.float32)


def select_geometry_indices(timestamps: Sequence[float], target_fps: Optional[float]) -> List[int]:
    """
    Choose frame indices approximating a desired FPS given monotonically increasing timestamps.
    """
    total = len(timestamps)
    if total == 0:
        return []
    if target_fps is None or target_fps <= 0:
        return list(range(total))
    interval = 1.0 / target_fps
    indices = [0]
    next_time = timestamps[0] + interval
    epsilon = 1e-6
    for idx in range(1, total):
        t = timestamps[idx]
        if t + epsilon >= next_time:
            indices.append(idx)
            while next_time <= t + epsilon:
                next_time += interval
    if indices[-1] != total - 1:
        indices.append(total - 1)
    # Remove potential duplicates while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    return unique_indices


def label_to_color(label: str) -> np.ndarray:
    digest = hashlib_md5(label)
    return np.frombuffer(digest[:3], dtype=np.uint8)


def hashlib_md5(text: str) -> bytes:
    import hashlib

    return hashlib.md5(text.encode("utf-8")).digest()


def subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def extract_world_points_from_mask(
    mask: np.ndarray,
    frame_idx: int,
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth_conf_threshold: float,
) -> np.ndarray:
    depth_frame = depth_map[frame_idx, ..., 0] if depth_map.shape[-1] == 1 else depth_map[frame_idx]
    world_points, _, valid_depth_mask = depth_to_world_coords_points(
        depth_frame, extrinsic[frame_idx], intrinsic[frame_idx]
    )
    combined_mask = valid_depth_mask
    if depth_conf is not None:
        conf_frame = depth_conf[frame_idx]
        combined_mask &= conf_frame >= depth_conf_threshold
    combined_mask &= mask
    if not np.any(combined_mask):
        return np.empty((0, 3), dtype=np.float32)
    pts = world_points[combined_mask]
    return pts.astype(np.float32)


def project_world_points_to_mask(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    mask_shape: Tuple[int, int],
    min_pixels: int,
    dilation: int,
) -> Optional[np.ndarray]:
    if points_world.size == 0:
        return None

    pts_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=points_world.dtype)], axis=1)
    cam_pts = pts_h @ extrinsic.T
    z = cam_pts[:, 2]
    positive = z > 1e-5
    if not np.any(positive):
        return None
    cam_pts = cam_pts[positive]
    pixels = (intrinsic @ cam_pts.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    xs = np.round(pixels[:, 0]).astype(int)
    ys = np.round(pixels[:, 1]).astype(int)
    height, width = mask_shape
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(in_bounds):
        return None
    xs = xs[in_bounds]
    ys = ys[in_bounds]

    mask_uint8 = np.zeros(mask_shape, dtype=np.uint8)
    coords = np.stack([xs, ys], axis=1)
    if len(coords) >= 3:
        hull = cv2.convexHull(coords.astype(np.int32))
        cv2.fillConvexPoly(mask_uint8, hull, 1)
    mask_uint8[ys, xs] = 1

    if dilation > 1:
        if dilation % 2 == 0:
            raise ValueError("projection-dilation must be odd.")
        kernel = np.ones((dilation, dilation), dtype=np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, kernel)

    if mask_uint8.sum() < max(min_pixels, 1):
        return None
    return mask_uint8.astype(bool)


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


def merge_points(existing: np.ndarray, new_points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if existing.size == 0:
        merged = new_points
    else:
        merged = np.concatenate([existing, new_points], axis=0)
    if max_points > 0 and len(merged) > max_points:
        idx = rng.choice(len(merged), size=max_points, replace=False)
        merged = merged[idx]
    return merged.astype(np.float32)


def render_annotated_frame(
    frame_bgr: np.ndarray,
    mask_list: List[np.ndarray],
    object_ids: List[int],
    objects: Dict[int, WorldObject],
) -> np.ndarray:
    if not mask_list:
        return frame_bgr
    mask_stack = np.stack(mask_list).astype(bool)
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(mask_stack),
        mask=mask_stack,
        class_id=np.array(object_ids, dtype=np.int32),
    )
    annotated = frame_bgr.copy()
    mask_annotator = sv.MaskAnnotator()
    annotated = mask_annotator.annotate(scene=annotated, detections=detections)
    box_annotator = sv.BoxAnnotator()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    label_annotator = sv.LabelAnnotator()
    labels = [objects[obj_id].label for obj_id in object_ids]
    annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
    return annotated


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
            continue
        results.append(
            SegmentMask(
                label=label.strip(),
                score=float(score),
                mask=mask_vggt,
                bbox_xyxy=bbox,
            )
        )
    return results


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_out_dir = args.output_dir / "annotated_frames"
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir = args.output_dir / "mask_cache"
    if args.save_mask_cache:
        masks_out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.frames_dir) == bool(args.video_path):
        raise ValueError("Provide exactly one of --frames-dir or --video-path.")

    temp_dir: Path | None = None
    try:
        if args.video_path is not None:
            temp_dir = Path(tempfile.mkdtemp(prefix="vggt_tracking_frames_"))
            tracking_dir = temp_dir / "tracking_frames"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            tracking_fps = args.tracking_video_fps if args.tracking_video_fps is not None else args.video_fps
            tracking_frame_paths, tracking_timestamps = extract_video_frames(
                args.video_path, tracking_dir, tracking_fps
            )
            geometry_fps = args.video_fps
            if (
                geometry_fps is not None
                and tracking_fps is not None
                and geometry_fps > tracking_fps + 1e-6
            ):
                raise ValueError("--tracking-video-fps must be >= --video-fps when both are provided.")
            geometry_indices = select_geometry_indices(tracking_timestamps, geometry_fps)
            geometry_frame_paths = [tracking_frame_paths[idx] for idx in geometry_indices]
            geometry_timestamps = [tracking_timestamps[idx] for idx in geometry_indices]
            source_desc = str(args.video_path)
        else:
            tracking_frame_paths = list_frame_paths(args.frames_dir)  # type: ignore[arg-type]
            tracking_timestamps = [float(i) for i in range(len(tracking_frame_paths))]
            geometry_frame_paths = tracking_frame_paths
            geometry_timestamps = tracking_timestamps
            geometry_indices = list(range(len(tracking_frame_paths)))
            source_desc = str(args.frames_dir)

        if not tracking_frame_paths:
            raise RuntimeError("No frames available for tracking.")
        if not geometry_frame_paths:
            raise RuntimeError("No frames selected for VGGT geometry estimation.")

        print(f"Found {len(tracking_frame_paths)} tracking frames from {source_desc}")
        if len(geometry_frame_paths) != len(tracking_frame_paths):
            print(f"Using {len(geometry_frame_paths)} frames for VGGT geometry (downsampled subset).")

        geometry_index_by_tracking = {track_idx: geo_idx for geo_idx, track_idx in enumerate(geometry_indices)}

        # VGGT setup on geometry subset
        vggt_device, dtype, autocast_ctx = select_device_dtype(args.vggt_device)
        print(f"Running VGGT on device {vggt_device} with dtype {dtype}")
        vggt_model = load_vggt_model(args.vggt_checkpoint, vggt_device)

        geometry_images_tensor, geometry_coords_tensor = load_and_preprocess_images_square(
            [str(p) for p in geometry_frame_paths], target_size=args.img_load_resolution
        )
        geometry_images_tensor = geometry_images_tensor.to(device=vggt_device, dtype=dtype)

        extrinsic_geo, intrinsic_geo, depth_map_geo, depth_conf_geo = run_vggt(
            vggt_model, geometry_images_tensor, autocast_ctx, args.vggt_resolution
        )
        print("VGGT inference complete on geometry subset.")

        geometry_original_coords = geometry_coords_tensor.cpu().numpy()
        geometry_extrinsic = extrinsic_geo.astype(np.float32)
        geometry_intrinsic = intrinsic_geo.astype(np.float32)
        geometry_depth_map = depth_map_geo.astype(np.float32)
        geometry_depth_conf = depth_conf_geo.astype(np.float32)

        # Prepare per-frame metadata for tracking timeline
        total_frames = len(tracking_frame_paths)
        tracking_original_coords = np.zeros((total_frames, 6), dtype=np.float32)
        for geo_idx, track_idx in enumerate(geometry_indices):
            tracking_original_coords[track_idx] = geometry_original_coords[geo_idx]
        for idx in range(total_frames):
            if idx not in geometry_index_by_tracking:
                tracking_original_coords[idx] = compute_original_coords(
                    tracking_frame_paths[idx], args.img_load_resolution
                )

        geometry_timestamps_np = np.array(geometry_timestamps, dtype=np.float64)
        tracking_timestamps_np = np.array(tracking_timestamps, dtype=np.float64)
        tracking_extrinsics = np.zeros((total_frames, 3, 4), dtype=np.float32)
        tracking_intrinsics = np.zeros((total_frames, 3, 3), dtype=np.float32)

        for idx in range(total_frames):
            t = tracking_timestamps_np[idx]
            geo_idx = geometry_index_by_tracking.get(idx)
            if geo_idx is not None:
                tracking_extrinsics[idx] = geometry_extrinsic[geo_idx]
                tracking_intrinsics[idx] = geometry_intrinsic[geo_idx]
                continue

            if len(geometry_extrinsic) == 1:
                tracking_extrinsics[idx] = geometry_extrinsic[0]
                tracking_intrinsics[idx] = geometry_intrinsic[0]
                continue

            pos = int(np.searchsorted(geometry_timestamps_np, t))
            if pos <= 0:
                tracking_extrinsics[idx] = geometry_extrinsic[0]
                tracking_intrinsics[idx] = geometry_intrinsic[0]
            elif pos >= len(geometry_timestamps_np):
                tracking_extrinsics[idx] = geometry_extrinsic[-1]
                tracking_intrinsics[idx] = geometry_intrinsic[-1]
            else:
                t0 = geometry_timestamps_np[pos - 1]
                t1 = geometry_timestamps_np[pos]
                if abs(t1 - t0) < 1e-6:
                    alpha = 0.0
                else:
                    alpha = float((t - t0) / (t1 - t0))
                tracking_extrinsics[idx] = interpolate_extrinsic(
                    geometry_extrinsic[pos - 1], geometry_extrinsic[pos], alpha
                )
                tracking_intrinsics[idx] = (
                    (1.0 - alpha) * geometry_intrinsic[pos - 1] + alpha * geometry_intrinsic[pos]
                ).astype(np.float32)

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

        mask_shape = (args.vggt_resolution, args.vggt_resolution)
        rng = np.random.default_rng(args.random_seed)

        objects: Dict[int, WorldObject] = {}
        next_object_id = 0
        frame_index_entries: List[str] = []

        for frame_idx, frame_path in enumerate(tracking_frame_paths):
            geo_idx = geometry_index_by_tracking.get(frame_idx)
            detection_trigger = geo_idx is not None and (geo_idx % max(args.detection_interval, 1) == 0)
            frame_masks_square: Dict[int, np.ndarray] = {}

            if detection_trigger:
                geo_idx_int = int(geo_idx)
                segments = run_grounded_sam2_on_frame(
                    frame_path=frame_path,
                    prompt=args.prompt,
                    predictor=sam2_predictor,
                    grounding_model=grounding_model,
                    device=seg_device,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    box_shrink_ratio=args.box_shrink_ratio,
                    morph_kernel=args.morph_kernel,
                    target_size=args.vggt_resolution,
                    processor=dino_processor,
                )
                print(
                    f"[Frame {frame_idx:05d} | geom {geo_idx_int:05d}] Grounded SAM 2 detections: {len(segments)}"
                )

                existing_projection: Dict[int, np.ndarray] = {}
                for obj_id, obj in objects.items():
                    if obj.first_frame > frame_idx:
                        continue
                    projected = project_world_points_to_mask(
                        obj.world_points,
                        tracking_extrinsics[frame_idx],
                        tracking_intrinsics[frame_idx],
                        mask_shape,
                        args.min_project_pixels,
                        args.projection_dilation,
                    )
                    if projected is not None:
                        existing_projection[obj_id] = projected

                occupied_ids = set()
                for seg in segments:
                    label = seg.label if seg.label else args.prompt.strip()
                    world_pts = extract_world_points_from_mask(
                        seg.mask,
                        geo_idx_int,
                        geometry_depth_map,
                        geometry_depth_conf,
                        geometry_extrinsic,
                        geometry_intrinsic,
                        args.depth_confidence_threshold,
                    )
                    if world_pts.size == 0:
                        continue
                    world_pts = subsample_points(world_pts, args.max_points_per_object, rng).astype(np.float32)

                    best_id = None
                    best_iou = 0.0
                    for obj_id, proj_mask in existing_projection.items():
                        iou = mask_iou(seg.mask, proj_mask)
                        if iou > best_iou:
                            best_iou = iou
                            best_id = obj_id

                    if best_id is not None and best_iou >= args.track_iou_threshold:
                        obj = objects[best_id]
                        obj.world_points = merge_points(obj.world_points, world_pts, args.max_points_per_object, rng)
                        obj.score = max(obj.score, seg.score)
                        obj.last_update_frame = frame_idx
                        frame_masks_square[best_id] = seg.mask
                        occupied_ids.add(best_id)
                    else:
                        obj_id = next_object_id
                        next_object_id += 1
                        objects[obj_id] = WorldObject(
                            object_id=obj_id,
                            label=label,
                            score=seg.score,
                            world_points=world_pts,
                            first_frame=frame_idx,
                            last_update_frame=frame_idx,
                        )
                        frame_masks_square[obj_id] = seg.mask
                        occupied_ids.add(obj_id)

                for obj_id, obj in objects.items():
                    if obj.first_frame > frame_idx or obj_id in occupied_ids:
                        continue
                    projected = existing_projection.get(obj_id)
                    if projected is None:
                        projected = project_world_points_to_mask(
                            obj.world_points,
                            tracking_extrinsics[frame_idx],
                            tracking_intrinsics[frame_idx],
                            mask_shape,
                            args.min_project_pixels,
                            args.projection_dilation,
                        )
                    if projected is not None:
                        frame_masks_square[obj_id] = projected
            else:
                for obj_id, obj in objects.items():
                    if obj.first_frame > frame_idx:
                        continue
                    projected = project_world_points_to_mask(
                        obj.world_points,
                        tracking_extrinsics[frame_idx],
                        tracking_intrinsics[frame_idx],
                        mask_shape,
                        args.min_project_pixels,
                        args.projection_dilation,
                    )
                    if projected is not None:
                        frame_masks_square[obj_id] = projected

            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read frame {frame_path}")

            mask_original_list: List[np.ndarray] = []
            mask_obj_ids: List[int] = []
            coords_row = tracking_original_coords[frame_idx]
            for obj_id, mask_sq in frame_masks_square.items():
                mask_original = square_mask_to_original(mask_sq, coords_row)
                if mask_original.size == 0:
                    continue
                if mask_original.sum() < max(args.min_project_pixels, 1):
                    continue
                mask_original_list.append(mask_original)
                mask_obj_ids.append(obj_id)

            annotated = render_annotated_frame(frame_bgr, mask_original_list, mask_obj_ids, objects)
            output_path = frames_out_dir / f"frame_{frame_idx:05d}.jpg"
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"Failed to write annotated frame {output_path}")
            frame_index_entries.append(str(output_path))

            if args.save_mask_cache and frame_masks_square:
                cache_path = masks_out_dir / f"frame_{frame_idx:05d}.npz"
                obj_ids_arr = np.array(list(frame_masks_square.keys()), dtype=np.int32)
                masks_stack = np.stack(list(frame_masks_square.values()), axis=0).astype(bool)
                np.savez_compressed(cache_path, object_ids=obj_ids_arr, masks=masks_stack)

            print(f"[Frame {frame_idx:05d}] Tracked objects: {len(mask_obj_ids)} / {len(objects)} total")

        metadata = [
            {
                "object_id": obj_id,
                "label": obj.label,
                "score": float(obj.score),
                "first_frame": int(obj.first_frame),
                "last_update_frame": int(obj.last_update_frame),
                "points": int(len(obj.world_points)),
            }
            for obj_id, obj in objects.items()
        ]
        with (args.output_dir / "tracked_objects.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if frame_index_entries:
            index_path = args.output_dir / "annotated_frames_index.json"
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(frame_index_entries, f, indent=2)

        if args.output_video is not None:
            output_video_path = args.output_video
        else:
            output_video_path = str(args.output_dir / "vggt_gsam2_tracking.mp4")

        print(f"Writing video to {output_video_path}")
        create_video_from_images(str(frames_out_dir), output_video_path, frame_rate=args.frame_rate)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
