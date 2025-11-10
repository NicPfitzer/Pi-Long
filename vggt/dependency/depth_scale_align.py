import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AffineScale:
    s: float
    b: float


def bilinear_sample(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample depth at floating-point coordinates.

    depth: HxW
    xy: Nx2 in pixel coords (x, y)
    returns: N array
    """
    H, W = depth.shape[:2]
    x = xy[:, 0]
    y = xy[:, 1]

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    Ia = depth[y0, x0]
    Ib = depth[y1, x0]
    Ic = depth[y0, x1]
    Id = depth[y1, x1]

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def ransac_affine_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_iters: int = 2000,
    thresh: float = 0.02,
    min_inliers: int = 200,
    positive_scale: bool = True,
    scale_bounds: Tuple[float, float] | None = (1e-3, 1e3),
    random_state: int | None = 42,
) -> tuple[AffineScale, np.ndarray]:
    """Robustly fit y ≈ s*x + b with RANSAC.

    - thresh is relative error threshold on y: |y - (s x + b)| <= thresh * max(1, |y|).
    - Returns best (s, b) and boolean inlier mask.
    """
    assert x.ndim == 1 and y.ndim == 1 and x.shape == y.shape
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points for affine fit")
    rng = np.random.default_rng(random_state)
    best_inliers = None
    best_model = None
    best_count = -1

    for _ in range(max_iters):
        idx = rng.choice(n, size=2, replace=False)
        x_samp, y_samp = x[idx], y[idx]
        A = np.stack([x_samp, np.ones_like(x_samp)], axis=1)
        try:
            s, b = np.linalg.lstsq(A, y_samp, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        if positive_scale and s <= 0:
            continue
        if scale_bounds is not None and not (scale_bounds[0] <= s <= scale_bounds[1]):
            continue

        y_hat = s * x + b
        denom = np.maximum(1.0, np.abs(y))
        residual = np.abs(y - y_hat) / denom
        inliers = residual <= thresh
        count = int(inliers.sum())
        if count > best_count and count >= min_inliers:
            best_count = count
            best_inliers = inliers
            best_model = AffineScale(float(s), float(b))

    if best_inliers is None:
        # Fall back to ordinary least squares on all points
        A = np.stack([x, np.ones_like(x)], axis=1)
        s, b = np.linalg.lstsq(A, y, rcond=None)[0]
        best_model = AffineScale(float(s), float(b))
        best_inliers = np.ones_like(x, dtype=bool)
        return best_model, best_inliers

    # Refit on inliers
    A = np.stack([x[best_inliers], np.ones_like(x[best_inliers])], axis=1)
    s, b = np.linalg.lstsq(A, y[best_inliers], rcond=None)[0]
    return AffineScale(float(s), float(b)), best_inliers


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def per_frame_scale_correction(
    mono_depths: List[np.ndarray],
    sfm_depth_samples: Dict[int, Tuple[np.ndarray, np.ndarray]],
    global_affine: AffineScale,
    *,
    max_rel_adjust: float = 0.25,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-frame scale and optional shift corrections with smoothing.

    sfm_depth_samples maps frame_idx -> (x_mono, y_sfm) arrays for the observed points in that frame.
    We compute s_f from robust median ratio of (y - b)/x, then smooth and clamp relative change.

    Returns arrays s_factors and b_offsets of length = #frames.
    """
    num_frames = len(mono_depths)
    s0, b0 = global_affine.s, global_affine.b
    s_factors = np.ones(num_frames, dtype=np.float32)
    b_offsets = np.zeros(num_frames, dtype=np.float32)

    for f in range(num_frames):
        samples = sfm_depth_samples.get(f)
        if samples is None:
            s_factors[f] = 1.0
            b_offsets[f] = 0.0
            continue
        x_mono, y_sfm = samples
        mask = (x_mono > 0) & np.isfinite(x_mono) & np.isfinite(y_sfm)
        x_mono, y_sfm = x_mono[mask], y_sfm[mask]
        if x_mono.size < 20:
            s_factors[f] = 1.0
            b_offsets[f] = 0.0
            continue
        # Estimate per-frame scale around global affine
        # Prefer scale-only correction to avoid jittery shifts
        local_scale = np.median((y_sfm - b0) / np.maximum(x_mono, 1e-6))
        rel = np.clip(local_scale / max(s0, 1e-6), 1.0 - max_rel_adjust, 1.0 + max_rel_adjust)
        s_factors[f] = float(rel)
        b_offsets[f] = 0.0

    s_factors = moving_average(s_factors, smooth_window).astype(np.float32)
    return s_factors, b_offsets

