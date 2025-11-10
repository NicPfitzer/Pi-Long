"""Utilities for estimating the global up-direction by fitting a floor plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return vec
    return vec / norm


@dataclass
class FloorDetectorConfig:
    enable: bool = True
    max_points: int = 120_000
    max_iterations: int = 600
    distance_threshold: float = 0.08
    min_inlier_ratio: float = 0.02
    min_inliers: int = 2_000
    min_candidate_points: int = 5_000
    confidence_coef: float = 0.4
    alignment_weight: float = 0.2
    early_stop_ratio: float = 0.75
    random_seed: Optional[int] = None
    fallback_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    max_refine_points: int = 300_000

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, object]]) -> "FloorDetectorConfig":
        if mapping is None:
            return cls()
        return cls(
            enable=bool(mapping.get("enable", True)),
            max_points=int(mapping.get("max_points", cls.max_points)),
            max_iterations=int(mapping.get("max_iterations", cls.max_iterations)),
            distance_threshold=float(mapping.get("distance_threshold", cls.distance_threshold)),
            min_inlier_ratio=float(mapping.get("min_inlier_ratio", cls.min_inlier_ratio)),
            min_inliers=int(mapping.get("min_inliers", cls.min_inliers)),
            min_candidate_points=int(mapping.get("min_candidate_points", cls.min_candidate_points)),
            confidence_coef=float(mapping.get("confidence_coef", cls.confidence_coef)),
            alignment_weight=float(mapping.get("alignment_weight", cls.alignment_weight)),
            early_stop_ratio=float(mapping.get("early_stop_ratio", cls.early_stop_ratio)),
            random_seed=(
                int(mapping["random_seed"])
                if mapping.get("random_seed") is not None
                else None
            ),
            fallback_up=tuple(mapping.get("fallback_up", cls.fallback_up)),
            max_refine_points=int(mapping.get("max_refine_points", cls.max_refine_points)),
        )

    @property
    def fallback_up_array(self) -> np.ndarray:
        return _normalize(np.asarray(self.fallback_up, dtype=np.float32))


def _prepare_points(
    points: np.ndarray,
    conf: Optional[np.ndarray],
    cfg: FloorDetectorConfig,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])
    mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[mask]
    if conf is not None:
        conf_arr = np.asarray(conf, dtype=np.float32).reshape(-1)
        conf_arr = conf_arr[mask]
    else:
        conf_arr = None

    if conf_arr is not None and cfg.confidence_coef > 0:
        finite_mask = np.isfinite(conf_arr)
        conf_arr = conf_arr[finite_mask]
        pts = pts[finite_mask]
        if conf_arr.size > 0:
            threshold = float(np.nanmean(conf_arr) * cfg.confidence_coef)
            keep = conf_arr >= threshold
            pts = pts[keep]
            conf_arr = conf_arr[keep]

    return pts


def _best_plane_ransac(
    pts: np.ndarray,
    cfg: FloorDetectorConfig,
    target_dir: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray], int]:
    if len(pts) < 3:
        return None, None, None, 0
    best_normal = None
    best_offset = None
    best_mask = None
    best_score = -np.inf
    min_required = max(int(len(pts) * cfg.min_inlier_ratio), 3)
    iterations = 0

    for iterations in range(1, cfg.max_iterations + 1):
        idx = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        offset = float(np.dot(normal, p0))
        distances = np.abs(pts @ normal - offset)
        mask = distances <= cfg.distance_threshold
        inliers = int(mask.sum())
        if inliers < max(min_required, cfg.min_inliers):
            continue
        alignment = (
            abs(float(np.dot(normal, target_dir))) if target_dir is not None else 0.0
        )
        score = inliers + alignment * cfg.alignment_weight * len(pts)
        if score > best_score:
            best_score = score
            best_normal = normal.copy()
            best_offset = offset
            best_mask = mask
            if inliers / len(pts) >= cfg.early_stop_ratio:
                break

    return best_normal, best_offset, best_mask, iterations


def _refine_normal(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) < 3:
        return _normalize(centered.mean(axis=0))
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    return _normalize(normal)


def estimate_floor_up_direction(
    points: np.ndarray,
    conf: Optional[np.ndarray],
    cfg: Optional[FloorDetectorConfig] = None,
    prev_up: Optional[Sequence[float]] = None,
) -> Tuple[Optional[np.ndarray], dict]:
    cfg = cfg or FloorDetectorConfig()
    pts = _prepare_points(points, conf, cfg)
    result_meta = {
        "candidate_points": int(len(pts)),
        "distance_threshold": float(cfg.distance_threshold),
    }
    if len(pts) < cfg.min_candidate_points:
        result_meta["reason"] = "insufficient_points"
        result_meta["success"] = False
        return None, result_meta

    rng = np.random.default_rng(cfg.random_seed)
    if len(pts) > cfg.max_points:
        idx = rng.choice(len(pts), size=cfg.max_points, replace=False)
        pts_sample = pts[idx]
    else:
        pts_sample = pts

    target_dir = None
    if prev_up is not None:
        target_dir = _normalize(np.asarray(prev_up, dtype=np.float32))
    elif cfg.fallback_up is not None:
        target_dir = cfg.fallback_up_array

    normal, offset, mask, iterations = _best_plane_ransac(pts_sample, cfg, target_dir, rng)
    result_meta["iterations"] = int(iterations)
    if normal is None or offset is None or mask is None:
        result_meta["reason"] = "no_plane"
        result_meta["success"] = False
        return None, result_meta

    full_distances = np.abs(pts @ normal - offset)
    inlier_mask = full_distances <= cfg.distance_threshold
    inliers = pts[inlier_mask]
    result_meta["inliers"] = int(len(inliers))
    result_meta["inlier_ratio"] = float(len(inliers) / len(pts))
    if len(inliers) < cfg.min_inliers:
        result_meta["reason"] = "not_enough_inliers"
        result_meta["success"] = False
        return None, result_meta

    if len(inliers) > cfg.max_refine_points:
        idx = rng.choice(len(inliers), size=cfg.max_refine_points, replace=False)
        inliers = inliers[idx]

    refined = _refine_normal(inliers)
    if target_dir is not None and np.dot(refined, target_dir) < 0:
        refined = -refined

    result_meta["success"] = True
    return refined.astype(np.float32), result_meta
