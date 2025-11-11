import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from loop_utils.segmentation_postprocess import load_points_from_ply
from loop_utils.wire_fitting import (
    PoleInstance,
    _align_bbox_height_axis,
    _compute_oriented_bbox,
    _density_trim_points,
    _normalize_vector,
)


logger = logging.getLogger(__name__)
EPS = 1e-6


@dataclass
class PoleFilterConfig:
    min_points: int = 150
    height_bounds: Optional[Tuple[float, float]] = None
    slenderness_min: float = 5.0
    profile_height_bins: int = 12
    profile_radial_bins: int = 6
    min_reference: int = 4
    reference_feature_z: float = 1.0
    feature_z_thresh: float = 3.0
    profile_z_thresh: float = 3.5
    enforce_features: Tuple[str, ...] = ("height", "slenderness")
    strict_slenderness: float = 6.0
    min_profile_mad: float = 5e-4
    min_feature_mad: float = 5e-4
    summary_filename: str = "{label_slug}_filtering.json"
    density_trim_enable: bool = True
    density_trim_kwargs: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "PoleFilterConfig":
        if not mapping:
            return cls()
        enforce_features = mapping.get("enforce_features")
        if enforce_features is not None:
            enforce_tuple = tuple(str(name) for name in enforce_features)
        else:
            enforce_tuple = cls.enforce_features
        height_bounds = mapping.get("height_bounds")
        hb: Optional[Tuple[float, float]]
        if height_bounds is not None:
            if len(height_bounds) == 2:
                hb = (float(height_bounds[0]), float(height_bounds[1]))
            else:
                raise ValueError(
                    "height_bounds must provide exactly two values (min, max) when specified."
                )
        else:
            hb = cls.height_bounds
        density_cfg = mapping.get("density_trim", cls.density_trim_enable)
        density_enable = cls.density_trim_enable
        density_kwargs: Dict[str, object] = {}
        if isinstance(density_cfg, Mapping):
            density_enable = bool(density_cfg.get("enable", cls.density_trim_enable))
            allowed_keys = {
                "min_points_for_filter",
                "k_neighbors",
                "low_density_quantile",
                "radius_scale",
                "min_radius",
                "min_keep_ratio",
            }
            for key in allowed_keys:
                if key in density_cfg:
                    density_kwargs[key] = density_cfg[key]
        else:
            density_enable = bool(density_cfg)
        return cls(
            min_points=int(mapping.get("min_points", cls.min_points)),
            height_bounds=hb,
            slenderness_min=float(mapping.get("slenderness_min", cls.slenderness_min)),
            profile_height_bins=int(mapping.get("profile_height_bins", cls.profile_height_bins)),
            profile_radial_bins=int(mapping.get("profile_radial_bins", cls.profile_radial_bins)),
            min_reference=int(mapping.get("min_reference", cls.min_reference)),
            reference_feature_z=float(mapping.get("reference_feature_z", cls.reference_feature_z)),
            feature_z_thresh=float(mapping.get("feature_z_thresh", cls.feature_z_thresh)),
            profile_z_thresh=float(mapping.get("profile_z_thresh", cls.profile_z_thresh)),
            enforce_features=enforce_tuple,
            strict_slenderness=float(mapping.get("strict_slenderness", cls.strict_slenderness)),
            min_profile_mad=float(mapping.get("min_profile_mad", cls.min_profile_mad)),
            min_feature_mad=float(mapping.get("min_feature_mad", cls.min_feature_mad)),
            summary_filename=str(mapping.get("summary_filename", cls.summary_filename)),
            density_trim_enable=density_enable,
            density_trim_kwargs=density_kwargs,
        )


@dataclass
class FeatureStat:
    median: float
    mad: float


@dataclass
class PoleProfile:
    pole: PoleInstance
    path: Path
    profile: np.ndarray
    features: Dict[str, float]
    num_points: int
    profile_score: float = 0.0
    profile_peak: float = 0.0
    feature_z: Dict[str, float] = field(default_factory=dict)
    keep: bool = False
    reason: str = ""


@dataclass
class TemplateStats:
    profile_median: np.ndarray
    profile_mad: np.ndarray
    feature_stats: Dict[str, FeatureStat]
    reference_indices: np.ndarray


@dataclass
class PoleFilterResult:
    kept_profiles: List[PoleProfile]
    dropped_profiles: List[PoleProfile]
    summary_path: Path
    template: TemplateStats

    @property
    def kept_instances(self) -> List[PoleInstance]:
        return [profile.pole for profile in self.kept_profiles]

    @property
    def num_candidates(self) -> int:
        return len(self.kept_profiles) + len(self.dropped_profiles)


def _label_slug(label: str) -> str:
    return label.replace(" ", "_")


def _localize_points(points: np.ndarray, pole: PoleInstance) -> np.ndarray:
    return (points - pole.centroid[None, :]) @ pole.axes


def _compute_profile(points: np.ndarray, pole: PoleInstance, cfg: PoleFilterConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    local = _localize_points(points, pole)
    height_axis = pole.height_axis
    height_vals = local[:, height_axis]
    height_min = float(pole.local_min[height_axis])
    height_extent = float(pole.local_max[height_axis] - pole.local_min[height_axis])
    height_extent = max(height_extent, EPS)
    normalized_height = np.clip((height_vals - height_min) / height_extent, 0.0, 1.0)
    lateral_axes = [idx for idx in range(3) if idx != height_axis]
    if len(lateral_axes) != 2:
        lateral_axes = [idx for idx in range(3)]
        lateral_axes.remove(height_axis)
    lateral_coords = local[:, lateral_axes]
    radial = np.linalg.norm(lateral_coords, axis=1)
    span_values = [float(pole.local_max[idx] - pole.local_min[idx]) for idx in lateral_axes]
    max_span = max(max(span_values), EPS)
    min_span = max(min(span_values), EPS)
    radius_scale = max_span * 0.5
    radius_scale = max(radius_scale, 0.05)
    normalized_radius = np.clip(radial / radius_scale, 0.0, 1.0)
    hist, _, _ = np.histogram2d(
        normalized_height,
        normalized_radius,
        bins=(cfg.profile_height_bins, cfg.profile_radial_bins),
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    if total > 0:
        hist /= total
    profile_vec = hist.reshape(-1)
    features = {
        "height": height_extent,
        "max_span": max_span,
        "min_span": min_span,
        "slenderness": height_extent / max_span,
        "span_ratio": max_span / min_span,
        "density": float(points.shape[0]) / max(height_extent * max_span * min_span, EPS),
        "top_fraction": float(np.mean(normalized_height >= 0.7)),
        "mid_fraction": float(np.mean((normalized_height >= 0.35) & (normalized_height <= 0.65))),
        "radial_mean": float(np.mean(normalized_radius)),
        "points": int(points.shape[0]),
    }
    return profile_vec, features


def _load_profiles(
    label_dir: Path,
    cfg: PoleFilterConfig,
    preferred_up: Optional[np.ndarray],
) -> List[PoleProfile]:
    profiles: List[PoleProfile] = []
    for ply_path in sorted(label_dir.glob("*.ply")):
        if ply_path.name.endswith("_bbox.ply") or ply_path.name.endswith("_wires.ply"):
            continue
        points, _ = load_points_from_ply(ply_path)
        if len(points) < cfg.min_points:
            logger.debug(
                "Skipping %s (%d pts < min_points=%d)",
                ply_path.name,
                len(points),
                cfg.min_points,
            )
            continue
        if cfg.density_trim_enable:
            trimmed = _density_trim_points(points, **cfg.density_trim_kwargs)
            if len(trimmed) < cfg.min_points:
                logger.debug(
                    "Density trim dropped %s below min_points (%d -> %d < %d)",
                    ply_path.name,
                    len(points),
                    len(trimmed),
                    cfg.min_points,
                )
                continue
            if len(trimmed) != len(points):
                logger.debug(
                    "Density trim kept %d/%d pts for %s",
                    len(trimmed),
                    len(points),
                    ply_path.name,
                )
            points = trimmed
        centroid, axes, local_min, local_max = _compute_oriented_bbox(points)
        extents = (local_max - local_min).astype(np.float32)
        height_axis = int(np.argmax(extents))
        axes_aligned = axes
        local_min_aligned = local_min
        local_max_aligned = local_max
        if preferred_up is not None:
            axes_aligned, local_min_aligned, local_max_aligned, height_axis = _align_bbox_height_axis(
                axes,
                local_min,
                local_max,
                preferred_up,
                height_axis,
            )
            extents = (local_max_aligned - local_min_aligned).astype(np.float32)
            up_dir = preferred_up.copy()
        else:
            up_dir = axes_aligned[:, height_axis]
            if up_dir[2] < 0:
                up_dir = -up_dir
            norm = np.linalg.norm(up_dir)
            if norm < EPS:
                up_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                up_dir = up_dir / norm
        pole = PoleInstance(
            path=ply_path,
            centroid=centroid,
            axes=axes_aligned,
            local_min=local_min_aligned,
            local_max=local_max_aligned,
            extents=extents,
            height_axis=height_axis,
            up_direction=up_dir,
        )
        profile_vec, features = _compute_profile(points, pole, cfg)
        profiles.append(
            PoleProfile(
                pole=pole,
                path=ply_path,
                profile=profile_vec,
                features=features,
                num_points=len(points),
            )
        )
    return profiles


def _build_template(profiles: List[PoleProfile], cfg: PoleFilterConfig) -> TemplateStats:
    if not profiles:
        raise ValueError("No profiles provided for template creation")
    profile_matrix = np.stack([prof.profile for prof in profiles], axis=0)
    feature_arrays: Dict[str, np.ndarray] = {}
    for key in profiles[0].features.keys():
        feature_arrays[key] = np.asarray([prof.features.get(key, 0.0) for prof in profiles], dtype=np.float32)
    feature_stats: Dict[str, FeatureStat] = {}
    for name, arr in feature_arrays.items():
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad < cfg.min_feature_mad:
            mad = cfg.min_feature_mad
        feature_stats[name] = FeatureStat(median=median, mad=mad)
    reference_mask = np.ones(len(profiles), dtype=bool)
    if cfg.reference_feature_z > 0:
        for name in cfg.enforce_features:
            stats = feature_stats.get(name)
            if stats is None or stats.mad <= 0:
                continue
            diffs = np.abs(feature_arrays[name] - stats.median)
            reference_mask &= diffs <= (cfg.reference_feature_z * stats.mad)
    if "slenderness" in feature_arrays and cfg.strict_slenderness > 0:
        reference_mask &= feature_arrays["slenderness"] >= cfg.strict_slenderness
    reference_indices = np.nonzero(reference_mask)[0]
    if reference_indices.size < cfg.min_reference:
        sorted_indices = np.argsort(-feature_arrays.get("slenderness", np.arange(len(profiles), dtype=np.float32)))
        reference_indices = sorted_indices[: cfg.min_reference]
    profile_subset = profile_matrix[reference_indices]
    profile_median = np.median(profile_subset, axis=0)
    profile_mad = np.median(np.abs(profile_subset - profile_median), axis=0)
    profile_mad = np.maximum(profile_mad, cfg.min_profile_mad)
    return TemplateStats(
        profile_median=profile_median.astype(np.float32),
        profile_mad=profile_mad.astype(np.float32),
        feature_stats=feature_stats,
        reference_indices=reference_indices,
    )


def _evaluate_profiles(
    profiles: List[PoleProfile],
    template: TemplateStats,
    cfg: PoleFilterConfig,
) -> Tuple[List[PoleProfile], List[PoleProfile]]:
    kept: List[PoleProfile] = []
    dropped: List[PoleProfile] = []
    profile_mad = template.profile_mad
    for profile in profiles:
        diff = np.abs(profile.profile - template.profile_median)
        scores = diff / profile_mad
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            profile_score = 0.0
            profile_peak = 0.0
        else:
            profile_score = float(np.mean(finite_scores))
            profile_peak = float(np.max(finite_scores))
        profile.profile_score = profile_score
        profile.profile_peak = profile_peak
        feature_failure: Optional[str] = None
        for name, stats in template.feature_stats.items():
            value = profile.features.get(name)
            if value is None:
                continue
            if stats.mad <= 0:
                z = 0.0
            else:
                z = abs(value - stats.median) / stats.mad
            profile.feature_z[name] = float(z)
            if name in cfg.enforce_features and z > cfg.feature_z_thresh and feature_failure is None:
                feature_failure = f"{name}_z={z:.2f}"
        height = profile.features.get("height", 0.0)
        slenderness = profile.features.get("slenderness", 0.0)
        if cfg.height_bounds and not (cfg.height_bounds[0] <= height <= cfg.height_bounds[1]):
            profile.keep = False
            profile.reason = "height_bounds"
        elif slenderness < cfg.slenderness_min:
            profile.keep = False
            profile.reason = "slenderness_min"
        elif feature_failure:
            profile.keep = False
            profile.reason = feature_failure
        elif profile_score > cfg.profile_z_thresh:
            profile.keep = False
            profile.reason = f"profile_z={profile_score:.2f}"
        else:
            profile.keep = True
            profile.reason = "kept"
        if profile.keep:
            kept.append(profile)
        else:
            dropped.append(profile)
    return kept, dropped


def _write_summary(
    label_dir: Path,
    label: str,
    profiles: List[PoleProfile],
    cfg: PoleFilterConfig,
    template: TemplateStats,
) -> Path:
    summary_filename = cfg.summary_filename.format(label_slug=_label_slug(label))
    summary_path = label_dir / summary_filename
    summary = {
        "label": label,
        "config": asdict(cfg),
        "num_candidates": len(profiles),
        "num_kept": int(sum(1 for p in profiles if p.keep)),
        "num_dropped": int(sum(1 for p in profiles if not p.keep)),
        "template": {
            "reference_count": int(template.reference_indices.size),
            "feature_stats": {
                name: {"median": stat.median, "mad": stat.mad}
                for name, stat in template.feature_stats.items()
            },
        },
        "instances": [],
    }
    for profile in profiles:
        summary["instances"].append(
            {
                "path": profile.path.name,
                "keep": profile.keep,
                "reason": profile.reason,
                "features": {key: float(val) for key, val in profile.features.items()},
                "feature_z": {key: float(val) for key, val in profile.feature_z.items()},
                "profile_score": profile.profile_score,
                "profile_peak": profile.profile_peak,
            }
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def filter_electric_pole_instances(
    *,
    instance_root: Path,
    label: str = "electric pole",
    config: Optional[Mapping[str, object]] = None,
    global_up: Optional[Sequence[float]] = None,
) -> Optional[PoleFilterResult]:
    cfg = PoleFilterConfig.from_mapping(config or {})
    label_dir = instance_root / _label_slug(label)
    if not label_dir.exists():
        logger.warning("Label directory %s does not exist; skipping pole filtering", label_dir)
        return None
    preferred_up = _normalize_vector(global_up)
    profiles = _load_profiles(label_dir, cfg, preferred_up)
    if not profiles:
        logger.info("No pole instances to filter under %s", label_dir)
        return None
    template = _build_template(profiles, cfg)
    kept, dropped = _evaluate_profiles(profiles, template, cfg)
    summary_path = _write_summary(label_dir, label, profiles, cfg, template)
    logger.info(
        "Pole filter kept %d/%d '%s' instances (summary: %s)",
        len(kept),
        len(profiles),
        label,
        summary_path,
    )
    return PoleFilterResult(
        kept_profiles=kept,
        dropped_profiles=dropped,
        summary_path=summary_path,
        template=template,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter electric pole instances before wire fitting.")
    parser.add_argument("--instance-root", type=Path, required=True, help="Path to clustered instance root directory.")
    parser.add_argument("--label", type=str, default="electric pole", help="Label name to filter.")
    parser.add_argument("--min-points", type=int, default=None, help="Minimum points per instance.")
    parser.add_argument("--height-bounds", type=float, nargs=2, default=None, help="Expected height range.")
    parser.add_argument("--slenderness-min", type=float, default=None, help="Minimum slenderness ratio.")
    parser.add_argument("--profile-z-thresh", type=float, default=None, help="Maximum allowed profile z-score.")
    parser.add_argument("--feature-z-thresh", type=float, default=None, help="Maximum allowed feature z-score.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    args = parser.parse_args()
    config: Dict[str, object] = {}
    if args.min_points is not None:
        config["min_points"] = args.min_points
    if args.height_bounds is not None:
        config["height_bounds"] = args.height_bounds
    if args.slenderness_min is not None:
        config["slenderness_min"] = args.slenderness_min
    if args.profile_z_thresh is not None:
        config["profile_z_thresh"] = args.profile_z_thresh
    if args.feature_z_thresh is not None:
        config["feature_z_thresh"] = args.feature_z_thresh
    result = filter_electric_pole_instances(
        instance_root=args.instance_root,
        label=args.label,
        config=config,
    )
    if not result:
        print("Pole filtering finished with no candidates.")
    else:
        print(
            f"Pole filtering kept {len(result.kept_profiles)}/{result.num_candidates} "
            f"instances. Summary saved to {result.summary_path}"
        )


if __name__ == "__main__":
    main()
