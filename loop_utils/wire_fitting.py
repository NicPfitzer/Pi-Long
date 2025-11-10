import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from grounded_sam2_vggt_pointcloud import save_ply_ascii
from loop_utils.segmentation_postprocess import load_points_from_ply


@dataclass
class PoleInstance:
    path: Path
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    @property
    def height(self) -> float:
        return float(self.bbox_max[2] - self.bbox_min[2])

    @property
    def span_xy(self) -> Tuple[float, float]:
        dims = self.bbox_max - self.bbox_min
        return float(dims[0]), float(dims[1])

    def top_corners(self) -> np.ndarray:
        x0, y0, z0 = self.bbox_min
        x1, y1, z1 = self.bbox_max
        return np.asarray(
            [
                [x0, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
                [x1, y0, z1],
            ],
            dtype=np.float32,
        )


@dataclass
class WireConnection:
    source: PoleInstance
    target: PoleInstance
    distance: float


@dataclass
class WireFittingResult:
    wire_path: Path
    metadata_path: Path
    num_connections: int


SUGGESTED_HEURISTICS = [
    "Fit a 2D minimum spanning tree across pole centroids to better capture gentle turns.",
    "Leverage camera pose ordering to break the network into monotonic sub-sequences before linking.",
    "Score candidate edges by both distance and pole heading similarity to avoid spurious cross-street wires.",
]


def _mad_mask(values: np.ndarray, z_thresh: float) -> np.ndarray:
    if z_thresh <= 0 or values.size == 0:
        return np.ones_like(values, dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-6:
        return np.ones_like(values, dtype=bool)
    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) <= z_thresh


def _load_poles(instance_dir: Path, min_points: int) -> List[PoleInstance]:
    poles: List[PoleInstance] = []
    for ply_path in sorted(instance_dir.glob("*.ply")):
        if ply_path.name.endswith("_wires.ply"):
            continue
        points, _ = load_points_from_ply(ply_path)
        if len(points) < min_points:
            continue
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        centroid = points.mean(axis=0)
        poles.append(
            PoleInstance(
                path=ply_path,
                centroid=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            )
        )
    return poles


def _filter_outlier_poles(poles: Sequence[PoleInstance], z_thresh: float) -> List[PoleInstance]:
    if len(poles) < 3 or z_thresh <= 0:
        return list(poles)
    heights = np.array([p.height for p in poles], dtype=np.float32)
    span_x = np.array([p.span_xy[0] for p in poles], dtype=np.float32)
    span_y = np.array([p.span_xy[1] for p in poles], dtype=np.float32)
    mask = _mad_mask(heights, z_thresh)
    mask &= _mad_mask(span_x, z_thresh)
    mask &= _mad_mask(span_y, z_thresh)
    return [pole for pole, keep in zip(poles, mask) if keep]


def _order_poles(poles: Sequence[PoleInstance]) -> Tuple[List[PoleInstance], np.ndarray]:
    if len(poles) <= 1:
        return list(poles), np.zeros(len(poles))
    xy = np.stack([pole.centroid[:2] for pole in poles], axis=0)
    centered = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    scores = centered @ axis
    order = np.argsort(scores)
    ordered = [poles[i] for i in order]
    return ordered, scores[order]


def _connect_poles(
    poles: Sequence[PoleInstance],
    spacing_factor: Optional[float],
) -> List[WireConnection]:
    if len(poles) < 2:
        return []
    distances = [
        float(np.linalg.norm(poles[i + 1].centroid - poles[i].centroid))
        for i in range(len(poles) - 1)
    ]
    median_spacing = float(np.median(distances)) if distances else None
    edges: List[WireConnection] = []
    for i in range(len(poles) - 1):
        dist = distances[i]
        if (
            spacing_factor
            and spacing_factor > 0
            and median_spacing
            and median_spacing > 0
            and dist > spacing_factor * median_spacing
        ):
            continue
        edges.append(WireConnection(source=poles[i], target=poles[i + 1], distance=dist))
    return edges


def _sample_wire(p0: np.ndarray, p1: np.ndarray, samples: int, sag_fraction: float) -> np.ndarray:
    samples = max(samples, 2)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    pts = (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]
    if sag_fraction > 0:
        horizontal = np.linalg.norm((p0 - p1)[:2])
        sag_depth = sag_fraction * horizontal
        sag_curve = 4.0 * t * (1.0 - t)
        pts[:, 2] -= sag_depth * sag_curve
    return pts


def _build_wire_cloud(
    connections: Sequence[WireConnection],
    samples_per_segment: int,
    sag_fraction: float,
    color: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    if not connections:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), []
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    metadata: List[dict] = []
    color_arr = np.asarray(color, dtype=np.uint8)
    for conn_idx, conn in enumerate(connections):
        corners_a = conn.source.top_corners()
        corners_b = conn.target.top_corners()
        for corner_idx in range(4):
            pts = _sample_wire(
                corners_a[corner_idx],
                corners_b[corner_idx],
                samples_per_segment,
                sag_fraction,
            )
            all_points.append(pts)
            all_colors.append(np.repeat(color_arr[None, :], len(pts), axis=0))
            metadata.append(
                {
                    "connection_id": conn_idx,
                    "corner": corner_idx,
                    "from_instance": conn.source.path.name,
                    "to_instance": conn.target.path.name,
                    "distance": conn.distance,
                }
            )
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0), metadata


def fit_electric_pole_wires(
    instance_root: Path,
    label_name: str = "electric pole",
    *,
    min_points: int = 150,
    outlier_z_thresh: float = 2.5,
    samples_per_segment: int = 32,
    sag_fraction: float = 0.025,
    spacing_factor: Optional[float] = 2.5,
) -> Optional[WireFittingResult]:
    label_slug = label_name.replace(" ", "_")
    label_dir = instance_root / label_slug
    if not label_dir.exists():
        return None

    poles = _load_poles(label_dir, min_points=min_points)
    if len(poles) < 2:
        return None

    filtered = _filter_outlier_poles(poles, outlier_z_thresh)
    ordered, _ = _order_poles(filtered)
    connections = _connect_poles(ordered, spacing_factor=spacing_factor)
    if not connections:
        return None

    points, colors, wire_meta = _build_wire_cloud(
        connections,
        samples_per_segment=samples_per_segment,
        sag_fraction=sag_fraction,
        color=(90, 90, 90),
    )
    wire_path = label_dir / f"{label_slug}_wires.ply"
    save_ply_ascii(wire_path, points, colors)

    metadata = {
        "label": label_name,
        "num_poles": len(filtered),
        "num_connections": len(connections),
        "wire_points": int(points.shape[0]),
        "connections": wire_meta,
        "heuristic_suggestions": SUGGESTED_HEURISTICS,
    }
    metadata_path = label_dir / f"{label_slug}_wires.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return WireFittingResult(
        wire_path=wire_path,
        metadata_path=metadata_path,
        num_connections=len(connections),
    )
