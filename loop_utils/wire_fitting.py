import itertools
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from grounded_sam2_vggt_pointcloud import save_ply_ascii
from loop_utils.segmentation_postprocess import load_points_from_ply


logger = logging.getLogger(__name__)


@dataclass
class PoleInstance:
    path: Path
    centroid: np.ndarray
    axes: np.ndarray  # columns are orthonormal basis vectors
    local_min: np.ndarray
    local_max: np.ndarray
    extents: np.ndarray
    height_axis: int

    @property
    def height(self) -> float:
        return float(self.extents[self.height_axis])

    @property
    def span_xy(self) -> Tuple[float, float]:
        lateral_axes = [idx for idx in range(3) if idx != self.height_axis]
        return tuple(float(self.extents[idx]) for idx in lateral_axes)

    def _local_to_world(self, local_pts: np.ndarray) -> np.ndarray:
        return self.centroid[None, :] + local_pts @ self.axes.T

    def top_corners(self) -> np.ndarray:
        lateral_axes = [idx for idx in range(3) if idx != self.height_axis]
        local_corners: List[np.ndarray] = []
        for bit_combo in range(4):
            corner = np.empty(3, dtype=np.float32)
            corner[self.height_axis] = self.local_max[self.height_axis]
            for axis_offset, axis_idx in enumerate(lateral_axes):
                use_max = (bit_combo >> axis_offset) & 1
                corner[axis_idx] = (
                    self.local_max[axis_idx] if use_max else self.local_min[axis_idx]
                )
            local_corners.append(corner)
        return self._local_to_world(np.stack(local_corners, axis=0))

    def all_corners(self) -> np.ndarray:
        local_corners = []
        for combo in itertools.product((0, 1), repeat=3):
            selector = np.array(combo, dtype=bool)
            local_corners.append(np.where(selector, self.local_max, self.local_min))
        return self._local_to_world(np.stack(local_corners, axis=0))


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


def _compute_oriented_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0).astype(np.float32)
    centered = points - centroid
    if len(points) < 2:
        axes = np.eye(3, dtype=np.float32)
    else:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axes = eigvecs[:, order].astype(np.float32)
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
    local = centered @ axes
    local_min = local.min(axis=0).astype(np.float32)
    local_max = local.max(axis=0).astype(np.float32)
    return centroid, axes, local_min, local_max


def _save_bbox_mesh(path: Path, corners: np.ndarray, color: Sequence[int]) -> None:
    base_color = np.asarray(color, dtype=np.uint8)[None, :]
    colors = np.repeat(base_color, len(corners), axis=0)
    save_ply_ascii(path, corners, colors)


def _load_poles(instance_dir: Path, min_points: int, bbox_color: Sequence[int]) -> List[PoleInstance]:
    poles: List[PoleInstance] = []
    skipped = 0
    for ply_path in sorted(instance_dir.glob("*.ply")):
        if ply_path.name.endswith("_wires.ply") or ply_path.name.endswith("_bbox.ply"):
            continue
        points, _ = load_points_from_ply(ply_path)
        if len(points) < min_points:
            skipped += 1
            logger.debug(
                "Skipping %s (only %d points, requires >= %d)",
                ply_path.name,
                len(points),
                min_points,
            )
            continue
        centroid, axes, local_min, local_max = _compute_oriented_bbox(points)
        extents = (local_max - local_min).astype(np.float32)
        height_axis = int(np.argmax(extents))
        pole = PoleInstance(
            path=ply_path,
            centroid=centroid,
            axes=axes,
            local_min=local_min,
            local_max=local_max,
            extents=extents,
            height_axis=height_axis,
        )
        bbox_path = ply_path.with_name(f"{ply_path.stem}_bbox.ply")
        _save_bbox_mesh(bbox_path, pole.all_corners(), bbox_color)
        logger.debug("Saved oriented bbox for %s to %s", ply_path.name, bbox_path.name)
        poles.append(pole)
    logger.info(
        "Loaded %d pole instances from %s (skipped %d underpopulated candidates)",
        len(poles),
        instance_dir,
        skipped,
    )
    return poles


def _filter_outlier_poles(poles: Sequence[PoleInstance], z_thresh: float) -> List[PoleInstance]:
    if len(poles) < 3 or z_thresh <= 0:
        logger.debug(
            "Skipping outlier filtering (num_poles=%d, z_thresh=%.2f)",
            len(poles),
            z_thresh,
        )
        return list(poles)
    heights = np.array([p.height for p in poles], dtype=np.float32)
    span_x = np.array([p.span_xy[0] for p in poles], dtype=np.float32)
    span_y = np.array([p.span_xy[1] for p in poles], dtype=np.float32)
    mask = _mad_mask(heights, z_thresh)
    mask &= _mad_mask(span_x, z_thresh)
    mask &= _mad_mask(span_y, z_thresh)
    filtered = [pole for pole, keep in zip(poles, mask) if keep]
    logger.info(
        "Outlier filter kept %d/%d poles using z_thresh=%.2f",
        len(filtered),
        len(poles),
        z_thresh,
    )
    return filtered


def _order_poles(poles: Sequence[PoleInstance]) -> Tuple[List[PoleInstance], np.ndarray]:
    if len(poles) <= 1:
        return list(poles), np.zeros(len(poles), dtype=np.int32)
    centroids = np.stack([pole.centroid for pole in poles], axis=0)
    start_idx = int(np.argmin(np.sum(centroids**2, axis=1)))
    visited = np.zeros(len(poles), dtype=bool)
    order_indices = [start_idx]
    visited[start_idx] = True
    logger.debug(
        "Ordering poles via nearest-neighbor walk starting from %s (idx=%d, centroid=%s)",
        poles[start_idx].path.name,
        start_idx,
        np.array2string(centroids[start_idx], precision=3),
    )
    for _ in range(len(poles) - 1):
        last_idx = order_indices[-1]
        deltas = centroids - centroids[last_idx]
        distances = np.linalg.norm(deltas, axis=1)
        distances[visited] = np.inf
        next_idx = int(np.argmin(distances))
        if not np.isfinite(distances[next_idx]):
            break
        order_indices.append(next_idx)
        visited[next_idx] = True
    ordered = [poles[i] for i in order_indices]
    return ordered, np.asarray(order_indices, dtype=np.int32)


def _connect_poles(
    poles: Sequence[PoleInstance],
    spacing_factor: Optional[float],
) -> List[WireConnection]:
    if len(poles) < 2:
        logger.warning("Cannot connect poles: need >=2 but have %d", len(poles))
        return []
    distances = [
        float(np.linalg.norm(poles[i + 1].centroid - poles[i].centroid))
        for i in range(len(poles) - 1)
    ]
    median_spacing = float(np.median(distances)) if distances else None
    edges: List[WireConnection] = []
    skipped = 0
    for i in range(len(poles) - 1):
        dist = distances[i]
        if (
            spacing_factor
            and spacing_factor > 0
            and median_spacing
            and median_spacing > 0
            and dist > spacing_factor * median_spacing
        ):
            skipped += 1
            continue
        edges.append(WireConnection(source=poles[i], target=poles[i + 1], distance=dist))
    median_str = f"{median_spacing:.3f}" if median_spacing is not None else "n/a"
    spacing_str = spacing_factor if spacing_factor is not None else "n/a"
    logger.info(
        "Created %d wire connections (median_spacing=%s, spacing_factor=%s, skipped=%d)",
        len(edges),
        median_str,
        spacing_str,
        skipped,
    )
    return edges


def _pole_up_direction(pole: PoleInstance) -> np.ndarray:
    axis = pole.axes[:, pole.height_axis].astype(np.float32)
    if axis[2] < 0:
        axis = -axis
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return axis / norm


def _connection_sag_direction(conn: WireConnection) -> np.ndarray:
    up_a = _pole_up_direction(conn.source)
    up_b = _pole_up_direction(conn.target)
    up = up_a + up_b
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        up = up_a
        norm = np.linalg.norm(up)
    if norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        norm = 1.0
    return -up / norm


def _compute_degree_map(connections: Sequence[WireConnection]) -> Dict[Path, int]:
    degree = Counter()
    for conn in connections:
        degree[conn.source.path] += 1
        degree[conn.target.path] += 1
    return dict(degree)


def _corner_indices_for_connection(
    conn: WireConnection,
    degree_map: Dict[Path, int],
) -> Sequence[int]:
    deg_source = degree_map.get(conn.source.path, 0)
    deg_target = degree_map.get(conn.target.path, 0)
    if deg_source > 1 and deg_target > 1:
        return (0, 1, 2, 3)
    direction = conn.target.centroid - conn.source.centroid
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return (0, 1)
    direction /= norm
    corners_a = conn.source.top_corners()
    corners_b = conn.target.top_corners()
    centroid_a = conn.source.centroid
    centroid_b = conn.target.centroid
    scores = []
    for idx in range(4):
        vec_a = corners_a[idx] - centroid_a
        vec_b = corners_b[idx] - centroid_b
        score = float(np.dot(vec_a, direction) - np.dot(vec_b, direction))
        scores.append(score)
    best = np.argsort(scores)[-2:]
    return tuple(int(i) for i in sorted(best))


def _sample_wire(
    p0: np.ndarray,
    p1: np.ndarray,
    samples: int,
    sag_fraction: float,
    sag_direction: np.ndarray,
) -> np.ndarray:
    samples = max(samples, 2)
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    pts = (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]
    if sag_fraction > 0:
        chord = p1 - p0
        vertical_component = np.dot(chord, sag_direction) * sag_direction
        horizontal_vec = chord - vertical_component
        horizontal = np.linalg.norm(horizontal_vec)
        sag_depth = sag_fraction * horizontal
        sag_curve = 4.0 * t * (1.0 - t)
        pts -= sag_direction[None, :] * (sag_depth * sag_curve)[:, None]
    return pts


def _build_wire_cloud(
    connections: Sequence[WireConnection],
    degree_map: Dict[Path, int],
    samples_per_segment: int,
    sag_fraction: float,
    color: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    if not connections:
        logger.debug("No connections provided; returning empty wire cloud")
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), []
    logger.info(
        "Sampling %d connections with %d samples/segment and sag_fraction=%.4f",
        len(connections),
        samples_per_segment,
        sag_fraction,
    )
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    metadata: List[dict] = []
    color_arr = np.asarray(color, dtype=np.uint8)
    for conn_idx, conn in enumerate(connections):
        sag_direction = _connection_sag_direction(conn)
        corners_a = conn.source.top_corners()
        corners_b = conn.target.top_corners()
        corner_indices = _corner_indices_for_connection(conn, degree_map)
        if len(corner_indices) < 4:
            logger.debug(
                "Connection %s -> %s uses %d corners (endpoint adjustment)",
                conn.source.path.name,
                conn.target.path.name,
                len(corner_indices),
            )
        for corner_idx in corner_indices:
            pts = _sample_wire(
                corners_a[corner_idx],
                corners_b[corner_idx],
                samples_per_segment,
                sag_fraction,
                sag_direction,
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
    wire_color: Sequence[int] = (90, 90, 90),
    bbox_color: Sequence[int] = (255, 85, 0),
) -> Optional[WireFittingResult]:
    logger.info(
        "Starting wire fitting for label='%s' (root=%s)",
        label_name,
        instance_root,
    )
    label_slug = label_name.replace(" ", "_")
    label_dir = instance_root / label_slug
    if not label_dir.exists():
        logger.warning("Label directory %s does not exist; skipping", label_dir)
        return None

    poles = _load_poles(label_dir, min_points=min_points, bbox_color=bbox_color)
    if len(poles) < 2:
        logger.warning(
            "Need at least two poles to fit wires; got %d in %s",
            len(poles),
            label_dir,
        )
        return None

    filtered = _filter_outlier_poles(poles, outlier_z_thresh)
    ordered, _ = _order_poles(filtered)
    connections = _connect_poles(ordered, spacing_factor=spacing_factor)
    if not connections:
        logger.warning("No valid connections found after filtering; aborting wire fit")
        return None

    degree_map = _compute_degree_map(connections)
    points, colors, wire_meta = _build_wire_cloud(
        connections,
        degree_map=degree_map,
        samples_per_segment=samples_per_segment,
        sag_fraction=sag_fraction,
        color=wire_color,
    )
    wire_path = label_dir / f"{label_slug}_wires.ply"
    save_ply_ascii(wire_path, points, colors)
    logger.info(
        "Saved wire cloud %s (%d points across %d connections)",
        wire_path,
        points.shape[0],
        len(connections),
    )

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
    logger.debug("Wire metadata written to %s", metadata_path)

    return WireFittingResult(
        wire_path=wire_path,
        metadata_path=metadata_path,
        num_connections=len(connections),
    )
