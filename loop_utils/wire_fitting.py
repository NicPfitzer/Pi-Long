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
    up_direction: np.ndarray

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
        up_dir = _pole_up_direction(self)
        all_corners = self.all_corners()
        if len(all_corners) <= 4:
            corners = all_corners
        else:
            local_faces = {True: [], False: []}
            for combo in itertools.product((0, 1), repeat=3):
                selector = np.array(combo, dtype=bool)
                local_corner = np.where(selector, self.local_max, self.local_min)
                local_faces[bool(combo[self.height_axis])].append(local_corner)
            face_candidates = []
            for key in (True, False):
                face = local_faces[key]
                if face:
                    face_candidates.append(self._local_to_world(np.stack(face, axis=0)))
            if face_candidates:
                face_scores = [float(np.mean(face @ up_dir)) for face in face_candidates]
                best_idx = int(np.argmax(face_scores))
                corners = face_candidates[best_idx]
            else:
                corners = all_corners
        projections = corners @ up_dir
        order = np.argsort(projections)[::-1]
        print(f"Corner Projections: {projections}")
        print(f"Corner Order: {order}")
        print(f"Corners: {corners}")
        keep = min(4, len(order))
        return corners[order[:keep]]

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



def _mad_mask(values: np.ndarray, z_thresh: float) -> np.ndarray:
    if z_thresh <= 0 or values.size == 0:
        return np.ones_like(values, dtype=bool)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-6:
        return np.ones_like(values, dtype=bool)
    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) <= z_thresh


def _normalize_vector(vector: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        logger.warning("Ignoring malformed up vector with shape %s", arr.shape)
        return None
    norm = np.linalg.norm(arr)
    if norm < 1e-6:
        logger.warning("Ignoring near-zero up vector %s", arr.tolist())
        return None
    return arr / norm


def _density_trim_points(
    points: np.ndarray,
    *,
    min_points_for_filter: int = 120,
    k_neighbors: int = 10,
    low_density_quantile: float = 0.2,
    radius_scale: float = 1.15,
    min_radius: float = 0.04,
    min_keep_ratio: float = 0.35,
) -> np.ndarray:
    num_points = len(points)
    if num_points == 0 or num_points < max(min_points_for_filter, k_neighbors + 1):
        return points
    try:
        from sklearn.neighbors import KDTree
    except ImportError:  # pragma: no cover - defensive fallback
        logger.debug("scikit-learn unavailable; skipping density trimming")
        return points
    pts = np.asarray(points, dtype=np.float32, order="C")
    k = min(k_neighbors, num_points - 1)
    if k < 1:
        return pts
    centered = pts - pts.mean(axis=0, keepdims=True)
    try:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return pts
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].astype(np.float32)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = centered @ axes
    spreads = np.ptp(local, axis=0)
    height_axis_idx = int(np.argmax(spreads))
    lateral_axes = [idx for idx in range(3) if idx != height_axis_idx]
    if len(lateral_axes) != 2:
        return pts
    lateral_coords = np.ascontiguousarray(local[:, lateral_axes], dtype=np.float32)
    tree = KDTree(lateral_coords)
    dists, _ = tree.query(lateral_coords, k=k + 1)
    kth = dists[:, -1]
    finite = kth[np.isfinite(kth)]
    if finite.size == 0:
        return pts
    typical_scale = float(np.median(finite))
    if typical_scale < 1e-6:
        typical_scale = float(np.max(finite))
    radius = max(typical_scale * radius_scale, min_radius)
    counts = tree.query_radius(lateral_coords, r=radius, count_only=True)
    if counts.size == 0:
        return pts
    density_threshold = int(np.floor(np.percentile(counts, low_density_quantile * 100.0)))
    density_threshold = max(density_threshold, 4)
    mask = counts >= density_threshold
    keep = int(np.count_nonzero(mask))
    min_keep = max(int(num_points * min_keep_ratio), 30)
    if keep < min_keep:
        logger.debug(
            "Density trimming skipped (%d/%d points < min_keep=%d)",
            keep,
            num_points,
            min_keep,
        )
        return pts
    logger.debug(
        "Density trimming kept %d/%d points before bbox fit (radius=%.3f, threshold=%d, lateral_axes=%s)",
        keep,
        num_points,
        radius,
        density_threshold,
        lateral_axes,
    )
    return pts[mask]


def _compute_oriented_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid_source = _density_trim_points(points)
    if len(centroid_source) == 0:
        centroid_source = points
    centroid = centroid_source.mean(axis=0).astype(np.float32)
    centered = centroid_source - centroid
    if len(centroid_source) < 2:
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


def _align_bbox_height_axis(
    axes: np.ndarray,
    local_min: np.ndarray,
    local_max: np.ndarray,
    preferred_up: np.ndarray,
    height_axis: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    up = np.asarray(preferred_up, dtype=np.float32).reshape(3)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        return axes, local_min, local_max, height_axis
    up /= norm
    aligned_axes = axes.copy()
    aligned_min = local_min.copy()
    aligned_max = local_max.copy()
    axis_alignment = np.abs(aligned_axes.T @ up)
    best_axis = int(np.argmax(axis_alignment))
    if best_axis != height_axis:
        aligned_axes[:, [height_axis, best_axis]] = aligned_axes[:, [best_axis, height_axis]]
        aligned_min[[height_axis, best_axis]] = aligned_min[[best_axis, height_axis]]
        aligned_max[[height_axis, best_axis]] = aligned_max[[best_axis, height_axis]]
        height_axis = best_axis
    if np.dot(aligned_axes[:, height_axis], up) < 0:
        aligned_axes[:, height_axis] *= -1.0
        min_val = aligned_min[height_axis].copy()
        max_val = aligned_max[height_axis].copy()
        aligned_min[height_axis] = -max_val
        aligned_max[height_axis] = -min_val
    return aligned_axes, aligned_min, aligned_max, height_axis


def _save_bbox_mesh(path: Path, corners: np.ndarray, color: Sequence[int]) -> None:
    base_color = np.asarray(color, dtype=np.uint8)[None, :]
    colors = np.repeat(base_color, len(corners), axis=0)
    save_ply_ascii(path, corners, colors)

def _load_poles(
    instance_dir: Path,
    min_points: int,
    bbox_color: Sequence[int],
    preferred_up: Optional[np.ndarray] = None,
) -> List[PoleInstance]:
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
        if preferred_up is not None:
            axes, local_min, local_max, height_axis = _align_bbox_height_axis(
                axes,
                local_min,
                local_max,
                preferred_up,
                height_axis,
            )
            extents = (local_max - local_min).astype(np.float32)
            up_dir = preferred_up.copy()
        else:
            up_dir = axes[:, height_axis].astype(np.float32)
            if up_dir[2] < 0:
                up_dir = -up_dir
            norm = np.linalg.norm(up_dir)
            if norm < 1e-6:
                up_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                up_dir /= norm
        pole = PoleInstance(
            path=ply_path,
            centroid=centroid,
            axes=axes,
            local_min=local_min,
            local_max=local_max,
            extents=extents,
            height_axis=height_axis,
            up_direction=up_dir,
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
    metric_masks = np.stack(
        (
            _mad_mask(heights, z_thresh),
            _mad_mask(span_x, z_thresh),
            _mad_mask(span_y, z_thresh),
        ),
        axis=1,
    )
    allowed_failures = 1
    fail_counts = np.count_nonzero(~metric_masks, axis=1)
    mask = fail_counts <= allowed_failures
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
    return pole.up_direction


def _connection_sag_direction(
    conn: WireConnection,
    *,
    world_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    if world_up is not None:
        up = np.asarray(world_up, dtype=np.float32).reshape(3)
        norm = np.linalg.norm(up)
        if norm >= 1e-6:
            return up / norm
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
    return up / norm


def _compute_degree_map(connections: Sequence[WireConnection]) -> Dict[Path, int]:
    degree = Counter()
    for conn in connections:
        degree[conn.source.path] += 1
        degree[conn.target.path] += 1
    return dict(degree)


def _match_corners_by_distance(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
    max_pairs: int,
) -> Sequence[Tuple[int, int]]:
    num_pairs = min(max_pairs, len(corners_a), len(corners_b))
    if num_pairs <= 0:
        return tuple()
    distances = np.linalg.norm(
        corners_a[:, None, :] - corners_b[None, :, :],
        axis=2,
    )
    indices_a = range(len(corners_a))
    indices_b = range(len(corners_b))
    best_pairs: Optional[Tuple[Tuple[int, int], ...]] = None
    best_cost = np.inf
    for subset_a in itertools.combinations(indices_a, num_pairs):
        for perm_b in itertools.permutations(indices_b, num_pairs):
            cost = 0.0
            for idx_a, idx_b in zip(subset_a, perm_b):
                cost += float(distances[idx_a, idx_b])
                if cost >= best_cost:
                    break
            else:
                best_cost = cost
                pairs = tuple((idx_a, idx_b) for idx_a, idx_b in zip(subset_a, perm_b))
                best_pairs = pairs
    if best_pairs is None:
        return ((0, 0),)
    return tuple(
        pair
        for pair, _ in sorted(
            ((pair, float(distances[pair[0], pair[1]])) for pair in best_pairs),
            key=lambda item: item[1],
        )
    )


def _corner_indices_for_connection(
    conn: WireConnection,
    degree_map: Dict[Path, int],
) -> Sequence[Tuple[int, int]]:
    deg_source = degree_map.get(conn.source.path, 0)
    deg_target = degree_map.get(conn.target.path, 0)
    corners_a = conn.source.top_corners()
    corners_b = conn.target.top_corners()
    if len(corners_a) != 4 or len(corners_b) != 4:
        return ((0, 0),)
    # Limit high-degree pole pairs to a single pair of span wires to avoid duplicates.
    max_pairs = 2
    if deg_source > 1 and deg_target > 1:
        logger.debug(
            "Clamping %s -> %s to %d corner pairs (deg_source=%d, deg_target=%d)",
            conn.source.path.name,
            conn.target.path.name,
            max_pairs,
            deg_source,
            deg_target,
        )
    return _match_corners_by_distance(corners_a, corners_b, max_pairs=max_pairs)


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
    *,
    world_up: Optional[np.ndarray] = None,
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
        sag_direction = _connection_sag_direction(conn, world_up=world_up)
        corners_a = conn.source.top_corners()
        corners_b = conn.target.top_corners()
        corner_pairs = _corner_indices_for_connection(conn, degree_map)
        expected_pairs = min(2, len(corners_a), len(corners_b))
        if len(corner_pairs) < expected_pairs:
            logger.debug(
                "Connection %s -> %s limited to %d/%d corner pairs: %s",
                conn.source.path.name,
                conn.target.path.name,
                len(corner_pairs),
                expected_pairs,
                corner_pairs,
            )
        for corner_src, corner_tgt in corner_pairs:
            pts = _sample_wire(
                corners_a[corner_src],
                corners_b[corner_tgt],
                samples_per_segment,
                sag_fraction,
                sag_direction,
            )
            all_points.append(pts)
            all_colors.append(np.repeat(color_arr[None, :], len(pts), axis=0))
            metadata.append(
                {
                    "connection_id": conn_idx,
                    "corner": corner_src,
                    "corner_source": corner_src,
                    "corner_target": corner_tgt,
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
    global_up: Optional[Sequence[float]] = None,
) -> Optional[WireFittingResult]:
    logger.info(
        "Starting wire fitting for label='%s' (root=%s)",
        label_name,
        instance_root,
    )
    normalized_up = _normalize_vector(global_up)
    print(normalized_up)
    if normalized_up is not None:
        logger.info("Using provided global up vector %s", np.array2string(normalized_up, precision=3))
    label_slug = label_name.replace(" ", "_")
    label_dir = instance_root / label_slug
    if not label_dir.exists():
        logger.warning("Label directory %s does not exist; skipping", label_dir)
        return None

    poles = _load_poles(
        label_dir,
        min_points=min_points,
        bbox_color=bbox_color,
        preferred_up=normalized_up,
    )
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
        world_up=normalized_up,
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
        "global_up_vector": normalized_up.tolist() if normalized_up is not None else None,
        "global_up_source": "reconstruction_chunks" if normalized_up is not None else "pole_bbox_principal_axis",
    }
    metadata_path = label_dir / f"{label_slug}_wires.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.debug("Wire metadata written to %s", metadata_path)

    return WireFittingResult(
        wire_path=wire_path,
        metadata_path=metadata_path,
        num_connections=len(connections),
    )
