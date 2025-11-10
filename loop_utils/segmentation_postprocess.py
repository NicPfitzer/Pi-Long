import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData

from grounded_sam2_vggt_pointcloud import label_to_color, save_ply_ascii


@dataclass
class ClusteringParams:
    eps: float = 0.4
    min_samples: int = 30
    min_cluster_points: int = 500
    voxel_size: Optional[float] = None
    max_instances_per_label: Optional[int] = None


def load_points_from_ply(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    pts = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    color_available = all(ch in vertex.data.dtype.names for ch in ("red", "green", "blue"))
    cols = None
    if color_available:
        cols = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)
    return pts, cols


def _voxel_downsample(
    points: np.ndarray, colors: Optional[np.ndarray], voxel_size: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size is None or voxel_size <= 0:
        return points, colors
    if len(points) == 0:
        return points, colors
    scaled = np.floor(points / voxel_size).astype(np.int64)
    _, inverse_indices, counts = np.unique(
        scaled, axis=0, return_inverse=True, return_counts=True
    )
    accum_pts = np.zeros((counts.size, 3), dtype=np.float64)
    np.add.at(accum_pts, inverse_indices, points)
    accum_pts /= counts[:, None]
    if colors is not None:
        accum_cols = np.zeros((counts.size, 3), dtype=np.float64)
        np.add.at(accum_cols, inverse_indices, colors)
        accum_cols = np.clip(accum_cols / counts[:, None], 0, 255).astype(np.uint8)
    else:
        accum_cols = None
    return accum_pts.astype(np.float32), accum_cols


def _cluster_points(points: np.ndarray, cfg: ClusteringParams) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    if len(points) == 0:
        return np.empty(0, dtype=np.int32)
    db = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, n_jobs=-1)
    return db.fit_predict(points)


def run_segmentation_clustering(
    per_label_dir: Path,
    instance_root: Path,
    cfg_mapping: Mapping[str, object],
) -> Dict[str, List[Dict[str, object]]]:
    instance_root.mkdir(parents=True, exist_ok=True)
    cfg = ClusteringParams(
        eps=float(cfg_mapping.get("eps", ClusteringParams.eps)),
        min_samples=int(cfg_mapping.get("min_samples", ClusteringParams.min_samples)),
        min_cluster_points=int(
            cfg_mapping.get("min_cluster_points", ClusteringParams.min_cluster_points)
        ),
        voxel_size=(
            float(cfg_mapping["voxel_size"]) if cfg_mapping.get("voxel_size") else None
        ),
        max_instances_per_label=(
            int(cfg_mapping["max_instances_per_label"])
            if cfg_mapping.get("max_instances_per_label")
            else None
        ),
    )

    print(
        "[Clustering] Running DBSCAN with "
        f"eps={cfg.eps} (max neighborhood radius), "
        f"min_samples={cfg.min_samples}, "
        f"min_cluster_points={cfg.min_cluster_points}, "
        f"voxel_size={cfg.voxel_size}, "
        f"max_instances_per_label={cfg.max_instances_per_label}"
    )

    instance_metadata: Dict[str, List[Dict[str, object]]] = {}
    for label_ply in sorted(per_label_dir.glob("*.ply")):
        label_name = label_ply.stem.replace("_", " ")
        pts, cols = load_points_from_ply(label_ply)
        print(f"[Clustering] Label '{label_name}': {len(pts)} raw points from {label_ply.name}")
        pts, cols = _voxel_downsample(pts, cols, cfg.voxel_size or 0.0)
        print(f"[Clustering] Label '{label_name}': {len(pts)} points after voxelization")
        if len(pts) == 0:
            print(f"[Clustering] Label '{label_name}': no points after preprocessing, skipping.")
            continue

        labels = _cluster_points(pts, cfg)
        if labels.size == 0:
            print(f"[Clustering] Label '{label_name}': DBSCAN returned no labels, skipping.")
            continue

        clusters = [
            cluster_label
            for cluster_label in np.unique(labels)
            if cluster_label != -1
        ]
        if not clusters:
            print(f"[Clustering] Label '{label_name}': only noise detected, skipping.")
            continue

        clusters_sorted = sorted(
            clusters, key=lambda cl: int(np.sum(labels == cl)), reverse=True
        )

        if cfg.max_instances_per_label:
            clusters_sorted = clusters_sorted[: cfg.max_instances_per_label]

            label_dir = instance_root / label_ply.stem
        label_dir.mkdir(parents=True, exist_ok=True)
        label_color = np.asarray(label_to_color(label_name), dtype=np.uint8)

        instances: List[Dict[str, object]] = []
        for inst_idx, cluster_label in enumerate(clusters_sorted):
            mask = labels == cluster_label
            cluster_size = int(mask.sum())
            if cluster_size < cfg.min_cluster_points:
                print(
                    f"[Clustering] Label '{label_name}' cluster {cluster_label} "
                    f"dropped ({cluster_size} pts < min_cluster_points={cfg.min_cluster_points})"
                )
                continue
            inst_points = pts[mask]
            if cols is not None:
                inst_colors = cols[mask]
            else:
                inst_colors = np.repeat(label_color[None, :], cluster_size, axis=0)

            instance_path = label_dir / f"{label_ply.stem}_inst_{inst_idx:03d}.ply"
            save_ply_ascii(instance_path, inst_points, inst_colors)
            print(
                f"[Clustering] Label '{label_name}': saved instance {inst_idx} with {cluster_size} points "
                f"to {instance_path.relative_to(instance_root.parent)}"
            )
            instances.append(
                {
                    "instance": inst_idx,
                    "points": cluster_size,
                    "centroid": inst_points.mean(axis=0).tolist(),
                    "path": str(instance_path.relative_to(instance_root.parent)),
                }
            )
        if instances:
            instance_metadata[label_name] = instances
        else:
            print(f"[Clustering] Label '{label_name}': no clusters met the size threshold.")
    return instance_metadata


def save_instance_metadata(base_dir: Path, instances: Dict[str, List[Dict[str, object]]]) -> Path:
    metadata_path = base_dir / "instances.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(instances, fp, indent=2)
    return metadata_path


def aggregate_instance_cloud(instance_root: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not instance_root.exists():
        return None
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    for ply_path in sorted(instance_root.rglob("*.ply")):
        pts, cols = load_points_from_ply(ply_path)
        if len(pts) == 0:
            continue
        if cols is None:
            cols = np.zeros((len(pts), 3), dtype=np.uint8)
        all_points.append(pts)
        all_colors.append(cols)
    if not all_points:
        return None
    return np.concatenate(all_points, axis=0), np.concatenate(all_colors, axis=0)
