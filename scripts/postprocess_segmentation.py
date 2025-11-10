#!/usr/bin/env python3

import argparse
from pathlib import Path

from loop_utils.segmentation_postprocess import (
    run_segmentation_clustering,
    save_instance_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DBSCAN clustering over segmented per-label point clouds."
    )
    parser.add_argument(
        "--segmentation_dir",
        type=Path,
        required=True,
        help="Path to Pi-Long segmentation output (contains 'per_label').",
    )
    parser.add_argument("--eps", type=float, default=0.4, help="DBSCAN epsilon radius.")
    parser.add_argument(
        "--min-samples", type=int, default=30, help="DBSCAN min_samples parameter."
    )
    parser.add_argument(
        "--min-cluster-points",
        type=int,
        default=500,
        help="Drop clusters smaller than this size.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Optional voxel downsampling size before clustering.",
    )
    parser.add_argument(
        "--max-instances-per-label",
        type=int,
        default=None,
        help="Upper-bound the number of clusters saved per label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seg_dir = args.segmentation_dir
    per_label_dir = seg_dir / "per_label"
    if not per_label_dir.exists():
        raise FileNotFoundError(f"No per_label directory found in {seg_dir}")

    cfg = {
        "eps": args.eps,
        "min_samples": args.min_samples,
        "min_cluster_points": args.min_cluster_points,
        "voxel_size": args.voxel_size,
        "max_instances_per_label": args.max_instances_per_label,
    }
    instance_root = seg_dir / "instances"
    summary = run_segmentation_clustering(per_label_dir, instance_root, cfg)
    if summary:
        metadata_path = save_instance_metadata(seg_dir, summary)
        print(f"[Clustering] Saved instance metadata to {metadata_path}")
    else:
        print("[Clustering] No clusters produced; nothing saved.")


if __name__ == "__main__":
    main()
