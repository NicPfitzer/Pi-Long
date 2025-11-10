import argparse
import gc
import os
import shutil
import sys
from datetime import datetime
from typing import Optional, Sequence

import torch

from loop_utils.config_utils import load_config
from loop_utils.sim3utils import merge_ply_files, warmup_numba

from .pipeline import Pi_Long


def copy_file(src_path: str, dst_dir: str) -> Optional[str]:
    try:
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        print(f"Config yaml file has been copied to: {dst_path}")
        return dst_path
    except FileNotFoundError:
        print("File not found")
    except PermissionError:
        print("Permission error while copying config file")
    except Exception as exc:
        print(f"Copy error: {exc}")
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT-Long")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/base_config.yaml",
        help="Path to pipeline configuration yaml",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="./exps",
        help="Root directory where experiment outputs are placed",
    )
    return parser.parse_args(argv)


def prepare_save_dir(image_dir: str, config_path: str, exp_root: str) -> str:
    os.makedirs(exp_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    safe_image_name = image_dir.replace("/", "_")
    save_dir = os.path.join(exp_root, safe_image_name, timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"The experiment will be saved under: {save_dir}")
        copy_file(config_path, save_dir)
    return save_dir


def run_from_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    save_dir = prepare_save_dir(args.image_dir, args.config, args.exp_root)

    if config["Model"]["align_method"] == "numba":
        warmup_numba()

    pipeline = Pi_Long(args.image_dir, save_dir, config)
    pipeline.run()
    pipeline.close()

    del pipeline
    torch.cuda.empty_cache()
    gc.collect()

    all_ply_path = os.path.join(save_dir, "pcd", "combined_pcd.ply")
    input_dir = os.path.join(save_dir, "pcd")
    print("Saving all the point clouds")
    merge_ply_files(input_dir, all_ply_path)
    print("VGGT Long done.")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> None:
    sys.exit(run_from_cli(argv))

