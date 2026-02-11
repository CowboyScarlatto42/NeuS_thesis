#!/usr/bin/env python3
"""
Dataset builder for NeuS (from COLMAP subset).

Input:
  src_dataset/
    images/
    masks/
    preprocessed/cameras_sphere.npz

Output:
  out_dir/
    image/        (singolare)
    mask/         (singolare)
    cameras_spe3r.npz
"""

import argparse
import json
import re
import shutil
from pathlib import Path
import numpy as np


def parse_split_file(split_path: Path) -> list[int]:
    idxs = []
    for raw in split_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.search(r"(\d+)", line)
        if not m:
            raise ValueError(f"No index found in line: '{line}'")
        idxs.append(int(m.group(1)))
    return idxs


def copy_reindexed(src_folder: Path, dst_folder: Path, orig_indices: list[int]):
    dst_folder.mkdir(parents=True, exist_ok=True)
    for new_i, orig_i in enumerate(orig_indices):
        src = src_folder / f"{orig_i:03d}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing file: {src}")
        dst = dst_folder / f"{new_i:03d}.png"
        shutil.copy2(src, dst)


def subset_cameras_npz(src_npz: Path, dst_npz: Path, orig_indices: list[int]):
    data = np.load(src_npz)
    out = {}

    for new_i, orig_i in enumerate(orig_indices):
        w_key = f"world_mat_{orig_i}"
        s_key = f"scale_mat_{orig_i}"

        if w_key not in data or s_key not in data:
            raise KeyError(f"Missing keys '{w_key}'/'{s_key}'")

        out[f"world_mat_{new_i}"] = data[w_key]
        out[f"scale_mat_{new_i}"] = data[s_key]

        # optional inverse mats
        w_inv_key = f"world_mat_inv_{orig_i}"
        s_inv_key = f"scale_mat_inv_{orig_i}"

        if w_inv_key in data:
            out[f"world_mat_inv_{new_i}"] = data[w_inv_key]
        if s_inv_key in data:
            out[f"scale_mat_inv_{new_i}"] = data[s_inv_key]

    out["index_map"] = np.array(orig_indices, dtype=np.int32)
    np.savez(dst_npz, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset", type=str, required=True)
    ap.add_argument("--split_txt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_dataset = Path(args.src_dataset)
    split_txt = Path(args.split_txt)
    out_dir = Path(args.out_dir)

    src_images = src_dataset / "images"
    src_masks = src_dataset / "masks"
    src_npz = src_dataset / "preprocessed" / "cameras_sphere.npz"

    if not src_images.exists() or not src_masks.exists() or not src_npz.exists():
        raise FileNotFoundError("Source dataset structure invalid.")

    if not split_txt.exists():
        raise FileNotFoundError("split_txt not found.")

    orig_indices = parse_split_file(split_txt)

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"{out_dir} exists. Use --overwrite")

    out_dir.mkdir(parents=True, exist_ok=True)

    dst_images = out_dir / "image"   # singolare
    dst_masks  = out_dir / "mask"    # singolare
    dst_npz    = out_dir / "cameras_spe3r.npz"

    print(f"=== BUILD SUBSET | N={len(orig_indices)} ===")

    copy_reindexed(src_images, dst_images, orig_indices)
    copy_reindexed(src_masks,  dst_masks,  orig_indices)
    subset_cameras_npz(src_npz, dst_npz, orig_indices)

    (out_dir / "split_used.txt").write_text(split_txt.read_text())
    (out_dir / "index_map.json").write_text(json.dumps({"orig_indices": orig_indices}, indent=2))

    print("✅ Done.")
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
