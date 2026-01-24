#!/usr/bin/env python3
"""
make_neus_datasets_from_txt.py

Given:
  src_dataset/
    images/000.png..999.png
    masks/000.png..999.png
    cameras_spe3r.npz (world_mat_0..999, scale_mat_0..999)

And split txt files containing indices or filenames (e.g., 123 or 000123.png),
creates for each split:
  out_data_dir/<split_name>/
    image/000.png.. (reindexed 0..N-1)
    mask/000.png..
    cameras_spe3r.npz (subset + remapped mats)
    split_used.txt
    index_map.json  (just orig_indices in order)
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
            raise ValueError(f"No index found in line: '{line}' ({split_path})")
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
            raise KeyError(f"Missing keys '{w_key}'/'{s_key}' in {src_npz}")
        out[f"world_mat_{new_i}"] = data[w_key]
        out[f"scale_mat_{new_i}"] = data[s_key]
    out["index_map"] = np.array(orig_indices, dtype=np.int32)
    np.savez(dst_npz, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--out_data_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.txt", help="Glob for split files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing split folders")
    args = ap.parse_args()

    src_dataset = Path(args.src_dataset)
    splits_dir = Path(args.splits_dir)
    out_data_dir = Path(args.out_data_dir)

    src_images = src_dataset / "image"
    src_masks = src_dataset / "mask"
    src_npz = src_dataset / "cameras_spe3r.npz"

    if not src_images.exists() or not src_masks.exists() or not src_npz.exists():
        raise FileNotFoundError(
            "src_dataset must contain: images/, masks/, cameras_spe3r.npz\n"
            f"Checked:\n- {src_images} exists={src_images.exists()}\n"
            f"- {src_masks} exists={src_masks.exists()}\n"
            f"- {src_npz} exists={src_npz.exists()}"
        )

    split_files = sorted(splits_dir.glob(args.pattern))
    if not split_files:
        raise FileNotFoundError(f"No split files in {splits_dir} matching '{args.pattern}'")

    out_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(split_files)} split files.")

    for split_path in split_files:
        split_name = split_path.stem
        orig_indices = parse_split_file(split_path)

        if len(orig_indices) != len(set(orig_indices)):
            raise ValueError(f"Duplicate indices in {split_path}")
        if min(orig_indices) < 0 or max(orig_indices) > 999:
            raise ValueError(f"Index out of range in {split_path}: min={min(orig_indices)}, max={max(orig_indices)}")

        dst_root = out_data_dir / split_name
        dst_images = dst_root / "image"
        dst_masks = dst_root / "mask"
        dst_npz = dst_root / "cameras_spe3r.npz"

        print(f"\n=== {split_name} | N={len(orig_indices)} | idx range [{min(orig_indices)}, {max(orig_indices)}] ===")
        print(f"-> {dst_root}")

        if dst_root.exists():
            if args.overwrite:
                shutil.rmtree(dst_root)
            else:
                raise FileExistsError(f"{dst_root} exists. Use --overwrite to replace it.")

        dst_root.mkdir(parents=True, exist_ok=True)

        copy_reindexed(src_images, dst_images, orig_indices)
        copy_reindexed(src_masks, dst_masks, orig_indices)
        subset_cameras_npz(src_npz, dst_npz, orig_indices)

        (dst_root / "split_used.txt").write_text(split_path.read_text())
        (dst_root / "index_map.json").write_text(json.dumps({"orig_indices": orig_indices}, indent=2))

        print("Done.")

    print("\nAll per-split datasets created.")


if __name__ == "__main__":
    main()
