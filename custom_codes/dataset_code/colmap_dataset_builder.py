#!/usr/bin/env python3
"""
colmap_dataset_builder.py

Given a src_dataset with:
  src_dataset/
    image/000.png..999.png
    mask/000.png..999.png
    cameras_spe3r.npz  (world_mat_0..999, scale_mat_0..999)  OR (world_mat_<orig_id>...)

and a split txt file with lines like:
  123
  000123.png
  img000123
  img000123.png

Create:
  out_dir/
    image/ (reindexed 0..K-1)
    mask/
    cameras_spe3r.npz (subset + reindexed mats)
    split_used.txt
    index_map.json

Assumptions:
- Source images/masks are named with fixed 3 digits: 000.png, 001.png, ...
- NPZ keys are world_mat_<orig_id>, scale_mat_<orig_id>
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


def dst_name(new_i: int, dst_digits: int) -> str:
    """
    dst_digits:
      - 3  -> 000.png style
      - 0  -> 0.png style
      - 6  -> 000000.png style
    """
    if dst_digits <= 0:
        return f"{new_i}.png"
    return f"{new_i:0{dst_digits}d}.png"


def copy_reindexed(src_folder: Path, dst_folder: Path, orig_indices: list[int], dst_digits: int):
    dst_folder.mkdir(parents=True, exist_ok=True)
    for new_i, orig_i in enumerate(orig_indices):
        src = src_folder / f"{orig_i:03d}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing file: {src}")
        dst = dst_folder / dst_name(new_i, dst_digits)
        shutil.copy2(src, dst)


def subset_cameras_npz(src_npz: Path, dst_npz: Path, orig_indices: list[int]):
    data = np.load(src_npz)
    out = {}

    has_w_inv = any(k.startswith("world_mat_inv_") for k in data.keys())
    has_s_inv = any(k.startswith("scale_mat_inv_") for k in data.keys())

    for new_i, orig_i in enumerate(orig_indices):
        # forward
        w_key = f"world_mat_{orig_i}"
        s_key = f"scale_mat_{orig_i}"
        if w_key not in data or s_key not in data:
            raise KeyError(f"Missing keys '{w_key}'/'{s_key}' in {src_npz}")
        out[f"world_mat_{new_i}"] = data[w_key]
        out[f"scale_mat_{new_i}"] = data[s_key]

        # inverse (optional but present in your case)
        if has_w_inv:
            w_inv = f"world_mat_inv_{orig_i}"
            if w_inv not in data:
                raise KeyError(f"Missing key '{w_inv}' in {src_npz}")
            out[f"world_mat_inv_{new_i}"] = data[w_inv]

        if has_s_inv:
            s_inv = f"scale_mat_inv_{orig_i}"
            if s_inv not in data:
                raise KeyError(f"Missing key '{s_inv}' in {src_npz}")
            out[f"scale_mat_inv_{new_i}"] = data[s_inv]

    out["index_map"] = np.array(orig_indices, dtype=np.int32)
    np.savez(dst_npz, **out)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset", type=str, required=True,
                    help="Folder containing image/, mask/, cameras_spe3r.npz")
    ap.add_argument("--split_txt", type=str, required=True,
                    help="TXT containing selected original ids/names (img000123, 000123.png, 123, ...)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder to create (will contain image/, mask/, cameras_spe3r.npz)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing out_dir if it exists")
    ap.add_argument("--dst_digits", type=int, default=3,
                    help="Digits for destination filenames (default 3 -> 000.png). Use 0 for 0.png style.")
    args = ap.parse_args()

    src_dataset = Path(args.src_dataset)
    split_txt = Path(args.split_txt)
    out_dir = Path(args.out_dir)

    src_images = src_dataset / "image"
    src_masks = src_dataset / "mask"
    src_npz = src_dataset / "cameras_spe3r.npz"

    if not src_images.exists() or not src_masks.exists() or not src_npz.exists():
        raise FileNotFoundError(
            "src_dataset must contain: image/, mask/, cameras_spe3r.npz\n"
            f"Checked:\n- {src_images} exists={src_images.exists()}\n"
            f"- {src_masks} exists={src_masks.exists()}\n"
            f"- {src_npz} exists={src_npz.exists()}"
        )
    if not split_txt.exists():
        raise FileNotFoundError(f"split_txt not found: {split_txt}")

    orig_indices = parse_split_file(split_txt)

    if len(orig_indices) != len(set(orig_indices)):
        raise ValueError(f"Duplicate indices in {split_txt}")
    if min(orig_indices) < 0:
        raise ValueError(f"Negative index found in {split_txt}")

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"{out_dir} exists. Use --overwrite to replace it.")

    out_dir.mkdir(parents=True, exist_ok=True)
    dst_images = out_dir / "image"
    dst_masks = out_dir / "mask"
    dst_npz = out_dir / "cameras_spe3r.npz"

    print(f"=== BUILD SUBSET | N={len(orig_indices)} | idx range [{min(orig_indices)}, {max(orig_indices)}] ===")
    print(f"src_dataset: {src_dataset}")
    print(f"split_txt:   {split_txt}")
    print(f"out_dir:     {out_dir}")
    print(f"dst_digits:  {args.dst_digits}")

    copy_reindexed(src_images, dst_images, orig_indices, dst_digits=args.dst_digits)
    copy_reindexed(src_masks, dst_masks, orig_indices, dst_digits=args.dst_digits)
    subset_cameras_npz(src_npz, dst_npz, orig_indices)

    (out_dir / "split_used.txt").write_text(split_txt.read_text(), encoding="utf-8")
    (out_dir / "index_map.json").write_text(json.dumps({"orig_indices": orig_indices}, indent=2), encoding="utf-8")

    print("[OK] Done.")
    print(f"images: {dst_images}")
    print(f"masks:  {dst_masks}")
    print(f"npz:    {dst_npz}")


if __name__ == "__main__":
    main()
