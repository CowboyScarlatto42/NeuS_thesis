#!/usr/bin/env python3
"""
make_neus_datasets_all_splits_pad3.py

SRC dataset already in NeuS format with:
  images/  (000.png .. 999.png)
  masks/   (000.png .. 999.png)
  cameras_spe3r.npz  (world_mat_0..999, scale_mat_0..999)

Split files contain 0-based indices:
  BLACK: 0..499
  EARTH: 500..999

For each split txt (e.g. hst_black_nmc_4.txt, hst_earth_nmc_32.txt), creates:
  out_data_dir/<split_name>/
    images/ (000.png..)
    masks/  (000.png..)
    cameras_spe3r.npz (subset + remapped mats)
    split_used.txt
    index_map.json
"""

import argparse
import json
import re
import shutil
from pathlib import Path
import numpy as np


def parse_split_file(split_path: Path) -> list[int]:
    """Extract first integer from each non-empty line."""
    idxs = []
    for raw in split_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.search(r"(\d+)", line)
        if not m:
            raise ValueError(f"Could not find an index in line: '{line}' ({split_path})")
        idxs.append(int(m.group(1)))
    return idxs


def path_for_idx(folder: Path, idx: int) -> Path:
    """Your dataset uses 3-digit padding, png only."""
    p = folder / f"{idx:03d}.png"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def copy_and_reindex_png(src_folder: Path, dst_folder: Path, original_indices: list[int]) -> dict:
    """Copy PNGs from src_folder using original idx, save as 000.png.. in split order."""
    dst_folder.mkdir(parents=True, exist_ok=True)
    mapping = {}

    for new_i, orig_i in enumerate(original_indices):
        src_path = path_for_idx(src_folder, orig_i)
        dst_path = dst_folder / f"{new_i:03d}.png"
        shutil.copy2(src_path, dst_path)

        mapping[str(new_i)] = {
            "orig_idx": int(orig_i),
            "orig_file": src_path.name,
            "new_file": dst_path.name,
        }
    return mapping


def subset_cameras_npz(src_npz: Path, dst_npz: Path, original_indices: list[int]):
    """Create reduced cameras_spe3r.npz with remapped world/scale mats to 0..N-1 + index_map."""
    data = np.load(src_npz)
    out = {}

    for new_i, orig_i in enumerate(original_indices):
        w_key = f"world_mat_{orig_i}"
        s_key = f"scale_mat_{orig_i}"
        if w_key not in data or s_key not in data:
            raise KeyError(f"Missing keys '{w_key}'/'{s_key}' in {src_npz}")
        out[f"world_mat_{new_i}"] = data[w_key]
        out[f"scale_mat_{new_i}"] = data[s_key]

    out["index_map"] = np.array(original_indices, dtype=np.int32)
    np.savez(dst_npz, **out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset", type=str, required=True,
                    help="Full NeuS dataset with images/000.png..999.png, masks/, cameras_spe3r.npz")
    ap.add_argument("--splits_dir", type=str, required=True,
                    help="Directory containing all split txt files (black + earth)")
    ap.add_argument("--out_data_dir", type=str, required=True,
                    help="Output directory where per-split datasets will be created")
    ap.add_argument("--pattern", type=str, default="hst_*_nmc_*.txt",
                    help="Glob for split files (default: hst_*_nmc_*.txt)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src_dataset = Path(args.src_dataset)
    splits_dir = Path(args.splits_dir)
    out_data_dir = Path(args.out_data_dir)

    src_images = src_dataset / "images"
    src_masks = src_dataset / "masks"
    src_npz = src_dataset / "cameras_spe3r.npz"

    if not src_images.exists() or not src_masks.exists() or not src_npz.exists():
        raise FileNotFoundError(
            "src_dataset must contain images/, masks/, cameras_spe3r.npz\n"
            f"Got:\n- {src_images} exists={src_images.exists()}\n"
            f"- {src_masks} exists={src_masks.exists()}\n"
            f"- {src_npz} exists={src_npz.exists()}"
        )

    split_files = sorted(splits_dir.glob(args.pattern))
    if not split_files:
        raise FileNotFoundError(f"No split files found in {splits_dir} matching '{args.pattern}'")

    out_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(split_files)} split files.")

    for split_path in split_files:
        split_name = split_path.stem  # e.g. hst_black_nmc_4
        idxs = parse_split_file(split_path)  # 0-based as per your setup

        if len(idxs) != len(set(idxs)):
            raise ValueError(f"Duplicate indices in {split_path}")
        if min(idxs) < 0 or max(idxs) > 999:
            raise ValueError(f"Index out of range in {split_path}: min={min(idxs)}, max={max(idxs)}")

        dst_root = out_data_dir / split_name
        dst_images = dst_root / "images"
        dst_masks = dst_root / "masks"
        dst_npz = dst_root / "cameras_spe3r.npz"

        print(f"\n=== {split_name} | N={len(idxs)} | idx range [{min(idxs)}, {max(idxs)}] ===")
        print(f"-> {dst_root}")

        if args.dry_run:
            continue

        # overwrite existing folder
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)

        img_map = copy_and_reindex_png(src_images, dst_images, idxs)
        msk_map = copy_and_reindex_png(src_masks, dst_masks, idxs)
        subset_cameras_npz(src_npz, dst_npz, idxs)

        # traceability
        (dst_root / "split_used.txt").write_text(split_path.read_text())
        meta = {
            "split_file": str(split_path),
            "src_dataset": str(src_dataset),
            "n_views": len(idxs),
            "orig_index_min": int(min(idxs)),
            "orig_index_max": int(max(idxs)),
            "images_map": img_map,
            "masks_map": msk_map,
            "note": "orig_idx are original indices (0-based, 000..999). New indices are 0..N-1 saved as 000.. in split order."
        }
        (dst_root / "index_map.json").write_text(json.dumps(meta, indent=2))

        print("Done.")

    print("\nAll per-split datasets created.")


if __name__ == "__main__":
    main()
