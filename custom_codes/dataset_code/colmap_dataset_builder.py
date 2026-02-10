#!/usr/bin/env python3
"""
build_colmap_subset.py

Input:
- N .txt files, each line contains an image name like: 000385.png

Output:
out_dir/
  images/
    000.png, 001.png, ..., (M-1).png     (copied + renamed)
  mapping.json                           (original -> renamed)

Usage (Colab):
  !python build_colmap_subset.py \
      --txt_paths /content/list1.txt /content/list2.txt \
      --src_images_dir /content/SPE3R/images \
      --out_dir /content/colmap_subset
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List


def read_txt_list(txt_path: Path) -> List[str]:
    """Read image names from a txt file, ignoring blank lines and comment lines starting with #."""
    names: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # keep only basename in case the txt contains paths
            names.append(Path(s).name)
    return names


def unique_preserve_order(seq: List[str]) -> List[str]:
    """Remove duplicates while preserving first occurrence order."""
    seen = set()
    out: List[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def infer_digits(n: int) -> int:
    """Choose a sensible zero-padding (at least 3)."""
    return max(3, len(str(max(0, n - 1))))


def main():
    parser = argparse.ArgumentParser(
        description="Build a COLMAP-ready subset folder from SPE3R image lists."
    )
    parser.add_argument(
        "--txt_paths",
        nargs="+",
        required=True,
        help="One or more .txt files listing SPE3R image names (e.g. 000385.png).",
    )
    parser.add_argument(
        "--src_images_dir",
        required=True,
        help="Directory containing the original SPE3R images.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to create (will contain images/ and mapping.json).",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="Zero-padding digits for renamed files (default: auto, at least 3).",
    )

    args = parser.parse_args()

    txt_paths = [Path(p) for p in args.txt_paths]
    src_images_dir = Path(args.src_images_dir)
    out_dir = Path(args.out_dir)

    if not src_images_dir.exists():
        raise FileNotFoundError(f"src_images_dir not found: {src_images_dir}")

    # ----------------------------
    # Read and merge all txt lists
    # ----------------------------
    all_names: List[str] = []
    for p in txt_paths:
        if not p.exists():
            raise FileNotFoundError(f"TXT file not found: {p}")
        all_names.extend(read_txt_list(p))

    if not all_names:
        raise ValueError("No image names found in the provided txt files.")

    # Remove duplicates (keep first)
    all_names = unique_preserve_order(all_names)

    # ----------------------------
    # Prepare output
    # ----------------------------
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    n_listed = len(all_names)
    digits = args.digits if args.digits is not None else infer_digits(n_listed)

    mapping = {}   # original -> renamed
    missing = []   # originals that were listed but not found

    # ----------------------------
    # Copy + rename
    # ----------------------------
    i_out = 0
    for orig_name in all_names:
        src = src_images_dir / orig_name
        if not src.exists():
            missing.append(orig_name)
            continue

        new_name = f"{i_out:0{digits}d}{src.suffix.lower()}"
        dst = images_out / new_name

        shutil.copy2(src, dst)
        mapping[orig_name] = new_name
        i_out += 1

    # ----------------------------
    # Save mapping
    # ----------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "src_images_dir": str(src_images_dir),
                "txt_paths": [str(p) for p in txt_paths],
                "num_listed_unique": n_listed,
                "num_copied": len(mapping),
                "num_missing": len(missing),
                "missing": missing,
                "digits": digits,
                "original_to_renamed": mapping,
            },
            f,
            indent=2,
        )

    print(f"[OK] Listed unique: {n_listed}")
    print(f"[OK] Copied:        {len(mapping)}  -> {images_out}")
    if missing:
        print(f"[WARN] Missing:     {len(missing)} (see mapping.json)")
    print(f"[OK] Mapping saved: {mapping_path}")


if __name__ == "__main__":
    main()
