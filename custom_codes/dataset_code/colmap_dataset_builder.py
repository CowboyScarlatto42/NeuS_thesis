#!/usr/bin/env python3
"""
build_colmap_subset.py

Input:
- N .txt files, each line contains an image name like: 000385.png (can have leading zeros)

Source dataset naming:
- Files in src_images_dir are named with fixed digits, e.g. 001.png, 385.png, etc.
  (default src_digits=3)

Output:
out_dir/
  images/
    000.png, 001.png, ..., (M-1).png     (copied + renamed)
  mapping.json                           (original_txt_name -> renamed)

Usage (Colab):
  !python build_colmap_subset.py \
      --txt_paths /content/orbit64.txt \
      --src_images_dir /content/drive/MyDrive/SPE3R/images \
      --out_dir /content/drive/MyDrive/colmap_subset \
      --src_digits 3
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
            names.append(Path(s).name)  # keeps e.g. 000385.png
    return names


def unique_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def infer_digits(n: int) -> int:
    """Choose a sensible zero-padding for output renamed images (at least 3)."""
    return max(3, len(str(max(0, n - 1))))


def normalize_to_src_name(original_txt_name: str, src_digits: int) -> str:
    """
    Convert something like '000385.png' -> '385.png' (then pad to src_digits: '385.png' or '001.png', etc.)
    Keeps extension from txt.
    """
    p = Path(original_txt_name)
    stem = p.stem  # '000385'
    suffix = p.suffix.lower()  # '.png'

    # Convert stem to int to drop leading zeros safely
    try:
        idx = int(stem)
    except ValueError:
        # If stem isn't numeric, fall back to original (no normalization possible)
        return original_txt_name

    return f"{idx:0{src_digits}d}{suffix}"


def main():
    parser = argparse.ArgumentParser(
        description="Build a COLMAP-ready subset folder from SPE3R image lists."
    )
    parser.add_argument(
        "--txt_paths",
        nargs="+",
        required=True,
        help="One or more .txt files listing image names (e.g. 000385.png).",
    )
    parser.add_argument(
        "--src_images_dir",
        required=True,
        help="Directory containing the original SPE3R images (e.g. 001.png naming).",
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
        help="Zero-padding digits for renamed output files (default: auto, at least 3).",
    )
    parser.add_argument(
        "--src_digits",
        type=int,
        default=3,
        help="Zero-padding digits used by the source dataset filenames (default: 3, e.g. 001.png).",
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

    mapping = {}   # original_txt_name -> renamed_output_name
    missing = []   # original_txt_name that couldn't be found in src_images_dir

    # ----------------------------
    # Copy + rename
    # ----------------------------
    i_out = 0
    for orig_txt_name in all_names:
        # normalize to how files are named in src_images_dir (e.g., 000385.png -> 385.png -> 385 padded -> 385.png)
        src_name = normalize_to_src_name(orig_txt_name, src_digits=args.src_digits)

        src = src_images_dir / src_name
        if not src.exists():
            missing.append(orig_txt_name)
            continue

        new_name = f"{i_out:0{digits}d}{Path(src_name).suffix.lower()}"  # output is png
        dst = images_out / new_name

        shutil.copy2(src, dst)
        mapping[orig_txt_name] = new_name
        i_out += 1

    # ----------------------------
    # Save mapping
    # ----------------------------
    mapping_path = out_dir / "mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "src_images_dir": str(src_images_dir),
                "txt_paths": [str(p) for p in txt_paths],
                "src_digits": args.src_digits,
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
    print(f"[OK] Copied:        {len(mapping)} -> {images_out}")
    if missing:
        print(f"[WARN] Missing:     {len(missing)} (see mapping.json). Example: {missing[:5]}")
    print(f"[OK] Mapping saved: {mapping_path}")


if __name__ == "__main__":
    main()
