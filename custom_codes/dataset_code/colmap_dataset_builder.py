#!/usr/bin/env python3
"""
build_colmap_subset.py

Input:
- N .txt files, each line contains an image id/name like:
    img000385
    img000385.png
    000385.png
    /path/to/000385.png
  (we extract numeric id robustly)

Source dataset naming:
- Files in src_images_dir are named with fixed digits, e.g. 001.png, 385.png, etc.
  (default src_digits=3)
- (Optional) Masks in src_masks_dir have THE SAME filenames as images in src_images_dir.

Output:
out_dir/
  images/
    000.png, 001.png, ..., (M-1).png     (copied + renamed)
  masks/                                 (only if --src_masks_dir is provided)
    000.png, 001.png, ..., (M-1).png     (copied + renamed, same names as images)
  mapping.json                           (source_id -> renamed)

Usage (Colab):
  !python build_colmap_subset.py \
      --txt_paths /content/orbit230.txt \
      --src_images_dir /content/drive/MyDrive/SPE3R/images \
      --src_masks_dir  /content/drive/MyDrive/SPE3R/masks \
      --out_dir /content/drive/MyDrive/colmap_subset_230 \
      --src_digits 3
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict


def extract_numeric_id(name: str) -> int:
    s = Path(name).stem
    digits = "".join([c for c in s if c.isdigit()])
    if not digits:
        raise ValueError(f"Cannot extract numeric id from: {name}")
    return int(digits[-6:])


def read_txt_list(txt_path: Path) -> List[str]:
    names: List[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(Path(s).name)
    return names


def unique_preserve_order(seq: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def infer_digits(n: int) -> int:
    return max(3, len(str(max(0, n - 1))))


def id_to_src_filename(idx: int, src_digits: int, ext: str = ".png") -> str:
    return f"{idx:0{src_digits}d}{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Build a COLMAP-ready subset folder from image lists (optionally with masks)."
    )
    parser.add_argument("--txt_paths", nargs="+", required=True)
    parser.add_argument("--src_images_dir", required=True)
    parser.add_argument("--src_masks_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--src_digits", type=int, default=3)
    parser.add_argument("--src_ext", type=str, default=".png")
    args = parser.parse_args()

    txt_paths = [Path(p) for p in args.txt_paths]
    src_images_dir = Path(args.src_images_dir)
    src_masks_dir = Path(args.src_masks_dir) if args.src_masks_dir is not None else None
    out_dir = Path(args.out_dir)

    if not src_images_dir.exists():
        raise FileNotFoundError(f"src_images_dir not found: {src_images_dir}")
    if src_masks_dir is not None and not src_masks_dir.exists():
        raise FileNotFoundError(f"src_masks_dir not found: {src_masks_dir}")

    listed_raw: List[str] = []
    for p in txt_paths:
        if not p.exists():
            raise FileNotFoundError(f"TXT file not found: {p}")
        listed_raw.extend(read_txt_list(p))

    if not listed_raw:
        raise ValueError("No image names found in the provided txt files.")

    listed_ids: List[int] = []
    bad_lines: List[str] = []
    for s in listed_raw:
        try:
            listed_ids.append(extract_numeric_id(s))
        except ValueError:
            bad_lines.append(s)
    if not listed_ids:
        raise ValueError("Could not parse any numeric ids from txt lists.")
    listed_ids = unique_preserve_order(listed_ids)

    if bad_lines:
        print(f"[WARN] {len(bad_lines)} unparseable lines (showing up to 10): {bad_lines[:10]}")

    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    masks_out = None
    if src_masks_dir is not None:
        masks_out = out_dir / "masks"
        masks_out.mkdir(parents=True, exist_ok=True)

    digits = args.digits if args.digits is not None else infer_digits(len(listed_ids))

    source_id_to_renamed: Dict[str, str] = {}
    missing_ids: List[int] = []

    i_out = 0
    for idx in listed_ids:
        src_name = id_to_src_filename(idx, src_digits=args.src_digits, ext=args.src_ext)
        src_img = src_images_dir / src_name
        if not src_img.exists():
            missing_ids.append(idx)
            continue

        new_name = f"{i_out:0{digits}d}{src_img.suffix.lower()}"
        shutil.copy2(src_img, images_out / new_name)

        if src_masks_dir is not None:
            src_mask = src_masks_dir / src_name
            if not src_mask.exists():
                raise FileNotFoundError(f"Mask missing for image '{src_name}' in {src_masks_dir}")
            shutil.copy2(src_mask, masks_out / new_name)

        source_id_to_renamed[str(idx)] = new_name
        i_out += 1

    mapping_path = out_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "src_images_dir": str(src_images_dir),
                "src_masks_dir": str(src_masks_dir) if src_masks_dir else None,
                "txt_paths": [str(p) for p in txt_paths],
                "src_digits": args.src_digits,
                "src_ext": args.src_ext,
                "digits": digits,
                "num_listed_unique_ids": len(listed_ids),
                "num_copied": len(source_id_to_renamed),
                "missing_image_ids": missing_ids,
                "listed_ids_in_order": listed_ids,
                "source_id_to_renamed": source_id_to_renamed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Listed unique ids: {len(listed_ids)}")
    print(f"[OK] Copied images: {len(source_id_to_renamed)} -> {images_out}")
    if src_masks_dir is not None:
        print(f"[OK] Copied masks:  {len(source_id_to_renamed)} -> {masks_out}")
    if missing_ids:
        print(f"[WARN] Missing source images: {len(missing_ids)} (see mapping.json)")
    print(f"[OK] Mapping saved: {mapping_path}")


if __name__ == "__main__":
    main()
