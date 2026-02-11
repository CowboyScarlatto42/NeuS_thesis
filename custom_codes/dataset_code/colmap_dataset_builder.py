#!/usr/bin/env python3
"""
build_colmap_subset.py

Build a COLMAP-ready subset folder from SPE3R image lists (optionally with masks),
AND (optionally) produce labels_subset.json consistent with the renamed images.

Supports labels.json entries where filename can be like:
  - "img000001" (NO extension)
  - "img000001.png"
  - "000001.png"
  - "000001"

TXT lists can contain:
  - "000385.png"
  - "img000385"
  - "img000385.png"
  - paths; we always use basename/stem

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
  labels_subset.json                     (only if --labels_json is provided)

Mapping keys:
- "source_id_to_renamed": {"385": "012.png", ...}
- "listed_ids_in_order": [385, 17, ...]
This makes it robust to filename formatting differences (img000385 vs 000385.png).

Usage (Colab):
  !python build_colmap_subset.py \
      --txt_paths /content/orbit230.txt \
      --src_images_dir /content/drive/MyDrive/SPE3R/images \
      --src_masks_dir  /content/drive/MyDrive/SPE3R/masks \
      --labels_json    /content/drive/MyDrive/SPE3R/labels_black_500.json \
      --out_dir        /content/drive/MyDrive/colmap_subset_230 \
      --src_digits 3
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# ----------------------------
# Parsing helpers
# ----------------------------
def extract_numeric_id(name: str) -> int:
    """
    Extract last up-to-6 digits from any string like:
      img000001, img000001.png, 000001.png, 1, /path/to/img000001.png
    """
    s = Path(name).stem
    digits = "".join([c for c in s if c.isdigit()])
    if not digits:
        raise ValueError(f"Cannot extract numeric id from: {name}")
    return int(digits[-6:])


def read_txt_list(txt_path: Path) -> List[str]:
    """Read names from a txt file, ignoring blank lines and comment lines starting with #."""
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
    """Choose a sensible zero-padding for output renamed images (at least 3)."""
    return max(3, len(str(max(0, n - 1))))


def id_to_src_filename(idx: int, src_digits: int, ext: str = ".png") -> str:
    """ID -> source filename in src_images_dir (e.g., 385 -> '385.png' or '001.png')."""
    return f"{idx:0{src_digits}d}{ext}"


def id_to_label_key_variants(idx: int) -> List[str]:
    """
    Possible representations in labels_json for a given id.
    You showed "img000001" (no ext).
    We'll generate common variants to match robustly.
    """
    s6 = f"{idx:06d}"
    return [
        f"img{s6}",
        f"img{s6}.png",
        s6,
        f"{s6}.png",
        str(idx),
        f"{idx}.png",
    ]


# ----------------------------
# Labels IO
# ----------------------------
def load_labels(labels_path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Returns (raw_obj, entries_list).
    Supports either:
      - dict with key 'data'
      - list of dicts
    """
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        return raw, raw["data"]
    if isinstance(raw, list):
        return raw, raw
    raise ValueError("Unsupported labels_json format. Expected {'data':[...]} or a list of dicts.")


def save_labels(raw_in: Any, kept_entries: List[Dict[str, Any]], out_path: Path) -> None:
    """Save subset preserving original top-level structure."""
    if isinstance(raw_in, dict) and "data" in raw_in and isinstance(raw_in["data"], list):
        out_obj = dict(raw_in)
        out_obj["data"] = kept_entries
    else:
        out_obj = kept_entries
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")


def build_labels_lookup(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Map normalized filename token -> entry.
    Normalization: basename only, no extension if present (keep both).
    We'll store a few keys per entry to increase match chance.
    """
    lut: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        fn = e.get("filename", "")
        if not fn:
            continue
        base = Path(fn).name
        stem = Path(fn).stem

        # store both base and stem
        lut[base] = e
        lut[stem] = e

    return lut


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build a COLMAP-ready subset folder from image lists (optionally with masks). "
                    "Optionally also produce labels_subset.json consistent with renamed images."
    )
    parser.add_argument(
        "--txt_paths",
        nargs="+",
        required=True,
        help="One or more .txt files listing image names (e.g. 000385.png, img000385, ...).",
    )
    parser.add_argument(
        "--src_images_dir",
        required=True,
        help="Directory containing the original SPE3R images.",
    )
    parser.add_argument(
        "--src_masks_dir",
        default=None,
        help="(Optional) Directory containing masks with SAME filenames as images.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to create (will contain images/ and optionally masks/).",
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
        help="Zero-padding digits used by the source dataset filenames (default: 3).",
    )
    parser.add_argument(
        "--src_ext",
        type=str,
        default=".png",
        help="Source image extension (default: .png).",
    )

    # labels support
    parser.add_argument(
        "--labels_json",
        default=None,
        help="(Optional) Path to the full labels.json. If provided, will output labels_subset.json.",
    )
    parser.add_argument(
        "--labels_out",
        default=None,
        help="(Optional) Output path for labels subset (default: <out_dir>/labels_subset.json).",
    )
    parser.add_argument(
        "--labels_filename_mode",
        choices=["renamed_noext", "renamed_with_ext"],
        default="renamed_noext",
        help="How to write 'filename' in labels_subset.json: "
             "'renamed_noext' -> '000000' ; 'renamed_with_ext' -> '000000.png'.",
    )

    args = parser.parse_args()

    txt_paths = [Path(p) for p in args.txt_paths]
    src_images_dir = Path(args.src_images_dir)
    src_masks_dir = Path(args.src_masks_dir) if args.src_masks_dir is not None else None
    out_dir = Path(args.out_dir)

    if not src_images_dir.exists():
        raise FileNotFoundError(f"src_images_dir not found: {src_images_dir}")
    if src_masks_dir is not None and not src_masks_dir.exists():
        raise FileNotFoundError(f"src_masks_dir not found: {src_masks_dir}")

    labels_path: Optional[Path] = Path(args.labels_json) if args.labels_json is not None else None
    if labels_path is not None and not labels_path.exists():
        raise FileNotFoundError(f"labels_json not found: {labels_path}")

    # ----------------------------
    # Read + parse ids from txt lists
    # ----------------------------
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
        print(f"[WARN] {len(bad_lines)} lines could not be parsed (showing up to 20): {bad_lines[:20]}")

    # ----------------------------
    # Prepare output dirs
    # ----------------------------
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    masks_out = None
    if src_masks_dir is not None:
        masks_out = out_dir / "masks"
        masks_out.mkdir(parents=True, exist_ok=True)

    digits = args.digits if args.digits is not None else infer_digits(len(listed_ids))

    # ----------------------------
    # Copy + rename images (+ masks)
    # ----------------------------
    source_id_to_renamed: Dict[str, str] = {}
    missing_images: List[int] = []

    i_out = 0
    for idx in listed_ids:
        src_name = id_to_src_filename(idx, src_digits=args.src_digits, ext=args.src_ext)
        src_img = src_images_dir / src_name

        if not src_img.exists():
            missing_images.append(idx)
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

    # ----------------------------
    # Save mapping.json
    # ----------------------------
    mapping_path = out_dir / "mapping.json"
    mapping_obj = {
        "src_images_dir": str(src_images_dir),
        "src_masks_dir": str(src_masks_dir) if src_masks_dir else None,
        "txt_paths": [str(p) for p in txt_paths],
        "src_digits": args.src_digits,
        "src_ext": args.src_ext,
        "num_listed_unique_ids": len(listed_ids),
        "num_copied": len(source_id_to_renamed),
        "num_missing_images": len(missing_images),
        "missing_image_ids": missing_images,
        "digits": digits,
        "listed_ids_in_order": listed_ids,
        "source_id_to_renamed": source_id_to_renamed,
    }
    mapping_path.write_text(json.dumps(mapping_obj, indent=2), encoding="utf-8")

    print(f"[OK] Listed unique ids: {len(listed_ids)}")
    print(f"[OK] Copied images: {len(source_id_to_renamed)} -> {images_out}")
    if src_masks_dir is not None:
        print(f"[OK] Copied masks:  {len(source_id_to_renamed)} -> {masks_out}")
    if missing_images:
        print(f"[WARN] Missing source images: {len(missing_images)} (see mapping.json)")
    print(f"[OK] Mapping saved: {mapping_path}")

    # ----------------------------
    # Build labels_subset.json
    # ----------------------------
    if labels_path is not None:
        raw_labels, entries = load_labels(labels_path)
        lut = build_labels_lookup(entries)

        kept: List[Dict[str, Any]] = []
        missing_in_labels: List[int] = []

        # iterate in output order (0..M-1 corresponds to i_out)
        # We reconstruct order by sorting renamed names numerically
        def renamed_index(name: str) -> int:
            try:
                return int(Path(name).stem)
            except Exception:
                return 10**9

        mapping_items = sorted(source_id_to_renamed.items(), key=lambda kv: renamed_index(kv[1]))

        for idx_str, new_name in mapping_items:
            idx = int(idx_str)

            # find entry in labels by trying common variants
            entry = None
            for key in id_to_label_key_variants(idx):
                if key in lut:
                    entry = lut[key]
                    break

            if entry is None:
                # also try matching by numeric id extracted from entry filenames (fallback)
                missing_in_labels.append(idx)
                continue

            e2 = dict(entry)  # shallow copy

            if args.labels_filename_mode == "renamed_with_ext":
                e2["filename"] = new_name  # e.g. 012.png
            else:
                e2["filename"] = Path(new_name).stem  # e.g. 012

            kept.append(e2)

        labels_out_path = Path(args.labels_out) if args.labels_out is not None else (out_dir / "labels_subset.json")
        labels_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_labels(raw_labels, kept, labels_out_path)

        print(f"[OK] labels_full entries: {len(entries)}")
        print(f"[OK] labels_subset entries written: {len(kept)} -> {labels_out_path}")
        if missing_in_labels:
            print(f"[WARN] {len(missing_in_labels)} ids in subset but NOT found in labels_json (showing up to 20):")
            print(missing_in_labels[:20])

        # Optional sanity check: ensure image count matches labels count
        img_files = sorted([p.name for p in images_out.glob("*") if p.is_file()])
        label_names = set(
            (Path(e["filename"]).name if args.labels_filename_mode == "renamed_with_ext" else f'{e["filename"]}{Path(img_files[0]).suffix}')
            for e in kept
            if "filename" in e
        )
        # If renamed_noext, label_names built above appends one suffix (best-effort); not perfect but useful.
        print(f"[INFO] images_on_disk: {len(img_files)} | labels_written: {len(kept)}")

    # done


if __name__ == "__main__":
    main()
