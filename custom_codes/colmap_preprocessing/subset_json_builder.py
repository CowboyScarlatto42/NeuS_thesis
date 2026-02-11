#!/usr/bin/env python3
"""
make_labels_subset.py

Assunzioni:
- labels.json: lista di dict con campo:
    {"filename": "img000001", ...}   # 1-based

- index_map.json: lista di dict con:
    {"new_index":0, "orig_index":385, ...}  # entrambi 0-based

Output:
- labels_subset.json (lista)
- filename rinominato come img{new_index+1:06d}
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def dump_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_label_index_1based(filename: str) -> int:
    """
    img000001 -> 1
    """
    m = re.fullmatch(r"img0*([0-9]+)", filename)
    if not m:
        raise ValueError(f"Invalid filename format: {filename}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_map", type=Path, required=True)
    ap.add_argument("--labels_in", type=Path, required=True)
    ap.add_argument("--labels_out", type=Path, required=True)
    args = ap.parse_args()

    mapping = load_json(args.index_map)
    labels = load_json(args.labels_in)

    if not isinstance(mapping, list):
        raise RuntimeError("index_map must be a list")
    if not isinstance(labels, list):
        raise RuntimeError("labels must be a list")

    # Build: orig_index_1based -> new_index_0based
    orig1_to_new0: Dict[int, int] = {}

    for row in mapping:
        if "orig_index" not in row:
            raise RuntimeError("index_map must contain 'orig_index' for this script")

        orig0 = int(row["orig_index"])   # orig_index is 0-based, convert to 1-based
        new0 = int(row["new_index"])      
        orig1 = orig0 + 1
        orig1_to_new0[orig1] = new0

    subset: List[dict] = []

    kept = 0
    dropped = 0

    for rec in labels:
        orig1 = parse_label_index_1based(rec["filename"])

        if orig1 not in orig1_to_new0:
            dropped += 1
            continue

        new0 = orig1_to_new0[orig1]
        new_fname = f"img{new0+1:06d}"   # 1-based in labels

        rec2 = dict(rec)
        rec2["filename"] = new_fname
        subset.append(rec2)
        kept += 1

    dump_json(subset, args.labels_out)

    print("✅ labels_subset.json written:", args.labels_out)
    print("Original labels:", len(labels))
    print("Subset size:", kept)
    print("Dropped:", dropped)
    print("Mapping size:", len(mapping))

    if kept != len(mapping):
        print("⚠️ WARNING: kept != mapping size (possible mismatch in indexing assumptions)")


if __name__ == "__main__":
    main()
