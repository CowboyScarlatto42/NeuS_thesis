#!/usr/bin/env python3
"""
views_selector.py

Select views on a tilted circular orbit and write:
- views txt (e.g., 000123.png or img000123 depending on --filename_out_mode)
- labels_subset.json containing ONLY the selected entries (filtrated from labels_json),
  with 'filename' rewritten consistently with --filename_out_mode.

labels_json format supported:
- list of dicts OR {"data":[...]}
- each entry filename can be like "img000001" (no ext) etc.

"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> rotation matrix."""
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w) + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),       2*(xz+wy)],
        [2*(xy+wz),         1 - 2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),         2*(yz+wx),       1 - 2*(xx+yy)]
    ], dtype=np.float64)


def extract_numeric_id(name: str):
    """Extract last up-to-6 digits from any string like img000001 / 000001.png / etc."""
    s = Path(str(name)).stem
    digits = "".join([c for c in s if c.isdigit()])
    return int(digits[-6:]) if digits else None


def id_to_name(idx: int, mode: str) -> str:
    """
    mode:
      - 'png6' -> 000123.png
      - 'img6' -> img000123
    """
    if mode == "png6":
        return f"{idx:06d}.png"
    if mode == "img6":
        return f"img{idx:06d}"
    raise ValueError(f"Unknown filename mode: {mode}")


def read_train_ids(train_txt: str) -> set:
    train_ids = set()
    with open(train_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            i = extract_numeric_id(s)
            if i is not None:
                train_ids.add(int(i))
    return train_ids


def read_many_train_ids(train_txt_list) -> set:
    exclude = set()
    for p in train_txt_list:
        if p is None:
            continue
        if not os.path.isfile(p):
            raise FileNotFoundError(f"--train_txt file not found: {p}")
        exclude.update(read_train_ids(p))
    return exclude


def orbit_frame_from_tilt_x(tilt_deg: float):
    tilted_angle = np.deg2rad(tilt_deg)
    c, s = np.cos(tilted_angle), np.sin(tilted_angle)
    n = np.array([0.0, -s, c], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    u = u / (np.linalg.norm(u) + 1e-12)
    return u, v, n


def select_validation_from_bin(idxs_in_bin, ids, dists, exclude_ids):
    if idxs_in_bin.size == 0:
        return None
    idx_candidates_sorted = idxs_in_bin[np.argsort(dists[idxs_in_bin])]
    for j in idx_candidates_sorted:
        if int(ids[j]) not in exclude_ids:
            return int(j)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val"], required=True)
    parser.add_argument("--labels_json", type=str, required=True)
    parser.add_argument("--split", choices=["black", "earth"], required=True)
    parser.add_argument("--views_counter", type=int, required=True)
    parser.add_argument("--orbit_samples", type=int, default=256)
    parser.add_argument("--tilt_deg", type=float, required=True)
    parser.add_argument("--train_txt", action="append", default=[],
                        help="(val only) repeatable: --train_txt a.txt --train_txt b.txt")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--out_name", type=str, default=None)

    # NEW
    parser.add_argument("--labels_out", type=str, default=None,
                        help="Output path for labels subset (default: out_dir/labels_subset_*.json)")
    parser.add_argument("--filename_out_mode", choices=["png6", "img6"], default="png6",
                        help="How to write filenames in txt AND in labels_subset.json")

    args = parser.parse_args()

    if args.mode == "val" and len(args.train_txt) == 0:
        raise ValueError("--train_txt is required when --mode val")

    IDX_MIN, IDX_MAX = (0, 499) if args.split == "black" else (500, 999)
    K_SELECT = int(args.views_counter)
    K_ORBIT_SAMPLES = int(args.orbit_samples)
    tilt_deg = float(args.tilt_deg)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.out_name is not None:
        out_name = args.out_name
    else:
        out_name = f"{args.mode}_views_{args.split}_tilt{tilt_deg:g}_K{K_SELECT}.txt"
    out_path = os.path.join(args.out_dir, out_name)

    # labels_out default
    if args.labels_out is None:
        labels_out = os.path.join(args.out_dir, f"labels_subset_{args.mode}_{args.split}_tilt{tilt_deg:g}_K{K_SELECT}.json")
    else:
        labels_out = args.labels_out

    exclude_ids = set()
    if args.mode == "val":
        exclude_ids = read_many_train_ids(args.train_txt)
        print(f"[INFO] loaded {len(exclude_ids)} training ids from {len(args.train_txt)} txt files")
    else:
        print("[INFO] mode=train (no exclusions)")

    if not os.path.isfile(args.labels_json):
        raise FileNotFoundError(f"labels_json not found: {args.labels_json}")

    with open(args.labels_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    if not isinstance(entries, list):
        raise ValueError("labels_json must be a list or a dict with key 'data'")

    ids = []
    pT_C = []
    entry_by_id = {}

    for e in entries:
        i = extract_numeric_id(e.get("filename", ""))
        if i is None or i < IDX_MIN or i > IDX_MAX:
            continue

        q = np.array(e["q_vbs2tango_true"], dtype=np.float64)
        t = np.array(e["r_Vo2To_vbs_true"], dtype=np.float64)

        R_CT = quat_xyzw_to_R(q)
        p = -R_CT.T @ t

        ids.append(i)
        pT_C.append(p)
        entry_by_id[int(i)] = e

    if len(ids) == 0:
        raise RuntimeError(f"No poses found in range {IDX_MIN}..{IDX_MAX}.")

    ids = np.array(ids, dtype=int)
    pT_C = np.stack(pT_C, axis=0)

    print(f"[INFO] loaded {len(ids)} poses in range {ids.min()}..{ids.max()}")

    u, v, n = orbit_frame_from_tilt_x(tilt_deg)
    pu = pT_C @ u
    pv = pT_C @ v
    pn = pT_C @ n

    rad_uv = np.sqrt(pu*pu + pv*pv)
    r0 = float(np.median(rad_uv))
    dists = np.sqrt((rad_uv - r0)**2 + pn**2)

    az = np.arctan2(pv, pu)
    az = (az + 2*np.pi) % (2*np.pi)
    bin_id = np.floor(K_SELECT * az / (2*np.pi)).astype(int)
    bin_id = np.clip(bin_id, 0, K_SELECT-1)

    selected_indices = []
    for k in range(K_SELECT):
        idxs_in_bin = np.where(bin_id == k)[0]
        if idxs_in_bin.size == 0:
            continue
        idx_selected = select_validation_from_bin(
            idxs_in_bin=idxs_in_bin, ids=ids, dists=dists, exclude_ids=exclude_ids
        )
        if idx_selected is not None:
            selected_indices.append(idx_selected)

    selected_indices = np.array(selected_indices, dtype=int)
    selected_ids = ids[selected_indices]
    selected_ids_sorted = sorted(selected_ids.tolist())

    print(f"[INFO] selected {len(selected_ids_sorted)}/{K_SELECT} bins")
    print("[SELECTED IDS]", selected_ids_sorted)

    # ---- Write views txt ----
    selected_filenames = [id_to_name(i, args.filename_out_mode) for i in selected_ids_sorted]
    with open(out_path, "w", encoding="utf-8") as f:
        for name in selected_filenames:
            f.write(name + "\n")
    print(f"[INFO] Saved views txt: {out_path}")

    # ---- Write labels_subset.json (filtered + filename rewritten) ----
    labels_subset = []
    for i in selected_ids_sorted:
        e = entry_by_id.get(int(i))
        if e is None:
            continue
        e2 = dict(e)
        e2["filename"] = id_to_name(int(i), args.filename_out_mode) if args.filename_out_mode == "png6" else id_to_name(int(i), "img6")
        # if mode png6 -> filename "000123.png"
        # if mode img6 -> filename "img000123"
        labels_subset.append(e2)

    # preserve original wrapper
    if isinstance(raw, dict) and "data" in raw:
        out_obj = dict(raw)
        out_obj["data"] = labels_subset
    else:
        out_obj = labels_subset

    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[INFO] Saved labels subset: {labels_out} ({len(labels_subset)} entries)")

    # ---- (plots unchanged - you can keep your plotting code as before) ----
    # (omitted here for brevity)


if __name__ == "__main__":
    main()
