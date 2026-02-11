#!/usr/bin/env python3
"""
views_selector.py

Uso: genera SOLO il file .txt con la lista immagini per NeuS (subset COLMAP).

Input:
- labels_json: labels_subset.json del SUBSET, con entry:
    {"filename":"img000001", "q_vbs2tango_true":[...], "r_Vo2To_vbs_true":[...]}
  dove filename è 1-based (img000001..img000N)

Output:
- .txt con nomi immagini NeuS subset (0-based): 000.png, 001.png, ...

NOTA:
- Non usa split black/earth e non usa range 0..499. Lavora sul subset.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path


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


def extract_subset_id_1based(filename: str):
    """img000001 -> 1"""
    s = Path(str(filename)).stem
    digits = "".join([c for c in s if c.isdigit()])
    return int(digits) if digits else None


def subset_id_to_neus_png(subset_id_1based: int, digits_png: int = 3) -> str:
    """1-based id -> 0-based png name"""
    new0 = subset_id_1based - 1
    return f"{new0:0{digits_png}d}.png"


def read_txt_as_subset_ids_1based(txt_path: str) -> set:
    """
    Legge un txt NeuS (righe tipo 000.png) e ritorna gli id subset 1-based:
      000.png -> 1
      001.png -> 2
    """
    out = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            stem = Path(s).stem
            digits = "".join([c for c in stem if c.isdigit()])
            if not digits:
                continue
            idx0 = int(digits)           # 0-based
            out.add(idx0 + 1)            # -> 1-based
    return out


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


def select_best_in_bin(idxs_in_bin, ids_1based, dists, exclude_ids_1based):
    if idxs_in_bin.size == 0:
        return None
    idx_sorted = idxs_in_bin[np.argsort(dists[idxs_in_bin])]
    for j in idx_sorted:
        if int(ids_1based[j]) not in exclude_ids_1based:
            return int(j)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "val"], required=True)
    p.add_argument("--labels_json", type=str, required=True, help="labels_subset.json (subset-only, img000001..)")
    p.add_argument("--views_counter", type=int, required=True)
    p.add_argument("--tilt_deg", type=float, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--out_name", type=str, default=None)
    p.add_argument("--orbit_samples", type=int, default=256)  # kept for compatibility, not used here
    p.add_argument("--train_txt", action="append", default=[],
                   help="(val only) txt di train da escludere (righe tipo 000.png)")

    args = p.parse_args()

    if args.mode == "val" and len(args.train_txt) == 0:
        raise ValueError("--train_txt is required when --mode val")

    os.makedirs(args.out_dir, exist_ok=True)

    K = int(args.views_counter)
    tilt_deg = float(args.tilt_deg)

    if args.out_name is None:
        out_name = f"{args.mode}_views_tilt{tilt_deg:g}_K{K}.txt"
    else:
        out_name = args.out_name
    out_path = os.path.join(args.out_dir, out_name)

    exclude_ids = set()
    if args.mode == "val":
        for t in args.train_txt:
            exclude_ids |= read_txt_as_subset_ids_1based(t)
        print(f"[INFO] val mode: excluding {len(exclude_ids)} ids from {len(args.train_txt)} train txt files")
    else:
        print("[INFO] train mode: no exclusions")

    with open(args.labels_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    if not isinstance(entries, list):
        raise ValueError("labels_json must be a list or a dict with key 'data'")

    ids_1based = []
    pT_C = []

    for e in entries:
        sid = extract_subset_id_1based(e.get("filename", ""))
        if sid is None:
            continue

        q = np.array(e["q_vbs2tango_true"], dtype=np.float64)
        t = np.array(e["r_Vo2To_vbs_true"], dtype=np.float64)

        R_CT = quat_xyzw_to_R(q)
        p_cam = -R_CT.T @ t

        ids_1based.append(int(sid))
        pT_C.append(p_cam)

    if len(ids_1based) == 0:
        raise RuntimeError("No valid poses found in labels_json.")

    ids_1based = np.array(ids_1based, dtype=int)
    pT_C = np.stack(pT_C, axis=0)

    print(f"[INFO] loaded {len(ids_1based)} poses (subset ids 1-based), range {ids_1based.min()}..{ids_1based.max()}")

    # Orbit selection logic (same as your original)
    u, v, n = orbit_frame_from_tilt_x(tilt_deg)
    pu = pT_C @ u
    pv = pT_C @ v
    pn = pT_C @ n

    rad_uv = np.sqrt(pu*pu + pv*pv)
    r0 = float(np.median(rad_uv))
    dists = np.sqrt((rad_uv - r0)**2 + pn**2)

    az = np.arctan2(pv, pu)
    az = (az + 2*np.pi) % (2*np.pi)
    bin_id = np.floor(K * az / (2*np.pi)).astype(int)
    bin_id = np.clip(bin_id, 0, K-1)

    selected_idx = []
    for k in range(K):
        idxs_in_bin = np.where(bin_id == k)[0]
        if idxs_in_bin.size == 0:
            continue
        j = select_best_in_bin(idxs_in_bin, ids_1based, dists, exclude_ids)
        if j is not None:
            selected_idx.append(j)

    selected_idx = np.array(selected_idx, dtype=int)
    selected_ids = ids_1based[selected_idx]
    selected_ids_sorted = sorted(selected_ids.tolist())

    print(f"[INFO] selected {len(selected_ids_sorted)}/{K} bins")
    print("[SELECTED subset ids (1-based)]", selected_ids_sorted)

    # Write NeuS txt (000.png etc)
    with open(out_path, "w", encoding="utf-8") as f:
        for sid in selected_ids_sorted:
            f.write(subset_id_to_neus_png(sid) + "\n")

    print(f"[INFO] Saved txt for NeuS: {out_path}")


if __name__ == "__main__":
    main()
