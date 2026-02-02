#!/usr/bin/env python3
"""
views_selector.py

Modes:
- --mode train: select training views on a tilted circular orbit (no exclusion).
- --mode val  : select validation views on a tilted circular orbit, excluding ids listed in --train_txt.

Selection:
- Build an orbit plane by tilting the equatorial plane by --tilt_deg about X_T.
- Project camera positions into the orbit frame (u, v, n).
- Define reference circle radius r0 = median(sqrt(pu^2 + pv^2)).
- Distance metric: sqrt((rad_uv - r0)^2 + pn^2).
- Divide the circular orbit [azimuth angle = atan2(pv, pu)] into K_SELECT sectors and pick best per non-empty sector.
- In val mode, skip candidates whose id is in training set; pick next best.

Output:
- Writes selected image filenames (e.g., 000123.png) to a .txt file.
- Plots at screen: all cameras + tilted orbit + selected views + origin.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


# -------------------------
# Helpers
# -------------------------
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


def parse_idx(filename: str):
    """Extract last up-to-6 digits from filename."""
    digits = "".join([c for c in filename if c.isdigit()])
    return int(digits[-6:]) if len(digits) >= 1 else None


def idx_to_png(idx: int) -> str:
    """Format index as 6-digit png filename."""
    return f"{idx:06d}.png"


def read_train_ids(train_txt: str) -> set:
    """Read training filenames (000123.png) and return set of integer ids."""
    train_ids = set()
    with open(train_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            i = parse_idx(s)
            if i is not None:
                train_ids.add(int(i))
    return train_ids


def orbit_frame_from_tilt_x(tilt_deg: float):
    """
    Build orbit frame (u, v, n) by rotating the equatorial plane about X axis.
    Equatorial plane normal is Z = (0,0,1).
    """
    tilted_angle = np.deg2rad(tilt_deg)
    c, s = np.cos(tilted_angle), np.sin(tilted_angle)

    # Normal obtained by rotating Z about X
    n = np.array([0.0, -s, c], dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # keep X as u
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    u = u / (np.linalg.norm(u) + 1e-12)

    return u, v, n


def select_validation_from_bin(
    idxs_in_bin: np.ndarray,
    ids: np.ndarray,
    dists: np.ndarray,
    exclude_ids: set,
):
    """
    Select one view from a single azimuth bin.

    Rule:
    - sort candidates by increasing distance
    - skip candidates whose image id is in exclude_ids
    - return the first valid candidate, else None
    """
    if idxs_in_bin.size == 0:
        return None

    idx_candidates_sorted = idxs_in_bin[np.argsort(dists[idxs_in_bin])]
    for j in idx_candidates_sorted:
        if int(ids[j]) not in exclude_ids:
            return int(j)
    return None


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "val"], required=True,
                        help="train: build training split; val: build validation split (excludes --train_txt ids)")
    parser.add_argument("--labels_json", type=str, required=True, help="Path to labels.json")
    parser.add_argument("--split", type=str, choices=["black", "earth"], required=True,
                        help="black: use idx 0..499, earth: use idx 500..999")
    parser.add_argument("--views_counter", type=int, required=True, help="Number of azimuth sectors (K_SELECT)")
    parser.add_argument("--orbit_samples", type=int, default=256, help="K_ORBIT_SAMPLES for plotting/orbit discretization")
    parser.add_argument("--tilt_deg", type=float, required=True, help="Orbit plane tilt angle in degrees (about X_T)")
    parser.add_argument("--train_txt", type=str, default=None,
                        help="(val mode only) training views txt (000123.png per line) to exclude")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where output txt will be saved")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Optional output filename (default depends on mode)")
    args = parser.parse_args()

    if args.mode == "val" and not args.train_txt:
        raise ValueError("--train_txt is required when --mode val")

    # Split config
    if args.split == "black":
        IDX_MIN, IDX_MAX = 0, 499
    else:
        IDX_MIN, IDX_MAX = 500, 999

    K_SELECT = int(args.views_counter)
    K_ORBIT_SAMPLES = int(args.orbit_samples)
    tilt_deg = float(args.tilt_deg)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.out_name is not None:
        out_name = args.out_name
    else:
        if args.mode == "train":
            out_name = f"train_views_{args.split}_tilt{tilt_deg:g}_K{K_SELECT}.txt"
        else:
            out_name = f"val_views_{args.split}_tilt{tilt_deg:g}_K{K_SELECT}.txt"

    out_path = os.path.join(args.out_dir, out_name)

    # Exclusions (only for val mode)
    exclude_ids = set()
    if args.mode == "val":
        exclude_ids = read_train_ids(args.train_txt)
        print(f"[INFO] loaded {len(exclude_ids)} training ids from: {args.train_txt}")
    else:
        print("[INFO] mode=train (no exclusions)")

    # ---- load labels ----
    with open(args.labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data["data"] if isinstance(data, dict) and "data" in data else data

    ids = []
    pT_C = []

    for e in entries:
        i = parse_idx(e.get("filename", ""))
        if i is None or i < IDX_MIN or i > IDX_MAX:
            continue

        q = np.array(e["q_vbs2tango_true"], dtype=np.float64)   # target -> camera
        t = np.array(e["r_Vo2To_vbs_true"], dtype=np.float64)   # target origin in camera frame

        R_CT = quat_xyzw_to_R(q)
        p = -R_CT.T @ t          # camera position in target frame

        ids.append(i)
        pT_C.append(p)

    if len(ids) == 0:
        raise RuntimeError(f"No poses found in range {IDX_MIN}..{IDX_MAX}. Check split and labels_json.")

    ids = np.array(ids, dtype=int)
    pT_C = np.stack(pT_C, axis=0)

    print(f"[INFO] loaded {len(ids)} poses in range {ids.min()}..{ids.max()}")
    print(f"[INFO] orbit plane tilt_deg={tilt_deg:g} (about X_T), K_ORBIT_SAMPLES={K_ORBIT_SAMPLES}")

    # ------------------------------------------------
    # Build orbit frame (u,v,n) and project points
    # ------------------------------------------------
    u, v, n = orbit_frame_from_tilt_x(tilt_deg)

    pu = pT_C @ u
    pv = pT_C @ v
    pn = pT_C @ n

    rad_uv = np.sqrt(pu*pu + pv*pv)
    r0 = float(np.median(rad_uv))

    # distance metric: radial-to-circle in plane + out-of-plane distance
    dists = np.sqrt((rad_uv - r0)**2 + pn**2)

    # Oribit division into K_SELECT azimuth sectors
    az = np.arctan2(pv, pu)                 # (-pi, pi]
    az = (az + 2*np.pi) % (2*np.pi)         # [0, 2pi)
    bin_id = np.floor(K_SELECT * az / (2*np.pi)).astype(int)
    bin_id = np.clip(bin_id, 0, K_SELECT-1)

    counts = np.bincount(bin_id, minlength=K_SELECT)
    empty_bins = np.where(counts == 0)[0].tolist()

    print(f"[INFO] Binning into K={K_SELECT} sectors (orbit plane)")
    print(f"[INFO] Empty sectors BEFORE exclusion ({len(empty_bins)}): {empty_bins if empty_bins else 'None'}")

    # ------------------------------------------------
    # Pick best per bin (with optional exclusion)
    # ------------------------------------------------
    selected_indices = []
    empty_after_exclusion = []

    # exclude_ids = empty in train mode --> no effect
    # exclude_ids = non-empty in val mode --> skip those ids

    for k in range(K_SELECT):
        idxs_in_bin = np.where(bin_id == k)[0]
        if idxs_in_bin.size == 0:
            continue

        idx_selected = select_validation_from_bin(
            idxs_in_bin=idxs_in_bin,
            ids=ids,
            dists=dists,
            exclude_ids=exclude_ids,
        )

        if idx_selected is None:
            empty_after_exclusion.append(k)
            continue

        selected_indices.append(idx_selected)

    selected_indices = np.array(selected_indices, dtype=int)
    selected_ids = ids[selected_indices]

    if args.mode == "val":
        print(f"[INFO] excluded ids (from train_txt): {len(exclude_ids)}")
    print(f"[INFO] selected {len(selected_indices)}/{K_SELECT} bins (1 per non-empty bin after exclusion)")
    if empty_after_exclusion:
        print(f"[INFO] Empty sectors AFTER exclusion ({len(empty_after_exclusion)}): {empty_after_exclusion}")
    if len(selected_indices) > 0:
        print(f"[INFO] mean d={dists[selected_indices].mean():.4f}, max d={dists[selected_indices].max():.4f}")
    print("[SELECTED IDS]", sorted(selected_ids.tolist()))

    # Sector summary
    print("\n[SECTOR SUMMARY] k: count | chosen_img_id | chosen_d")
    for k in range(K_SELECT):
        idxs_in_bin = np.where(bin_id == k)[0]

        idx_selected = select_validation_from_bin(
            idxs_in_bin=idxs_in_bin,
            ids=ids,
            dists=dists,
            exclude_ids=exclude_ids,
        )

        if idxs_in_bin.size == 0:
            print(f"{k:02d}: {0:3d} |   ---    |   ---")
        elif idx_selected is None:
            print(f"{k:02d}: {idxs_in_bin.size:3d} |   ---    |   ---")
        else:
            print(f"{k:02d}: {idxs_in_bin.size:3d} | {ids[idx_selected]:7d} | {dists[idx_selected]:.4f}")

    # ---- Write output txt: 000123.png ----
    selected_filenames = [idx_to_png(i) for i in sorted(selected_ids.tolist())]
    with open(out_path, "w", encoding="utf-8") as f:
        for name in selected_filenames:
            f.write(name + "\n")

    print(f"\n[INFO] Saved {len(selected_filenames)} selected views to: {out_path}")

    # ------------------------------------------------
    # Plot tilted orbit + selected views
    # ------------------------------------------------
    t = np.linspace(0.0, 2.0*np.pi, K_ORBIT_SAMPLES, endpoint=False)
    orbit = (r0 * np.cos(t))[:, None] * u[None, :] + (r0 * np.sin(t))[:, None] * v[None, :]

    selected_pts = pT_C[selected_indices] if len(selected_indices) > 0 else np.zeros((0, 3), dtype=np.float64)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pT_C[:, 0], pT_C[:, 1], pT_C[:, 2], s=8, alpha=0.15, label="All cameras (target frame)")
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], linewidth=2, alpha=0.9, label=f"Tilted orbit (tilt={tilt_deg:g}°)")
    ax.scatter(selected_pts[:, 0], selected_pts[:, 1], selected_pts[:, 2], s=50, alpha=0.95,
               label="Selected (train)" if args.mode == "train" else "Selected (val)")
    ax.scatter([0], [0], [0], s=100, marker="*", label="Target (origin)")

    ax.set_title(f"Selection ({args.mode}): tilted orbit + azimuth sectors (orbit plane)")
    ax.set_xlabel("X_T")
    ax.set_ylabel("Y_T")
    ax.set_zlabel("Z_T")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------
    # Plot 2D in the orbit plane (u-v coordinates)
    # ------------------------------------------------
    tt = np.linspace(0.0, 2.0*np.pi, K_ORBIT_SAMPLES, endpoint=False)
    circle_uv = np.stack([r0*np.cos(tt), r0*np.sin(tt)], axis=1)

    plt.figure(figsize=(7, 6))
    plt.scatter(pu, pv, s=8, alpha=0.15, label="All cameras (orbit plane)")
    plt.plot(circle_uv[:, 0], circle_uv[:, 1], linewidth=2, alpha=0.9,
             label="Reference orbit (u-v)")
    if len(selected_indices) > 0:
        plt.scatter(pu[selected_indices], pv[selected_indices],
                    s=50, alpha=0.95,
                    label="Selected (train)" if args.mode == "train" else "Selected (val)")
    plt.scatter([0], [0], s=120, marker="*", label="Target (origin)")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("u (orbit plane)")
    plt.ylabel("v (orbit plane)")
    plt.title(f"Orbit-plane projection (tilt={tilt_deg:g}°)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
