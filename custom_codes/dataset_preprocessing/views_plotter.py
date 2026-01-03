#!/usr/bin/env python3
"""
Plot camera view selections from SPE3R cameras_spe3r.npz + split txt files.

Supports:
- single selection: --train
- dual selection:   --train_a + --train_b (one plot, two markers, two dashed orbits)
Optional:
- --val overlay

Examples:
  %run plot_views.py --npz ... --train ... --prefix black
  %run plot_views.py --npz ... --train_a ... --train_b ... --prefix black --val ...
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Utilities
# -------------------------
def camera_center_from_P_svd(P3x4: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(P3x4)
    C_h = Vt[-1]
    if abs(C_h[-1]) > 1e-12:
        C_h = C_h / C_h[-1]
    return C_h[:3]


def load_centers(npz_path: Path):
    data = np.load(str(npz_path), allow_pickle=True)
    ids = sorted([int(k.split("_")[-1]) for k in data.keys() if k.startswith("world_mat_")])

    centers = []
    for i in ids:
        W = data[f"world_mat_{i}"]
        S = data[f"scale_mat_{i}"]
        P4 = W @ S
        P = P4[:3, :4].astype(np.float64)
        centers.append(camera_center_from_P_svd(P))
    return np.array(ids, dtype=int), np.stack(centers)


def pca_plane_basis(X: np.ndarray):
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    e2 = Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
    return e1, e2


def nmc_points(k: int, delta_x: float, omega: float = 1.0):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    x = -(delta_x / 2.0) * np.cos(omega * t)
    z = (delta_x / 4.0) * np.sin(omega * t)
    return x, z


def parse_vid_from_line(line: str) -> int:
    base = os.path.basename(line.strip())
    return int(os.path.splitext(base)[0])


def read_split_txt(path: Path):
    with open(path, "r") as f:
        vids = [parse_vid_from_line(x) for x in f if x.strip()]
    return vids


def subset_mask_from_prefix(ids_all: np.ndarray, prefix: str):
    p = prefix.lower()
    if p in ("black", "hst_black"):
        return (ids_all >= 0) & (ids_all <= 499)
    if p in ("earth", "hst_earth"):
        return (ids_all >= 500) & (ids_all <= 999)
    if p in ("all", "full", "both"):
        return np.ones_like(ids_all, dtype=bool)
    raise ValueError(f"Unknown prefix '{prefix}'. Use black|earth|all.")


def compute_orbit_from_centers(centers: np.ndarray, k_orbit: int = 64):
    """
    Build a 'desired orbit' reference curve by fitting PCA plane to provided centers,
    then drawing the NMC ellipse-like curve on that plane using median radius.
    """
    target = centers.mean(axis=0)
    X = centers - target
    e1, e2 = pca_plane_basis(X)

    r = np.linalg.norm(centers - target, axis=1)
    delta_x = 2.0 * float(np.median(r))

    x, z = nmc_points(k_orbit, delta_x, omega=1.0)
    desired = target[None, :] + x[:, None] * e1[None, :] + z[:, None] * e2[None, :]
    return target, desired


# -------------------------
# Plot
# -------------------------
def plot_single_or_dual(
    npz_path: Path,
    prefix: str,
    train_txt: Path | None = None,
    train_a: Path | None = None,
    train_b: Path | None = None,
    val_txt: Path | None = None,
    k_orbit: int = 64,
    show_orbits: bool = True,
    out_path: Path | None = None,
    title: str | None = None,
):
    ids_all, centers_all = load_centers(npz_path)
    id_to_idx = {int(i): j for j, i in enumerate(ids_all)}

    # Background subset
    mask = subset_mask_from_prefix(ids_all, prefix)
    centers_sub = centers_all[mask]

    # Validation (optional)
    val_centers = None
    if val_txt is not None:
        val_vids = read_split_txt(val_txt)
        val_centers = np.array([centers_all[id_to_idx[v]] for v in val_vids])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        centers_sub[:, 0], centers_sub[:, 1], centers_sub[:, 2],
        s=5, alpha=0.2, label="All camera centers"
    )

    # --- SINGLE mode ---
    if train_txt is not None:
        train_vids = read_split_txt(train_txt)
        train_centers = np.array([centers_all[id_to_idx[v]] for v in train_vids])

        # Orbit fitted on the selected centers (optional)
        target, orbit = compute_orbit_from_centers(train_centers, k_orbit=k_orbit)
        if show_orbits:
            ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], "k--", linewidth=2, label=f"Orbit ref (k={k_orbit})")

        ax.scatter(train_centers[:, 0], train_centers[:, 1], train_centers[:, 2],
                   s=70, marker="^", label=f"Train ({len(train_centers)})")
        ax.scatter(target[0], target[1], target[2], c="red", s=110, marker="*", label="Target center")

        if val_centers is not None:
            overlap = set(train_vids) & set(read_split_txt(val_txt))
            if overlap:
                print(f"[WARN] train/val overlap detected ({len(overlap)}). Example: {sorted(list(overlap))[:5]}")
            ax.scatter(val_centers[:, 0], val_centers[:, 1], val_centers[:, 2],
                       s=70, marker="o", label=f"Validation ({len(val_centers)})")

        if title is None:
            title = f"{prefix} — {train_txt.stem}" + (" (+val)" if val_txt else "")

    # --- DUAL mode ---
    else:
        if train_a is None or train_b is None:
            raise ValueError("Dual mode requires --train_a and --train_b (or use --train for single mode).")

        vids_a = read_split_txt(train_a)
        vids_b = read_split_txt(train_b)

        centers_a = np.array([centers_all[id_to_idx[v]] for v in vids_a])
        centers_b = np.array([centers_all[id_to_idx[v]] for v in vids_b])

        # Orbits fitted separately
        target_a, orbit_a = compute_orbit_from_centers(centers_a, k_orbit=k_orbit)
        target_b, orbit_b = compute_orbit_from_centers(centers_b, k_orbit=k_orbit)

        if show_orbits:
            ax.plot(orbit_a[:, 0], orbit_a[:, 1], orbit_a[:, 2], "k--", linewidth=2,
                    label=f"Orbit A (k={k_orbit})")
            ax.plot(orbit_b[:, 0], orbit_b[:, 1], orbit_b[:, 2], "k:", linewidth=2,
                    label=f"Orbit B (k={k_orbit})")

        ax.scatter(centers_a[:, 0], centers_a[:, 1], centers_a[:, 2],
                   s=70, marker="^", label=f"A: {train_a.stem} ({len(centers_a)})")
        ax.scatter(centers_b[:, 0], centers_b[:, 1], centers_b[:, 2],
                   s=70, marker="s", label=f"B: {train_b.stem} ({len(centers_b)})")

        ax.scatter(target_a[0], target_a[1], target_a[2], c="red", s=110, marker="*", label="Target A")
        ax.scatter(target_b[0], target_b[1], target_b[2], c="red", s=90, marker="x", label="Target B")

        if val_centers is not None:
            val_vids = read_split_txt(val_txt)
            overlap_a = set(vids_a) & set(val_vids)
            overlap_b = set(vids_b) & set(val_vids)
            if overlap_a:
                print(f"[WARN] A/val overlap detected ({len(overlap_a)}). Example: {sorted(list(overlap_a))[:5]}")
            if overlap_b:
                print(f"[WARN] B/val overlap detected ({len(overlap_b)}). Example: {sorted(list(overlap_b))[:5]}")

            ax.scatter(val_centers[:, 0], val_centers[:, 1], val_centers[:, 2],
                       s=70, marker="o", label=f"Validation ({len(val_centers)})")

        if title is None:
            title = f"{prefix} — dual selection: {train_a.stem} + {train_b.stem}" + (" (+val)" if val_txt else "")

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
        print(f"[OK] saved: {out_path}")
    else:
        plt.show()

    plt.close(fig)


# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Plot view selections (single or dual txt).")
    p.add_argument("--npz", type=str, required=True, help="Path to cameras_spe3r.npz")
    p.add_argument("--prefix", type=str, default="black", help="black | earth | all")

    # Single mode
    p.add_argument("--train", type=str, default=None, help="Single train split txt")

    # Dual mode
    p.add_argument("--train_a", type=str, default=None, help="Train split A txt")
    p.add_argument("--train_b", type=str, default=None, help="Train split B txt")

    # Optional validation overlay
    p.add_argument("--val", type=str, default=None, help="Optional validation split txt")

    p.add_argument("--k_orbit", type=int, default=64, help="Points used to draw orbit reference")
    p.add_argument("--no_orbits", action="store_true", help="Disable orbit overlays")
    p.add_argument("--out", type=str, default=None, help="Save figure to this path instead of showing")
    p.add_argument("--title", type=str, default=None, help="Custom title")

    args = p.parse_args()

    # Enforce: either single or dual
    single = args.train is not None
    dual = (args.train_a is not None) or (args.train_b is not None)

    if single and dual:
        raise ValueError("Use either --train OR (--train_a and --train_b), not both.")
    if not single and not (args.train_a and args.train_b):
        raise ValueError("Provide --train (single) OR both --train_a and --train_b (dual).")

    plot_single_or_dual(
        npz_path=Path(args.npz),
        prefix=args.prefix,
        train_txt=(Path(args.train) if args.train else None),
        train_a=(Path(args.train_a) if args.train_a else None),
        train_b=(Path(args.train_b) if args.train_b else None),
        val_txt=(Path(args.val) if args.val else None),
        k_orbit=args.k_orbit,
        show_orbits=(not args.no_orbits),
        out_path=(Path(args.out) if args.out else None),
        title=args.title,
    )


if __name__ == "__main__":
    main()
