#!/usr/bin/env python3
"""
Plot camera view selections (train / optional val) from SPE3R cameras_spe3r.npz + split txt files.

Usage examples:
  python plot_views.py --npz /path/cameras_spe3r.npz --train /path/hst_black_mix_32.txt --prefix black
  python plot_views.py --npz ... --train ... --val ... --prefix black --out plot.png
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
    """Camera center C from 3x4 projection matrix via nullspace (SVD)."""
    _, _, Vt = np.linalg.svd(P3x4)
    C_h = Vt[-1]
    if abs(C_h[-1]) > 1e-12:
        C_h = C_h / C_h[-1]
    return C_h[:3]


def load_centers_and_dirs(npz_path: Path):
    """Load camera centers and view directions from cameras_spe3r.npz."""
    data = np.load(str(npz_path), allow_pickle=True)

    ids = sorted(
        [int(k.split("_")[-1]) for k in data.keys() if k.startswith("world_mat_")]
    )

    centers, vdirs = [], []
    for i in ids:
        W = data[f"world_mat_{i}"]
        S = data[f"scale_mat_{i}"]
        P4 = W @ S
        P = P4[:3, :4].astype(np.float64)

        C = camera_center_from_P_svd(P)
        centers.append(C)

        # Approx view direction: -R^T z_cam
        R = P4[:3, :3].astype(np.float64)
        v = -R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-12)
        vdirs.append(v)

    return np.array(ids, dtype=int), np.stack(centers), np.stack(vdirs)


def pca_plane_basis(X: np.ndarray):
    """Return two principal directions spanning the best-fit plane."""
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    e2 = Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
    return e1, e2


def nmc_points(k: int, delta_x: float, omega: float = 1.0):
    """Parametric NMC ellipse-like orbit in (x,z)."""
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


# -------------------------
# Plotting
# -------------------------
def plot_selection(
    npz_path: Path,
    train_txt: Path,
    prefix: str,
    val_txt: Path | None = None,
    k_orbit: int = 64,
    show_orbit: bool = True,
    out_path: Path | None = None,
    title: str | None = None,
):
    # Load all cameras
    ids_all, centers_all, _ = load_centers_and_dirs(npz_path)
    id_to_idx_all = {int(i): j for j, i in enumerate(ids_all)}

    # Restrict to subset (black/earth/all) for background scatter + orbit basis
    mask = subset_mask_from_prefix(ids_all, prefix)
    ids_sub = ids_all[mask]
    centers_sub = centers_all[mask]

    # Target center + PCA plane for orbit reference
    target = centers_sub.mean(axis=0)
    X = centers_sub - target
    e1, e2 = pca_plane_basis(X)

    r = np.linalg.norm(centers_sub - target, axis=1)
    delta_x = 2.0 * float(np.median(r))

    desired_pos = None
    if show_orbit:
        x, z = nmc_points(k_orbit, delta_x, omega=1.0)
        desired_pos = target[None, :] + x[:, None] * e1[None, :] + z[:, None] * e2[None, :]

    # Load selections
    train_vids = read_split_txt(train_txt)
    train_centers = np.array([centers_all[id_to_idx_all[v]] for v in train_vids])

    val_centers = None
    if val_txt is not None:
        val_vids = read_split_txt(val_txt)
        overlap = set(train_vids) & set(val_vids)
        if overlap:
            print(f"[WARN] train/val overlap detected ({len(overlap)}). Example: {sorted(list(overlap))[:5]}")
        val_centers = np.array([centers_all[id_to_idx_all[v]] for v in val_vids])

    # Plot
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        centers_sub[:, 0], centers_sub[:, 1], centers_sub[:, 2],
        s=5, alpha=0.2, label="All camera centers"
    )

    if desired_pos is not None:
        ax.plot(
            desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2],
            "k--", linewidth=2, label=f"Desired NMC orbit (k={k_orbit})"
        )

    ax.scatter(
        train_centers[:, 0], train_centers[:, 1], train_centers[:, 2],
        s=70, marker="^", label=f"Train ({len(train_centers)})"
    )

    if val_centers is not None:
        ax.scatter(
            val_centers[:, 0], val_centers[:, 1], val_centers[:, 2],
            s=70, marker="o", label=f"Validation ({len(val_centers)})"
        )

    ax.scatter(target[0], target[1], target[2], c="red", s=110, marker="*", label="Target center")

    if title is None:
        base = train_txt.stem
        title = f"{prefix} — {base}"
        if val_txt is not None:
            title += " (+val overlay)"
    ax.set_title(title)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
        print(f"[OK] saved figure to: {out_path}")
    else:
        plt.show()

    plt.close(fig)


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot camera selections from split txt files.")
    parser.add_argument("--npz", type=str, required=True, help="Path to cameras_spe3r.npz")
    parser.add_argument("--train", type=str, required=True, help="Path to train split .txt")
    parser.add_argument("--prefix", type=str, default="black", help="black | earth | all (for background subset)")
    parser.add_argument("--val", type=str, default=None, help="Optional path to val split .txt (overlay)")
    parser.add_argument("--k_orbit", type=int, default=64, help="Number of points to draw desired NMC orbit")
    parser.add_argument("--no_orbit", action="store_true", help="Disable desired orbit overlay")
    parser.add_argument("--out", type=str, default=None, help="If provided, save figure instead of showing")
    parser.add_argument("--title", type=str, default=None, help="Optional custom plot title")

    args = parser.parse_args()

    plot_selection(
        npz_path=Path(args.npz),
        train_txt=Path(args.train),
        prefix=args.prefix,
        val_txt=(Path(args.val) if args.val else None),
        k_orbit=args.k_orbit,
        show_orbit=(not args.no_orbit),
        out_path=(Path(args.out) if args.out else None),
        title=args.title,
    )


if __name__ == "__main__":
    main()
