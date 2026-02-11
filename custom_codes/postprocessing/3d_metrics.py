#!/usr/bin/env python3
"""
mesh_metrics.py

Compute 3D surface distances (sample-based Chamfer, bidirectional) and
(pixel) reprojection error metrics (true 3D->2D projection, no rendering).

3D metrics:
- Sample N points on each mesh surface
- Nearest-neighbor distances pred->gt and gt->pred
- Stats + histograms (fraction-of-points, log-x)

Pixel metrics (optional):
- Uses cameras_*.npz (world_mat_i, scale_mat_i)
- For each sampled point, pair it with its NN on the other mesh (from KDTree)
- For each view, project both points, compute 2D pixel distance
- Validity: Z > 0 only (no FOV clipping, per your request)
- Aggregate per point as median over views
- Stats + histograms

Outputs (if --out_dir is given):
- pred_to_gt.npy, gt_to_pred.npy
- (optional) pred_to_gt_px.npy, gt_to_pred_px.npy
- hist_*.png
- stats.json
"""

import argparse
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# ============================================================
# Mesh utilities (loading, cleaning, sampling)
# ============================================================
def clean_mesh(m: trimesh.Trimesh) -> trimesh.Trimesh:
    if hasattr(m, "remove_infinite_values"):
        m.remove_infinite_values()
    if hasattr(m, "remove_unreferenced_vertices"):
        m.remove_unreferenced_vertices()
    if hasattr(m, "remove_duplicate_faces"):
        m.remove_duplicate_faces()

    # remove near-degenerate faces
    if hasattr(m, "area_faces") and hasattr(m, "update_faces"):
        mask = m.area_faces > 1e-16
        if mask.shape[0] == len(m.faces) and np.any(~mask):
            m.update_faces(mask)
            if hasattr(m, "remove_unreferenced_vertices"):
                m.remove_unreferenced_vertices()
    return m


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Could not load mesh from {path}")
    return clean_mesh(m)


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts


# ============================================================
# Stats + plotting
# ============================================================
def stats(d: np.ndarray) -> dict:
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "median": float(np.median(d)),
        "p95": float(np.quantile(d, 0.95)),
        "max": float(np.max(d)),
    }


def print_stats(title: str, s: dict):
    print(f"\n{title}")
    for k, v in s.items():
        print(f"  {k:>8s}: {v:.6e}")


def plot_histogram_fraction(
    d: np.ndarray,
    title: str,
    xlabel: str,
    save_path: Path | None = None,
    n_bins: int = 100,
):
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if d.size == 0:
        print(f"[WARN] No valid values for histogram: {title}")
        return

    # log-spaced bins
    bins = np.logspace(np.log10(d.min()), np.log10(d.max()), n_bins)
    weights = np.ones_like(d) / len(d)

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=bins, weights=weights, alpha=0.75)

    plt.axvline(np.median(d), color="black", linestyle="--", label="median")
    plt.axvline(np.quantile(d, 0.95), color="red", linestyle="--", label="p95")

    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Fraction of points")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


# ============================================================
# Camera utilities (for pixel metrics)
# ============================================================
def load_cameras(npz_path: Path):
    data = np.load(npz_path)
    world_mats, scale_mats = [], []
    i = 0
    while f"world_mat_{i}" in data:
        world_mats.append(data[f"world_mat_{i}"])
        scale_mats.append(data[f"scale_mat_{i}"])
        i += 1
    if i == 0:
        raise ValueError(f"No world_mat_i found in {npz_path}")
    return world_mats, scale_mats


def project_uvz(points_world: np.ndarray, world_mat: np.ndarray, scale_mat: np.ndarray):
    """
    Project points using P = (world_mat @ scale_mat)[:3,:4].

    Returns:
      uv: (N,2) pixel coordinates after perspective division
      z : (N,) depth (third component before division). Validity uses z>0 only.
    """
    N = points_world.shape[0]
    Xh = np.concatenate([points_world, np.ones((N, 1))], axis=1)  # (N,4)
    P = (world_mat @ scale_mat)[:3, :4]                           # (3,4)
    x = (P @ Xh.T).T                                              # (N,3)
    z = x[:, 2]
    uv = np.empty((N, 2), dtype=np.float64)
    uv[:, 0] = x[:, 0] / z
    uv[:, 1] = x[:, 1] / z
    return uv, z


def reprojection_pixel_errors_median(
    A: np.ndarray,          # (N,3) points A in world
    B: np.ndarray,          # (N,3) matched points B in world (same length)
    world_mats,
    scale_mats,
):
    """
    True pixel reprojection distance between paired 3D points (A_i, B_i),
    aggregated per-point as median over views.

    Validity criterion only: Z>0 for BOTH points in a given view.
    """
    V = len(world_mats)
    N = A.shape[0]
    E = np.full((V, N), np.nan, dtype=np.float64)

    for i, (Wm, Sm) in enumerate(zip(world_mats, scale_mats)):
        uvA, zA = project_uvz(A, Wm, Sm)
        uvB, zB = project_uvz(B, Wm, Sm)

        valid = np.isfinite(zA) & np.isfinite(zB) & (zA > 0) & (zB > 0)

        diff = uvA - uvB
        e = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

        E[i, valid] = e[valid]

    return np.nanmedian(E, axis=0)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_mesh", type=Path, required=True)
    ap.add_argument("--gt_mesh", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", type=Path, default=None,
                    help="If provided, save .npy arrays, stats.json and histogram PNGs")

    ap.add_argument("--cameras_npz", type=Path, default=None,
                    help="If provided, compute pixel reprojection metrics using this cameras_*.npz")
    ap.add_argument("--px_thresh", type=float, default=1.0,
                    help="Threshold in pixels for 'fraction within px' metric")

    args = ap.parse_args()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Load + sample meshes
    # --------------------
    pred = load_mesh(args.pred_mesh)
    gt = load_mesh(args.gt_mesh)

    P = sample_surface(pred, args.n, seed=args.seed)
    G = sample_surface(gt, args.n, seed=args.seed + 1)

    # --------------------
    # 3D Chamfer (linear distances) + store NN indices
    # --------------------
    treeG = cKDTree(G)
    dP, idxP = treeG.query(P, k=1, workers=-1)  # pred -> gt (idx into G)

    treeP = cKDTree(P)
    dG, idxG = treeP.query(G, k=1, workers=-1)  # gt -> pred (idx into P)

    sP = stats(dP)
    sG = stats(dG)
    chamfer_sym = float(np.mean(dP) + np.mean(dG))

    print_stats("Pred → GT statistics (3D)", sP)
    print_stats("GT → Pred statistics (3D)", sG)
    print(f"\nSymmetric Chamfer (mean, 3D): {chamfer_sym:.6e}")

    # Plot 3D histograms
    plot_histogram_fraction(
        dP,
        "Distance error distribution (pred → gt)",
        xlabel="Distance (normalized units)",
        save_path=(args.out_dir / "hist_pred_to_gt.png" if args.out_dir else None),
    )
    plot_histogram_fraction(
        dG,
        "Distance error distribution (gt → pred)",
        xlabel="Distance (normalized units)",
        save_path=(args.out_dir / "hist_gt_to_pred.png" if args.out_dir else None),
    )

    payload = None
    if args.out_dir is not None:
        np.save(args.out_dir / "pred_to_gt.npy", dP)
        np.save(args.out_dir / "gt_to_pred.npy", dG)
        payload = {
            "pred_to_gt_3d": sP,
            "gt_to_pred_3d": sG,
            "symmetric_chamfer_3d": chamfer_sym,
            "n": args.n,
            "seed": args.seed,
        }

    # --------------------
    # Pixel reprojection metrics (true projection, no rendering)
    # --------------------
    if args.cameras_npz is not None:
        world_mats, scale_mats = load_cameras(args.cameras_npz)

        # Build matched pairs via NN indices from Chamfer computation
        P_match = G[idxP]  # for each pred point, its NN GT point
        G_match = P[idxG]  # for each GT point, its NN pred point

        eP = reprojection_pixel_errors_median(P, P_match, world_mats, scale_mats)  # pred->gt px
        eG = reprojection_pixel_errors_median(G, G_match, world_mats, scale_mats)  # gt->pred px

        # Drop NaNs: points invalid in all views (Z<=0 for all)
        eP = eP[np.isfinite(eP)]
        eG = eG[np.isfinite(eG)]

        sP_px = stats(eP)
        sG_px = stats(eG)

        fracP = float(np.mean(eP <= args.px_thresh)) if eP.size else float("nan")
        fracG = float(np.mean(eG <= args.px_thresh)) if eG.size else float("nan")

        print_stats("Pred → GT statistics (pixel reprojection)", sP_px)
        print_stats("GT → Pred statistics (pixel reprojection)", sG_px)
        print(f"\nFraction ≤ {args.px_thresh:.2f} px (pred→gt): {100*fracP:.2f}%")
        print(f"Fraction ≤ {args.px_thresh:.2f} px (gt→pred): {100*fracG:.2f}%")

        # Plot pixel histograms
        plot_histogram_fraction(
            eP,
            "Pixel reprojection error distribution (pred → gt)",
            xlabel="Pixel error [px]",
            save_path=(args.out_dir / "hist_pred_to_gt_px.png" if args.out_dir else None),
        )
        plot_histogram_fraction(
            eG,
            "Pixel reprojection error distribution (gt → pred)",
            xlabel="Pixel error [px]",
            save_path=(args.out_dir / "hist_gt_to_pred_px.png" if args.out_dir else None),
        )

        if args.out_dir is not None and payload is not None:
            np.save(args.out_dir / "pred_to_gt_px.npy", eP)
            np.save(args.out_dir / "gt_to_pred_px.npy", eG)
            payload.update({
                "pred_to_gt_px_reproj": sP_px,
                "gt_to_pred_px_reproj": sG_px,
                "frac_within_px_thresh_pred_to_gt": fracP,
                "frac_within_px_thresh_gt_to_pred": fracG,
                "px_thresh": args.px_thresh,
                "n_views": len(world_mats),
            })

    # --------------------
    # Save stats.json
    # --------------------
    if args.out_dir is not None and payload is not None:
        with open(args.out_dir / "stats.json", "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
