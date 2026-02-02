#!/usr/bin/env python3
"""
3d_metrics_m.py

Per checkpoint:
(A) Colormap 3D (in unità mesh, assumendo metri):
    - distance d: pred->gt e gt->pred  (2 PLY)
    - chamfer-local d^2: pred->gt e gt->pred  (2 PLY)
    => totale 4 PLY

(B) Distribuzioni Monte Carlo (R ripetizioni) per metriche globali:
    - chamfer_sq
    - hausdorff
    - hausdorff_95
    Salva istogrammi direttamente come PNG.

Output:
- CSV con statistiche (mean/median/var/p95/p99/max) delle metriche su repeats
- PNG istogrammi per checkpoint
- PLY colormap per checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, distance


# -----------------------
# Robust loading (Scene -> Trimesh)
# -----------------------
def to_trimesh(mesh_or_scene) -> trimesh.Trimesh:
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = list(mesh_or_scene.geometry.values())
        if len(geoms) == 0:
            raise ValueError("Loaded an empty trimesh.Scene")
        return trimesh.util.concatenate(geoms)
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    raise TypeError(f"Unsupported type: {type(mesh_or_scene)}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    return to_trimesh(trimesh.load(path, force="mesh"))


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    return np.asarray(pts, dtype=np.float64)


# -----------------------
# Simple colormap (no matplotlib needed)
# -----------------------
def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def scalar_to_rgb(s: np.ndarray) -> np.ndarray:
    # blue -> cyan -> green -> yellow -> red
    s = _clamp01(s)
    r = _clamp01(1.5 * s - 0.5)
    g = _clamp01(1.5 - 3.0 * np.abs(s - 0.5))
    b = _clamp01(1.0 - 1.5 * s)
    rgb = np.stack([r, g, b], axis=1)
    return (255.0 * rgb).astype(np.uint8)


def save_colored_pointcloud(out_path: Path, points: np.ndarray, values: np.ndarray) -> None:
    """
    Save PLY point cloud with vertex colors from 'values'.
    Uses fixed clipping at 99th percentile for robust visualization.
    """
    v = np.asarray(values, dtype=np.float64)
    hi = float(np.percentile(v, 99.0))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.max(v)) if np.max(v) > 0 else 1.0

    colors = scalar_to_rgb(v / hi)
    pc = trimesh.points.PointCloud(points, colors=colors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pc.export(out_path)


# -----------------------
# Distances + Metrics
# -----------------------
def nn_distances(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    tree = cKDTree(p_to)
    d, _ = tree.query(p_from)
    return np.asarray(d, dtype=np.float64)


def chamfer_sq_from_dists(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray) -> float:
    return float(np.mean(d_pred_to_gt**2) + np.mean(d_gt_to_pred**2))


def hausdorff_scipy(p1: np.ndarray, p2: np.ndarray) -> float:
    f = distance.directed_hausdorff(p1, p2)[0]
    b = distance.directed_hausdorff(p2, p1)[0]
    return float(max(f, b))


def hd95_from_dists(d_pred_to_gt: np.ndarray, d_gt_to_pred: np.ndarray) -> float:
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


def stats_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "var": float(np.var(x, ddof=1)) if x.size > 1 else 0.0,
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


def save_hist(out_path: Path, values: np.ndarray, title: str, xlabel: str, bins: int = 30) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path", type=Path, required=True)
    p.add_argument("--mesh_dir", type=Path, required=True)
    p.add_argument("--checks", nargs="+", required=True)

    p.add_argument("--n_points", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0, help="Base seed. Set -1 to disable seeding.")
    p.add_argument("--repeats", type=int, default=30, help="Monte Carlo repeats (>=1).")

    p.add_argument("--out_dir", type=Path, required=True, help="Output directory (PLY + PNG per checkpoint).")
    p.add_argument("--out_csv", type=Path, required=True, help="CSV output path.")
    p.add_argument("--hist_bins", type=int, default=30, help="Histogram bins.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.gt_path.exists():
        raise FileNotFoundError(f"gt_path not found: {args.gt_path}")
    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"mesh_dir not found: {args.mesh_dir}")

    pred_paths = [args.mesh_dir / ck for ck in args.checks]
    for pth in pred_paths:
        if not pth.exists():
            raise FileNotFoundError(f"Missing predicted mesh: {pth}")

    gt_mesh = load_mesh(args.gt_path)

    rows = []

    for ck, pred_path in zip(args.checks, pred_paths):
        name = Path(ck).stem
        print(f"\n[CHECKPOINT] {name}")

        pred_mesh = load_mesh(pred_path)

        out_base = args.out_dir / name
        out_base.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # (A) 4 colormaps (single reference sampling)
        # -----------------------------
        if args.seed != -1:
            np.random.seed(int(args.seed))

        gt_pts_ref = sample_surface_points(gt_mesh, args.n_points)
        pred_pts_ref = sample_surface_points(pred_mesh, args.n_points)

        d_pred_to_gt_ref = nn_distances(pred_pts_ref, gt_pts_ref)
        d_gt_to_pred_ref = nn_distances(gt_pts_ref, pred_pts_ref)

        # Distance maps (d)
        save_colored_pointcloud(out_base / "pred_to_gt_dist.ply", pred_pts_ref, d_pred_to_gt_ref)
        save_colored_pointcloud(out_base / "gt_to_pred_dist.ply", gt_pts_ref, d_gt_to_pred_ref)

        # Chamfer-local maps (d^2) — now genuinely different from distance maps
        save_colored_pointcloud(out_base / "pred_to_gt_chamfer_local.ply", pred_pts_ref, d_pred_to_gt_ref**2)
        save_colored_pointcloud(out_base / "gt_to_pred_chamfer_local.ply", gt_pts_ref, d_gt_to_pred_ref**2)

        # -----------------------------
        # (B) Monte Carlo distributions (metrics)
        # -----------------------------
        R = int(max(1, args.repeats))
        cd_vals = np.zeros(R, dtype=np.float64)
        hd_vals = np.zeros(R, dtype=np.float64)
        hd95_vals = np.zeros(R, dtype=np.float64)

        for r in range(R):
            if args.seed != -1:
                np.random.seed(int(args.seed) + r)

            gt_pts = sample_surface_points(gt_mesh, args.n_points)
            pred_pts = sample_surface_points(pred_mesh, args.n_points)

            d_pred_to_gt = nn_distances(pred_pts, gt_pts)
            d_gt_to_pred = nn_distances(gt_pts, pred_pts)

            cd_vals[r] = chamfer_sq_from_dists(d_pred_to_gt, d_gt_to_pred)
            hd_vals[r] = hausdorff_scipy(gt_pts, pred_pts)
            hd95_vals[r] = hd95_from_dists(d_pred_to_gt, d_gt_to_pred)

        # Save histograms (PNG)
        save_hist(
            out_base / "hist_chamfer_sq.png",
            cd_vals,
            title=f"{name} | Chamfer_sq (R={R})",
            xlabel="chamfer_sq",
            bins=args.hist_bins,
        )
        save_hist(
            out_base / "hist_hausdorff.png",
            hd_vals,
            title=f"{name} | Hausdorff (R={R})",
            xlabel="hausdorff",
            bins=args.hist_bins,
        )
        save_hist(
            out_base / "hist_hausdorff95.png",
            hd95_vals,
            title=f"{name} | Hausdorff95 (R={R})",
            xlabel="hausdorff_95",
            bins=args.hist_bins,
        )

        # Stats + covariance (Chamfer vs Hausdorff)
        cd_stats = stats_1d(cd_vals)
        hd_stats = stats_1d(hd_vals)
        hd95_stats = stats_1d(hd95_vals)
        cov_cd_hd = float(np.cov(cd_vals, hd_vals, ddof=1)[0, 1]) if R > 1 else 0.0

        print(f"  chamfer_sq: mean={cd_stats['mean']:.6e} median={cd_stats['median']:.6e} var={cd_stats['var']:.6e}")
        print(f"  hausdorff : mean={hd_stats['mean']:.6e} median={hd_stats['median']:.6e} var={hd_stats['var']:.6e}")
        print(f"  cov(chamfer_sq, hausdorff): {cov_cd_hd:.6e}")
        print(f"  [saved] 4 colormaps + 3 histograms -> {out_base}")

        rows.append(
            {
                "checkpoint": name,
                "n_points": args.n_points,
                "seed": args.seed,
                "repeats": R,

                "chamfer_sq_mean": cd_stats["mean"],
                "chamfer_sq_median": cd_stats["median"],
                "chamfer_sq_var": cd_stats["var"],
                "chamfer_sq_p95": cd_stats["p95"],
                "chamfer_sq_p99": cd_stats["p99"],
                "chamfer_sq_max": cd_stats["max"],

                "hausdorff_mean": hd_stats["mean"],
                "hausdorff_median": hd_stats["median"],
                "hausdorff_var": hd_stats["var"],
                "hausdorff_p95": hd_stats["p95"],
                "hausdorff_p99": hd_stats["p99"],
                "hausdorff_max": hd_stats["max"],

                "hausdorff95_mean": hd95_stats["mean"],
                "hausdorff95_median": hd95_stats["median"],
                "hausdorff95_var": hd95_stats["var"],
                "hausdorff95_p95": hd95_stats["p95"],
                "hausdorff95_p99": hd95_stats["p99"],
                "hausdorff95_max": hd95_stats["max"],

                "cov_chamfer_sq_hausdorff": cov_cd_hd,
            }
        )

    df = pd.DataFrame(rows).sort_values("checkpoint")
    pd.options.display.float_format = "{:.6e}".format
    print("\n" + df.to_string(index=False))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[Saved] {args.out_csv}")


if __name__ == "__main__":
    main()
