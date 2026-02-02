#!/usr/bin/env python3
"""
3d_metrics_px.py

Pixel-analogue of 3d_metrics_m.py using validation views.

Per checkpoint:
(A) 4 colormaps (PLY point clouds) in pixel units:
    - distance d_px: pred->gt and gt->pred
    - chamfer-local d_px^2: pred->gt and gt->pred

(B) Monte Carlo distributions over repeats for global pixel metrics:
    - chamfer_px_sq
    - hausdorff_px
    - hausdorff95_px
    Saves histograms as PNG.

Assumes cameras_npz contains world_mat_i and scale_mat_i.
Uses validation images directory to determine which view indices exist and image size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree


# -----------------------
# Robust mesh loading
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
# Simple RGB ramp for PLY (no matplotlib needed)
# -----------------------
def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def scalar_to_rgb(s: np.ndarray) -> np.ndarray:
    s = _clamp01(s)
    r = _clamp01(1.5 * s - 0.5)
    g = _clamp01(1.5 - 3.0 * np.abs(s - 0.5))
    b = _clamp01(1.0 - 1.5 * s)
    rgb = np.stack([r, g, b], axis=1)
    return (255.0 * rgb).astype(np.uint8)


def save_colored_pointcloud(out_path: Path, points: np.ndarray, values: np.ndarray) -> None:
    # fixed 99th percentile clip for robustness
    v = np.asarray(values, dtype=np.float64)
    v = np.nan_to_num(v, nan=np.nan)  # keep nans for clipping logic below

    finite = np.isfinite(v)
    if not np.any(finite):
        raise ValueError("All values are NaN/Inf; cannot save colormap.")

    hi = float(np.percentile(v[finite], 99.0))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.max(v[finite])) if np.max(v[finite]) > 0 else 1.0

    s = np.zeros_like(v, dtype=np.float64)
    s[finite] = v[finite] / hi
    colors = scalar_to_rgb(s)

    pc = trimesh.points.PointCloud(points, colors=colors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pc.export(out_path)


# -----------------------
# Camera / projection
# -----------------------
def list_val_view_ids(val_image_dir: Path) -> list[int]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    ids = []
    for p in sorted(val_image_dir.iterdir()):
        if p.suffix.lower() in exts and p.stem.isdigit():
            ids.append(int(p.stem))
    if len(ids) == 0:
        raise ValueError(f"No numeric image filenames found in {val_image_dir} (expected 000.png style).")
    return ids


def get_image_size(val_image_dir: Path, view_id: int) -> tuple[int, int]:
    # use the first existing extension
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        p = val_image_dir / f"{view_id:03d}{ext}"
        if p.exists():
            with Image.open(p) as im:
                w, h = im.size
            return int(w), int(h)
    raise FileNotFoundError(f"Could not find image for view_id={view_id:03d} in {val_image_dir}.")


def projection_matrix(npz, view_id: int) -> np.ndarray:
    W = np.asarray(npz[f"world_mat_{view_id}"], dtype=np.float64)  # 4x4
    S = np.asarray(npz[f"scale_mat_{view_id}"], dtype=np.float64)  # 4x4
    return W @ S


def project_points(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    P: (4,4), X: (N,3)
    returns:
      uv: (N,2) pixel coords
      z:  (N,) depth after projection (the homogeneous z used for division)
    """
    N = X.shape[0]
    Xh = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)  # (N,4)
    Y = (P @ Xh.T).T  # (N,4)
    z = Y[:, 2]
    # avoid divide-by-zero; keep invalid flagged via z<=0 later
    uv = np.zeros((N, 2), dtype=np.float64)
    ok = z != 0
    uv[ok, 0] = Y[ok, 0] / z[ok]
    uv[ok, 1] = Y[ok, 1] / z[ok]
    return uv, z


def nn_pairs(p_from: np.ndarray, p_to: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each point in p_from, return:
      d (distance to NN in p_to), idx (index of NN in p_to)
    """
    tree = cKDTree(p_to)
    d, idx = tree.query(p_from)
    return np.asarray(d, dtype=np.float64), np.asarray(idx, dtype=np.int64)


def pixel_distances_multiview(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    view_ids: list[int],
    npz,
    W: int,
    H: int,
    aggregate: str = "median",
) -> np.ndarray:
    """
    For each pair (pts_a[i], pts_b[i]), compute pixel distance across views,
    filtering invalid projections. Aggregate across views (median recommended).
    Returns d_px (N,), NaN if a pair is never valid in any view.
    """
    N = pts_a.shape[0]
    all_d = []

    for vid in view_ids:
        P = projection_matrix(npz, vid)
        ua, za = project_points(P, pts_a)
        ub, zb = project_points(P, pts_b)

        valid = (za > 0) & (zb > 0)
        valid &= (ua[:, 0] >= 0) & (ua[:, 0] < W) & (ua[:, 1] >= 0) & (ua[:, 1] < H)
        valid &= (ub[:, 0] >= 0) & (ub[:, 0] < W) & (ub[:, 1] >= 0) & (ub[:, 1] < H)

        d = np.full(N, np.nan, dtype=np.float64)
        diff = ua - ub
        d[valid] = np.sqrt(np.sum(diff[valid] ** 2, axis=1))
        all_d.append(d)

    D = np.stack(all_d, axis=1)  # (N, V)

    if aggregate == "median":
        return np.nanmedian(D, axis=1)
    if aggregate == "mean":
        return np.nanmean(D, axis=1)
    raise ValueError("aggregate must be 'median' or 'mean'")


# -----------------------
# Metrics + stats + hist
# -----------------------
def stats_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "median": np.nan, "var": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan, "min": np.nan}
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
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(v, bins=bins)
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

    p.add_argument("--cameras_npz", type=Path, required=True)
    p.add_argument("--val_image_dir", type=Path, required=True)

    p.add_argument("--n_points", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0, help="Base seed. Set -1 to disable seeding.")
    p.add_argument("--repeats", type=int, default=30)

    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--hist_bins", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.gt_path.exists():
        raise FileNotFoundError(f"gt_path not found: {args.gt_path}")
    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"mesh_dir not found: {args.mesh_dir}")
    if not args.cameras_npz.exists():
        raise FileNotFoundError(f"cameras_npz not found: {args.cameras_npz}")
    if not args.val_image_dir.exists():
        raise FileNotFoundError(f"val_image_dir not found: {args.val_image_dir}")

    pred_paths = [args.mesh_dir / ck for ck in args.checks]
    for pth in pred_paths:
        if not pth.exists():
            raise FileNotFoundError(f"Missing predicted mesh: {pth}")

    view_ids = list_val_view_ids(args.val_image_dir)
    W, H = get_image_size(args.val_image_dir, view_ids[0])

    npz = np.load(args.cameras_npz)

    gt_mesh = load_mesh(args.gt_path)

    rows = []
    R = int(max(1, args.repeats))

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

        _, idx_pred2gt = nn_pairs(pred_pts_ref, gt_pts_ref)
        _, idx_gt2pred = nn_pairs(gt_pts_ref, pred_pts_ref)

        dpx_pred2gt = pixel_distances_multiview(
            pred_pts_ref,
            gt_pts_ref[idx_pred2gt],
            view_ids=view_ids,
            npz=npz,
            W=W,
            H=H,
            aggregate="median",
        )
        dpx_gt2pred = pixel_distances_multiview(
            gt_pts_ref,
            pred_pts_ref[idx_gt2pred],
            view_ids=view_ids,
            npz=npz,
            W=W,
            H=H,
            aggregate="median",
        )

        # 4 maps
        save_colored_pointcloud(out_base / "pred_to_gt_dist_px.ply", pred_pts_ref, dpx_pred2gt)
        save_colored_pointcloud(out_base / "gt_to_pred_dist_px.ply", gt_pts_ref, dpx_gt2pred)
        save_colored_pointcloud(out_base / "pred_to_gt_chamfer_local_px.ply", pred_pts_ref, dpx_pred2gt**2)
        save_colored_pointcloud(out_base / "gt_to_pred_chamfer_local_px.ply", gt_pts_ref, dpx_gt2pred**2)

        # -----------------------------
        # (B) Monte Carlo distributions (metrics in pixel)
        # -----------------------------
        cd_vals = np.zeros(R, dtype=np.float64)
        hd_vals = np.zeros(R, dtype=np.float64)
        hd95_vals = np.zeros(R, dtype=np.float64)

        for r in range(R):
            if args.seed != -1:
                np.random.seed(int(args.seed) + r)

            gt_pts = sample_surface_points(gt_mesh, args.n_points)
            pred_pts = sample_surface_points(pred_mesh, args.n_points)

            _, idx_pred2gt = nn_pairs(pred_pts, gt_pts)
            _, idx_gt2pred = nn_pairs(gt_pts, pred_pts)

            dpx_pred2gt = pixel_distances_multiview(pred_pts, gt_pts[idx_pred2gt], view_ids, npz, W, H, aggregate="median")
            dpx_gt2pred = pixel_distances_multiview(gt_pts, pred_pts[idx_gt2pred], view_ids, npz, W, H, aggregate="median")

            a = dpx_pred2gt[np.isfinite(dpx_pred2gt)]
            b = dpx_gt2pred[np.isfinite(dpx_gt2pred)]
            if a.size == 0 or b.size == 0:
                cd_vals[r] = np.nan
                hd_vals[r] = np.nan
                hd95_vals[r] = np.nan
                continue

            cd_vals[r] = float(np.mean(a**2) + np.mean(b**2))
            hd_vals[r] = float(max(np.max(a), np.max(b)))
            hd95_vals[r] = float(max(np.percentile(a, 95), np.percentile(b, 95)))

        # Histograms
        save_hist(out_base / "hist_chamfer_px_sq.png", cd_vals, title=f"{name} | Chamfer_px_sq (R={R})", xlabel="chamfer_px_sq", bins=args.hist_bins)
        save_hist(out_base / "hist_hausdorff_px.png", hd_vals, title=f"{name} | Hausdorff_px (R={R})", xlabel="hausdorff_px", bins=args.hist_bins)
        save_hist(out_base / "hist_hd95_px.png", hd95_vals, title=f"{name} | HD95_px (R={R})", xlabel="hd95_px", bins=args.hist_bins)

        # Stats
        cd_stats = stats_1d(cd_vals)
        hd_stats = stats_1d(hd_vals)
        hd95_stats = stats_1d(hd95_vals)

        cov_cd_hd = float(np.cov(cd_vals[np.isfinite(cd_vals)], hd_vals[np.isfinite(hd_vals)], ddof=1)[0, 1]) if np.isfinite(cd_vals).sum() > 1 else 0.0

        print(f"  chamfer_px_sq: mean={cd_stats['mean']:.6e} median={cd_stats['median']:.6e} var={cd_stats['var']:.6e}")
        print(f"  hausdorff_px : mean={hd_stats['mean']:.6e} median={hd_stats['median']:.6e} var={hd_stats['var']:.6e}")
        print(f"  cov(chamfer_px_sq, hausdorff_px): {cov_cd_hd:.6e}")
        print(f"  [saved] 4 colormaps + 3 histograms -> {out_base}")

        rows.append(
            {
                "checkpoint": name,
                "n_points": args.n_points,
                "seed": args.seed,
                "repeats": R,
                "W": W,
                "H": H,
                "n_views": len(view_ids),

                "chamfer_px_sq_mean": cd_stats["mean"],
                "chamfer_px_sq_median": cd_stats["median"],
                "chamfer_px_sq_var": cd_stats["var"],
                "chamfer_px_sq_p95": cd_stats["p95"],
                "chamfer_px_sq_p99": cd_stats["p99"],
                "chamfer_px_sq_max": cd_stats["max"],

                "hausdorff_px_mean": hd_stats["mean"],
                "hausdorff_px_median": hd_stats["median"],
                "hausdorff_px_var": hd_stats["var"],
                "hausdorff_px_p95": hd_stats["p95"],
                "hausdorff_px_p99": hd_stats["p99"],
                "hausdorff_px_max": hd_stats["max"],

                "hd95_px_mean": hd95_stats["mean"],
                "hd95_px_median": hd95_stats["median"],
                "hd95_px_var": hd95_stats["var"],
                "hd95_px_p95": hd95_stats["p95"],
                "hd95_px_p99": hd95_stats["p99"],
                "hd95_px_max": hd95_stats["max"],

                "cov_chamfer_px_sq_hausdorff_px": cov_cd_hd,
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
