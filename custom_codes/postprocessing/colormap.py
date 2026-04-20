#!/usr/bin/env python3
"""
colormap.py

Per checkpoint:
(A) Colormap 3D (in unità mesh normalizzate, scala assoluta NON metrica):
    - 2 point cloud:
        * pred_to_gt.ply
        * gt_to_pred.ply
    Ogni cloud contiene:
        - RGB (per visualizzazione immediata, con clipping p99)
        - scalar field 'dist_raw' (distanza grezza in unità mesh)
          per legenda in CloudCompare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plyfile import PlyData, PlyElement

from metrics_utils import load_mesh, nn_distances, sample_surface_points


# ============================================================
# Simple colormap (RGB only for visualization)
# ============================================================
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


# ============================================================
# Save point cloud with scalar field 'dist_raw'
# ============================================================
def save_pointcloud_with_dist(out_path: Path, points: np.ndarray, dist_raw: np.ndarray) -> None:
    """
    Save PLY point cloud with:
      - xyz
      - uchar rgb (from dist_raw clipped at p99, for visualization)
      - scalar field:
          * dist_raw (mesh units, RAW values)
    """
    pts = np.asarray(points, dtype=np.float32)
    d   = np.asarray(dist_raw, dtype=np.float64)

    # p99 clipping ONLY for colors
    hi = float(np.percentile(d, 99.0))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.max(d)) if np.max(d) > 0 else 1.0

    colors = scalar_to_rgb(d / hi)

    vertex = np.empty(len(pts), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("dist_raw", "f4"),
    ])

    vertex["x"]         = pts[:, 0]
    vertex["y"]         = pts[:, 1]
    vertex["z"]         = pts[:, 2]
    vertex["red"]       = colors[:, 0]
    vertex["green"]     = colors[:, 1]
    vertex["blue"]      = colors[:, 2]
    vertex["dist_raw"] = d.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(out_path))


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path",   type=Path, required=True)
    p.add_argument("--mesh_dir",  type=Path, required=True)
    p.add_argument("--checks",    nargs="+", required=True)
    p.add_argument("--n_points",  type=int,  default=50_000)
    p.add_argument("--seed",      type=int,  default=0,
                   help="Set -1 to disable seeding.")
    p.add_argument("--out_dir",   type=Path, required=True)
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

    print("Assuming predicted meshes are already in the same frame as GT.")

    for ck_idx, (ck, pred_path) in enumerate(zip(args.checks, pred_paths)):
        name = Path(ck).stem
        print(f"\n[CHECKPOINT] {name}")

        pred_mesh = load_mesh(pred_path)

        out_base = args.out_dir / name
        out_base.mkdir(parents=True, exist_ok=True)

        if args.seed == -1:
            gt_seed = None
            pred_seed = None
        else:
            # Keep deterministic and explicit sampling per checkpoint.
            gt_seed = int(args.seed) + (2 * ck_idx)
            pred_seed = gt_seed + 1

        gt_pts   = sample_surface_points(gt_mesh,   args.n_points, seed=gt_seed)
        pred_pts = sample_surface_points(pred_mesh, args.n_points, seed=pred_seed)

        d_pred_to_gt = nn_distances(pred_pts, gt_pts)
        d_gt_to_pred = nn_distances(gt_pts, pred_pts)

        save_pointcloud_with_dist(out_base / "pred_to_gt.ply", pred_pts, d_pred_to_gt)
        save_pointcloud_with_dist(out_base / "gt_to_pred.ply", gt_pts,   d_gt_to_pred)

        print(f"  [saved] pred_to_gt.ply, gt_to_pred.ply -> {out_base}")


if __name__ == "__main__":
    main()
