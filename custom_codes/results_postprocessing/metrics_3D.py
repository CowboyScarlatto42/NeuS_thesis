#!/usr/bin/env python3
"""
Compute 3D mesh distance metrics between a GT mesh and multiple predicted meshes.

IMPORTANT:
- No normalization
- No ICP
- Assumes GT and predictions are already aligned in the same coordinate frame.

Metrics (matching the reference script definitions):
- Chamfer (squared): mean(d1^2) + mean(d2^2) using KDTree NN distances
- Hausdorff: max(directed_hausdorff(A,B), directed_hausdorff(B,A)) from SciPy
- Hausdorff95: max(95th percentile NN distances both directions)

Example:
python mesh_metrics_chamfer_hausdorff.py \
  --mesh_dir "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/meshes" \
  --gt_path  "/content/drive/MyDrive/Tesi/neus/serious_analysis/data/hst_neus/model/model_normalized.obj" \
  --checks 00025000.ply 00050000.ply 00075000.ply 00100000.ply \
  --n_points 50000 \
  --seed 0 \
  --out_csv "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/post/metrics_3d.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree, distance


# -----------------------
# Mesh loading (robust for .obj loading as Scene)
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


def mesh_to_pointcloud(mesh_path: Path, n_points: int, seed: int | None) -> np.ndarray:
    """
    Sample points uniformly from mesh surface.
    NOTE: trimesh sampling relies on numpy global RNG in many versions; we seed it for reproducibility.
    """
    if seed is not None:
        np.random.seed(int(seed))
    mesh = load_mesh(mesh_path)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pts, dtype=np.float64)


# -----------------------
# Metrics (match reference script)
# -----------------------
def chamfer_distance_squared(pcd1: np.ndarray, pcd2: np.ndarray) -> float:
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)
    d1, _ = tree1.query(pcd2)  # NN from pcd2 to pcd1
    d2, _ = tree2.query(pcd1)  # NN from pcd1 to pcd2
    return float(np.mean(d1**2) + np.mean(d2**2))


def hausdorff_distance(pcd1: np.ndarray, pcd2: np.ndarray) -> float:
    forward = distance.directed_hausdorff(pcd1, pcd2)[0]
    backward = distance.directed_hausdorff(pcd2, pcd1)[0]
    return float(max(forward, backward))


def hausdorff95(pcd1: np.ndarray, pcd2: np.ndarray) -> float:
    tree1 = cKDTree(pcd1)
    tree2 = cKDTree(pcd2)
    d1, _ = tree1.query(pcd2)
    d2, _ = tree2.query(pcd1)
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path", type=Path, required=True, help="Path to GT mesh (.obj/.ply/...).")
    p.add_argument("--mesh_dir", type=Path, required=True, help="Directory containing predicted meshes.")
    p.add_argument("--checks", nargs="+", required=True, help="Predicted mesh filenames inside mesh_dir.")
    p.add_argument("--n_points", type=int, default=50_000, help="Number of sampled surface points per mesh.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed. Set -1 to disable seeding.")
    p.add_argument("--out_csv", type=Path, default=None, help="Optional output CSV path.")
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

    seed = None if args.seed == -1 else args.seed

    # Sample GT once
    gt_pts = mesh_to_pointcloud(args.gt_path, args.n_points, seed)

    rows = []
    for ck, pred_path in zip(args.checks, pred_paths):
        pred_pts = mesh_to_pointcloud(pred_path, args.n_points, seed)

        cd = chamfer_distance_squared(gt_pts, pred_pts)
        hd = hausdorff_distance(gt_pts, pred_pts)
        hd95 = hausdorff95(gt_pts, pred_pts)

        rows.append(
            {
                "checkpoint": Path(ck).stem,
                "n_points": args.n_points,
                "seed": args.seed,
                "chamfer_sq": cd,
                "hausdorff": hd,
                "hausdorff_95": hd95,
            }
        )

    df = pd.DataFrame(rows).sort_values("checkpoint")

    pd.options.display.float_format = "{:.6e}".format
    print(df.to_string(index=False))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\n[Saved] {args.out_csv}")


if __name__ == "__main__":
    main()
