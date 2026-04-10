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
        - scalar field 'dist_norm' (distanza normalizzata rispetto alla dimensione dell'oggetto)
          per legenda in CloudCompare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from plyfile import PlyData, PlyElement


# ============================================================
# Alignment
# ============================================================

# ── Paste your CloudCompare matrices here ───────────────────
_T_ALIGN = np.array([
    [-0.095,  0.075,  0.993, -3.440],
    [ 0.119,  0.991, -0.063,  0.195],
    [-0.988,  0.112, -0.103,  0.161],
    [ 0.000,  0.000,  0.000,  1.000],
])

_T_ICP = np.array([
    [ 1.000,  0.009,  0.007,  0.004],
    [-0.009,  1.000,  0.000,  0.003],
    [-0.007, -0.000,  1.000, -0.001],
    [ 0.000,  0.000,  0.000,  1.000],
])

# Combined roto-translation (ICP refines the manual alignment)
_T_ROT = _T_ICP @ _T_ALIGN


def align_mesh(pred: trimesh.Trimesh, gt: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Bring pred from COLMAP/NeuS frame into SPE3R frame:
      1. Apply combined roto-translation from CloudCompare.
      2. Estimate per-axis scale from bounding-box ratio.
      3. Apply non-uniform scale centred on the (already rotated) pred centroid.

    Returns a NEW mesh (original is not modified).
    """
    m = pred.copy()

    # ── Step 1: roto-translation ──────────────────────────────
    m.apply_transform(_T_ROT)

    # ── Step 2: per-axis scale ────────────────────────────────
    bb_pred = m.bounds[1] - m.bounds[0]
    bb_gt   = gt.bounds[1] - gt.bounds[0]
    scales  = bb_gt / bb_pred

    print("\n── Alignment ────────────────────────────────────────")
    print(f"  BB pred (post-rot): {bb_pred}")
    print(f"  BB gt:              {bb_gt}")
    print(f"  Scale per axis:  X={scales[0]:.4f}  Y={scales[1]:.4f}  Z={scales[2]:.4f}")

    cv = np.std(scales) / np.mean(scales)
    if cv > 0.05:
        print(f"  ⚠  Scale CV={cv:.2%} > 5% — check rotational alignment")
    else:
        print(f"  ✓  Scales consistent (CV={cv:.2%})")

    # ── Step 3: apply non-uniform scale centred on pred centroid
    c = m.centroid
    S = np.eye(4)
    S[0, 0] = scales[0]
    S[1, 1] = scales[1]
    S[2, 2] = scales[2]
    S[:3, 3] = c * (1.0 - scales)
    m.apply_transform(S)
    print("────────────────────────────────────────────────────\n")

    return m


# ============================================================
# Robust loading (Scene -> Trimesh)
# ============================================================
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
# Save point cloud with scalar field 'dist_norm'
# ============================================================
def save_pointcloud_with_dist(out_path: Path, points: np.ndarray, dist_norm: np.ndarray) -> None:
    """
    Save PLY point cloud with:
      - xyz
      - uchar rgb (from dist_norm clipped at p99, for visualization)
      - scalar field:
          * dist_norm (normalized mesh units, RAW values)
    """
    pts = np.asarray(points, dtype=np.float32)
    d   = np.asarray(dist_norm, dtype=np.float64)

    # p99 clipping ONLY for colors
    hi = float(np.percentile(d, 99.0))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.max(d)) if np.max(d) > 0 else 1.0

    colors = scalar_to_rgb(d / hi)

    vertex = np.empty(len(pts), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("dist_norm", "f4"),
    ])

    vertex["x"]         = pts[:, 0]
    vertex["y"]         = pts[:, 1]
    vertex["z"]         = pts[:, 2]
    vertex["red"]       = colors[:, 0]
    vertex["green"]     = colors[:, 1]
    vertex["blue"]      = colors[:, 2]
    vertex["dist_norm"] = d.astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(out_path))


# ============================================================
# Distances
# ============================================================
def nn_distances(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    tree = cKDTree(p_to)
    d, _ = tree.query(p_from)
    return np.asarray(d, dtype=np.float64)


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
    p.add_argument("--no_align",  action="store_true",
                   help="Skip alignment (mesh already in GT/SPE3R frame)")
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

    for ck, pred_path in zip(args.checks, pred_paths):
        name = Path(ck).stem
        print(f"\n[CHECKPOINT] {name}")

        pred_mesh = load_mesh(pred_path)

        # ── Allinea pred → SPE3R frame ────────────────────────
        if not args.no_align:
            pred_mesh = align_mesh(pred_mesh, gt_mesh)
        else:
            print("Alignment skipped (--no_align).")

        out_base = args.out_dir / name
        out_base.mkdir(parents=True, exist_ok=True)

        if args.seed != -1:
            np.random.seed(int(args.seed))

        gt_pts   = sample_surface_points(gt_mesh,   args.n_points)
        pred_pts = sample_surface_points(pred_mesh, args.n_points)

        d_pred_to_gt = nn_distances(pred_pts, gt_pts)
        d_gt_to_pred = nn_distances(gt_pts, pred_pts)

        save_pointcloud_with_dist(out_base / "pred_to_gt.ply", pred_pts, d_pred_to_gt)
        save_pointcloud_with_dist(out_base / "gt_to_pred.ply", gt_pts,   d_gt_to_pred)

        print(f"  [saved] pred_to_gt.ply, gt_to_pred.ply -> {out_base}")


if __name__ == "__main__":
    main()
