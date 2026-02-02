#!/usr/bin/env python3
"""
ply_cleaner.py

Keep only the connected component with the largest det(cov) computed from
N surface-sampled points (default 10k).

Output:
- Prints, for each component: det(cov) and eigenvalues of cov (descending).
- Saves the selected component as a cleaned mesh.

Example:
python ply_cleaner.py \
  --in_mesh  "/path/to/pred.ply" \
  --out_mesh "/path/to/pred_clean.ply" \
  --n_points 10000 \
  --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


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


def sample_points_on_surface(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """
    Sample points uniformly on the mesh surface.
    If mesh has no faces, fall back to vertices (rare for .ply outputs, but safe).
    """
    if mesh.faces is None or len(mesh.faces) == 0:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        if v.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)
        # sample up to n_points vertices without replacement
        m = min(int(n_points), v.shape[0])
        idx = np.random.choice(v.shape[0], size=m, replace=False)
        return v[idx]

    pts, _ = trimesh.sample.sample_surface(mesh, int(n_points))
    return np.asarray(pts, dtype=np.float64)


def det_cov_and_eigs(points: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute det(cov(points)) and eigenvalues of cov (descending).
    Uses eigvalsh for numerical stability.
    """
    if points.shape[0] < 3:
        return 0.0, np.zeros(3, dtype=np.float64)

    cov = np.cov(points.T, bias=False)  # 3x3
    cov = 0.5 * (cov + cov.T)  # symmetrize to avoid numerical issues

    eigvals = np.linalg.eigvalsh(cov)  # ascending
    eigvals = np.maximum(eigvals, 0.0)  # clamp tiny negatives from numerics
    eig_desc = eigvals[::-1]  # descending

    det_cov = float(np.prod(eigvals))  # product of eigenvalues
    return det_cov, eig_desc


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_mesh", type=Path, required=True, help="Input mesh (.ply/.obj/...).")
    p.add_argument("--out_mesh", type=Path, required=True, help="Output cleaned mesh (.ply recommended).")
    p.add_argument("--n_points", type=int, default=10_000, help="Surface points sampled per component.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed. Set -1 to disable seeding.")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.in_mesh.exists():
        raise FileNotFoundError(f"in_mesh not found: {args.in_mesh}")

    if args.seed != -1:
        np.random.seed(int(args.seed))

    mesh = load_mesh(args.in_mesh)

    # Split into connected components (no watertight requirement)
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        raise ValueError("No components found (empty mesh after loading).")

    best_idx = None
    best_score = -np.inf

    print(f"[INFO] components: {len(components)}")
    for i, comp in enumerate(components):
        pts = sample_points_on_surface(comp, args.n_points)
        det_cov, eigs = det_cov_and_eigs(pts)

        print(
            f"component {i:03d} | det_cov={det_cov:.6e} | "
            f"eig=[{eigs[0]:.6e}, {eigs[1]:.6e}, {eigs[2]:.6e}]"
        )

        if det_cov > best_score:
            best_score = det_cov
            best_idx = i

    if best_idx is None:
        raise ValueError("Could not select a component (all invalid).")

    cleaned = components[best_idx]
    args.out_mesh.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(args.out_mesh)

    print(f"\n[SELECTED] component {best_idx:03d} with det_cov={best_score:.6e}")
    print(f"[SAVED] {args.out_mesh}")


if __name__ == "__main__":
    main()
