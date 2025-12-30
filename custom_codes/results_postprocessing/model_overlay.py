#!/usr/bin/env python3
"""
Plot predicted meshes (multiple checkpoints) overlaid with a GT mesh using Matplotlib 3D.

Example (Colab / local):
python plot_mesh_overlays.py \
  --mesh_dir "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/meshes" \
  --gt_path  "/content/drive/MyDrive/Tesi/neus/serious_analysis/data/hst_neus/model/model_normalized.obj" \
  --checks 00025000.ply 00050000.ply 00075000.ply 00100000.ply \
  --out "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/post/overlay.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt


def to_trimesh(mesh_or_scene) -> trimesh.Trimesh:
    """Ensure a single trimesh.Trimesh even if the file loads as a Scene."""
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = list(mesh_or_scene.geometry.values())
        if len(geoms) == 0:
            raise ValueError("Loaded an empty trimesh.Scene")
        return trimesh.util.concatenate(geoms)
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    raise TypeError(f"Unsupported type: {type(mesh_or_scene)}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    return to_trimesh(m)


def compute_global_limits(meshes: list[trimesh.Trimesh]):
    verts = [m.vertices for m in meshes]
    xmin = min(v[:, 0].min() for v in verts)
    xmax = max(v[:, 0].max() for v in verts)
    ymin = min(v[:, 1].min() for v in verts)
    ymax = max(v[:, 1].max() for v in verts)
    zmin = min(v[:, 2].min() for v in verts)
    zmax = max(v[:, 2].max() for v in verts)
    return xmin, xmax, ymin, ymax, zmin, zmax


def plot_overlays(
    pred_meshes: list[trimesh.Trimesh],
    names: list[str],
    gt_mesh: trimesh.Trimesh,
    out_path: Path | None,
    elev: float,
    azim: float,
    gt_alpha: float,
    pred_alpha: float,
    figsize: int,
    dpi: int,
    global_limits: bool,
):
    n = len(pred_meshes)
    if n == 0:
        raise ValueError("No predicted meshes to plot.")

    # grid: square-ish
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # global limits
    if global_limits:
        xmin, xmax, ymin, ymax, zmin, zmax = compute_global_limits(pred_meshes + [gt_mesh])

    fig = plt.figure(figsize=(figsize, figsize))
    fig.patch.set_facecolor("black")

    for i, (mesh, name) in enumerate(zip(pred_meshes, names), start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        ax.set_facecolor("black")

        # Pred mesh (orange)
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.faces,
            mesh.vertices[:, 2],
            color="#ff7a18",
            alpha=pred_alpha,
            linewidth=0.0,
            antialiased=True,
        )

        # GT mesh (cyan)
        ax.plot_trisurf(
            gt_mesh.vertices[:, 0],
            gt_mesh.vertices[:, 1],
            gt_mesh.faces,
            gt_mesh.vertices[:, 2],
            color="#2ec4b6",
            alpha=gt_alpha,
            linewidth=0.0,
            antialiased=True,
        )

        # limits
        if global_limits:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)

        # cosmetics
        ax.set_title(name, color="white", fontsize=10)
        ax.tick_params(colors="white")
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0.25)

        ax.view_init(elev=elev, azim=azim)

    # hide empty subplots (if any)
    for j in range(n + 1, rows * cols + 1):
        ax = fig.add_subplot(rows, cols, j, projection="3d")
        ax.set_axis_off()
        ax.set_facecolor("black")

    plt.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[Saved] {out_path}")

    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_dir", type=Path, required=True, help="Directory containing predicted .ply meshes.")
    p.add_argument("--gt_path", type=Path, required=True, help="Path to GT mesh (.obj/.ply/...).")
    p.add_argument(
        "--checks",
        nargs="+",
        required=True,
        help="List of checkpoint mesh filenames inside mesh_dir (e.g. 00025000.ply ...).",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional output image path (png).")
    p.add_argument("--elev", type=float, default=20.0, help="Camera elevation.")
    p.add_argument("--azim", type=float, default=45.0, help="Camera azimuth.")
    p.add_argument("--gt_alpha", type=float, default=0.25, help="GT mesh transparency.")
    p.add_argument("--pred_alpha", type=float, default=1.0, help="Pred mesh transparency.")
    p.add_argument("--figsize", type=int, default=12, help="Figure size (inches, square).")
    p.add_argument("--dpi", type=int, default=200, help="Output DPI (if --out is set).")
    p.add_argument(
        "--no_global_limits",
        action="store_true",
        help="Disable global axis limits (each subplot autoscale).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"mesh_dir not found: {args.mesh_dir}")
    if not args.gt_path.exists():
        raise FileNotFoundError(f"gt_path not found: {args.gt_path}")

    gt_mesh = load_mesh(args.gt_path)

    pred_paths = [args.mesh_dir / ck for ck in args.checks]
    for p in pred_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing predicted mesh: {p}")

    pred_meshes = [load_mesh(p) for p in pred_paths]
    names = [Path(ck).stem for ck in args.checks]

    plot_overlays(
        pred_meshes=pred_meshes,
        names=names,
        gt_mesh=gt_mesh,
        out_path=args.out,
        elev=args.elev,
        azim=args.azim,
        gt_alpha=args.gt_alpha,
        pred_alpha=args.pred_alpha,
        figsize=args.figsize,
        dpi=args.dpi,
        global_limits=not args.no_global_limits,
    )


if __name__ == "__main__":
    main()
