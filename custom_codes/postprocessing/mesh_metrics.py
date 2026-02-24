#!/usr/bin/env python3
"""
mesh_metrics.py

Align predicted mesh (COLMAP/NeuS frame) to GT mesh (SPE3R frame),
then compute 3D surface distances (sample-based Chamfer, bidirectional).

Alignment pipeline:
  1. Apply T_align  (manual point-pairs from CloudCompare)
  2. Apply T_icp    (fine ICP from CloudCompare)
  3. Estimate uniform scale from bounding-box diagonal ratio and apply it

3D metrics:
- Sample N points on each mesh surface
- Nearest-neighbor distances pred->gt and gt->pred
- Stats + histograms (fraction-of-points, log-x)

Outputs (if --out_dir is given):
- pred_to_gt.npy, gt_to_pred.npy
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
      2. Estimate uniform scale from bounding-box diagonal ratio.
      3. Apply scale centred on the (already rotated) pred centroid.

    Returns a NEW mesh (original is not modified).
    """
    m = pred.copy()

    # ── Step 1: roto-translation ──────────────────────────────
    m.apply_transform(_T_ROT)

    # ── Step 2: estimate scale per axis, warn if inconsistent ─
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

    scale = float(np.mean(scales))
    print(f"  Using mean scale: {scale:.6f}")

    # ── Step 3: apply uniform scale centred on pred centroid ──
    c = m.centroid
    S = np.eye(4)
    S[:3, :3] *= scale
    S[:3, 3]   = c * (1.0 - scale)
    m.apply_transform(S)
    print("────────────────────────────────────────────────────\n")

    return m


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
        "mean":   float(np.mean(d)),
        "std":    float(np.std(d)),
        "median": float(np.median(d)),
        "p95":    float(np.quantile(d, 0.95)),
        "max":    float(np.max(d)),
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

    bins    = np.logspace(np.log10(d.min()), np.log10(d.max()), n_bins)
    weights = np.ones_like(d) / len(d)

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=bins, weights=weights, alpha=0.75)
    plt.axvline(np.median(d),         color="black", linestyle="--", label="median")
    plt.axvline(np.quantile(d, 0.95), color="red",   linestyle="--", label="p95")
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
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_mesh",    type=Path, required=True)
    ap.add_argument("--gt_mesh",      type=Path, required=True)
    ap.add_argument("--n",            type=int,  default=100_000)
    ap.add_argument("--seed",         type=int,  default=42)
    ap.add_argument("--no_align",     action="store_true",
                    help="Skip alignment (use if pred_mesh is already in SPE3R frame)")
    ap.add_argument("--save_aligned", type=Path, default=None,
                    help="If provided, save the aligned pred mesh to this path")
    ap.add_argument("--out_dir",      type=Path, default=None,
                    help="If provided, save .npy arrays, stats.json and histogram PNGs")
    args = ap.parse_args()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load meshes ───────────────────────────────────────────
    print(f"Loading pred mesh: {args.pred_mesh}")
    pred = load_mesh(args.pred_mesh)
    print(f"Loading GT mesh:   {args.gt_mesh}")
    gt   = load_mesh(args.gt_mesh)

    # ── Align pred → SPE3R frame ──────────────────────────────
    if not args.no_align:
        pred = align_mesh(pred, gt)
        if args.save_aligned is not None:
            pred.export(args.save_aligned)
            print(f"Aligned mesh saved to: {args.save_aligned}")
    else:
        print("Alignment skipped (--no_align).")

    # ── Sample surfaces ───────────────────────────────────────
    P = sample_surface(pred, args.n, seed=args.seed)
    G = sample_surface(gt,   args.n, seed=args.seed + 1)

    # ── 3D Chamfer ────────────────────────────────────────────
    treeG = cKDTree(G)
    dP, _ = treeG.query(P, k=1, workers=-1)   # pred → gt

    treeP = cKDTree(P)
    dG, _ = treeP.query(G, k=1, workers=-1)   # gt → pred

    sP = stats(dP)
    sG = stats(dG)
    chamfer_sym = float(np.mean(dP) + np.mean(dG))

    print_stats("Pred → GT statistics (3D)", sP)
    print_stats("GT → Pred statistics (3D)", sG)
    print(f"\nSymmetric Chamfer (mean, 3D): {chamfer_sym:.6e}")

    plot_histogram_fraction(
        dP, "Distance error distribution (pred → gt)",
        xlabel="Distance (SPE3R units)",
        save_path=(args.out_dir / "hist_pred_to_gt.png" if args.out_dir else None),
    )
    plot_histogram_fraction(
        dG, "Distance error distribution (gt → pred)",
        xlabel="Distance (SPE3R units)",
        save_path=(args.out_dir / "hist_gt_to_pred.png" if args.out_dir else None),
    )

    # ── Save outputs ──────────────────────────────────────────
    if args.out_dir is not None:
        np.save(args.out_dir / "pred_to_gt.npy", dP)
        np.save(args.out_dir / "gt_to_pred.npy", dG)
        payload = {
            "pred_to_gt_3d":        sP,
            "gt_to_pred_3d":        sG,
            "symmetric_chamfer_3d": chamfer_sym,
            "n":    args.n,
            "seed": args.seed,
        }
        with open(args.out_dir / "stats.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
