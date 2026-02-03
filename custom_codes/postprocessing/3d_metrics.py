import argparse
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# ============================================================
# Mesh utilities (version-safe)
# ============================================================
def clean_mesh(m: trimesh.Trimesh) -> trimesh.Trimesh:
    if hasattr(m, "remove_infinite_values"):
        m.remove_infinite_values()
    if hasattr(m, "remove_unreferenced_vertices"):
        m.remove_unreferenced_vertices()
    if hasattr(m, "remove_duplicate_faces"):
        m.remove_duplicate_faces()

    # remove near-degenerate faces (version-independent)
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
    """
    Histogram where each bar height is the fraction of points in that bin.
    Sum of bar heights (weighted by bin membership) equals 1.
    This matches the interpretation: "this bar contains X% of the points".
    """
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if d.size == 0:
        print(f"[WARN] No valid values for histogram: {title}")
        return

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


def focal_from_world_mat(world_mat: np.ndarray) -> float:
    # NeuS convention: world_mat ≈ K [R|t]
    K = world_mat[:3, :3]
    fx = np.linalg.norm(K[0, :])
    fy = np.linalg.norm(K[1, :])
    return 0.5 * (fx + fy)


def depths_for_points(points_world: np.ndarray, world_mat: np.ndarray, scale_mat: np.ndarray) -> np.ndarray:
    N = points_world.shape[0]
    Xh = np.concatenate([points_world, np.ones((N, 1))], axis=1)  # (N,4)
    M = world_mat @ scale_mat
    Xc = (M @ Xh.T).T
    return Xc[:, 2]


def pixel_errors_from_3d(
    points_world: np.ndarray,
    d_3d: np.ndarray,
    world_mats,
    scale_mats,
    z_min: float = 1e-6,
) -> np.ndarray:
    """
    Option B: approximate pixel error e ≈ (f/Z) * d_3d, aggregated with median across valid views.
    Returns per-point pixel error; points invalid in all views become NaN.
    """
    focals = np.array([focal_from_world_mat(W) for W in world_mats], dtype=np.float64)

    Z = np.stack(
        [depths_for_points(points_world, W, S) for W, S in zip(world_mats, scale_mats)],
        axis=0
    )  # (V, N)

    valid = Z > z_min
    E = (focals[:, None] / np.where(valid, Z, np.nan)) * d_3d[None, :]
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
                    help="If provided, compute pixel metrics (Option B) using this cameras_*.npz")
    ap.add_argument("--px_thresh", type=float, default=1.0,
                    help="Threshold in pixels for 'fraction within px' metric")
    ap.add_argument("--z_min", type=float, default=1e-6,
                    help="Minimum positive depth to consider a view valid")

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
    # 3D Chamfer (linear distances)
    # --------------------
    treeG = cKDTree(G)
    dP, _ = treeG.query(P, k=1, workers=-1)  # pred -> gt

    treeP = cKDTree(P)
    dG, _ = treeP.query(G, k=1, workers=-1)  # gt -> pred

    sP = stats(dP)
    sG = stats(dG)
    chamfer_sym = float(np.mean(dP) + np.mean(dG))

    print_stats("Pred → GT statistics (3D)", sP)
    print_stats("GT → Pred statistics (3D)", sG)
    print(f"\nSymmetric Chamfer (mean, 3D): {chamfer_sym:.6e}")

    # Plot 3D histograms (fraction-of-points)
    plot_histogram_fraction(
        dP,
        "Chamfer distribution (pred → gt)",
        xlabel="Distance (NeuS normalized units)",
        save_path=(args.out_dir / "hist_pred_to_gt.png" if args.out_dir else None),
    )
    plot_histogram_fraction(
        dG,
        "Chamfer distribution (gt → pred)",
        xlabel="Distance (NeuS normalized units)",
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
    # Pixel metrics (Option B)
    # --------------------
    if args.cameras_npz is not None:
        world_mats, scale_mats = load_cameras(args.cameras_npz)

        eP = pixel_errors_from_3d(P, dP, world_mats, scale_mats, z_min=args.z_min)  # pred->gt px
        eG = pixel_errors_from_3d(G, dG, world_mats, scale_mats, z_min=args.z_min)  # gt->pred px

        # Drop NaNs: points not valid in any view
        eP = eP[np.isfinite(eP)]
        eG = eG[np.isfinite(eG)]

        sP_px = stats(eP)
        sG_px = stats(eG)

        fracP = float(np.mean(eP <= args.px_thresh))
        fracG = float(np.mean(eG <= args.px_thresh))

        print_stats("Pred → GT statistics (pixel)", sP_px)
        print_stats("GT → Pred statistics (pixel)", sG_px)
        print(f"\nFraction ≤ {args.px_thresh:.2f} px (pred→gt): {100*fracP:.2f}%")
        print(f"Fraction ≤ {args.px_thresh:.2f} px (gt→pred): {100*fracG:.2f}%")

        # Plot pixel histograms (fraction-of-points)
        plot_histogram_fraction(
            eP,
            "Pixel error distribution (pred → gt)",
            xlabel="Pixel error [px]",
            save_path=(args.out_dir / "hist_pred_to_gt_px.png" if args.out_dir else None),
        )
        plot_histogram_fraction(
            eG,
            "Pixel error distribution (gt → pred)",
            xlabel="Pixel error [px]",
            save_path=(args.out_dir / "hist_gt_to_pred_px.png" if args.out_dir else None),
        )

        if args.out_dir is not None:
            np.save(args.out_dir / "pred_to_gt_px.npy", eP)
            np.save(args.out_dir / "gt_to_pred_px.npy", eG)
            payload.update({
                "pred_to_gt_px": sP_px,
                "gt_to_pred_px": sG_px,
                "frac_within_px_thresh_pred_to_gt": fracP,
                "frac_within_px_thresh_gt_to_pred": fracG,
                "px_thresh": args.px_thresh,
                "z_min": args.z_min,
            })

    # --------------------
    # Save stats.json
    # --------------------
    if args.out_dir is not None and payload is not None:
        with open(args.out_dir / "stats.json", "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
