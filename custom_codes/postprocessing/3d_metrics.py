import argparse
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# -----------------------------
# Mesh utilities
# -----------------------------
def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    m.remove_unreferenced_vertices()
    m.remove_degenerate_faces()
    return m


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts


# -----------------------------
# Statistics
# -----------------------------
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


# -----------------------------
# Histogram (paper-ready)
# -----------------------------
def plot_histogram(
    d: np.ndarray,
    title: str,
    save_path: Path | None = None,
    n_bins: int = 100,
):
    d = d[d > 0]  # safety for log-scale

    bins = np.logspace(
        np.log10(d.min()),
        np.log10(d.max()),
        n_bins
    )

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=bins, density=True, alpha=0.75)

    plt.axvline(np.median(d), color="black", linestyle="--", label="median")
    plt.axvline(np.quantile(d, 0.95), color="red", linestyle="--", label="p95")

    plt.xscale("log")
    plt.xlabel("Distance [m]")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_mesh", type=Path, required=True)
    ap.add_argument("--gt_mesh", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="If provided, save .npy, stats.json and histogram images"
    )
    args = ap.parse_args()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # load meshes
    pred = load_mesh(args.pred_mesh)
    gt = load_mesh(args.gt_mesh)

    # sample surfaces
    P = sample_surface(pred, args.n, seed=args.seed)
    G = sample_surface(gt, args.n, seed=args.seed + 1)

    # pred -> gt
    treeG = cKDTree(G)
    dP, _ = treeG.query(P, k=1, workers=-1)

    # gt -> pred
    treeP = cKDTree(P)
    dG, _ = treeP.query(G, k=1, workers=-1)

    # statistics
    sP = stats(dP)
    sG = stats(dG)
    symmetric_mean = float(np.mean(dP) + np.mean(dG))

    # print results
    print_stats("Pred → GT statistics", sP)
    print_stats("GT → Pred statistics", sG)
    print(f"\nSymmetric Chamfer (mean): {symmetric_mean:.6e}")

    # save arrays + stats only if out_dir is provided
    if args.out_dir is not None:
        np.save(args.out_dir / "pred_to_gt.npy", dP)
        np.save(args.out_dir / "gt_to_pred.npy", dG)

        with open(args.out_dir / "stats.json", "w") as f:
            json.dump(
                {
                    "pred_to_gt": sP,
                    "gt_to_pred": sG,
                    "symmetric_mean": symmetric_mean,
                    "n": args.n,
                    "seed": args.seed,
                },
                f,
                indent=2,
            )

    # histograms (always shown, optionally saved)
    plot_histogram(
        dP,
        "Chamfer distribution (pred → gt)",
        save_path=(
            args.out_dir / "hist_pred_to_gt.png"
            if args.out_dir is not None
            else None
        ),
    )

    plot_histogram(
        dG,
        "Chamfer distribution (gt → pred)",
        save_path=(
            args.out_dir / "hist_gt_to_pred.png"
            if args.out_dir is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
