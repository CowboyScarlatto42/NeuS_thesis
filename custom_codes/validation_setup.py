#!/usr/bin/env python3
"""
make_val_fixed16_nmc.py

Generate a fixed validation split (e.g. 16 views) that lies on the SAME
approximate NMC orbit used for training selection (PCA plane + NMC ellipse),
while ensuring the selected validation views are DISJOINT from the train_32 views.

Outputs a .txt with lines like:
  148.png
  021.png

Also (optional) saves/displays a 3D plot showing:
- all camera centers
- desired NMC orbit (for validation points)
- selected validation cameras
- (optional) train32 cameras
"""

import argparse
import re
from pathlib import Path
import numpy as np

# ---------- parsing + naming ----------
def parse_split_file(split_path: Path) -> list[int]:
    idxs = []
    for raw in split_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.search(r"(\d+)", line)
        if not m:
            raise ValueError(f"Could not find an index in line: '{line}' ({split_path})")
        idxs.append(int(m.group(1)))
    return idxs

def viewname(i: int) -> str:
    return f"{int(i):03d}.png"

# ---------- camera center + view dir ----------
def camera_center_from_P_svd(P3x4: np.ndarray) -> np.ndarray:
    _, _, Vt = np.linalg.svd(P3x4)
    C_h = Vt[-1]
    if abs(C_h[-1]) < 1e-12:
        return C_h[:3]
    C_h = C_h / C_h[-1]
    return C_h[:3]

def load_centers_and_dirs(npz_path: Path, ids: list[int]):
    data = np.load(npz_path, allow_pickle=True)
    centers, dirs = [], []
    for i in ids:
        W = data[f"world_mat_{i}"]
        S = data[f"scale_mat_{i}"]
        P4 = (W @ S).astype(np.float64)
        P  = P4[:3, :4]

        C = camera_center_from_P_svd(P)
        centers.append(C)

        R = P4[:3, :3]
        v = -R.T @ np.array([0., 0., 1.], dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-12)
        dirs.append(v)

    return np.stack(centers, axis=0), np.stack(dirs, axis=0)

# ---------- NMC orbit + plane ----------
def pca_plane_basis(X: np.ndarray):
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    e2 = Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
    return e1, e2

def look_at_dirs(positions: np.ndarray, target: np.ndarray):
    v = target[None, :] - positions
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

# ---------- matching ----------
def greedy_match_excluding(
    centers: np.ndarray,
    dirs: np.ndarray,
    ids: np.ndarray,
    desired_pos: np.ndarray,
    desired_dir: np.ndarray,
    exclude_ids: set[int],
    wp: float,
    wa: float,
):
    N = centers.shape[0]
    unused = np.ones(N, dtype=bool)

    for j in range(N):
        if int(ids[j]) in exclude_ids:
            unused[j] = False

    chosen = []
    for t in range(desired_pos.shape[0]):
        dp = np.linalg.norm(centers - desired_pos[t], axis=1)
        cosang = np.clip(dirs @ desired_dir[t], -1.0, 1.0)
        da = np.arccos(cosang)
        score = wp * dp + wa * da
        score[~unused] = np.inf

        j = int(np.argmin(score))
        if not np.isfinite(score[j]):
            raise RuntimeError("No available view left to match (too many exclusions).")
        chosen.append(j)
        unused[j] = False

    return chosen

# ---------- plotting ----------
def plot_selection(
    all_centers: np.ndarray,
    desired_orbit: np.ndarray,
    val_centers: np.ndarray,
    target: np.ndarray,
    title: str,
    train_centers: np.ndarray | None = None,
    plot_png: Path | None = None,
    show_plot: bool = False,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(all_centers[:, 0], all_centers[:, 1], all_centers[:, 2],
               s=5, alpha=0.2, label="All camera centers")

    ax.plot(desired_orbit[:, 0], desired_orbit[:, 1], desired_orbit[:, 2],
            "k--", linewidth=2, label="Desired NMC orbit")

    if train_centers is not None:
        ax.scatter(train_centers[:, 0], train_centers[:, 1], train_centers[:, 2],
                   s=45, marker="^", label="Train views (k=32)")

    ax.scatter(val_centers[:, 0], val_centers[:, 1], val_centers[:, 2],
               s=60, label="Validation views (k=16)")

    ax.scatter(target[0], target[1], target[2],
               s=90, marker="*", label="Target center")

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    if plot_png is not None:
        plot_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_png, dpi=200)
        print(f"[OK] Saved plot to: {plot_png}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", type=str, required=True)
    ap.add_argument("--train32_txt", type=str, required=True)
    ap.add_argument("--out_txt", type=str, required=True)
    ap.add_argument("--range_min", type=int, default=0)
    ap.add_argument("--range_max", type=int, default=499)
    ap.add_argument("--k_val", type=int, default=16)
    ap.add_argument("--k_train_orbit", type=int, default=32)
    ap.add_argument("--omega", type=float, default=1.0)

    # plotting options
    ap.add_argument("--plot_png", type=str, default="",
                    help="If set, saves a PNG plot at this path.")
    ap.add_argument("--show_plot", action="store_true",
                    help="Show plot window (works in notebooks/colab).")
    ap.add_argument("--plot_with_train", action="store_true",
                    help="Overlay train32 camera centers on the plot.")
    args = ap.parse_args()

    npz_path = Path(args.npz_path)
    train32_txt = Path(args.train32_txt)
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    if not train32_txt.exists():
        raise FileNotFoundError(f"Missing {train32_txt}")

    exclude_train = set(parse_split_file(train32_txt))

    all_ids = np.arange(args.range_min, args.range_max + 1, dtype=int)
    centers, vdirs = load_centers_and_dirs(npz_path, all_ids.tolist())

    target = centers.mean(axis=0)
    e1, e2 = pca_plane_basis(centers - target)

    r = np.linalg.norm(centers - target, axis=1)
    r_med = float(np.median(r))
    delta_x = 2.0 * r_med
    wp, wa = 1.0, r_med

    # Interleaved angles: between the 32 training orbit points
    k_train = int(args.k_train_orbit)
    k_val = int(args.k_val)
    t_val = (2.0 * np.pi / k_train) * (np.arange(k_val) + 0.5)

    x_val = -(delta_x / 2.0) * np.cos(args.omega * t_val)
    z_val =  (delta_x / 4.0) * np.sin(args.omega * t_val)

    desired_pos = target[None, :] + x_val[:, None] * e1[None, :] + z_val[:, None] * e2[None, :]
    desired_dir = look_at_dirs(desired_pos, target)

    chosen_idx = greedy_match_excluding(
        centers=centers,
        dirs=vdirs,
        ids=all_ids,
        desired_pos=desired_pos,
        desired_dir=desired_dir,
        exclude_ids=exclude_train,
        wp=wp,
        wa=wa,
    )
    chosen_ids = all_ids[chosen_idx].tolist()

    # sanity
    if set(chosen_ids) & exclude_train:
        raise RuntimeError("Validation intersects training (should not happen).")
    if len(set(chosen_ids)) != len(chosen_ids):
        raise RuntimeError("Duplicate ids in validation (should not happen).")

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(viewname(i) for i in chosen_ids) + "\n")

    print("[OK] Validation split generated")
    print(f" - range         : {args.range_min}..{args.range_max}")
    print(f" - excluded train: {len(exclude_train)} views from {train32_txt.name}")
    print(f" - k_val         : {k_val}")
    print(f" - saved to      : {out_txt}")
    print(" - val ids       :", chosen_ids)

    # plot if requested
    plot_path = Path(args.plot_png) if args.plot_png else None
    train_centers = None
    if args.plot_with_train:
        train_ids = sorted(list(exclude_train))
        train_centers, _ = load_centers_and_dirs(npz_path, train_ids)

    val_centers = centers[chosen_idx]
    plot_selection(
        all_centers=centers,
        desired_orbit=desired_pos,
        val_centers=val_centers,
        target=target,
        title=f"NMC orbit-like validation selection (range {args.range_min}-{args.range_max}) — k={k_val}",
        train_centers=train_centers,
        plot_png=plot_path,
        show_plot=args.show_plot,
    )

if __name__ == "__main__":
    main()
# ---------- end of file ----------