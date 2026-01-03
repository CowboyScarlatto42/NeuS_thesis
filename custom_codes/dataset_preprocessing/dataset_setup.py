"""
CLI utility to build Natural Motion Circumnavigation (NMC) view splits.

Minimal refactor of the original notebook-style script:
- keeps the original helper functions (load_centers_and_dirs, nmc_points, etc.)
- adds PCA helpers, orbit builders, and split IO utilities
- exposes a CLI for generating base/complement/both orbital subsets
"""

import argparse
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---- default paths ----
NPZ_PATH = "/content/drive/MyDrive/Tesi/neus/serious_analysis/data/hst_neus/cameras_spe3r.npz"
SPLIT_DIR = "/content/drive/MyDrive/Tesi/neus/serious_analysis/splits"


# ---- plotting settings ----
LABEL_STEP_TEXT = 40  # controls how many labels to show in the ordering plot


# ============================================================
# Original helper functions (kept with minimal adjustments)
# ============================================================

def camera_center_from_P_svd(P3x4: np.ndarray) -> np.ndarray:
    """Extract camera center as right singular vector associated with nullspace."""

    _, _, Vt = np.linalg.svd(P3x4)
    C_h = Vt[-1]
    if abs(C_h[-1]) < 1e-12:
        return C_h[:3]
    C_h = C_h / C_h[-1]
    return C_h[:3]


def load_centers_and_dirs(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load camera centers and viewing directions from a NeRF-style NPZ archive."""

    data = np.load(npz_path, allow_pickle=True)
    ids = sorted(int(k.split("_")[-1]) for k in data.keys() if k.startswith("world_mat_"))
    if not ids:
        raise RuntimeError("No world_mat_* keys found in NPZ.")

    centers, dirs = [], []
    for i in ids:
        W = data[f"world_mat_{i}"]
        S = data[f"scale_mat_{i}"]
        P4 = W @ S
        P = P4[:3, :4].astype(np.float64)

        centers.append(camera_center_from_P_svd(P))

        R = P4[:3, :3].astype(np.float64)
        v = -R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-12)
        dirs.append(v)

    return np.array(ids), np.stack(centers, axis=0), np.stack(dirs, axis=0)


def pca_basis_3d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three orthonormal PCA axes (e1 >= e2 >= e3 in variance)."""

    if X.shape[0] < 3:
        raise ValueError("Need at least three points for 3D PCA basis.")
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    basis = [vec / (np.linalg.norm(vec) + 1e-12) for vec in Vt[:3]]
    return basis[0], basis[1], basis[2]


def pca_plane_basis(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible helper returning the first two PCA axes."""

    e1, e2, _ = pca_basis_3d(X)
    return e1, e2


def nmc_points(k: int, delta_x: float, omega: float = 1.0, phase: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Natural Motion Circumnavigation trajectory points in a canonical plane."""

    t = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    t = omega * t + phase
    x = -(delta_x / 2.0) * np.cos(t)
    z = (delta_x / 4.0) * np.sin(t)
    return x, z


def look_at_dirs(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute normalized viewing directions aiming at target."""

    v = target[None, :] - positions
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def build_orbit_in_plane(
    target: np.ndarray,
    eA: np.ndarray,
    eB: np.ndarray,
    KMAX: int,
    delta_x: float,
    omega: float = 1.0,
    phase: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate NMC desired positions and directions embedded in plane (eA, eB)."""

    x, z = nmc_points(KMAX, delta_x, omega=omega, phase=phase)
    desired_pos = target[None, :] + x[:, None] * eA[None, :] + z[:, None] * eB[None, :]
    desired_dir = look_at_dirs(desired_pos, target)
    return desired_pos, desired_dir


def greedy_match(
    centers: np.ndarray,
    dirs: np.ndarray,
    desired_pos: np.ndarray,
    desired_dir: np.ndarray,
    wp: float = 1.0,
    wa: float = 1.0,
    forbidden_idx: Optional[Iterable[int]] = None,
) -> List[int]:
    """Greedy nearest-neighbor matching with optional forbidden indices."""

    N = centers.shape[0]
    unused = np.ones(N, dtype=bool)
    if forbidden_idx is not None:
        forbidden = np.array(list(forbidden_idx), dtype=int)
        forbidden = forbidden[(forbidden >= 0) & (forbidden < N)]
        unused[forbidden] = False

    chosen: List[int] = []
    for t in range(desired_pos.shape[0]):
        dp = np.linalg.norm(centers - desired_pos[t], axis=1)
        cosang = np.clip(dirs @ desired_dir[t], -1.0, 1.0)
        da = np.arccos(cosang)
        score = wp * dp + wa * da
        score[~unused] = np.inf
        if not np.isfinite(score).any():
            raise RuntimeError("Greedy match ran out of available views. Try reducing KMAX or forbidden set.")
        j = int(np.argmin(score))
        chosen.append(j)
        unused[j] = False

    return chosen


def viewname(i: int) -> str:
    return f"{int(i):03d}.png"


def nested_from_sequence(sequence: Sequence[int], k: int) -> List[int]:
    """Deterministic nested subsets from a base sequence."""

    if k <= 0:
        raise ValueError("k must be > 0")
    if len(sequence) < k:
        raise ValueError("Sequence shorter than requested subset size.")
    step = max(1, len(sequence) // k)
    return [sequence[i] for i in range(0, len(sequence), step)][:k]


def read_split_txt(path: str) -> List[int]:
    """Read view IDs from a txt file with lines like 000.png."""

    with open(path, "r", encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    ids = [int(name.replace(".png", "")) for name in entries]
    return ids


def write_split_txt(path: str, view_ids: Sequence[int]) -> None:
    """Write view IDs into a txt file with 3-digit names."""

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for vid in view_ids:
            handle.write(viewname(int(vid)) + "\n")


# ============================================================
# Plotting helpers (optional)
# ============================================================

def plot_dataset_ordering(ids: np.ndarray, centers: np.ndarray) -> None:
    idx = np.arange(ids.size)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=idx, s=8, alpha=0.9)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], linewidth=0.8, alpha=0.6)

    step = max(1, ids.size // LABEL_STEP_TEXT)
    for j in range(0, ids.size, step):
        ax.text(centers[j, 0], centers[j, 1], centers[j, 2], str(j), fontsize=7)

    ax.set_title("Camera centers colored by dataset index")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.1)
    cbar.set_label("View index")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

    if ids.size > 1:
        d = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
        plt.figure(figsize=(8, 3))
        plt.plot(d)
        plt.title("Distance between consecutive camera centers")
        plt.xlabel("i (between i and i+1)")
        plt.ylabel("distance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_orbit_diagnostics(
    all_centers: np.ndarray,
    all_dirs: np.ndarray,
    target: np.ndarray,
    desired_pos: np.ndarray,
    selected_idx: Sequence[int],
    name: str,
) -> None:
    selected_centers = all_centers[selected_idx]
    selected_dirs = all_dirs[selected_idx]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(all_centers[:, 0], all_centers[:, 1], all_centers[:, 2], s=5, alpha=0.2, label="All centers")
    ax.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], "k--", linewidth=2, label="Desired orbit")
    ax.scatter(selected_centers[:, 0], selected_centers[:, 1], selected_centers[:, 2], c="orange", s=50, label=f"Selected ({name})")
    ax.scatter(target[0], target[1], target[2], c="red", s=80, marker="*", label="Target")
    ax.set_title(f"NMC selection diagnostics — {name}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

    if selected_centers.shape[0] > 1:
        dist_seq = np.linalg.norm(selected_centers[1:] - selected_centers[:-1], axis=1)
        plt.figure(figsize=(7, 3))
        plt.plot(dist_seq, "-o", markersize=3)
        plt.xlabel("Step along orbit")
        plt.ylabel("Distance between consecutive views")
        plt.title(f"Continuity check — {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(all_dirs[:, 0], all_dirs[:, 1], all_dirs[:, 2], s=5, alpha=0.2, label="All directions")
    ax.scatter(selected_dirs[:, 0], selected_dirs[:, 1], selected_dirs[:, 2], c="orange", s=50, label=f"Selected ({name})")
    ax.set_title(f"Viewing directions (unit sphere) — {name}")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_base_complement(
    centers: np.ndarray,
    target: np.ndarray,
    base_desired: Optional[np.ndarray],
    complement_desired: Optional[np.ndarray],
    base_idx: Optional[Sequence[int]],
    complement_idx: Optional[Sequence[int]],
    plane_label: str,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=5, alpha=0.15, label="All centers")

    if base_desired is not None:
        ax.plot(base_desired[:, 0], base_desired[:, 1], base_desired[:, 2], "--", color="tab:blue", linewidth=2, label="Desired base orbit")
    if complement_desired is not None:
        ax.plot(complement_desired[:, 0], complement_desired[:, 1], complement_desired[:, 2], "--", color="tab:green", linewidth=2, label=f"Desired complement orbit ({plane_label})")

    if base_idx is not None and len(base_idx) > 0:
        base_centers = centers[list(base_idx)]
        ax.scatter(base_centers[:, 0], base_centers[:, 1], base_centers[:, 2], c="tab:blue", s=45, label="Selected base")
    if complement_idx is not None and len(complement_idx) > 0:
        comp_centers = centers[list(complement_idx)]
        ax.scatter(comp_centers[:, 0], comp_centers[:, 1], comp_centers[:, 2], c="tab:green", s=45, label="Selected complement")

    ax.scatter(target[0], target[1], target[2], c="red", s=80, marker="*", label="Target")
    ax.set_title("Base vs complement selections")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


# ============================================================
# CLI helpers
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NMC orbit view splits.")
    parser.add_argument("--npz", default=NPZ_PATH, help="Path to cameras_spe3r.npz")
    parser.add_argument("--out_dir", default=SPLIT_DIR, help="Directory where split txt files are written")
    parser.add_argument("--prefix", required=True, help="Output prefix, e.g. hst_black")
    parser.add_argument("--subset", choices=["black", "earth", "all"], default="black", help="Subset of views to use (0-499, 500-999, or all)")
    parser.add_argument("--k", type=int, default=16, help="Number of views per orbit")
    parser.add_argument("--mode", choices=["base", "complement", "both"], default="both", help="Generation mode")
    parser.add_argument("--plane", choices=["e1e3", "e2e3"], default="e1e3", help="Complement plane selection")
    parser.add_argument("--phase", type=float, default=0.0, help="Phase shift (radians) for complement orbit")
    parser.add_argument("--kmax", type=int, default=None, help="Number of desired orbit samples before nesting")
    parser.add_argument("--plot", action="store_true", help="Enable diagnostic plots")
    return parser.parse_args(argv)


def subset_mask(ids: np.ndarray, subset: str) -> np.ndarray:
    if subset == "black":
        return (ids >= 0) & (ids <= 499)
    if subset == "earth":
        return (ids >= 500) & (ids <= 999)
    return np.ones_like(ids, dtype=bool)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    ids, centers, vdirs = load_centers_and_dirs(args.npz)
    mask = subset_mask(ids, args.subset)

    ids_sub = ids[mask]
    centers_sub = centers[mask]
    vdirs_sub = vdirs[mask]

    if ids_sub.size == 0:
        raise ValueError("Selected subset is empty; check --subset filters.")

    print(f"[INFO] Loaded {ids.size} cameras (subset: {ids_sub.size} views, id range {ids_sub.min()}..{ids_sub.max()})")

    if centers_sub.shape[0] > 1:
        consecutive = np.linalg.norm(centers_sub[1:] - centers_sub[:-1], axis=1)
        print(
            f"[INFO] Consecutive-center distance: mean={float(consecutive.mean()):.3f} | std={float(consecutive.std()):.3f}"
        )
    else:
        print("[INFO] Consecutive-center distance: n/a (subset has one view)")

    target = centers_sub.mean(axis=0)
    centered = centers_sub - target
    e1, e2, e3 = pca_basis_3d(centered)

    r = np.linalg.norm(centered, axis=1)
    r_med = float(np.median(r))
    delta_x = 2.0 * r_med

    wp = 1.0
    wa = r_med

    KMAX_default = max(64, 4 * args.k)
    KMAX = args.kmax if args.kmax is not None else KMAX_default
    KMAX = max(KMAX, args.k)

    print(f"[INFO] delta_x={delta_x:.3f} | r_med={r_med:.3f} | wp={wp:.1f} | wa={wa:.3f} | KMAX={KMAX}")

    plane_map = {
        "e1e3": (e1, e3),
        "e2e3": (e2, e3),
    }

    outputs: List[Tuple[str, List[int]]] = []

    desired_base = None
    base_idx_full: Optional[List[int]] = None
    base_k: Optional[List[int]] = None

    if args.mode in ("base", "both"):
        desired_base, desired_dir_base = build_orbit_in_plane(target, e1, e2, KMAX, delta_x)
        base_idx_full = greedy_match(centers_sub, vdirs_sub, desired_base, desired_dir_base, wp=wp, wa=wa)
        base_view_ids = ids_sub[base_idx_full].tolist()
        base_k = nested_from_sequence(base_view_ids, args.k)
        base_path = os.path.join(args.out_dir, f"{args.prefix}_nmc_base_{args.k}.txt")
        write_split_txt(base_path, base_k)
        outputs.append((base_path, base_k))
        print(f"[OK] wrote base orbit split: {base_path}")

    desired_complement = None
    complement_idx_full: Optional[List[int]] = None
    complement_k: Optional[List[int]] = None

    if args.mode in ("complement", "both"):
        eA, eB = plane_map[args.plane]
        desired_complement, desired_dir_complement = build_orbit_in_plane(
            target,
            eA,
            eB,
            KMAX,
            delta_x,
            phase=args.phase,
        )
        forbidden = base_idx_full if args.mode == "both" else None
        complement_idx_full = greedy_match(
            centers_sub,
            vdirs_sub,
            desired_complement,
            desired_dir_complement,
            wp=wp,
            wa=wa,
            forbidden_idx=forbidden,
        )
        complement_view_ids = ids_sub[complement_idx_full].tolist()
        complement_k = nested_from_sequence(complement_view_ids, args.k)
        comp_path = os.path.join(args.out_dir, f"{args.prefix}_nmc_complement_{args.plane}_{args.k}.txt")
        write_split_txt(comp_path, complement_k)
        outputs.append((comp_path, complement_k))
        print(f"[OK] wrote complement orbit split: {comp_path}")

    if args.mode == "both" and base_k is not None and complement_k is not None:
        combined = base_k + complement_k
        if len(combined) != 2 * args.k:
            raise RuntimeError("Combined split does not contain exactly 2k views.")
        combined_path = os.path.join(args.out_dir, f"{args.prefix}_nmc_{2 * args.k}_{args.plane}.txt")
        write_split_txt(combined_path, combined)
        outputs.append((combined_path, combined))
        print(f"[OK] wrote combined orbit split: {combined_path}")

    if args.plot:
        plot_dataset_ordering(ids_sub, centers_sub)

        if desired_base is not None and base_idx_full is not None:
            plot_orbit_diagnostics(centers_sub, vdirs_sub, target, desired_base, base_idx_full, "base")

        if desired_complement is not None and complement_idx_full is not None:
            plot_orbit_diagnostics(centers_sub, vdirs_sub, target, desired_complement, complement_idx_full, f"complement ({args.plane})")

        if args.mode == "both":
            plot_base_complement(
                centers_sub,
                target,
                desired_base,
                desired_complement,
                base_idx_full,
                complement_idx_full,
                args.plane,
            )

    print("\n[SUMMARY]")
    print(f"- subset views: {ids_sub.size}")
    print(f"- delta_x={delta_x:.3f} | r_med={r_med:.3f}")
    print("- base plane: e1e2")
    if args.mode in ("complement", "both"):
        print(f"- complement plane: {args.plane} | phase={args.phase:.3f}")
    else:
        print("- complement plane: n/a")
    for path, items in outputs:
        print(f"- {path}: {len(items)} views")


if __name__ == "__main__":
    main()

