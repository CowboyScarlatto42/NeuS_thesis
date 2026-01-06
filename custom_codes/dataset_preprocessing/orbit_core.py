"""
Core geometry and selection utilities for Natural Motion Circumnavigation (NMC) splits.
"""

import os
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


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
        W = np.asarray(data[f"world_mat_{i}"], dtype=np.float64)
        S = np.asarray(data[f"scale_mat_{i}"], dtype=np.float64)
        P4 = W @ S
        P = P4[:3, :4]

        centers.append(camera_center_from_P_svd(P))

        R = P4[:3, :3]
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


def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector v around axis by theta radians using Rodrigues' formula."""

    k = np.asarray(axis, dtype=np.float64)
    norm_k = np.linalg.norm(k)
    if norm_k < 1e-12 or abs(theta) < 1e-12:
        return np.asarray(v, dtype=np.float64)
    k = k / norm_k
    v = np.asarray(v, dtype=np.float64)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return v * cos_t + np.cross(k, v) * sin_t + k * (k @ v) * (1.0 - cos_t)


def rotate_plane(
    eA: np.ndarray,
    eB: np.ndarray,
    axis: np.ndarray,
    theta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate plane basis (eA, eB) around axis by theta and re-orthonormalize."""

    if abs(theta) < 1e-12:
        return eA, eB

    rot_eA = rodrigues_rotate(eA, axis, theta)
    rot_eB = rodrigues_rotate(eB, axis, theta)

    u = rot_eA / (np.linalg.norm(rot_eA) + 1e-12)
    v = rot_eB - (u @ rot_eB) * u
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        raise ValueError("rotate_plane produced degenerate basis; adjust inputs.")
    v = v / v_norm
    return u, v


def get_plane_basis(plane: str, e1: np.ndarray, e2: np.ndarray, e3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Select PCA plane basis according to plane identifier."""

    plane_map = {
        "e1e2": (e1, e2),
        "e1e3": (e1, e3),
        "e2e3": (e2, e3),
    }
    if plane not in plane_map:
        raise ValueError(f"Unsupported plane identifier: {plane}")
    return plane_map[plane]


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
    """Read view IDs from a txt file with lines like 000.png or 000."""

    ids: List[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry:
                continue
            low = entry.lower()
            if low.endswith(".png"):
                entry = entry[:-4]
            ids.append(int(entry))
    return ids


def write_split_txt(path: str, view_ids: Sequence[int]) -> None:
    """Write view IDs into a txt file with 3-digit names."""

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for vid in view_ids:
            handle.write(viewname(int(vid)) + "\n")


def forbidden_local_indices(ids_sub: np.ndarray, forbid_abs_ids: Sequence[int]) -> List[int]:
    """Map absolute view IDs to local subset indices, skipping missing entries."""

    if not forbid_abs_ids:
        return []

    index_map = {int(vid): idx for idx, vid in enumerate(ids_sub.tolist())}
    local: List[int] = []
    seen = set()
    for view_id in forbid_abs_ids:
        idx = index_map.get(int(view_id))
        if idx is None or idx in seen:
            continue
        local.append(idx)
        seen.add(idx)
    return local
