"""CLI utility to generate validation NMC orbit split text files disjoint from training splits."""

import argparse
import os
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from orbit_core import (
    build_orbit_in_plane,
    forbidden_local_indices,
    get_plane_basis,
    greedy_match,
    load_centers_and_dirs,
    nested_from_sequence,
    pca_basis_3d,
    read_split_txt,
    rotate_plane,
    write_split_txt,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation NMC orbit splits disjoint from training splits.")
    parser.add_argument("--npz", default="cameras_spe3r.npz", help="Path to cameras_spe3r.npz")
    parser.add_argument("--out_dir", default="splits", help="Directory where validation split txt files are written")
    parser.add_argument("--prefix", required=True, help="Output prefix, e.g. hst_black")
    parser.add_argument("--subset", choices=["black", "earth", "all"], default="black", help="Subset of views to use (0-499, 500-999, or all)")
    parser.add_argument("--k", type=int, default=32, help="Number of views in the validation orbit")
    parser.add_argument("--kmax", type=int, default=None, help="Number of desired orbit samples before nesting")
    parser.add_argument("--plane", choices=["e1e2", "e1e3", "e2e3"], default="e1e2", help="Plane selection for validation orbit")
    parser.add_argument("--phase", type=float, default=0.0, help="Phase shift (radians) for validation orbit")
    parser.add_argument("--tilt", type=float, default=0.0, help="Tilt angle in radians applied to orbit plane")
    parser.add_argument("--tilt_axis", choices=["e1", "e2", "e3"], default="e1", help="Axis for tilt rotation")
    parser.add_argument("--forbid_txt", nargs="+", required=True, help="Training split txt files whose views must be forbidden")
    return parser.parse_args(argv)


def subset_mask(ids: np.ndarray, subset: str) -> np.ndarray:
    if subset == "black":
        return (ids >= 0) & (ids <= 499)
    if subset == "earth":
        return (ids >= 500) & (ids <= 999)
    return np.ones_like(ids, dtype=bool)


def apply_tilt(eA: np.ndarray, eB: np.ndarray, axis: np.ndarray, tilt: float) -> Tuple[np.ndarray, np.ndarray]:
    if abs(tilt) < 1e-12:
        return eA, eB
    return rotate_plane(eA, eB, axis, tilt)


def load_forbidden_ids(paths: Sequence[str]) -> List[int]:
    seen: Set[int] = set()
    collected: List[int] = []
    for path in paths:
        for view_id in read_split_txt(path):
            if view_id not in seen:
                seen.add(view_id)
                collected.append(view_id)
    return collected


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

    forbid_abs = load_forbidden_ids(args.forbid_txt)
    print(f"[INFO] forbid files: {len(args.forbid_txt)}")
    print(f"[INFO] forbid abs union: {len(forbid_abs)}")

    forbidden_local = forbidden_local_indices(ids_sub, forbid_abs)
    print(f"[INFO] forbid in subset: {len(forbidden_local)} / {ids_sub.size} (available: {ids_sub.size - len(forbidden_local)})")
    available = ids_sub.size - len(forbidden_local)
    if available < args.k:
        raise RuntimeError(f"Not enough available views for validation: need k={args.k}, available={available}")

    plane_eA, plane_eB = get_plane_basis(args.plane, e1, e2, e3)
    tilt_axis_map = {
        "e1": e1,
        "e2": e2,
        "e3": e3,
    }
    axis_vec = tilt_axis_map[args.tilt_axis]
    if abs(args.tilt) >= 1e-12:
        print(f"[INFO] Tilt applied: theta={args.tilt:.6f} rad around {args.tilt_axis}")
    plane_eA, plane_eB = apply_tilt(plane_eA, plane_eB, axis_vec, args.tilt)

    desired_pos, desired_dir = build_orbit_in_plane(
        target,
        plane_eA,
        plane_eB,
        KMAX,
        delta_x,
        phase=args.phase,
    )

    val_idx_full = greedy_match(
        centers_sub,
        vdirs_sub,
        desired_pos,
        desired_dir,
        wp=wp,
        wa=wa,
        forbidden_idx=forbidden_local,
    )
    val_abs_full = ids_sub[val_idx_full].tolist()
    val_k = nested_from_sequence(val_abs_full, args.k)

    forbid_set = set(forbid_abs)
    overlap = sorted(set(val_k).intersection(forbid_set))
    if overlap:
        raise RuntimeError(
            f"Validation split overlaps with forbidden views: {overlap[:10]} (showing up to 10)."
        )

    out_name = (
        f"{args.prefix}_val_nmc_{args.k}_"
        f"plane-{args.plane}_"
        f"tilt-{args.tilt_axis}-{args.tilt:.3f}_"
        f"phase-{args.phase:.3f}.txt"
    )
    out_path = os.path.join(args.out_dir, out_name)
    write_split_txt(out_path, val_k)

    print("\n[SUMMARY]")
    print(f"- subset views: {ids_sub.size}")
    print(f"- delta_x={delta_x:.3f} | r_med={r_med:.3f}")
    print(f"- plane: {args.plane} | phase={args.phase:.3f}")
    print(f"- tilt: theta={args.tilt:.3f} rad around {args.tilt_axis}")
    print(f"- forbidden abs IDs: {len(forbid_abs)}")
    print(f"- forbidden local indices: {len(forbidden_local)}")
    print(f"- output: {out_path} ({len(val_k)} views)")


if __name__ == "__main__":
    main()
