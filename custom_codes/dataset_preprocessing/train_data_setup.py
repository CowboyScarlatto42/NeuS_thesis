"""CLI utility to generate NMC orbit split text files for training/validation setups."""

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from orbit_core import (
    build_orbit_in_plane,
    get_plane_basis,
    greedy_match,
    load_centers_and_dirs,
    nested_from_sequence,
    pca_basis_3d,
    rotate_plane,
    write_split_txt,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Natural Motion Circumnavigation splits.")
    parser.add_argument("--npz", default="cameras_spe3r.npz", help="Path to cameras_spe3r.npz")
    parser.add_argument("--out_dir", default="splits", help="Directory where split txt files are written")
    parser.add_argument("--prefix", required=True, help="Output prefix, e.g. hst_black")
    parser.add_argument("--subset", choices=["black", "earth", "all"], default="black", help="Subset of views to use (0-499, 500-999, or all)")
    parser.add_argument("--mode", choices=["base", "complement", "both"], default="both", help="Selection mode")
    parser.add_argument("--k", type=int, default=16, help="Number of views per orbit")
    parser.add_argument("--kmax", type=int, default=None, help="Number of desired orbit samples before nesting")
    parser.add_argument("--plane", choices=["e1e2", "e1e3", "e2e3"], default="e1e3", help="Complement plane selection")
    parser.add_argument("--phase", type=float, default=0.0, help="Phase shift (radians) for complement orbit")
    parser.add_argument("--tilt", type=float, default=0.0, help="Tilt angle in radians applied to orbit planes")
    parser.add_argument("--tilt_axis", choices=["e1", "e2", "e3"], default="e1", help="Axis for tilt rotation")
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

    tilt_axis_map = {
        "e1": e1,
        "e2": e2,
        "e3": e3,
    }
    tilt_axis_vec = tilt_axis_map[args.tilt_axis]
    if abs(args.tilt) >= 1e-12:
        print(f"[INFO] Tilt applied: theta={args.tilt:.6f} rad around {args.tilt_axis}")

    outputs: List[Tuple[str, List[int]]] = []

    desired_base = None
    base_idx_full: Optional[List[int]] = None
    base_k: Optional[List[int]] = None

    if args.mode in ("base", "both"):
        base_eA, base_eB = apply_tilt(e1, e2, tilt_axis_vec, args.tilt)
        desired_base, desired_dir_base = build_orbit_in_plane(target, base_eA, base_eB, KMAX, delta_x)
        base_idx_full = greedy_match(
            centers_sub,
            vdirs_sub,
            desired_base,
            desired_dir_base,
            wp=wp,
            wa=wa,
        )
        base_view_ids = ids_sub[base_idx_full].tolist()
        base_k = nested_from_sequence(base_view_ids, args.k)
        base_path = os.path.join(args.out_dir, f"{args.prefix}_nmc_base_{args.k}.txt")
        write_split_txt(base_path, base_k)
        outputs.append((base_path, base_k))
        print(f"[OK] wrote base orbit split: {base_path}")

    complement_idx_full: Optional[List[int]] = None
    complement_k: Optional[List[int]] = None

    if args.mode in ("complement", "both"):
        comp_eA, comp_eB = get_plane_basis(args.plane, e1, e2, e3)
        comp_eA, comp_eB = apply_tilt(comp_eA, comp_eB, tilt_axis_vec, args.tilt)
        desired_comp, desired_dir_comp = build_orbit_in_plane(
            target,
            comp_eA,
            comp_eB,
            KMAX,
            delta_x,
            phase=args.phase,
        )
        forbidden_idx = base_idx_full if args.mode == "both" else None
        complement_idx_full = greedy_match(
            centers_sub,
            vdirs_sub,
            desired_comp,
            desired_dir_comp,
            wp=wp,
            wa=wa,
            forbidden_idx=forbidden_idx,
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
            raise RuntimeError("Combined split does not contain exactly 2 * k views.")
        combined_path = os.path.join(args.out_dir, f"{args.prefix}_nmc_{2 * args.k}_{args.plane}.txt")
        write_split_txt(combined_path, combined)
        outputs.append((combined_path, combined))
        print(f"[OK] wrote combined orbit split: {combined_path}")

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
