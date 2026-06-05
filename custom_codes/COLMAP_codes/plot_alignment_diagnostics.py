#!/usr/bin/env python3
"""Generate plots from align_colmap_orbits_to_corto.py outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class OrbitDiagnostics:
    tag: str
    names: List[str]
    gt_centers: np.ndarray
    aligned_centers: np.ndarray
    position_errors: np.ndarray
    view_errors_deg: np.ndarray
    sparse_points: Optional[np.ndarray]


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_ascii_ply(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    points: List[List[float]] = []
    in_header = True
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if in_header:
                if line == "end_header":
                    in_header = False
                continue
            if line:
                x, y, z = line.split()[:3]
                points.append([float(x), float(y), float(z)])
    return np.asarray(points, dtype=float).reshape(-1, 3)


def load_orbit(alignment_root: Path, tag: str) -> OrbitDiagnostics:
    orbit_dir = alignment_root / tag
    records = read_json(orbit_dir / "aligned_poses_all_fit.json")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Invalid or empty aligned pose list: {orbit_dir}")
    return OrbitDiagnostics(
        tag=tag,
        names=[str(row["stem"]) for row in records],
        gt_centers=np.asarray([row["camera_center_gt"] for row in records], dtype=float),
        aligned_centers=np.asarray([row["camera_center_aligned"] for row in records], dtype=float),
        position_errors=np.asarray([row["position_error"] for row in records], dtype=float),
        view_errors_deg=np.asarray([row["view_direction_error_deg"] for row in records], dtype=float),
        sparse_points=read_ascii_ply(orbit_dir / "aligned_sparse_points_all_fit.ply"),
    )


def downsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=max_points, replace=False)]


def plot_trajectories(
    results: Sequence[OrbitDiagnostics], output: Path, max_sparse_points: int
) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for result in results:
        ax.plot(*result.gt_centers.T, marker="o", markersize=2, linewidth=1, label=f"{result.tag} GT")
        ax.plot(
            *result.aligned_centers.T,
            marker="x",
            markersize=3,
            linewidth=1,
            linestyle="--",
            label=f"{result.tag} COLMAP aligned",
        )
        if result.sparse_points is not None and len(result.sparse_points):
            cloud = downsample(result.sparse_points, max_sparse_points)
            ax.scatter(*cloud.T, s=1, alpha=0.08, label=f"{result.tag} sparse")
    ax.scatter([0], [0], [0], marker="*", s=100, label="target origin")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="CORTO/Tango trajectories and aligned COLMAP reconstructions")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_errors(results: Sequence[OrbitDiagnostics], output: Path, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for result in results:
        values = result.position_errors if kind == "position" else result.view_errors_deg
        frame_indices = np.arange(len(values))
        ax.plot(frame_indices, values, marker="o", markersize=3, linewidth=1, label=result.tag)
    if kind == "position":
        ax.set_ylabel("camera-centre error [CORTO units]")
        ax.set_title("Camera-centre alignment residuals")
    else:
        ax.set_ylabel("view-direction error [deg]")
        ax.set_title("View-direction errors after alignment")
    ax.set_xlabel("registered frame index (filtered labels.json order)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-sparse-points", type=int, default=5000)
    args = parser.parse_args()

    alignment_root = args.alignment_root.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else alignment_root
    output.mkdir(parents=True, exist_ok=True)
    results = [load_orbit(alignment_root, tag) for tag in ("orbit1", "orbit2")]

    plot_trajectories(results, output / "trajectories_all_fit.png", args.max_sparse_points)
    plot_errors(results, output / "position_errors_all_fit.png", "position")
    plot_errors(results, output / "view_direction_errors_all_fit.png", "view")
    print(f"Plots written to: {output}")


if __name__ == "__main__":
    main()
