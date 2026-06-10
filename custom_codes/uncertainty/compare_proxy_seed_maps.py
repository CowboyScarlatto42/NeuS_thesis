import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

if 'MPLCONFIGDIR' not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / 'matplotlib'
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_cache_dir)

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
POSTPROCESSING_DIR = os.path.join(REPO_ROOT, 'custom_codes', 'postprocessing')
for path in [REPO_ROOT, POSTPROCESSING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from metrics_utils import load_mesh, sample_surface_points
from custom_codes.uncertainty.validate_geometry_proxy_on_mesh import interpolate_hessian_grid


def load_hessian_grid(path):
    grid = np.load(path).astype(np.float64)
    if grid.ndim != 3 or grid.shape[0] != grid.shape[1] or grid.shape[1] != grid.shape[2]:
        raise ValueError('grid must be cubic with shape [R, R, R], got {}'.format(grid.shape))
    if not np.all(np.isfinite(grid)):
        raise RuntimeError('grid contains NaN or Inf values: {}'.format(path))
    return grid


def stable_lowest_subset(values, fraction):
    n = len(values)
    k = int(round(fraction * n))
    if k < 1 or k > n:
        raise ValueError('lowest-confidence subset size must satisfy 1 <= k <= n, got k={} n={}'.format(k, n))
    order = np.argsort(values, kind='mergesort')
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask, k


def save_points_csv(path, points, h_a, h_b, log_a, log_b, low_a, low_b, label_a, label_b):
    fieldnames = [
        'point_index',
        'x_norm',
        'y_norm',
        'z_norm',
        'hessian_{}'.format(label_a),
        'hessian_{}'.format(label_b),
        'log10_hessian_{}'.format(label_a),
        'log10_hessian_{}'.format(label_b),
        'lowest_confidence_{}'.format(label_a),
        'lowest_confidence_{}'.format(label_b),
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, point in enumerate(points):
            writer.writerow({
                'point_index': idx,
                'x_norm': point[0],
                'y_norm': point[1],
                'z_norm': point[2],
                'hessian_{}'.format(label_a): h_a[idx],
                'hessian_{}'.format(label_b): h_b[idx],
                'log10_hessian_{}'.format(label_a): log_a[idx],
                'log10_hessian_{}'.format(label_b): log_b[idx],
                'lowest_confidence_{}'.format(label_a): bool(low_a[idx]),
                'lowest_confidence_{}'.format(label_b): bool(low_b[idx]),
            })


def save_scatter(path, log_a, log_b, label_a, label_b):
    plt.figure(figsize=(6, 6))
    plt.scatter(log_a, log_b, s=4, alpha=0.2)
    lo = float(min(log_a.min(), log_b.min()))
    hi = float(max(log_a.max(), log_b.max()))
    plt.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1)
    plt.xlabel('log10 H — {}'.format(label_a.replace('_', ' ')))
    plt.ylabel('log10 H — {}'.format(label_b.replace('_', ' ')))
    plt.title('Geometry-oriented proxy stability across ray-sampling seeds')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hessian_grid_a', type=str, required=True)
    parser.add_argument('--hessian_grid_b', type=str, required=True)
    parser.add_argument('--label_a', type=str, default='seed_42')
    parser.add_argument('--label_b', type=str, default='seed_123')
    parser.add_argument('--reconstruction_mesh', type=str, required=True)
    parser.add_argument('--num_surface_points', type=int, default=50000)
    parser.add_argument('--surface_sampling_seed', type=int, default=42)
    parser.add_argument('--lowest_confidence_fraction', type=float, default=0.25)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--eps', type=float, default=1e-12)
    args = parser.parse_args()

    if args.eps <= 0.0:
        raise ValueError('--eps must be positive')
    if args.lowest_confidence_fraction <= 0.0 or args.lowest_confidence_fraction > 1.0:
        raise ValueError('--lowest_confidence_fraction must be in (0, 1]')

    os.makedirs(args.output_dir, exist_ok=True)
    grid_a = load_hessian_grid(args.hessian_grid_a)
    grid_b = load_hessian_grid(args.hessian_grid_b)
    if grid_a.shape != grid_b.shape:
        raise RuntimeError('grid shapes differ: {} vs {}'.format(grid_a.shape, grid_b.shape))

    mesh = load_mesh(args.reconstruction_mesh)
    points = sample_surface_points(mesh, args.num_surface_points, seed=args.surface_sampling_seed)
    h_a, inside_a = interpolate_hessian_grid(grid_a, points)
    h_b, inside_b = interpolate_hessian_grid(grid_b, points)
    inside = inside_a & inside_b
    inside_fraction = float(np.mean(inside))
    print('fraction inside bounds:', inside_fraction)
    if inside_fraction < 0.99:
        raise RuntimeError('inside bounds fraction {:.6f} < 0.99'.format(inside_fraction))

    finite = inside & np.isfinite(h_a) & np.isfinite(h_b)
    if int(finite.sum()) < 2:
        raise RuntimeError('not enough finite inside-bounds samples')
    points = points[finite]
    h_a = h_a[finite]
    h_b = h_b[finite]
    if np.any(h_a < 0.0) or np.any(h_b < 0.0):
        raise RuntimeError('interpolated Hessian values must be non-negative')

    log_a = np.log10(h_a + args.eps)
    log_b = np.log10(h_b + args.eps)
    if not np.all(np.isfinite(log_a)) or not np.all(np.isfinite(log_b)):
        raise RuntimeError('non-finite log Hessian values')

    spearman = spearmanr(h_a, h_b)
    pearson = pearsonr(log_a, log_b)
    low_a, subset_size = stable_lowest_subset(h_a, args.lowest_confidence_fraction)
    low_b, subset_size_b = stable_lowest_subset(h_b, args.lowest_confidence_fraction)
    if subset_size != subset_size_b:
        raise RuntimeError('lowest-confidence subsets have different sizes')
    intersection_size = int(np.logical_and(low_a, low_b).sum())
    union_size = int(np.logical_or(low_a, low_b).sum())
    iou = float(intersection_size / union_size) if union_size > 0 else float('nan')
    overlap_fraction = float(intersection_size / subset_size)

    save_points_csv(
        os.path.join(args.output_dir, 'seed_map_comparison_points.csv'),
        points,
        h_a,
        h_b,
        log_a,
        log_b,
        low_a,
        low_b,
        args.label_a,
        args.label_b,
    )
    save_scatter(
        os.path.join(args.output_dir, 'seed_map_comparison_scatter.png'),
        log_a,
        log_b,
        args.label_a,
        args.label_b,
    )

    summary = {
        'hessian_grid_a': args.hessian_grid_a,
        'hessian_grid_b': args.hessian_grid_b,
        'label_a': args.label_a,
        'label_b': args.label_b,
        'grid_shape': list(grid_a.shape),
        'reconstruction_mesh': args.reconstruction_mesh,
        'num_sampled_points': int(len(points)),
        'num_requested_surface_points': int(args.num_surface_points),
        'fraction_inside_bounds': inside_fraction,
        'surface_sampling_seed': int(args.surface_sampling_seed),
        'lowest_confidence_fraction': float(args.lowest_confidence_fraction),
        'subset_size': int(subset_size),
        'intersection_size': intersection_size,
        'union_size': union_size,
        'iou': iou,
        'overlap_fraction': overlap_fraction,
        'spearman_rho': float(spearman.statistic),
        'spearman_p_value': float(spearman.pvalue),
        'pearson_log10_r': float(pearson.statistic),
        'pearson_log10_p_value': float(pearson.pvalue),
        'eps': float(args.eps),
        'tie_handling': (
            'Lowest-confidence subsets are selected by exact rank using np.argsort(kind="mergesort"); '
            'ties are resolved stably according to sampled point order.'
        ),
    }
    save_json(os.path.join(args.output_dir, 'seed_map_comparison_summary.json'), summary)

    print('sampled finite points:', len(points))
    print('Spearman rho:', summary['spearman_rho'])
    print('Spearman p-value:', summary['spearman_p_value'])
    print('Pearson log10 r:', summary['pearson_log10_r'])
    print('IoU lowest-confidence subsets:', iou)
    print('overlap fraction:', overlap_fraction)
    print('output_dir:', args.output_dir)


if __name__ == '__main__':
    main()
