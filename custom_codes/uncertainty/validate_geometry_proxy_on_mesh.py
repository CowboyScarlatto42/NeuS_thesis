import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from pyhocon import ConfigFactory
from scipy.stats import spearmanr

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


def resolve_conf_path(conf_path):
    if os.path.isabs(conf_path):
        return conf_path
    candidate = os.path.join(REPO_ROOT, conf_path)
    if os.path.isfile(candidate):
        return candidate
    return conf_path


def load_scale_mat_from_conf(conf_path, case):
    conf_path = resolve_conf_path(conf_path)
    with open(conf_path) as f:
        conf_text = f.read().replace('CASE_NAME', case)
    conf = ConfigFactory.parse_string(conf_text)
    data_dir = conf['dataset.data_dir'].replace('CASE_NAME', case)
    cameras_name = conf.get_string('dataset.render_cameras_name')
    cameras = np.load(os.path.join(data_dir, cameras_name))
    scale_mat = cameras['scale_mat_0'].astype(np.float64)
    return scale_mat


def normalized_to_gt_frame(points_norm, scale_mat):
    # Same convention used by Runner.validate_mesh(world_space=True):
    # vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3].
    return points_norm * scale_mat[0, 0] + scale_mat[:3, 3][None, :]


def interpolate_hessian_grid(hessian_grid, points_norm):
    if hessian_grid.ndim != 3 or hessian_grid.shape[0] != hessian_grid.shape[1] or hessian_grid.shape[1] != hessian_grid.shape[2]:
        raise ValueError('hessian grid must have shape [R, R, R]')

    resolution = hessian_grid.shape[0]
    points = np.asarray(points_norm, dtype=np.float64)
    normalized = (points + 1.0) * 0.5 * (resolution - 1)
    x = normalized[:, 0]
    y = normalized[:, 1]
    z = normalized[:, 2]

    inside = (
        (x >= 0.0) & (x <= resolution - 1) &
        (y >= 0.0) & (y <= resolution - 1) &
        (z >= 0.0) & (z <= resolution - 1)
    )
    values = np.zeros(points.shape[0], dtype=np.float64)
    if not np.any(inside):
        return values, inside

    xi = x[inside]
    yi = y[inside]
    zi = z[inside]
    x0 = np.floor(xi).astype(np.int64)
    y0 = np.floor(yi).astype(np.int64)
    z0 = np.floor(zi).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, resolution - 1)
    y1 = np.clip(y0 + 1, 0, resolution - 1)
    z1 = np.clip(z0 + 1, 0, resolution - 1)
    wx = xi - x0
    wy = yi - y0
    wz = zi - z0

    # Saved scalar Hessian grid is indexed [z, y, x], while points are (x, y, z).
    c000 = hessian_grid[z0, y0, x0]
    c001 = hessian_grid[z0, y0, x1]
    c010 = hessian_grid[z0, y1, x0]
    c011 = hessian_grid[z0, y1, x1]
    c100 = hessian_grid[z1, y0, x0]
    c101 = hessian_grid[z1, y0, x1]
    c110 = hessian_grid[z1, y1, x0]
    c111 = hessian_grid[z1, y1, x1]

    c00 = c000 * (1.0 - wx) + c001 * wx
    c01 = c010 * (1.0 - wx) + c011 * wx
    c10 = c100 * (1.0 - wx) + c101 * wx
    c11 = c110 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    values[inside] = c0 * (1.0 - wz) + c1 * wz
    return values, inside


def point_to_surface_distance(points, mesh):
    try:
        closest, distances, triangle_id = mesh.nearest.on_surface(points)
    except BaseException as exc:
        raise RuntimeError(
            'trimesh point-to-surface query failed. Install runtime dependencies such as rtree, '
            'or run in an environment where trimesh proximity queries are available.'
        ) from exc
    return np.asarray(distances, dtype=np.float64)


def spearman_payload(error, hessian):
    mask = np.isfinite(error) & np.isfinite(hessian)
    if int(mask.sum()) < 2:
        return {'rho': float('nan'), 'p_value': float('nan'), 'n': int(mask.sum())}
    rho, p_value = spearmanr(error[mask], hessian[mask])
    return {'rho': float(rho), 'p_value': float(p_value), 'n': int(mask.sum())}


def write_sampled_points_csv(path, points_norm, points_gt, hessian, error):
    fieldnames = ['x_norm', 'y_norm', 'z_norm', 'x_gt', 'y_gt', 'z_gt', 'hessian', 'gt_distance']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p_norm, p_gt, h, d in zip(points_norm, points_gt, hessian, error):
            writer.writerow({
                'x_norm': p_norm[0],
                'y_norm': p_norm[1],
                'z_norm': p_norm[2],
                'x_gt': p_gt[0],
                'y_gt': p_gt[1],
                'z_gt': p_gt[2],
                'hessian': h,
                'gt_distance': d,
            })


def sparsification_curve(error, hessian, seed, steps=101):
    n = len(error)
    fractions_removed = np.linspace(0.0, 0.99, steps)
    rng = np.random.default_rng(seed)
    orders = {
        'estimated_proxy': np.argsort(hessian),
        'random_baseline': rng.permutation(n),
        'oracle': np.argsort(-error),
    }
    rows = []
    for fraction in fractions_removed:
        remove_count = min(int(round(fraction * n)), n - 1)
        row = {'fraction_removed': float(fraction), 'remaining_points': int(n - remove_count)}
        for name, order in orders.items():
            keep = order[remove_count:]
            row['mean_error_{}'.format(name)] = float(np.mean(error[keep]))
        rows.append(row)
    return rows


def write_csv(path, rows, fieldnames=None):
    if len(rows) == 0:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_sparsification_curves(rows, tolerance=1e-12):
    estimated = np.asarray([row['mean_error_estimated_proxy'] for row in rows], dtype=np.float64)
    random = np.asarray([row['mean_error_random_baseline'] for row in rows], dtype=np.float64)
    oracle = np.asarray([row['mean_error_oracle'] for row in rows], dtype=np.float64)

    if estimated.shape != random.shape or estimated.shape != oracle.shape:
        raise RuntimeError(
            'sparsification curves have mismatched shapes: estimated={}, random={}, oracle={}'.format(
                estimated.shape, random.shape, oracle.shape
            )
        )
    if not (np.all(np.isfinite(estimated)) and np.all(np.isfinite(random)) and np.all(np.isfinite(oracle))):
        raise RuntimeError('sparsification curves contain NaN or Inf values')

    initial = np.asarray([estimated[0], random[0], oracle[0]], dtype=np.float64)
    if np.max(np.abs(initial - initial[0])) > tolerance:
        raise RuntimeError(
            'sparsification curves do not share the same initial mean error: {}'.format(initial.tolist())
        )
    if np.any(np.diff(oracle) > tolerance):
        raise RuntimeError('oracle sparsification curve is not non-increasing within tolerance {}'.format(tolerance))

    return estimated, random, oracle


def plot_sparsification(path, rows):
    x = np.asarray([row['fraction_removed'] for row in rows], dtype=np.float64)
    estimated, random, oracle = validate_sparsification_curves(rows)

    plt.figure(figsize=(7, 5))
    plt.plot(x, estimated, label='Estimated proxy')
    plt.plot(x, random, label='Random baseline')
    plt.plot(x, oracle, label='Oracle')
    plt.xlabel('Fraction of removed points')
    plt.ylabel('Mean geometric error of remaining points')
    plt.title('Sparsification curve — geometry-oriented sensitivity proxy')
    plt.xlim(0.0, 0.95)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def quartile_summary(error, hessian):
    quantiles = np.quantile(hessian, [0.0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    for idx in range(4):
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        if idx == 3:
            mask = (hessian >= lo) & (hessian <= hi)
        else:
            mask = (hessian >= lo) & (hessian < hi)
        values = error[mask]
        rows.append({
            'quartile': idx + 1,
            'hessian_min': float(lo),
            'hessian_max': float(hi),
            'count': int(mask.sum()),
            'mean_error': float(np.mean(values)) if len(values) > 0 else float('nan'),
            'median_error': float(np.median(values)) if len(values) > 0 else float('nan'),
        })
    return rows


def save_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hessian_grid', type=str, required=True)
    parser.add_argument('--reconstruction_mesh', type=str, required=True)
    parser.add_argument('--gt_mesh', type=str, required=True)
    parser.add_argument('--num_surface_points', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--conf', type=str, default='./confs/long_test.conf')
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hessian_grid = np.load(args.hessian_grid).astype(np.float64)
    recon_mesh = load_mesh(args.reconstruction_mesh)
    gt_mesh = load_mesh(args.gt_mesh)

    points_norm = sample_surface_points(recon_mesh, args.num_surface_points, seed=args.seed)
    hessian, inside = interpolate_hessian_grid(hessian_grid, points_norm)
    scale_mat = load_scale_mat_from_conf(args.conf, args.case)
    points_gt = normalized_to_gt_frame(points_norm, scale_mat)
    error = point_to_surface_distance(points_gt, gt_mesh)

    finite = np.isfinite(hessian) & np.isfinite(error)
    if int(finite.sum()) < 2:
        raise RuntimeError('not enough finite samples for validation')
    points_norm = points_norm[finite]
    points_gt = points_gt[finite]
    hessian = hessian[finite]
    error = error[finite]

    positive = hessian > 0.0
    spearman_all_h = spearman_payload(error, hessian)
    spearman_all_neg_h = spearman_payload(error, -hessian)
    spearman_pos_h = spearman_payload(error[positive], hessian[positive])
    spearman_pos_neg_h = spearman_payload(error[positive], -hessian[positive])

    sampled_csv = os.path.join(args.output_dir, 'sampled_mesh_points_with_proxy.csv')
    write_sampled_points_csv(sampled_csv, points_norm, points_gt, hessian, error)

    curve_rows = sparsification_curve(error, hessian, args.seed)
    write_csv(os.path.join(args.output_dir, 'sparsification_curve.csv'), curve_rows)
    plot_sparsification(os.path.join(args.output_dir, 'sparsification_curve.png'), curve_rows)

    quartile_rows = quartile_summary(error, hessian)
    write_csv(os.path.join(args.output_dir, 'quartile_error_summary.csv'), quartile_rows)

    summary = {
        'hessian_grid': args.hessian_grid,
        'reconstruction_mesh': args.reconstruction_mesh,
        'gt_mesh': args.gt_mesh,
        'num_sampled_points': int(len(error)),
        'num_requested_surface_points': int(args.num_surface_points),
        'fraction_hessian_gt_0': float(np.mean(positive)),
        'fraction_inside_hessian_bounds_before_finite_filter': float(np.mean(inside)),
        'spearman_error_vs_hessian_all': spearman_all_h,
        'spearman_error_vs_negative_hessian_all': spearman_all_neg_h,
        'spearman_error_vs_hessian_positive_h': spearman_pos_h,
        'spearman_error_vs_negative_hessian_positive_h': spearman_pos_neg_h,
        'mean_geometric_error': float(np.mean(error)),
        'median_geometric_error': float(np.median(error)),
        'quartile_error_summary': quartile_rows,
        'scale_mat_0': scale_mat.tolist(),
        'coordinate_note': (
            'Hessian interpolation is performed in normalized NeuS coordinates before '
            'mapping sampled points to the CORTO/GT frame with the NeuS scale_mat convention.'
        ),
    }
    save_json(os.path.join(args.output_dir, 'spearman_summary.json'), summary)

    print('sampled points:', len(error))
    print('fraction H > 0:', summary['fraction_hessian_gt_0'])
    print('Spearman error vs H:', spearman_all_h)
    print('Spearman error vs -H:', spearman_all_neg_h)
    print('Spearman H > 0 error vs H:', spearman_pos_h)
    print('Spearman H > 0 error vs -H:', spearman_pos_neg_h)
    print('mean geometric error:', summary['mean_geometric_error'])
    print('mean error by quartile:')
    for row in quartile_rows:
        print('  Q{}: {}'.format(row['quartile'], row['mean_error']))
    print('output_dir:', args.output_dir)


if __name__ == '__main__':
    main()
