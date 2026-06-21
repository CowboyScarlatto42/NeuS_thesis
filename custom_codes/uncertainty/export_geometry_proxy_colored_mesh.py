import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if 'MPLCONFIGDIR' not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / 'matplotlib'
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_cache_dir)

import matplotlib.pyplot as plt
from matplotlib import cm, colors

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
POSTPROCESSING_DIR = os.path.join(REPO_ROOT, 'custom_codes', 'postprocessing')
for path in [REPO_ROOT, POSTPROCESSING_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from metrics_utils import load_mesh
from custom_codes.uncertainty.validate_geometry_proxy_on_mesh import interpolate_hessian_grid


def valid_grid_values(values, inside):
    finite = np.isfinite(values)
    return inside & finite


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'min': None, 'max': None, 'mean': None, 'median': None}
    return {
        'min': float(values.min()),
        'max': float(values.max()),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
    }


def percentile_limits(values, lower_percentile, upper_percentile):
    if values.size == 0:
        raise RuntimeError('no valid values available for percentile color scaling')
    lower = float(np.percentile(values, lower_percentile))
    upper = float(np.percentile(values, upper_percentile))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise RuntimeError('non-finite percentile limits')
    if upper <= lower:
        delta = max(abs(lower) * 1e-6, 1e-12)
        lower -= delta
        upper += delta
    return lower, upper


def resolve_color_limits(values, lower_percentile, upper_percentile, manual_min, manual_max, name):
    if (manual_min is None) != (manual_max is None):
        raise ValueError('both --{}_vmin and --{}_vmax must be provided together'.format(name, name))
    if manual_min is not None:
        lower = float(manual_min)
        upper = float(manual_max)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError('{} manual color limits must be finite'.format(name))
        if upper <= lower:
            raise ValueError('--{}_vmax must be greater than --{}_vmin'.format(name, name))
        return lower, upper, 'manual'

    lower, upper = percentile_limits(values, lower_percentile, upper_percentile)
    return lower, upper, 'percentile'


def values_to_rgba(values, valid_mask, lower, upper, cmap_name, neutral_rgba=(160, 160, 160, 255)):
    rgba = np.zeros((len(values), 4), dtype=np.uint8)
    rgba[:] = np.asarray(neutral_rgba, dtype=np.uint8)
    norm = colors.Normalize(vmin=lower, vmax=upper, clip=True)
    cmap = cm.get_cmap(cmap_name)
    mapped = cmap(norm(values[valid_mask]))
    rgba[valid_mask] = (mapped * 255.0).round().astype(np.uint8)
    return rgba


def save_vertex_csv(path, vertices, inside, hessian, log_hessian, inverse_visual):
    fieldnames = [
        'vertex_index',
        'x_norm',
        'y_norm',
        'z_norm',
        'inside_bounds',
        'hessian_raw',
        'log10_hessian',
        'inverse_sensitivity_visual',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (vertex, is_inside, h_value, log_value, inverse_value) in enumerate(
            zip(vertices, inside, hessian, log_hessian, inverse_visual)
        ):
            writer.writerow({
                'vertex_index': idx,
                'x_norm': vertex[0],
                'y_norm': vertex[1],
                'z_norm': vertex[2],
                'inside_bounds': bool(is_inside),
                'hessian_raw': h_value,
                'log10_hessian': log_value,
                'inverse_sensitivity_visual': inverse_value,
            })


def save_colorbar(path, cmap_name, lower, upper, title, label):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.subplots_adjust(bottom=0.45)
    norm = colors.Normalize(vmin=lower, vmax=upper)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cmap_name))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label(label)
    ax.set_title(title)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hessian_grid', type=str, required=True)
    parser.add_argument('--reconstruction_mesh', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--lower_percentile', type=float, default=5.0)
    parser.add_argument('--upper_percentile', type=float, default=95.0)
    parser.add_argument('--confidence_vmin', type=float, default=None)
    parser.add_argument('--confidence_vmax', type=float, default=None)
    parser.add_argument('--inverse_vmin', type=float, default=None)
    parser.add_argument('--inverse_vmax', type=float, default=None)
    parser.add_argument('--eps', type=float, default=1e-12)
    args = parser.parse_args()

    if args.upper_percentile <= args.lower_percentile:
        raise ValueError('--upper_percentile must be greater than --lower_percentile')
    if args.eps <= 0.0:
        raise ValueError('--eps must be positive')

    os.makedirs(args.output_dir, exist_ok=True)

    hessian_grid = np.load(args.hessian_grid).astype(np.float64)
    if hessian_grid.ndim != 3 or len(set(hessian_grid.shape)) != 1:
        raise ValueError('expected scalar Hessian grid with shape [R, R, R], got {}'.format(hessian_grid.shape))

    mesh = load_mesh(args.reconstruction_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    hessian_raw, inside = interpolate_hessian_grid(hessian_grid, vertices)

    vertex_count = int(len(vertices))
    inside_count = int(inside.sum())
    outside_count = int(vertex_count - inside_count)
    inside_fraction = float(inside_count / vertex_count) if vertex_count > 0 else 0.0

    print('vertex_count:', vertex_count)
    print('inside_bounds_count:', inside_count)
    print('outside_bounds_count:', outside_count)
    print('inside_bounds_fraction:', inside_fraction)

    if inside_count == 0:
        raise RuntimeError('all mesh vertices are outside [-1, 1]^3')
    if inside_fraction < 0.99:
        raise RuntimeError(
            'inside bounds fraction {:.6f} < 0.99; the reconstruction mesh is likely not in normalized NeuS coordinates'.format(
                inside_fraction
            )
        )

    valid_mask = valid_grid_values(hessian_raw, inside)
    if not np.any(valid_mask):
        raise RuntimeError('no finite inside-bounds Hessian values were interpolated')
    if np.any(hessian_raw[valid_mask] < 0.0):
        raise RuntimeError('interpolated Hessian contains negative values inside bounds')

    log10_hessian = np.full(vertex_count, np.nan, dtype=np.float64)
    inverse_sensitivity_visual = np.full(vertex_count, np.nan, dtype=np.float64)
    log10_hessian[valid_mask] = np.log10(hessian_raw[valid_mask] + args.eps)
    inverse_sensitivity_visual[valid_mask] = -log10_hessian[valid_mask]

    if not np.all(np.isfinite(log10_hessian[valid_mask])):
        raise RuntimeError('non-finite log10_hessian values')
    if not np.all(np.isfinite(inverse_sensitivity_visual[valid_mask])):
        raise RuntimeError('non-finite inverse_sensitivity_visual values')

    confidence_lower, confidence_upper, confidence_limit_source = resolve_color_limits(
        log10_hessian[valid_mask],
        args.lower_percentile,
        args.upper_percentile,
        args.confidence_vmin,
        args.confidence_vmax,
        'confidence',
    )
    inverse_lower, inverse_upper, inverse_limit_source = resolve_color_limits(
        inverse_sensitivity_visual[valid_mask],
        args.lower_percentile,
        args.upper_percentile,
        args.inverse_vmin,
        args.inverse_vmax,
        'inverse',
    )

    confidence_rgba = values_to_rgba(
        log10_hessian,
        valid_mask,
        confidence_lower,
        confidence_upper,
        'viridis',
    )
    inverse_rgba = values_to_rgba(
        inverse_sensitivity_visual,
        valid_mask,
        inverse_lower,
        inverse_upper,
        'magma',
    )

    confidence_mesh = mesh.copy()
    inverse_mesh = mesh.copy()
    confidence_mesh.visual.vertex_colors = confidence_rgba
    inverse_mesh.visual.vertex_colors = inverse_rgba

    confidence_ply = os.path.join(args.output_dir, 'mesh_geometry_proxy_confidence.ply')
    inverse_ply = os.path.join(args.output_dir, 'mesh_geometry_proxy_inverse_sensitivity.ply')
    confidence_mesh.export(confidence_ply)
    inverse_mesh.export(inverse_ply)

    vertex_csv = os.path.join(args.output_dir, 'vertex_proxy_values.csv')
    save_vertex_csv(vertex_csv, vertices, inside, hessian_raw, log10_hessian, inverse_sensitivity_visual)

    confidence_colorbar = os.path.join(args.output_dir, 'confidence_colorbar.png')
    inverse_colorbar = os.path.join(args.output_dir, 'inverse_sensitivity_colorbar.png')
    save_colorbar(
        confidence_colorbar,
        'viridis',
        confidence_lower,
        confidence_upper,
        'Confidence visualization',
        'log10(H + eps)',
    )
    save_colorbar(
        inverse_colorbar,
        'magma',
        inverse_lower,
        inverse_upper,
        'Uncertainty score',
        r'$U = -\log_{10}(H + \epsilon)$',
    )

    h_positive_count = int((hessian_raw[valid_mask] > 0.0).sum())
    metadata = {
        'hessian_grid_path': args.hessian_grid,
        'reconstruction_mesh_path': args.reconstruction_mesh,
        'grid_shape': list(hessian_grid.shape),
        'vertex_count': vertex_count,
        'inside_bounds_count': inside_count,
        'outside_bounds_count': outside_count,
        'fraction_inside_bounds': inside_fraction,
        'hessian_gt_0_count': h_positive_count,
        'hessian_gt_0_fraction_valid': float(h_positive_count / int(valid_mask.sum())),
        'raw_hessian_stats_valid_inside': stats(hessian_raw[valid_mask]),
        'visualization_transform': {
            'confidence': 'log10(hessian_raw + eps)',
            'inverse_sensitivity': '-log10(hessian_raw + eps)',
        },
        'eps': float(args.eps),
        'percentile_settings': {
            'lower_percentile': float(args.lower_percentile),
            'upper_percentile': float(args.upper_percentile),
        },
        'manual_color_limits': {
            'confidence_vmin': args.confidence_vmin,
            'confidence_vmax': args.confidence_vmax,
            'inverse_vmin': args.inverse_vmin,
            'inverse_vmax': args.inverse_vmax,
        },
        'color_limit_source': {
            'confidence': confidence_limit_source,
            'inverse_sensitivity': inverse_limit_source,
        },
        'effective_percentile_limits': {
            'confidence': {
                'lower': confidence_lower,
                'upper': confidence_upper,
            },
            'inverse_sensitivity': {
                'lower': inverse_lower,
                'upper': inverse_upper,
            },
        },
        'colormaps': {
            'confidence': 'viridis',
            'inverse_sensitivity': 'magma',
            'outside_bounds_color': 'neutral gray rgba(160,160,160,255)',
        },
        'output_filenames': {
            'confidence_ply': os.path.basename(confidence_ply),
            'inverse_sensitivity_ply': os.path.basename(inverse_ply),
            'vertex_proxy_values_csv': os.path.basename(vertex_csv),
            'confidence_colorbar_png': os.path.basename(confidence_colorbar),
            'inverse_sensitivity_colorbar_png': os.path.basename(inverse_colorbar),
        },
        'note': (
            'Percentile clipping affects visualization colors only. Raw hessian_raw, '
            'log10_hessian, and inverse_sensitivity_visual values are preserved in CSV. '
            'Inverse sensitivity is an uncertainty-oriented proxy, not calibrated uncertainty.'
        ),
    }
    save_json(os.path.join(args.output_dir, 'colored_mesh_metadata.json'), metadata)

    print('vertices with H > 0:', h_positive_count)
    print('confidence PLY:', confidence_ply)
    print('inverse sensitivity PLY:', inverse_ply)
    print('output_dir:', args.output_dir)


if __name__ == '__main__':
    main()
