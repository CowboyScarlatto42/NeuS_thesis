import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp_runner import Runner
from models.deformation_grid import DenseDeformationGrid


SUPPORTED_MODES = ['full', 'geometry_only', 'appearance_only', 'appearance_no_normal_grad']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_neus(runner):
    modules = [
        runner.nerf_outside,
        runner.sdf_network,
        runner.deviation_network,
        runner.color_network,
    ]
    for module in modules:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)


def parse_modes(raw_modes):
    modes = []
    for item in raw_modes.split(','):
        mode = item.strip()
        if mode == '':
            continue
        if mode not in SUPPORTED_MODES:
            raise ValueError('Unsupported mode {}. Supported modes: {}'.format(mode, SUPPORTED_MODES))
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError('At least one mode must be requested')
    return modes


def choose_image_index(dataset, requested_idx):
    if requested_idx >= 0:
        return requested_idx

    best_idx = 0
    best_count = -1
    for idx in range(dataset.n_images):
        mask = dataset.masks[idx][..., 0]
        count = int((mask > 0.5).sum().item())
        if count > best_count:
            best_idx = idx
            best_count = count
    return best_idx


def rays_from_pixels(dataset, img_idx, pixels_y, pixels_x):
    pixels_x = pixels_x.float()
    pixels_y = pixels_y.float()
    p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
    p = torch.matmul(dataset.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
    rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
    rays_d = torch.matmul(dataset.pose_all[img_idx, None, :3, :3], rays_d[:, :, None]).squeeze()
    rays_o = dataset.pose_all[img_idx, None, :3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def candidate_rays_from_mask(dataset, img_idx, candidate_count, mask_threshold):
    mask = dataset.masks[img_idx][..., 0]
    coords = torch.nonzero(mask > mask_threshold, as_tuple=False)
    if len(coords) == 0:
        return None, None, None, 0

    perm = torch.randperm(len(coords))[:candidate_count]
    selected = coords[perm]
    pixels_y = selected[:, 0]
    pixels_x = selected[:, 1]
    rays_o, rays_d = rays_from_pixels(dataset, img_idx, pixels_y, pixels_x)
    return rays_o, rays_d, selected, len(coords)


def random_candidate_rays(dataset, img_idx, candidate_count):
    pixels_x = torch.randint(low=0, high=dataset.W, size=[candidate_count])
    pixels_y = torch.randint(low=0, high=dataset.H, size=[candidate_count])
    coords = torch.stack([pixels_y, pixels_x], dim=-1)
    rays_o, rays_d = rays_from_pixels(dataset, img_idx, pixels_y, pixels_x)
    return rays_o, rays_d, coords


def render_batch(runner, rays_o, rays_d, deformation_grid=None, diagnostic_mode=None):
    near, far = runner.dataset.near_far_from_sphere(rays_o.cpu(), rays_d.cpu())
    near = near.to(runner.device)
    far = far.to(runner.device)
    background_rgb = torch.ones([1, 3], device=runner.device) if runner.use_white_bkgd else None
    return runner.renderer.render(
        rays_o,
        rays_d,
        near,
        far,
        perturb_overwrite=0,
        background_rgb=background_rgb,
        cos_anneal_ratio=runner.get_cos_anneal_ratio(),
        deformation_grid=deformation_grid,
        diagnostic_mode=diagnostic_mode,
    )


def render_for_selection(runner, rays_o, rays_d):
    # Selection only needs detached opacity. NeuS still computes normals via
    # autograd, so re-enable grad inside no_grad and detach the result.
    with torch.no_grad():
        with torch.enable_grad():
            render_out = render_batch(runner, rays_o, rays_d)
        return {'weight_sum': render_out['weight_sum'].detach()}


def ensure_finite_tensor(mode, name, values, ray_idx=None, channel_idx=None, pixel_coord=None):
    if torch.isfinite(values).all():
        return

    details = ['mode={}'.format(mode), name]
    if ray_idx is not None:
        details.append('ray_idx={}'.format(ray_idx))
    if channel_idx is not None:
        details.append('channel_idx={}'.format(channel_idx))
    if pixel_coord is not None:
        details.append('pixel_yx={}'.format(tuple(int(v) for v in pixel_coord)))
    raise RuntimeError('non-finite value detected: ' + ', '.join(details))


def stats(values):
    return {
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean': float(values.mean().item()),
    }


def nonzero_stats(values, threshold):
    count = int((values > threshold).sum().item())
    total = int(values.numel())
    return {
        'count': count,
        'fraction': float(count / total) if total > 0 else 0.0,
    }


def percentile_from_sorted(values, fraction):
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * fraction
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def per_ray_summary(rows):
    totals = np.array([row['total_contribution'] for row in rows], dtype=np.float64)
    finite_mask = np.isfinite(totals)
    finite_totals = totals[finite_mask]
    finite_count = int(finite_mask.sum())
    non_finite_count = int(len(totals) - finite_count)
    if finite_count == 0:
        raise RuntimeError('no finite per-ray contributions were computed')
    total_proxy = float(finite_totals.sum())
    if total_proxy == 0.0:
        raise RuntimeError('final total Hessian proxy contribution is zero')

    sorted_totals = np.sort(finite_totals)
    descending = sorted_totals[::-1]
    top5_count = min(5, len(descending))
    top10_count = min(10, len(descending))
    return {
        'finite_rays': finite_count,
        'non_finite_rays': non_finite_count,
        'min': float(sorted_totals[0]),
        'median': percentile_from_sorted(sorted_totals, 0.5),
        'mean': float(finite_totals.mean()),
        'max': float(sorted_totals[-1]),
        'largest_fraction': float(descending[0] / total_proxy),
        'top5_fraction': float(descending[:top5_count].sum() / total_proxy),
        'top10_fraction': float(descending[:top10_count].sum() / total_proxy),
        'total_proxy': total_proxy,
    }


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_per_ray_csv(path, rows):
    fieldnames = [
        'ray_order_index',
        'pixel_y',
        'pixel_x',
        'pixel_linear_index',
        'opacity',
        'total_contribution',
        'max_single_voxel_trace_contribution',
        'r_contribution',
        'g_contribution',
        'b_contribution',
        'finite',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_csv(path, rows):
    fieldnames = [
        'mode',
        'total_trace_sum',
        'trace_max',
        'trace_mean',
        'finite_rays',
        'non_finite_rays',
        'per_ray_median',
        'per_ray_mean',
        'per_ray_max',
        'largest_ray_fraction',
        'top5_fraction',
        'top10_fraction',
        'top10_pixel_coordinates_yx',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row['top10_pixel_coordinates_yx'] = json.dumps(csv_row['top10_pixel_coordinates_yx'])
            writer.writerow(csv_row)


def identify_checkpoint_path(runner):
    candidate = os.path.join(
        runner.base_exp_dir,
        'checkpoints',
        'ckpt_{:0>6d}.pth'.format(runner.iter_step),
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def run_full_rgb_regression_check(runner, rays_o, rays_d, grid_resolution, tolerance):
    baseline_grid = DenseDeformationGrid(resolution=grid_resolution).to(runner.device)
    full_grid = DenseDeformationGrid(resolution=grid_resolution).to(runner.device)
    baseline = render_batch(runner, rays_o, rays_d, deformation_grid=baseline_grid, diagnostic_mode=None)
    full = render_batch(runner, rays_o, rays_d, deformation_grid=full_grid, diagnostic_mode='full')
    max_diff = (baseline['color_fine'].detach() - full['color_fine'].detach()).abs().max().item()
    print('full RGB regression max_abs_diff:', max_diff)
    if max_diff > tolerance:
        raise RuntimeError(
            "diagnostic_mode='full' RGB regression failed: max diff {} > {}".format(max_diff, tolerance)
        )
    return max_diff


def accumulate_mode(mode, runner, rays_o, rays_d, pixel_coords, opacity, grid_resolution, output_dir):
    mode_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    deformation_grid = DenseDeformationGrid(resolution=grid_resolution).to(runner.device)
    grid_param = deformation_grid.offsets
    components = torch.zeros_like(grid_param.detach()[0], dtype=torch.float32, device=runner.device)

    render_out = render_batch(
        runner,
        rays_o,
        rays_d,
        deformation_grid=deformation_grid,
        diagnostic_mode=mode,
    )
    color = render_out['color_fine']
    ensure_finite_tensor(mode, 'rendered_color', color)
    if color.shape[1] != 3:
        raise RuntimeError('mode {} expected RGB with 3 channels, got {}'.format(mode, color.shape[1]))

    num_rays = color.shape[0]
    num_scalars = num_rays * 3
    scalar_index = 0
    per_ray_rows = []

    # Each scalar RGB gradient is squared independently. Using grad(rgb.sum())
    # would introduce cross terms and would not diagnose per-channel sensitivity.
    for ray_idx in range(num_rays):
        pixel_coord = pixel_coords[ray_idx].tolist()
        ray_components = torch.zeros_like(components)
        channel_contributions = []
        for channel_idx in range(3):
            scalar_index += 1
            retain_graph = scalar_index < num_scalars
            scalar = color[ray_idx, channel_idx]
            ensure_finite_tensor(mode, 'rgb_scalar', scalar, ray_idx, channel_idx, pixel_coord)
            grad = torch.autograd.grad(
                outputs=scalar,
                inputs=grid_param,
                retain_graph=retain_graph,
                create_graph=False,
                only_inputs=True,
            )[0]
            ensure_finite_tensor(mode, 'grid_gradient', grad, ray_idx, channel_idx, pixel_coord)
            squared = grad.detach()[0].float() ** 2
            components += squared
            ray_components += squared
            channel_contributions.append(float(squared.sum().item()))

        ray_trace = ray_components.sum(dim=0)
        ray_total = float(ray_components.sum().item())
        ray_max = float(ray_trace.max().item())
        ray_values = torch.tensor([ray_total, ray_max] + channel_contributions, device=runner.device)
        ensure_finite_tensor(mode, 'per_ray_contribution', ray_values, ray_idx, pixel_coord=pixel_coord)
        per_ray_rows.append({
            'ray_order_index': int(ray_idx),
            'pixel_y': int(pixel_coord[0]),
            'pixel_x': int(pixel_coord[1]),
            'pixel_linear_index': int(pixel_coord[0] * runner.dataset.W + pixel_coord[1]),
            'opacity': float(opacity[ray_idx].item()),
            'total_contribution': ray_total,
            'max_single_voxel_trace_contribution': ray_max,
            'r_contribution': channel_contributions[0],
            'g_contribution': channel_contributions[1],
            'b_contribution': channel_contributions[2],
            'finite': True,
        })

        processed = ray_idx + 1
        if processed == num_rays or processed % 8 == 0:
            elapsed = time.time() - mode_start
            cumulative_trace_sum = float(components.sum().item())
            print(
                '[{}] processed rays: {}/{} | elapsed: {:.2f} s | cumulative trace sum: {:.8g}'.format(
                    mode,
                    processed,
                    num_rays,
                    elapsed,
                    cumulative_trace_sum,
                )
            )

    trace = components.sum(dim=0)
    ensure_finite_tensor(mode, 'hessian_components', components)
    ensure_finite_tensor(mode, 'hessian_trace', trace)
    components_cpu = components.detach().cpu()
    trace_cpu = trace.detach().cpu()
    summary = per_ray_summary(per_ray_rows)
    top_rows = sorted(per_ray_rows, key=lambda row: row['total_contribution'], reverse=True)[:min(10, len(per_ray_rows))]
    top_pixels = [[row['pixel_y'], row['pixel_x']] for row in top_rows]

    np.save(os.path.join(output_dir, 'hessian_proxy_components.npy'), components_cpu.numpy())
    np.save(os.path.join(output_dir, 'hessian_proxy_trace.npy'), trace_cpu.numpy())
    save_per_ray_csv(os.path.join(output_dir, 'per_ray_hessian_contributions.csv'), per_ray_rows)
    save_json(os.path.join(output_dir, 'per_ray_hessian_contributions.json'), per_ray_rows)

    trace_stats = stats(trace_cpu)
    metadata = {
        'mode': mode,
        'grid_resolution': int(grid_resolution),
        'component_shape': list(components_cpu.shape),
        'trace_shape': list(trace_cpu.shape),
        'trace_stats': trace_stats,
        'trace_sum': float(trace_cpu.sum().item()),
        'trace_voxels_gt_0': nonzero_stats(trace_cpu, 0.0),
        'trace_voxels_gt_1e-12': nonzero_stats(trace_cpu, 1e-12),
        'per_ray_contribution_summary': summary,
        'top10_pixel_coordinates_yx': top_pixels,
        'finite_validation': True,
        'elapsed_time_seconds': float(time.time() - mode_start),
    }
    save_json(os.path.join(output_dir, 'metadata.json'), metadata)

    print('[{}] total trace sum: {}'.format(mode, metadata['trace_sum']))
    print('[{}] per-ray median: {}'.format(mode, summary['median']))
    print('[{}] per-ray max: {}'.format(mode, summary['max']))
    print('[{}] largest-ray fraction: {}'.format(mode, summary['largest_fraction']))
    print('[{}] top-5 fraction: {}'.format(mode, summary['top5_fraction']))
    print('[{}] top-10 fraction: {}'.format(mode, summary['top10_fraction']))
    print('[{}] top 10 pixels yx: {}'.format(mode, top_pixels))

    return {
        'mode': mode,
        'total_trace_sum': metadata['trace_sum'],
        'trace_max': trace_stats['max'],
        'trace_mean': trace_stats['mean'],
        'finite_rays': summary['finite_rays'],
        'non_finite_rays': summary['non_finite_rays'],
        'per_ray_median': summary['median'],
        'per_ray_mean': summary['mean'],
        'per_ray_max': summary['max'],
        'largest_ray_fraction': summary['largest_fraction'],
        'top5_fraction': summary['top5_fraction'],
        'top10_fraction': summary['top10_fraction'],
        'top10_pixel_coordinates_yx': top_pixels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--img_idx', type=int, default=-1)
    parser.add_argument('--candidate_rays', type=int, default=128)
    parser.add_argument('--num_rays', type=int, default=128)
    parser.add_argument('--grid_resolution', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--modes', type=str, default=','.join(SUPPORTED_MODES))
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--opacity_threshold', type=float, default=1e-3)
    parser.add_argument('--rgb_tolerance', type=float, default=1e-5)
    parser.add_argument('--skip_full_regression_check', default=False, action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    modes = parse_modes(args.modes)
    os.makedirs(args.output_dir, exist_ok=True)

    runner = Runner(args.conf, mode='validate_image', case=args.case, is_continue=True)
    freeze_neus(runner)

    img_idx = choose_image_index(runner.dataset, args.img_idx)
    candidate_count = max(args.candidate_rays, args.num_rays)
    rays_o, rays_d, pixel_coords, mask_count = candidate_rays_from_mask(
        runner.dataset,
        img_idx,
        candidate_count,
        args.mask_threshold,
    )
    source = 'mask'
    if rays_o is None:
        rays_o, rays_d, pixel_coords = random_candidate_rays(runner.dataset, img_idx, candidate_count)
        mask_count = 0
        source = 'random'

    rays_o = rays_o.to(runner.device)
    rays_d = rays_d.to(runner.device)

    baseline_candidates = render_for_selection(runner, rays_o, rays_d)
    candidate_opacity = baseline_candidates['weight_sum'].reshape(-1).detach()
    valid = torch.nonzero(candidate_opacity > args.opacity_threshold, as_tuple=False).reshape(-1)
    if len(valid) == 0:
        raise RuntimeError('no candidate rays exceeded the opacity threshold')
    selected = valid[:min(args.num_rays, len(valid))]
    selected_cpu = selected.cpu()
    rays_o = rays_o[selected]
    rays_d = rays_d[selected]
    selected_pixel_coords = pixel_coords[selected_cpu].cpu()
    selected_opacity = candidate_opacity[selected].cpu()

    selected_metadata = {
        'conf': args.conf,
        'case': args.case,
        'checkpoint_path': identify_checkpoint_path(runner),
        'image_index': int(img_idx),
        'seed': int(args.seed),
        'candidate_source': source,
        'mask_pixels_above_threshold': int(mask_count),
        'candidate_rays': int(len(candidate_opacity)),
        'candidate_valid_rays': int(len(valid)),
        'selected_rays': int(len(selected)),
        'pixel_coordinates_yx': selected_pixel_coords.tolist(),
        'pixel_linear_indices': (
            selected_pixel_coords[:, 0] * runner.dataset.W + selected_pixel_coords[:, 1]
        ).tolist(),
        'opacity': [float(v) for v in selected_opacity.tolist()],
        'device': str(runner.device),
    }
    save_json(os.path.join(args.output_dir, 'selected_rays_metadata.json'), selected_metadata)

    full_rgb_regression_max_abs_diff = None
    if not args.skip_full_regression_check:
        full_rgb_regression_max_abs_diff = run_full_rgb_regression_check(
            runner,
            rays_o,
            rays_d,
            args.grid_resolution,
            args.rgb_tolerance,
        )

    summary_rows = []
    for mode in modes:
        print('running sensitivity mode:', mode)
        mode_dir = os.path.join(args.output_dir, mode)
        summary_rows.append(
            accumulate_mode(
                mode,
                runner,
                rays_o,
                rays_d,
                selected_pixel_coords,
                selected_opacity,
                args.grid_resolution,
                mode_dir,
            )
        )

    summary = {
        'conf': args.conf,
        'case': args.case,
        'image_index': int(img_idx),
        'seed': int(args.seed),
        'modes': modes,
        'selected_rays': int(len(selected)),
        'grid_resolution': int(args.grid_resolution),
        'full_rgb_regression_max_abs_diff': full_rgb_regression_max_abs_diff,
        'rows': summary_rows,
    }
    save_json(os.path.join(args.output_dir, 'sensitivity_modes_summary.json'), summary)
    save_summary_csv(os.path.join(args.output_dir, 'sensitivity_modes_summary.csv'), summary_rows)


if __name__ == '__main__':
    main()
