import argparse
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


def render_batch(runner, rays_o, rays_d, deformation_grid=None):
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
    )


def render_for_selection(runner, rays_o, rays_d):
    # The baseline selection render only needs detached opacity values. The NeuS
    # renderer still needs autograd internally for SDF normals, so temporarily
    # re-enable grad inside no_grad and detach immediately after the render.
    with torch.no_grad():
        with torch.enable_grad():
            render_out = render_batch(runner, rays_o, rays_d, deformation_grid=None)
        return {
            'weight_sum': render_out['weight_sum'].detach(),
        }


def opacity_stats(values):
    return {
        'min': float(values.min().item()),
        'max': float(values.max().item()),
        'mean': float(values.mean().item()),
    }


def tensor_stats(values):
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


def identify_checkpoint_path(runner):
    candidate = os.path.join(
        runner.base_exp_dir,
        'checkpoints',
        'ckpt_{:0>6d}.pth'.format(runner.iter_step),
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--img_idx', type=int, default=-1)
    parser.add_argument('--candidate_rays', type=int, default=128)
    parser.add_argument('--num_rays', type=int, default=8)
    parser.add_argument('--grid_resolution', type=int, default=16)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--opacity_threshold', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    start_time = time.time()
    set_seed(args.seed)
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
    candidate_stats = opacity_stats(candidate_opacity)
    valid = torch.nonzero(candidate_opacity > args.opacity_threshold, as_tuple=False).reshape(-1)
    if len(valid) == 0:
        raise RuntimeError(
            'no candidate rays exceeded the opacity threshold; '
            'increase --candidate_rays or lower --opacity_threshold'
        )
    selected = valid[:min(args.num_rays, len(valid))]
    selected_cpu = selected.cpu()

    rays_o = rays_o[selected]
    rays_d = rays_d[selected]
    selected_pixel_coords = pixel_coords[selected_cpu].cpu()
    selected_opacity = candidate_opacity[selected].cpu()
    selected_stats = opacity_stats(selected_opacity)

    deformation_grid = DenseDeformationGrid(resolution=args.grid_resolution).to(runner.device)
    grid_param = deformation_grid.offsets
    hessian_components = torch.zeros_like(grid_param.detach()[0], dtype=torch.float32, device=runner.device)

    render_out = render_batch(runner, rays_o, rays_d, deformation_grid=deformation_grid)
    color = render_out['color_fine']

    num_selected = color.shape[0]
    num_channels = color.shape[1]
    num_scalars = num_selected * num_channels

    # Compute each d RGB[ray, channel] / d grid separately. Squaring gradients
    # after summing RGB first would introduce cross terms, which is not the
    # requested trace-style Hessian proxy.
    scalar_index = 0
    for ray_idx in range(num_selected):
        for channel_idx in range(num_channels):
            scalar_index += 1
            retain_graph = scalar_index < num_scalars
            grad = torch.autograd.grad(
                outputs=color[ray_idx, channel_idx],
                inputs=grid_param,
                retain_graph=retain_graph,
                create_graph=False,
                only_inputs=True,
            )[0]
            hessian_components += grad.detach()[0].float() ** 2

    hessian_trace = hessian_components.sum(dim=0)
    components_cpu = hessian_components.detach().cpu()
    trace_cpu = hessian_trace.detach().cpu()

    components_path = os.path.join(args.output_dir, 'hessian_proxy_components.npy')
    trace_path = os.path.join(args.output_dir, 'hessian_proxy_trace.npy')
    metadata_path = os.path.join(args.output_dir, 'hessian_proxy_metadata.json')
    selected_rays_path = os.path.join(args.output_dir, 'selected_rays_metadata.json')

    np.save(components_path, components_cpu.numpy())
    np.save(trace_path, trace_cpu.numpy())

    trace_nonzero_gt0 = nonzero_stats(trace_cpu, 0.0)
    trace_nonzero_gt_eps = nonzero_stats(trace_cpu, 1e-12)
    elapsed_time = time.time() - start_time
    checkpoint_path = identify_checkpoint_path(runner)

    metadata = {
        'conf': args.conf,
        'case': args.case,
        'checkpoint_path': checkpoint_path,
        'image_index': int(img_idx),
        'seed': int(args.seed),
        'candidate_source': source,
        'mask_pixels_above_threshold': int(mask_count),
        'candidate_rays': int(len(candidate_opacity)),
        'candidate_valid_rays': int(len(valid)),
        'selected_rays': int(num_selected),
        'grid_resolution': int(args.grid_resolution),
        'component_shape': list(components_cpu.shape),
        'trace_shape': list(trace_cpu.shape),
        'components_stats': tensor_stats(components_cpu),
        'trace_stats': tensor_stats(trace_cpu),
        'trace_voxels_gt_0': trace_nonzero_gt0,
        'trace_voxels_gt_1e-12': trace_nonzero_gt_eps,
        'candidate_opacity_stats': candidate_stats,
        'selected_opacity_stats': selected_stats,
        'elapsed_time_seconds': float(elapsed_time),
        'device': str(runner.device),
    }
    save_json(metadata_path, metadata)

    selected_rays_metadata = {
        'image_index': int(img_idx),
        'seed': int(args.seed),
        'num_rays': int(num_selected),
        'pixel_coordinates_yx': selected_pixel_coords.tolist(),
        'pixel_linear_indices': (
            selected_pixel_coords[:, 0] * runner.dataset.W + selected_pixel_coords[:, 1]
        ).tolist(),
        'opacity': [float(v) for v in selected_opacity.tolist()],
    }
    save_json(selected_rays_path, selected_rays_metadata)

    pct_gt0 = 100.0 * trace_nonzero_gt0['fraction']
    print('selected_rays:', num_selected)
    print('image_index:', img_idx)
    print('output_dir:', args.output_dir)
    print('component_grid_shape:', tuple(components_cpu.shape))
    print('trace_grid_shape:', tuple(trace_cpu.shape))
    print('trace_min:', metadata['trace_stats']['min'])
    print('trace_max:', metadata['trace_stats']['max'])
    print('trace_mean:', metadata['trace_stats']['mean'])
    print('trace_voxels_gt_0:', trace_nonzero_gt0['count'], '({:.6f}%)'.format(pct_gt0))
    print('elapsed_time_seconds:', elapsed_time)


if __name__ == '__main__':
    main()
