import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp_runner import Runner
from models.deformation_grid import DenseDeformationGrid


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
        return None, None, 0

    perm = torch.randperm(len(coords))[:candidate_count]
    selected = coords[perm]
    pixels_y = selected[:, 0]
    pixels_x = selected[:, 1]
    rays_o, rays_d = rays_from_pixels(dataset, img_idx, pixels_y, pixels_x)
    return rays_o, rays_d, len(coords)


def random_candidate_rays(dataset, img_idx, candidate_count):
    data = dataset.gen_random_rays_at(img_idx, candidate_count)
    return data[:, :3], data[:, 3:6]


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


def opacity_stats(values):
    return {
        'min': values.min().item(),
        'max': values.max().item(),
        'mean': values.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--img_idx', type=int, default=-1)
    parser.add_argument('--candidate_rays', type=int, default=128)
    parser.add_argument('--num_rays', type=int, default=16)
    parser.add_argument('--grid_resolution', type=int, default=16)
    parser.add_argument('--mask_threshold', type=float, default=0.5)
    parser.add_argument('--opacity_threshold', type=float, default=1e-3)
    args = parser.parse_args()

    runner = Runner(args.conf, mode='validate_image', case=args.case, is_continue=True)
    freeze_neus(runner)

    img_idx = choose_image_index(runner.dataset, args.img_idx)
    candidate_count = max(args.candidate_rays, args.num_rays)
    rays_o, rays_d, mask_count = candidate_rays_from_mask(
        runner.dataset,
        img_idx,
        candidate_count,
        args.mask_threshold,
    )
    source = 'mask'
    if rays_o is None:
        rays_o, rays_d = random_candidate_rays(runner.dataset, img_idx, candidate_count)
        source = 'random'

    rays_o = rays_o.to(runner.device)
    rays_d = rays_d.to(runner.device)

    baseline_candidates = render_batch(runner, rays_o, rays_d, deformation_grid=None)
    candidate_opacity = baseline_candidates['weight_sum'].reshape(-1).detach()
    candidate_stats = opacity_stats(candidate_opacity)
    valid = torch.nonzero(candidate_opacity > args.opacity_threshold, as_tuple=False).reshape(-1)
    if len(valid) == 0:
        raise RuntimeError(
            'no candidate rays exceeded the opacity threshold; '
            'increase --candidate_rays or lower --opacity_threshold'
        )
    selected = valid[:min(args.num_rays, len(valid))]

    del baseline_candidates

    rays_o = rays_o[selected]
    rays_d = rays_d[selected]

    baseline = render_batch(runner, rays_o, rays_d, deformation_grid=None)
    baseline_color = baseline['color_fine'].detach()
    selected_opacity = baseline['weight_sum'].reshape(-1).detach()
    selected_stats = opacity_stats(selected_opacity)
    del baseline

    deformation_grid = DenseDeformationGrid(resolution=args.grid_resolution).to(runner.device)
    deformation_grid.zero_grad(set_to_none=True)

    deformed = render_batch(runner, rays_o, rays_d, deformation_grid=deformation_grid)
    deformed_color = deformed['color_fine']
    max_rgb_diff = (baseline_color - deformed_color).abs().max().item()
    deformed_color.sum().backward()

    grad = deformation_grid.offsets.grad
    if grad is None:
        raise RuntimeError('deformation grid gradient is None')

    print('image_index:', img_idx)
    print('candidate_source:', source)
    print('mask_pixels_above_threshold:', mask_count)
    print('candidate_rays:', len(candidate_opacity))
    print('candidate_valid_rays:', len(valid))
    print('selected_valid_rays:', len(selected))
    print('candidate_baseline_opacity_min:', candidate_stats['min'])
    print('candidate_baseline_opacity_max:', candidate_stats['max'])
    print('candidate_baseline_opacity_mean:', candidate_stats['mean'])
    print('selected_baseline_opacity_min:', selected_stats['min'])
    print('selected_baseline_opacity_max:', selected_stats['max'])
    print('selected_baseline_opacity_mean:', selected_stats['mean'])
    print('max_rgb_diff_baseline_vs_zero_grid:', max_rgb_diff)
    print('deformation_grid_grad_shape:', tuple(grad.shape))
    print('deformation_grid_grad_max_abs:', grad.abs().max().item())
    print('deformation_grid_grad_mean_abs:', grad.abs().mean().item())


if __name__ == '__main__':
    main()
