#!/usr/bin/env python3
"""
Render-based metrics (PSNR / SSIM) by rendering GT and predicted meshes from
multiple yaw angles using pyrender (headless EGL) and comparing images.

Default behavior uses a canonical rendering normalization (center + scale)
to make the fixed camera comparable across meshes.

Metrics:
- PSNR (dB): per-angle, then mean/std across angles
- SSIM: grayscale SSIM per-angle, then mean/std across angles

Example:
python render_metrics_psnr_ssim.py \
  --mesh_dir "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/meshes" \
  --gt_path  "/content/drive/MyDrive/Tesi/neus/serious_analysis/data/hst_neus/model/model_normalized.obj" \
  --checks 00025000.ply 00050000.ply 00075000.ply 00100000.ply \
  --angles_step 30 \
  --w 640 --h 480 \
  --out_csv "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/post/render_metrics.csv"

If you are 100% sure meshes are already aligned and scaled for the fixed camera,
disable canonical normalization with:
  --no_canonical_norm
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
import cv2
from skimage.metrics import structural_similarity as ssim
import pyrender

# Headless rendering (must be set before creating pyrender contexts)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# -----------------------
# Metrics
# -----------------------
def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return float(20 * np.log10(PIXEL_MAX / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # Convert to grayscale (OpenCV expects BGR, but these are RGB; for grayscale it doesn't matter much,
    # still, keep it consistent by converting RGB->GRAY explicitly)
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    dr = float(g2.max() - g2.min())
    if dr <= 1e-8:
        # degenerate image (almost constant); SSIM is not meaningful
        return float("nan")
    return float(ssim(g1, g2, data_range=dr))


# -----------------------
# Rendering utilities
# -----------------------
def to_trimesh(mesh_or_scene):
    """Ensure we end up with a single trimesh.Trimesh."""
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = list(mesh_or_scene.geometry.values())
        if len(geoms) == 0:
            raise ValueError("Empty trimesh.Scene")
        return trimesh.util.concatenate(geoms)
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene
    raise TypeError(f"Unsupported type: {type(mesh_or_scene)}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    return to_trimesh(trimesh.load(path, force="mesh"))


def canonical_normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Center + scale based on bbox extents, to make fixed-camera renders comparable.

    This is NOT ICP/PCA. It is only for canonical rendering.
    """
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    ext = m.bounding_box.extents
    scale = 1.0 / (np.linalg.norm(ext) + 1e-12)
    m.apply_scale(scale)
    return m


def render_mesh_rgb(
    mesh: trimesh.Trimesh,
    angle_deg: float,
    w: int,
    h: int,
    cam_dist: float,
    yfov: float,
    ambient: float,
    light_intensity: float,
) -> np.ndarray:
    """Render mesh with pyrender after rotating around Y-axis. Returns RGB uint8 image."""
    m = mesh.copy()
    angle = np.radians(angle_deg)
    R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    m.apply_transform(R)

    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 255],
        ambient_light=[ambient, ambient, ambient],
    )
    pm = pyrender.Mesh.from_trimesh(m, smooth=False)
    scene.add(pm)

    camera = pyrender.PerspectiveCamera(yfov=yfov)
    cam_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, cam_dist],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(w, h)
    color, _ = r.render(scene)
    r.delete()
    return color  # RGB uint8


# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_path", type=Path, required=True)
    p.add_argument("--mesh_dir", type=Path, required=True)
    p.add_argument("--checks", nargs="+", required=True)

    p.add_argument("--angles_step", type=int, default=30, help="Angle step in degrees (0..330).")
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=480)

    p.add_argument("--cam_dist", type=float, default=1.2)
    p.add_argument("--yfov", type=float, default=float(np.pi / 3.0))
    p.add_argument("--ambient", type=float, default=0.15)
    p.add_argument("--light_intensity", type=float, default=3.0)

    p.add_argument(
        "--no_canonical_norm",
        action="store_true",
        help="Disable canonical normalization (center+scale) before rendering.",
    )

    p.add_argument("--out_csv", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.gt_path.exists():
        raise FileNotFoundError(f"gt_path not found: {args.gt_path}")
    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"mesh_dir not found: {args.mesh_dir}")

    pred_paths = [args.mesh_dir / ck for ck in args.checks]
    for pth in pred_paths:
        if not pth.exists():
            raise FileNotFoundError(f"Missing predicted mesh: {pth}")

    angles = list(range(0, 360, args.angles_step))

    # load meshes
    gt_mesh = load_mesh(args.gt_path)
    if not args.no_canonical_norm:
        gt_mesh = canonical_normalize(gt_mesh)

    rows = []
    for ck, pred_path in zip(args.checks, pred_paths):
        pred_mesh = load_mesh(pred_path)
        if not args.no_canonical_norm:
            pred_mesh = canonical_normalize(pred_mesh)

        psnr_vals, ssim_vals = [], []
        for ang in angles:
            gt_img = render_mesh_rgb(
                gt_mesh, ang, args.w, args.h,
                args.cam_dist, args.yfov, args.ambient, args.light_intensity
            )
            pr_img = render_mesh_rgb(
                pred_mesh, ang, args.w, args.h,
                args.cam_dist, args.yfov, args.ambient, args.light_intensity
            )

            if pr_img.shape != gt_img.shape:
                pr_img = cv2.resize(pr_img, (gt_img.shape[1], gt_img.shape[0]))

            psnr_vals.append(calculate_psnr(gt_img, pr_img))
            ssim_vals.append(calculate_ssim(gt_img, pr_img))

        psnr_vals = np.array(psnr_vals, dtype=np.float64)
        ssim_vals = np.array(ssim_vals, dtype=np.float64)

        rows.append(
            {
                "checkpoint": Path(ck).stem,
                "angles_step": args.angles_step,
                "w": args.w,
                "h": args.h,
                "canonical_norm": (not args.no_canonical_norm),
                "PSNR_mean_dB": float(np.nanmean(psnr_vals)),
                "SSIM_mean": float(np.nanmean(ssim_vals)),
                "PSNR_std": float(np.nanstd(psnr_vals)),
                "SSIM_std": float(np.nanstd(ssim_vals)),
            }
        )

    df = pd.DataFrame(rows).sort_values("checkpoint")
    pd.options.display.float_format = "{:.4f}".format
    print(df.to_string(index=False))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\n[Saved] {args.out_csv}")


if __name__ == "__main__":
    main()
