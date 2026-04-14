"""
NeuS Pre-Training Validation Script
===================================
Script di validazione dataset NeuS, semplice e orientato ai failure mode reali.

Checks principali:
  1. Struttura NPZ + consistenza delle inverse
  2. Pose in spazio normalizzato
  3. Direzione camere verso origine
  4. Reproiezione target normalizzato (0,0,0)
  5. Reproiezione mesh (se fornita)
  6. Qualita' maschere (coverage + binarieta')
  7. Near/Far rays check (coerente con Dataset.near_far_from_sphere)
  8. Pose consistency cross-pipeline:
       - COLMAP (se sparse points disponibili)
       - GT SPE3R (se labels/camera/scale_mat disponibili)
       - Generic fallback
"""

import argparse
import json
import os
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_npz(npz_path):
    return dict(np.load(npz_path))


def count_cameras(cam_dict):
    index = 0
    while f"world_mat_{index}" in cam_dict:
        index += 1
    return index


def list_pngs(folder):
    if folder is None or not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def ensure_exists(path, label, strict=True):
    ok = path is not None and os.path.exists(path)
    if not ok:
        msg = f"  ⚠️  {label} non trovato: {path}"
        print(msg)
        if strict:
            raise FileNotFoundError(msg)
    return ok


def load_K_pose_from_P(P):
    """Replica la logica NeuS (dataset.py) via cv.decomposeProjectionMatrix."""
    out = cv.decomposeProjectionMatrix(P.astype(np.float64))
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, pose


def extract_camera_pose(P):
    _, pose = load_K_pose_from_P(P[:3, :4])
    R = pose[:3, :3]
    cam_pos = pose[:3, 3]
    view_dir = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
    return cam_pos, view_dir, R


def choose_best_forward_axis(rotations, positions):
    axes = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    target_dirs = -positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 1e-12)

    best_mean = None
    best_axis = None
    best_dirs = None
    best_angles = None

    for axis in axes:
        dirs = (rotations @ axis.reshape(3, 1)).squeeze(-1)
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        dots = np.sum(dirs * target_dirs, axis=1)
        angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        mean_ang = float(np.mean(angles))

        if best_mean is None or mean_ang < best_mean:
            best_mean = mean_ang
            best_axis = axis
            best_dirs = dirs
            best_angles = angles

    return best_axis, best_dirs, best_angles


def project_points(P, points_3d):
    n_points = points_3d.shape[0]
    points_h = np.hstack([points_3d, np.ones((n_points, 1), dtype=np.float64)])
    proj = (P @ points_h.T).T

    z = proj[:, 2]
    valid = np.abs(z) > 1e-8

    px = np.full(n_points, np.nan)
    py = np.full(n_points, np.nan)
    px[valid] = proj[valid, 0] / z[valid]
    py[valid] = proj[valid, 1] / z[valid]

    return np.stack([px, py], axis=1), z


def load_mesh_vertices(mesh_path, max_points=5000):
    ext = Path(mesh_path).suffix.lower()

    if ext == ".obj":
        vertices = []
        with open(mesh_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        vertices = np.asarray(vertices, dtype=np.float64)
    elif ext == ".ply":
        import trimesh
        mesh = trimesh.load(mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
    else:
        raise ValueError(f"Formato mesh non supportato: {ext}")

    if len(vertices) == 0:
        raise ValueError("Mesh senza vertici")

    if len(vertices) > max_points:
        idx = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[idx]

    return vertices


def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        raise ValueError("Quaternion quasi nullo")
    q = q / qn
    w, x, y, z = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_camera_matrix_3x3(camera_json_path):
    with open(camera_json_path, "r", encoding="utf-8") as f:
        cam = json.load(f)

    if "cameraMatrix" in cam:
        K = np.array(cam["cameraMatrix"], dtype=np.float64)
    elif "K" in cam:
        K = np.array(cam["K"], dtype=np.float64)
    else:
        fx = float(cam.get("fx", cam.get("focal")))
        fy = float(cam.get("fy", fx))
        cx = float(cam.get("cx", cam.get("ccx", cam.get("ppx"))))
        cy = float(cam.get("cy", cam.get("ccy", cam.get("ppy"))))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError(f"Intrinseca 3x3 attesa, trovata {K.shape}")
    return K


def load_labels(labels_json_path):
    with open(labels_json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError("labels.json vuoto o non-lista")
    return sorted(labels, key=lambda x: x["filename"])


def build_w2c_from_label(label):
    R = quat_wxyz_to_rotmat(label["q_vbs2tango_true"])
    t = np.asarray(label["r_Vo2To_vbs_true"], dtype=np.float64).reshape(3)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def auto_pose_source(args):
    if args.pose_source != "auto":
        return args.pose_source

    if args.colmap_points_ply and os.path.exists(args.colmap_points_ply):
        return "colmap"

    gt_ready = (
        args.gt_labels_json and os.path.exists(args.gt_labels_json)
        and args.gt_camera_json and os.path.exists(args.gt_camera_json)
        and args.gt_scale_mat_json and os.path.exists(args.gt_scale_mat_json)
    )
    if gt_ready:
        return "gt"

    return "generic"


def check_npz_structure(cam_dict, n_cams, image_files, mask_files):
    print("\n" + "=" * 60)
    print("CHECK 1: Struttura NPZ + inverse consistency")
    print("=" * 60)

    ok = True
    required = ["world_mat", "world_mat_inv", "scale_mat", "scale_mat_inv", "camera_mat", "camera_mat_inv"]

    for i in range(n_cams):
        for key in required:
            fk = f"{key}_{i}"
            if fk not in cam_dict:
                print(f"  ⚠️  Chiave mancante: {fk}")
                ok = False
                continue

            m = cam_dict[fk]
            if m.shape != (4, 4):
                print(f"  ⚠️  {fk} shape {m.shape}, atteso (4,4)")
                ok = False
            if np.any(np.isnan(m)) or np.any(np.isinf(m)):
                print(f"  ⚠️  {fk} contiene NaN/Inf")
                ok = False

    for i in range(n_cams):
        pairs = [("world_mat", "world_mat_inv"), ("camera_mat", "camera_mat_inv"), ("scale_mat", "scale_mat_inv")]
        for a, b in pairs:
            A = cam_dict[f"{a}_{i}"].astype(np.float64)
            B = cam_dict[f"{b}_{i}"].astype(np.float64)
            err = np.linalg.norm(A @ B - np.eye(4), ord="fro")
            if err > 1e-2:
                print(f"  ⚠️  Inversa incoerente {a}_{i}/{b}_{i}, fro_err={err:.3e}")
                ok = False

    if len(image_files) != n_cams:
        print(f"  ⚠️  Immagini ({len(image_files)}) != camere NPZ ({n_cams})")
        ok = False
    else:
        print(f"  ✓ Numero immagini coerente: {len(image_files)}")

    if len(mask_files) > 0 and len(mask_files) != n_cams:
        print(f"  ⚠️  Maschere ({len(mask_files)}) != camere NPZ ({n_cams})")
        ok = False

    scale_0 = cam_dict["scale_mat_0"]
    all_same = all(np.allclose(cam_dict[f"scale_mat_{i}"], scale_0) for i in range(n_cams))
    if all_same:
        print(f"  ✓ scale_mat identica per tutte le {n_cams} viste")
    else:
        print("  ⚠️  scale_mat non identica tra le viste")
        ok = False

    return ok


def check_camera_positions(cam_dict, n_cams, out_dir):
    print("\n" + "=" * 60)
    print("CHECK 2: Pose in spazio normalizzato")
    print("=" * 60)

    positions, view_dirs, rotations, rot_errs, det_errs = [], [], [], [], []

    for i in range(n_cams):
        P = (cam_dict[f"world_mat_{i}"].astype(np.float64) @ cam_dict[f"scale_mat_{i}"].astype(np.float64))[:3, :4]
        cam_pos, view_dir, R = extract_camera_pose(P)
        positions.append(cam_pos)
        view_dirs.append(view_dir)
        rotations.append(R)
        rot_errs.append(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        det_errs.append(abs(np.linalg.det(R) - 1.0))

    positions = np.asarray(positions)
    view_dirs = np.asarray(view_dirs)
    rotations = np.asarray(rotations)
    dists = np.linalg.norm(positions, axis=1)

    print(f"  Distanza min/max/mean: {dists.min():.4f} / {dists.max():.4f} / {dists.mean():.4f}")
    print(f"  Ortonormalita' R (max fro): {np.max(rot_errs):.2e}")
    print(f"  det(R) error (max):        {np.max(det_errs):.2e}")

    ok = True
    if dists.min() < 1.0:
        print(f"  ⚠️  Camere dentro unit sphere (min={dists.min():.4f})")
        ok = False
    if np.max(rot_errs) > 5e-2 or np.max(det_errs) > 5e-2:
        print("  ⚠️  Rotazioni non coerenti")
        ok = False

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=dists, cmap="coolwarm", s=18)
    ax.scatter(0, 0, 0, color="red", marker="*", s=80)
    step = max(1, n_cams // 20)
    for i in range(0, n_cams, step):
        p = positions[i]
        d = view_dirs[i]
        ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], length=0.3, color="green", alpha=0.5)
    ax.set_title("Camera positions (normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check2_camera_positions.png"), dpi=150)
    plt.close()

    return ok, positions, view_dirs, rotations


def check_view_directions(positions, rotations):
    print("\n" + "=" * 60)
    print("CHECK 3: Direzioni di vista verso l'origine")
    print("=" * 60)

    best_axis, view_dirs, angles = choose_best_forward_axis(rotations, positions)
    print(f"  Asse forward stimato: [{best_axis[0]:.0f}, {best_axis[1]:.0f}, {best_axis[2]:.0f}]")

    print(f"  Angolo mean/p95/max: {angles.mean():.2f} / {np.percentile(angles,95):.2f} / {angles.max():.2f} deg")

    ok = angles.max() <= 20.0
    if ok:
        print("  ✓ Camere orientate verso l'origine")
    else:
        print("  ⚠️  Alcune camere non orientate verso il target")
    return ok, view_dirs


def check_target_reprojection(cam_dict, n_cams, image_dir, out_dir):
    print("\n" + "=" * 60)
    print("CHECK 4: Reproiezione target normalizzato (0,0,0)")
    print("=" * 60)

    img_files = list_pngs(image_dir)
    if not img_files:
        print("  ⚠️  Nessuna immagine trovata")
        return False

    sample = cv.imread(os.path.join(image_dir, img_files[0]))
    if sample is None:
        print("  ⚠️  Impossibile leggere immagine campione")
        return False

    H, W = sample.shape[:2]
    cx_exp, cy_exp = W / 2.0, H / 2.0

    errs, depths = [], []
    ok = True

    scale_mat = cam_dict["scale_mat_0"].astype(np.float64)
    scale_inv = np.linalg.inv(scale_mat)
    target_world_origin_norm = scale_inv @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    target_candidates = [
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        np.array([target_world_origin_norm[0], target_world_origin_norm[1], target_world_origin_norm[2], 1.0], dtype=np.float64),
    ]

    for i in range(n_cams):
        P = (cam_dict[f"world_mat_{i}"].astype(np.float64) @ cam_dict[f"scale_mat_{i}"].astype(np.float64))[:3, :4]
        best_err = None
        best_depth = None
        for target in target_candidates:
            proj = P @ target
            if abs(proj[2]) < 1e-8:
                continue
            px = proj[0] / proj[2]
            py = proj[1] / proj[2]
            depth = proj[2]
            if depth <= 0:
                continue
            err = np.hypot(px - cx_exp, py - cy_exp)
            if best_err is None or err < best_err:
                best_err = err
                best_depth = depth

        if best_err is None:
            continue

        errs.append(best_err)
        depths.append(best_depth)

    if len(errs) == 0:
        print("  ⚠️  Nessuna reproiezione valida")
        return False

    errs = np.asarray(errs)
    depths = np.asarray(depths)

    print(f"  Err mean/p95/max: {errs.mean():.2f} / {np.percentile(errs,95):.2f} / {errs.max():.2f} px")
    print(f"  Depth min: {depths.min():.4f}")

    if errs.max() > 80:
        ok = False

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(errs, "o-", markersize=2)
    ax[0].set_title("Target reprojection error")
    ax[1].plot(depths, "o-", markersize=2, color="orange")
    ax[1].axhline(0, color="red", linestyle="--", alpha=0.5)
    ax[1].set_title("Target depth")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check4_target_reprojection.png"), dpi=150)
    plt.close()

    return ok


def check_mesh_reprojection(cam_dict, n_cams, image_dir, mesh_path, out_dir, n_views=4):
    print("\n" + "=" * 60)
    print("CHECK 5: Reproiezione mesh")
    print("=" * 60)

    if mesh_path is None or not os.path.exists(mesh_path):
        print("  ⏭  Mesh non fornita, skip")
        return True

    verts_world = load_mesh_vertices(mesh_path, max_points=3000)
    scale_inv = np.linalg.inv(cam_dict["scale_mat_0"].astype(np.float64))
    verts_h = np.hstack([verts_world, np.ones((len(verts_world), 1), dtype=np.float64)])
    verts_norm = (scale_inv @ verts_h.T).T[:, :3]

    img_files = list_pngs(image_dir)
    if not img_files:
        print("  ⚠️  Immagini mancanti")
        return False

    view_indices = np.linspace(0, n_cams - 1, min(n_views, n_cams), dtype=int)
    fig, axes = plt.subplots(1, len(view_indices), figsize=(5 * len(view_indices), 4))
    if len(view_indices) == 1:
        axes = [axes]

    ok = True
    for col, vi in enumerate(view_indices):
        P = (cam_dict[f"world_mat_{vi}"].astype(np.float64) @ cam_dict[f"scale_mat_{vi}"].astype(np.float64))[:3, :4]
        pxy, z = project_points(P, verts_norm)
        valid = z > 0

        img = cv.imread(os.path.join(image_dir, img_files[vi]))
        if img is None:
            ok = False
            continue

        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        px = pxy[valid, 0]
        py = pxy[valid, 1]
        in_frame = (px >= 0) & (px < W) & (py >= 0) & (py < H)

        axes[col].imshow(rgb)
        axes[col].scatter(px[in_frame], py[in_frame], s=0.3, c="lime", alpha=0.5)
        axes[col].set_title(f"Cam {vi} in-frame {100*in_frame.mean():.1f}%")
        axes[col].axis("off")

        if in_frame.mean() < 0.15:
            ok = False

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check5_mesh_reprojection.png"), dpi=150)
    plt.close()
    return ok


def check_masks(n_cams, mask_dir, image_dir, out_dir, n_views=4):
    print("\n" + "=" * 60)
    print("CHECK 6: Qualita' maschere")
    print("=" * 60)

    if mask_dir is None or not os.path.isdir(mask_dir):
        print("  ⏭  Cartella maschere non fornita, skip")
        return True

    mask_files = list_pngs(mask_dir)
    img_files = list_pngs(image_dir)

    if len(mask_files) == 0:
        print("  ⚠️  Nessuna maschera trovata")
        return False

    if len(mask_files) != n_cams:
        print(f"  ⚠️  Numero maschere {len(mask_files)} != camere {n_cams}")

    coverages = []
    non_binary = []
    all_black = 0
    all_white = 0

    n_eval = min(n_cams, len(mask_files))
    for i in range(n_eval):
        m = cv.imread(os.path.join(mask_dir, mask_files[i]), cv.IMREAD_GRAYSCALE)
        if m is None:
            continue
        fg = m > 127
        cov = fg.mean()
        coverages.append(cov)

        gray = ((m > 10) & (m < 245)).mean()
        non_binary.append(gray)

        if cov < 0.01:
            all_black += 1
        if cov > 0.99:
            all_white += 1

    if len(coverages) == 0:
        print("  ⚠️  Maschere non leggibili")
        return False

    coverages = np.asarray(coverages)
    non_binary = np.asarray(non_binary)

    print(f"  Coverage mean/p5/p95: {coverages.mean():.3f} / {np.percentile(coverages,5):.3f} / {np.percentile(coverages,95):.3f}")
    print(f"  Non-binary mean:      {non_binary.mean():.3f}")
    print(f"  All-black: {all_black}, all-white: {all_white}")

    ok = True
    if all_black > 0:
        ok = False
    if np.percentile(coverages, 95) > 0.98:
        ok = False
    if np.percentile(coverages, 5) < 0.005:
        ok = False
    if non_binary.mean() > 0.20:
        ok = False

    show_n = min(n_views, n_eval, 4)
    view_indices = np.linspace(0, n_eval - 1, show_n, dtype=int)
    fig, axes = plt.subplots(1, show_n, figsize=(5 * show_n, 4))
    if show_n == 1:
        axes = [axes]

    for col, vi in enumerate(view_indices):
        img = cv.imread(os.path.join(image_dir, img_files[vi]))
        m = cv.imread(os.path.join(mask_dir, mask_files[vi]), cv.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        overlay = rgb.copy()
        fg = m > 127
        overlay[fg] = (0.5 * overlay[fg] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        axes[col].imshow(overlay)
        axes[col].set_title(f"Cam {vi} cov={fg.mean():.2f}")
        axes[col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check6_masks.png"), dpi=150)
    plt.close()

    return ok


def check_near_far(cam_dict, n_cams, image_dir, mask_dir=None, sample_step=64):
    print("\n" + "=" * 60)
    print("CHECK 7: Near/Far rays check")
    print("=" * 60)

    img_files = list_pngs(image_dir)
    if len(img_files) == 0:
        print("  ⚠️  Immagini mancanti")
        return False

    sample = cv.imread(os.path.join(image_dir, img_files[0]))
    if sample is None:
        return False
    H, W = sample.shape[:2]

    mask_files = list_pngs(mask_dir)
    use_mask = len(mask_files) >= n_cams

    invalid_rate = []
    hit_rate = []
    fg_hit_rate = []
    bg_hit_rate = []
    for i in range(n_cams):
        P = (cam_dict[f"world_mat_{i}"].astype(np.float64) @ cam_dict[f"scale_mat_{i}"].astype(np.float64))[:3, :4]
        K, pose = load_K_pose_from_P(P)
        K_inv = np.linalg.inv(K)

        ys = np.arange(0, H, sample_step)
        xs = np.arange(0, W, sample_step)
        px, py = np.meshgrid(xs, ys)
        p = np.stack([px.ravel(), py.ravel(), np.ones(px.size)], axis=1)

        rays_v_cam = (K_inv @ p.T).T
        rays_v_cam = rays_v_cam / (np.linalg.norm(rays_v_cam, axis=1, keepdims=True) + 1e-12)

        R = pose[:3, :3]
        rays_d = (R @ rays_v_cam.T).T
        rays_o = np.tile(pose[:3, 3], (len(rays_d), 1))

        a = np.sum(rays_d ** 2, axis=1, keepdims=True)
        b = 2.0 * np.sum(rays_o * rays_d, axis=1, keepdims=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0

        invalid = (far <= near) | (far <= 0)
        invalid_rate.append(invalid.mean())

        closest = rays_o + mid * rays_d
        hit = np.linalg.norm(closest, axis=1) <= 1.0
        hit_rate.append(hit.mean())

        if use_mask:
            m = cv.imread(os.path.join(mask_dir, mask_files[i]), cv.IMREAD_GRAYSCALE)
            if m is not None:
                xi = np.clip(px.ravel().astype(np.int32), 0, W - 1)
                yi = np.clip(py.ravel().astype(np.int32), 0, H - 1)
                fg = m[yi, xi] > 127
                if np.any(fg):
                    fg_hit_rate.append(hit[fg].mean())
                if np.any(~fg):
                    bg_hit_rate.append(hit[~fg].mean())

    invalid_rate = np.asarray(invalid_rate)
    hit_rate = np.asarray(hit_rate)
    print(f"  Invalid near/far rate mean/max: {invalid_rate.mean():.5f} / {invalid_rate.max():.5f}")
    print(f"  Sphere hit-rate mean/max:       {hit_rate.mean():.5f} / {hit_rate.max():.5f}")

    if len(fg_hit_rate) > 0:
        fg_hit_rate = np.asarray(fg_hit_rate)
        print(f"  FG hit-rate mean/min:           {fg_hit_rate.mean():.5f} / {fg_hit_rate.min():.5f}")
    if len(bg_hit_rate) > 0:
        bg_hit_rate = np.asarray(bg_hit_rate)
        print(f"  BG hit-rate mean/max:           {bg_hit_rate.mean():.5f} / {bg_hit_rate.max():.5f}")

    if len(fg_hit_rate) > 0:
        ok = fg_hit_rate.mean() > 0.55 and invalid_rate.max() < 0.25
    else:
        ok = hit_rate.mean() > 0.02 and invalid_rate.max() < 0.25

    if ok:
        print("  ✓ Near/Far coerenti")
    else:
        print("  ⚠️  Near/Far problematici in alcune viste")
    return ok


def check_pose_consistency_colmap(cam_dict, n_cams, image_dir, mask_dir, colmap_points_ply):
    print("\n" + "=" * 60)
    print("CHECK 8: Pose consistency (COLMAP)")
    print("=" * 60)

    try:
        import trimesh
    except ImportError:
        print("  ⚠️  trimesh non installato, skip check COLMAP")
        return True

    if not ensure_exists(colmap_points_ply, "sparse_points_interest.ply", strict=False):
        print("  ⏭  sparse points non disponibili, skip")
        return True

    pcd = trimesh.load(colmap_points_ply)
    pts_world = np.asarray(pcd.vertices, dtype=np.float64)
    if len(pts_world) == 0:
        print("  ⚠️  sparse points vuoti")
        return False

    if len(pts_world) > 6000:
        idx = np.random.choice(len(pts_world), 6000, replace=False)
        pts_world = pts_world[idx]

    scale_inv = np.linalg.inv(cam_dict["scale_mat_0"].astype(np.float64))
    pts_h = np.hstack([pts_world, np.ones((len(pts_world), 1), dtype=np.float64)])
    pts_norm = (scale_inv @ pts_h.T).T[:, :3]

    img_files = list_pngs(image_dir)
    if not img_files:
        print("  ⚠️  Immagini mancanti")
        return False

    mask_files = list_pngs(mask_dir)
    use_mask = len(mask_files) >= n_cams

    in_frame_rates = []
    front_rates = []
    mask_support_rates = []

    for i in range(n_cams):
        P = (cam_dict[f"world_mat_{i}"].astype(np.float64) @ cam_dict[f"scale_mat_{i}"].astype(np.float64))[:3, :4]
        pxy, z = project_points(P, pts_norm)
        front = z > 0
        front_rates.append(front.mean())

        img = cv.imread(os.path.join(image_dir, img_files[i]))
        if img is None:
            continue
        H, W = img.shape[:2]

        px = pxy[front, 0]
        py = pxy[front, 1]
        in_frame = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        in_frame_rates.append(in_frame.mean() if len(in_frame) else 0.0)

        if use_mask and len(px) > 0:
            m = cv.imread(os.path.join(mask_dir, mask_files[i]), cv.IMREAD_GRAYSCALE)
            if m is not None:
                xi = np.clip(px[in_frame].astype(np.int32), 0, W - 1)
                yi = np.clip(py[in_frame].astype(np.int32), 0, H - 1)
                if len(xi) > 0:
                    support = (m[yi, xi] > 127).mean()
                    mask_support_rates.append(support)

    if len(in_frame_rates) == 0:
        print("  ⚠️  Nessuna vista valida nel check COLMAP")
        return False

    in_frame_rates = np.asarray(in_frame_rates)
    front_rates = np.asarray(front_rates)
    mask_support = np.asarray(mask_support_rates) if len(mask_support_rates) > 0 else None

    print(f"  Front rate mean:    {front_rates.mean():.3f}")
    print(f"  In-frame mean:      {in_frame_rates.mean():.3f}")
    if mask_support is not None:
        print(f"  Mask support mean:  {mask_support.mean():.3f}")

    ok = True
    if front_rates.mean() < 0.7:
        ok = False
    if in_frame_rates.mean() < 0.25:
        ok = False
    if mask_support is not None and mask_support.mean() < 0.55:
        ok = False

    return ok


def check_pose_consistency_gt(cam_dict, n_cams, gt_labels_json, gt_camera_json, gt_scale_mat_json):
    print("\n" + "=" * 60)
    print("CHECK 8: Pose consistency (GT SPE3R)")
    print("=" * 60)

    if not ensure_exists(gt_labels_json, "labels.json", strict=False):
        print("  ⏭  labels GT non disponibili, skip")
        return True
    if not ensure_exists(gt_camera_json, "camera.json", strict=False):
        print("  ⏭  camera GT non disponibile, skip")
        return True
    if not ensure_exists(gt_scale_mat_json, "scale_mat.json", strict=False):
        print("  ⏭  scale_mat GT non disponibile, skip")
        return True

    labels = load_labels(gt_labels_json)
    K_gt = load_camera_matrix_3x3(gt_camera_json)

    with open(gt_scale_mat_json, "r", encoding="utf-8") as f:
        sm = json.load(f)
    scale_gt = np.asarray(sm.get("scale_mat", sm), dtype=np.float64)

    n = min(n_cams, len(labels))
    if n == 0:
        return False

    rot_err_deg = []
    trans_err = []
    proj_fro_rel = []

    for i in range(n):
        w2c_gt = build_w2c_from_label(labels[i])
        world_gt = np.eye(4, dtype=np.float64)
        world_gt[:3, :3] = K_gt
        world_gt = world_gt @ w2c_gt

        world_npz = cam_dict[f"world_mat_{i}"].astype(np.float64)

        R_gt = w2c_gt[:3, :3]
        t_gt = w2c_gt[:3, 3]

        # Recover w2c from npz using K from npz for robustness.
        K_npz = cam_dict[f"camera_mat_{i}"][:3, :3].astype(np.float64)
        w2c_npz = np.linalg.inv(K_npz) @ world_npz[:3, :4]
        R_npz = w2c_npz[:3, :3]
        t_npz = w2c_npz[:3, 3]

        R_delta = R_npz @ R_gt.T
        trace_val = np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)
        ang = np.degrees(np.arccos(trace_val))
        rot_err_deg.append(ang)
        trans_err.append(np.linalg.norm(t_npz - t_gt))

        proj_fro_rel.append(np.linalg.norm(world_npz - world_gt, ord="fro") / (np.linalg.norm(world_gt, ord="fro") + 1e-12))

    scale_npz = cam_dict["scale_mat_0"].astype(np.float64)
    center_err = np.linalg.norm(scale_npz[:3, 3] - scale_gt[:3, 3])
    radius_err_rel = abs(scale_npz[0, 0] - scale_gt[0, 0]) / (abs(scale_gt[0, 0]) + 1e-12)

    rot_err_deg = np.asarray(rot_err_deg)
    trans_err = np.asarray(trans_err)
    proj_fro_rel = np.asarray(proj_fro_rel)

    print(f"  Rot err median/p95:   {np.median(rot_err_deg):.3f} / {np.percentile(rot_err_deg,95):.3f} deg")
    print(f"  Trans err median/p95: {np.median(trans_err):.5f} / {np.percentile(trans_err,95):.5f}")
    print(f"  Proj fro rel median:  {np.median(proj_fro_rel):.3e}")
    print(f"  Scale center err:     {center_err:.5f}")
    print(f"  Scale radius rel err: {radius_err_rel:.5f}")

    ok = True
    if np.percentile(rot_err_deg, 95) > 2.0:
        ok = False
    if np.percentile(trans_err, 95) > 0.10:
        ok = False
    if radius_err_rel > 0.05:
        ok = False

    return ok


def check_pose_consistency_generic(positions, view_dirs):
    print("\n" + "=" * 60)
    print("CHECK 8: Pose consistency (Generic fallback)")
    print("=" * 60)

    d = np.linalg.norm(positions, axis=1)
    target_dirs = -positions / (d[:, None] + 1e-12)
    angles = np.degrees(np.arccos(np.clip(np.sum(target_dirs * view_dirs, axis=1), -1.0, 1.0)))

    print(f"  Camera dist CV: {d.std() / (d.mean() + 1e-12):.3f}")
    print(f"  Angle p95/max:  {np.percentile(angles,95):.2f}/{angles.max():.2f} deg")

    ok = True
    if angles.max() > 25.0:
        ok = False
    if d.min() < 1.0:
        ok = False
    return ok


def check_pose_consistency_dispatch(cam_dict, n_cams, args, positions, view_dirs):
    source = auto_pose_source(args)
    print(f"\nPose source mode: {source}")

    if source == "colmap":
        return check_pose_consistency_colmap(
            cam_dict=cam_dict,
            n_cams=n_cams,
            image_dir=args.images,
            mask_dir=args.masks,
            colmap_points_ply=args.colmap_points_ply,
        )

    if source == "gt":
        return check_pose_consistency_gt(
            cam_dict=cam_dict,
            n_cams=n_cams,
            gt_labels_json=args.gt_labels_json,
            gt_camera_json=args.gt_camera_json,
            gt_scale_mat_json=args.gt_scale_mat_json,
        )

    return check_pose_consistency_generic(positions, view_dirs)


def print_summary(results):
    print("\n" + "=" * 60)
    print("SOMMARIO")
    print("=" * 60)

    all_ok = True
    for name, passed in results.items():
        status = "✓" if passed else "⚠️ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("  ✅ TUTTI I CHECK PASSATI")
    else:
        print("  ❌ ALCUNI CHECK FALLITI")
    print("=" * 60)


def build_parser():
    parser = argparse.ArgumentParser(description="NeuS pre-training validation")

    # Base inputs.
    parser.add_argument("--npz", type=str, required=True, help="Path a cameras_sphere.npz")
    parser.add_argument("--images", type=str, required=True, help="Cartella immagini (image/)")
    parser.add_argument("--masks", type=str, default=None, help="Cartella maschere (mask/)")
    parser.add_argument("--mesh", type=str, default=None, help="Mesh OBJ/PLY per reprojection check")
    parser.add_argument("--out", type=str, default="./neus_checks", help="Cartella output")
    parser.add_argument("--n_views", type=int, default=4, help="Numero viste per overlay")

    # Pose source and optional paths.
    parser.add_argument(
        "--pose_source",
        type=str,
        default="auto",
        choices=["auto", "colmap", "gt", "generic"],
        help="Sorgente pose per check avanzato",
    )
    parser.add_argument("--strict", action="store_true", help="Fail se mancano file richiesti dalla modalita'")

    # COLMAP optional inputs.
    parser.add_argument("--colmap_points_ply", type=str, default=None, help="Path sparse_points_interest.ply")
    parser.add_argument("--colmap_images_txt", type=str, default=None, help="Path sparse_txt/images.txt (opzionale)")
    parser.add_argument("--colmap_poses_npy", type=str, default=None, help="Path poses.npy (opzionale)")

    # GT SPE3R optional inputs.
    parser.add_argument("--gt_labels_json", type=str, default=None, help="Path labels.json")
    parser.add_argument("--gt_camera_json", type=str, default=None, help="Path camera.json")
    parser.add_argument("--gt_scale_mat_json", type=str, default=None, help="Path scale_mat.json")
    parser.add_argument("--geometry_json", type=str, default=None, help="Path geometry.json (opzionale)")

    return parser


def validate_mode_requirements(args):
    source = auto_pose_source(args)

    if source == "colmap":
        if args.strict and not ensure_exists(args.colmap_points_ply, "colmap_points_ply", strict=False):
            raise ValueError("Modalita' colmap richiede --colmap_points_ply in strict mode")

    if source == "gt":
        req = [
            (args.gt_labels_json, "gt_labels_json"),
            (args.gt_camera_json, "gt_camera_json"),
            (args.gt_scale_mat_json, "gt_scale_mat_json"),
        ]
        missing = [name for path, name in req if not (path and os.path.exists(path))]
        if args.strict and missing:
            raise ValueError(f"Modalita' gt richiede file mancanti: {missing}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("NeuS Pre-Training Validation")
    print(f"NPZ:         {args.npz}")
    print(f"Images:      {args.images}")
    print(f"Masks:       {args.masks}")
    print(f"Mesh:        {args.mesh}")
    print(f"Output:      {args.out}")
    print(f"Pose source: {args.pose_source}")

    validate_mode_requirements(args)

    cam_dict = load_npz(args.npz)
    n_cams = count_cameras(cam_dict)
    print(f"\nCamere trovate: {n_cams}")

    image_files = list_pngs(args.images)
    mask_files = list_pngs(args.masks)

    results = {}
    results["Struttura NPZ"] = check_npz_structure(cam_dict, n_cams, image_files, mask_files)

    ok2, positions, view_dirs, rotations = check_camera_positions(cam_dict, n_cams, args.out)
    results["Pose normalized"] = ok2

    ok3, view_dirs = check_view_directions(positions, rotations)
    results["Direzioni vista"] = ok3
    results["Target reprojection"] = check_target_reprojection(cam_dict, n_cams, args.images, args.out)
    results["Mesh reprojection"] = check_mesh_reprojection(cam_dict, n_cams, args.images, args.mesh, args.out, n_views=args.n_views)
    results["Maschere"] = check_masks(n_cams, args.masks, args.images, args.out, n_views=args.n_views)
    results["Near/Far"] = check_near_far(cam_dict, n_cams, args.images, args.masks)
    results["Pose consistency"] = check_pose_consistency_dispatch(cam_dict, n_cams, args, positions, view_dirs)

    print_summary(results)


if __name__ == "__main__":
    main()
