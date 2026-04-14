"""
NeuS Pre-Training Validation Script
===================================
Controlli minimi per verificare che il dataset sia pronto per NeuS.

Checks:
  1. Struttura NPZ
  2. Posizioni camera in spazio normalizzato
  3. Direzioni di vista verso l'origine
  4. Reproiezione del target (0,0,0)
  5. Reproiezione mesh su immagini
  6. Coerenza maschere
  7. Near/Far plane check
"""

import argparse
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


def extract_camera_pose(P, camera_mat):
    K = camera_mat[:3, :3].astype(np.float64)
    extrinsic = np.linalg.inv(K) @ P[:3, :4].astype(np.float64)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    cam_pos = -R.T @ t
    view_dir = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)

    return cam_pos, view_dir


def project_points(P, points_3d):
    n_points = points_3d.shape[0]
    points_h = np.hstack([points_3d, np.ones((n_points, 1))])
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
        with open(mesh_path, "r") as f:
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
        raise ValueError(f"Formato non supportato: {ext}")

    if len(vertices) > max_points:
        indices = np.random.choice(len(vertices), max_points, replace=False)
        vertices = vertices[indices]

    return vertices


def check_npz_structure(cam_dict, n_cams):
    print("\n" + "=" * 60)
    print("CHECK 1: Struttura NPZ")
    print("=" * 60)

    ok = True
    required_keys = ["world_mat", "world_mat_inv", "scale_mat", "scale_mat_inv", "camera_mat", "camera_mat_inv"]

    for i in range(n_cams):
        for key in required_keys:
            full_key = f"{key}_{i}"
            if full_key not in cam_dict:
                print(f"  ⚠️  Chiave mancante: {full_key}")
                ok = False
                continue

            mat = cam_dict[full_key]
            if mat.shape != (4, 4):
                print(f"  ⚠️  {full_key} ha shape {mat.shape}, atteso (4,4)")
                ok = False
            if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
                print(f"  ⚠️  {full_key} contiene NaN o Inf!")
                ok = False

    scale_0 = cam_dict["scale_mat_0"]
    all_same = all(np.allclose(cam_dict[f"scale_mat_{i}"], scale_0) for i in range(n_cams))
    if all_same:
        print(f"  ✓ scale_mat identica per tutte le {n_cams} viste")
        print(f"    Raggio: {scale_0[0, 0]:.4f}")
        print(f"    Centro: {scale_0[:3, 3]}")
    else:
        print("  ⚠️  scale_mat NON identica tra le viste!")
        ok = False

    if ok:
        print(f"  ✓ Struttura OK — {n_cams} camere, tutte le chiavi presenti")

    return ok


def check_camera_positions(cam_dict, n_cams, out_dir):
    print("\n" + "=" * 60)
    print("CHECK 2: Posizioni camera in spazio normalizzato")
    print("=" * 60)

    positions = []
    view_dirs = []

    for i in range(n_cams):
        world_mat = cam_dict[f"world_mat_{i}"].astype(np.float64)
        scale_mat = cam_dict[f"scale_mat_{i}"].astype(np.float64)
        camera_mat = cam_dict[f"camera_mat_{i}"].astype(np.float64)
        P = world_mat @ scale_mat
        cam_pos, view_dir = extract_camera_pose(P, camera_mat)
        positions.append(cam_pos)
        view_dirs.append(view_dir)

    positions = np.asarray(positions)
    view_dirs = np.asarray(view_dirs)
    dists = np.linalg.norm(positions, axis=1)

    print(f"  Distanza min:  {dists.min():.4f}")
    print(f"  Distanza max:  {dists.max():.4f}")
    print(f"  Distanza mean: {dists.mean():.4f}")

    ok = True
    if dists.min() < 1.0:
        print(f"  ⚠️  Alcune camere sono DENTRO la unit sphere! (min={dists.min():.4f})")
        ok = False
    else:
        print("  ✓ Tutte le camere sono fuori dalla unit sphere")

    if dists.max() > 8.0:
        print(f"  ⚠️  Alcune camere troppo lontane (max={dists.max():.4f} > 8.0)")
        ok = False
    else:
        print("  ✓ Nessuna camera troppo lontana")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.1, color="gray")

    scatter = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=dists, cmap="coolwarm", s=30, edgecolors="k", linewidths=0.3,
    )
    plt.colorbar(scatter, ax=ax, label="Distanza dal centro", shrink=0.6)
    ax.scatter(0, 0, 0, color="red", s=100, marker="*", label="Target (origin)")

    step = max(1, n_cams // 20)
    for i in range(0, n_cams, step):
        p = positions[i]
        d = view_dirs[i]
        ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2], length=0.3, normalize=True, color="green", alpha=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera positions (normalized space)")
    ax.legend()

    max_range = max(dists.max(), 1.5)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check2_camera_positions.png"), dpi=150)
    plt.close()
    print("  → Salvato: check2_camera_positions.png")

    return ok, positions, view_dirs


def check_view_directions(positions, view_dirs):
    print("\n" + "=" * 60)
    print("CHECK 3: Direzioni di vista verso l'origine")
    print("=" * 60)

    target_dirs = -positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 1e-12)
    dots = np.sum(view_dirs * target_dirs, axis=1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))

    print(f"  Angolo medio camera→origine: {angles.mean():.2f}°")
    print(f"  Angolo max camera→origine:   {angles.max():.2f}°")

    if angles.max() > 15.0:
        print("  ⚠️  Alcune camere non puntano bene verso l'origine")
        return False

    print("  ✓ Le camere puntano verso l'origine")
    return True


def check_target_reprojection(cam_dict, n_cams, image_dir, out_dir):
    print("\n" + "=" * 60)
    print("CHECK 4: Reproiezione target (0,0,0) → centro immagine")
    print("=" * 60)

    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    if not img_files:
        print("  ⚠️  Nessuna immagine trovata!")
        return False

    sample_img = cv.imread(os.path.join(image_dir, img_files[0]))
    if sample_img is None:
        print("  ⚠️  Impossibile leggere la prima immagine")
        return False

    H, W = sample_img.shape[:2]
    cx_expected, cy_expected = W / 2.0, H / 2.0

    print(f"  Dimensioni immagine: {W}x{H}")
    print(f"  Centro atteso: ({cx_expected:.1f}, {cy_expected:.1f})")

    errors = []
    depths = []
    ok = True

    scale_mat = cam_dict["scale_mat_0"].astype(np.float64)
    scale_inv = np.linalg.inv(scale_mat)
    target_norm = (scale_inv @ np.array([0, 0, 0, 1]))[:3]

    print(f"  Target in spazio normalizzato: {target_norm}")

    for i in range(n_cams):
        world_mat = cam_dict[f"world_mat_{i}"].astype(np.float64)
        P = world_mat @ scale_mat

        target_h = np.append(target_norm, 1.0)
        proj = P @ target_h

        if abs(proj[2]) < 1e-8:
            print(f"  ⚠️  Camera {i}: target a profondità ~0 (proj[2]={proj[2]:.6f})")
            ok = False
            continue

        px = proj[0] / proj[2]
        py = proj[1] / proj[2]
        depth = proj[2]

        err = np.sqrt((px - cx_expected) ** 2 + (py - cy_expected) ** 2)
        errors.append(err)
        depths.append(depth)

        if depth < 0:
            print(f"  ⚠️  Camera {i}: target DIETRO la camera (depth={depth:.4f})")
            ok = False

    errors = np.asarray(errors)
    depths = np.asarray(depths)

    print("\n  Errore reproiezione target (pixel):")
    print(f"    Mean: {errors.mean():.2f}")
    print(f"    Max:  {errors.max():.2f}")
    print(f"    Min:  {errors.min():.2f}")

    if errors.max() > 50:
        print(f"  ⚠️  Target non al centro in alcune viste (errore max={errors.max():.1f}px)")
        worst = np.argmax(errors)
        print(f"      Peggiore: camera {worst} con errore {errors[worst]:.1f}px")
        ok = False
    else:
        print("  ✓ Target proietta al centro in tutte le viste")

    if np.all(depths > 0):
        print(f"  ✓ Target davanti a tutte le camere (depth min={depths.min():.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(errors, "o-", markersize=3)
    ax1.axhline(y=5, color="g", linestyle="--", alpha=0.5, label="5px threshold")
    ax1.set_xlabel("Camera index")
    ax1.set_ylabel("Reprojection error (px)")
    ax1.set_title("Target reprojection error")
    ax1.legend()

    ax2.plot(depths, "o-", markersize=3, color="orange")
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Camera index")
    ax2.set_ylabel("Depth")
    ax2.set_title("Target depth in each camera")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check4_target_reprojection.png"), dpi=150)
    plt.close()
    print("  → Salvato: check4_target_reprojection.png")

    return ok


def check_mesh_reprojection(cam_dict, n_cams, image_dir, mesh_path, out_dir, n_views=4):
    print("\n" + "=" * 60)
    print("CHECK 5: Reproiezione mesh su immagini")
    print("=" * 60)

    if mesh_path is None or not os.path.exists(mesh_path):
        print("  ⏭  Mesh non fornito, skip")
        return True

    vertices_world = load_mesh_vertices(mesh_path, max_points=3000)
    print(f"  Vertici caricati: {len(vertices_world)}")
    print(f"  Bbox mondo: {vertices_world.min(0)} → {vertices_world.max(0)}")

    scale_mat = cam_dict["scale_mat_0"].astype(np.float64)
    scale_inv = np.linalg.inv(scale_mat)

    verts_h = np.hstack([vertices_world, np.ones((len(vertices_world), 1))])
    verts_norm = (scale_inv @ verts_h.T).T[:, :3]

    print(f"  Bbox normalizzato: {verts_norm.min(0)} → {verts_norm.max(0)}")
    print(f"  Raggio normalizzato: {np.linalg.norm(verts_norm, axis=1).max():.4f}")

    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    view_indices = np.linspace(0, n_cams - 1, n_views, dtype=int)

    fig, axes = plt.subplots(2, n_views, figsize=(5 * n_views, 10))
    if n_views == 1:
        axes = axes.reshape(-1, 1)

    for col, vi in enumerate(view_indices):
        world_mat = cam_dict[f"world_mat_{vi}"].astype(np.float64)
        P = world_mat @ scale_mat

        pixels, depths = project_points(P, verts_norm)
        valid = depths > 0
        px_valid = pixels[valid, 0]
        py_valid = pixels[valid, 1]

        img_path = os.path.join(image_dir, img_files[vi])
        img = cv.imread(img_path)
        if img is None:
            print(f"  ⚠️  Impossibile leggere {img_path}")
            continue

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        axes[0, col].imshow(img_rgb)
        in_frame = (px_valid > 0) & (px_valid < W) & (py_valid > 0) & (py_valid < H)
        axes[0, col].scatter(px_valid[in_frame], py_valid[in_frame], c="lime", s=0.5, alpha=0.5)
        axes[0, col].set_title(f"Camera {vi}")
        axes[0, col].set_xlim(0, W)
        axes[0, col].set_ylim(H, 0)
        axes[0, col].axis("off")

        axes[1, col].set_facecolor("black")
        axes[1, col].scatter(px_valid[in_frame], py_valid[in_frame], c=depths[valid][in_frame], cmap="plasma", s=0.5, alpha=0.7)
        axes[1, col].set_xlim(0, W)
        axes[1, col].set_ylim(H, 0)
        axes[1, col].set_title(f"Depth map cam {vi}")
        axes[1, col].set_aspect("equal")
        axes[1, col].axis("off")

        n_in = in_frame.sum()
        n_total = valid.sum()
        print(f"  Camera {vi}: {n_in}/{n_total} punti nel frame ({100 * n_in / max(n_total, 1):.0f}%)")

    plt.suptitle("Mesh reprojection overlay (top) + depth (bottom)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check5_mesh_reprojection.png"), dpi=150)
    plt.close()
    print("  → Salvato: check5_mesh_reprojection.png")

    return True


def check_masks(cam_dict, n_cams, mask_dir, image_dir, out_dir, n_views=4):
    print("\n" + "=" * 60)
    print("CHECK 6: Coerenza maschere")
    print("=" * 60)

    if mask_dir is None or not os.path.exists(mask_dir):
        print("  ⏭  Cartella maschere non fornita, skip")
        return True

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    if len(mask_files) != n_cams:
        print(f"  ⚠️  Numero maschere ({len(mask_files)}) ≠ numero camere ({n_cams})")

    all_white = 0
    all_black = 0
    for mf in mask_files:
        mask = cv.imread(os.path.join(mask_dir, mf), cv.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        ratio = mask.mean() / 255.0
        if ratio > 0.99:
            all_white += 1
        elif ratio < 0.01:
            all_black += 1

    print(f"  Maschere totali: {len(mask_files)}")
    print(f"  Tutte bianche: {all_white}")
    print(f"  Tutte nere: {all_black}")

    if all_black > 0:
        print(f"  ⚠️  {all_black} maschere completamente nere — nessun oggetto visibile!")

    view_indices = np.linspace(0, min(n_cams, len(mask_files)) - 1, min(n_views, 4), dtype=int)
    fig, axes = plt.subplots(1, len(view_indices), figsize=(5 * len(view_indices), 5))
    if len(view_indices) == 1:
        axes = [axes]

    for col, vi in enumerate(view_indices):
        img = cv.imread(os.path.join(image_dir, img_files[vi]))
        mask = cv.imread(os.path.join(mask_dir, mask_files[vi]), cv.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        overlay = img_rgb.copy()
        overlay[mask > 127] = (overlay[mask > 127] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

        axes[col].imshow(overlay)
        axes[col].set_title(f"Cam {vi} — mask ratio: {mask.mean() / 255:.2f}")
        axes[col].axis("off")

    plt.suptitle("Maschere (verde) su immagini", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "check6_masks.png"), dpi=150)
    plt.close()
    print("  → Salvato: check6_masks.png")

    return True


def check_near_far(cam_dict, n_cams):
    print("\n" + "=" * 60)
    print("CHECK 7: Near/Far plane (raggiungibilità oggetto)")
    print("=" * 60)

    scale_mat = cam_dict["scale_mat_0"].astype(np.float64)
    dists = []

    for i in range(n_cams):
        world_mat = cam_dict[f"world_mat_{i}"].astype(np.float64)
        camera_mat = cam_dict[f"camera_mat_{i}"].astype(np.float64)
        P = world_mat @ scale_mat
        cam_pos, _ = extract_camera_pose(P, camera_mat)
        dists.append(np.linalg.norm(cam_pos))

    dists = np.asarray(dists)
    print(f"  Distanze camera (normalized): {dists.min():.3f} — {dists.max():.3f}")
    print(f"  Near plane tipico: {(dists.min() - 1.5):.3f}")
    print(f"  Far plane tipico:  {(dists.max() + 1.5):.3f}")
    print(f"  Oggetto (raggio in norm space): ~{scale_mat[0, 0]:.2f}⁻¹ del raggio mondo")
    print("  ✓ I raggi possono raggiungere l'oggetto nella unit sphere")

    return True


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
        print("  ✅ TUTTI I CHECK PASSATI — puoi lanciare il training!")
    else:
        print("  ❌ ALCUNI CHECK FALLITI — correggi prima di trainare!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NeuS pre-training validation")
    parser.add_argument("--npz", type=str, required=True, help="Path a cameras_sphere.npz")
    parser.add_argument("--images", type=str, required=True, help="Cartella immagini (image/)")
    parser.add_argument("--masks", type=str, default=None, help="Cartella maschere (mask/)")
    parser.add_argument("--mesh", type=str, default=None, help="Mesh OBJ/PLY per reprojection check")
    parser.add_argument("--out", type=str, default="./neus_checks", help="Cartella output per immagini di check")
    parser.add_argument("--n_views", type=int, default=4, help="Numero di viste per overlay (default 4)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("NeuS Pre-Training Validation")
    print(f"NPZ:    {args.npz}")
    print(f"Images: {args.images}")
    print(f"Masks:  {args.masks}")
    print(f"Mesh:   {args.mesh}")
    print(f"Output: {args.out}")

    cam_dict = load_npz(args.npz)
    n_cams = count_cameras(cam_dict)
    print(f"\nCamere trovate: {n_cams}")

    results = {}
    results["Struttura NPZ"] = check_npz_structure(cam_dict, n_cams)

    ok2, positions, view_dirs = check_camera_positions(cam_dict, n_cams, args.out)
    results["Posizioni camera"] = ok2

    results["Direzioni vista"] = check_view_directions(positions, view_dirs)
    results["Reproiezione target"] = check_target_reprojection(cam_dict, n_cams, args.images, args.out)
    results["Reproiezione mesh"] = check_mesh_reprojection(cam_dict, n_cams, args.images, args.mesh, args.out, n_views=args.n_views)
    results["Maschere"] = check_masks(cam_dict, n_cams, args.masks, args.images, args.out, n_views=args.n_views)
    results["Near/Far plane"] = check_near_far(cam_dict, n_cams)

    print_summary(results)


if __name__ == "__main__":
    main()
