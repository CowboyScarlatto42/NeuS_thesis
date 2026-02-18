"""
gen_cameras_with_masks.py
=========================
Adattamento MINIMALE di gen_cameras.py (NeuS/preprocess_custom_data/colmap_preprocess/).
Tutta la logica originale e' invariata. L'unica modifica e' nella sezione
di copia immagini: invece di glob('images/*.png') su tutte le 500,
usa sparse_txt/images.txt per prendere SOLO le N registrate da COLMAP,
nello stesso ordine (sorted per nome) usato da pose_utils.py per poses.npy.
Le maschere vengono copiate con lo stesso nome dell'immagine corrispondente.

USO:
    python gen_cameras_with_masks.py <images_dir> <masks_dir> [out_dir] [colmap_dir]

    images_dir : directory con le immagini originali
    masks_dir  : directory con le maschere originali (stesso nome delle immagini)
    out_dir    : (opzionale) cartella di output. Default: <colmap_dir>/preprocessed
    colmap_dir : (opzionale) directory con poses.npy, sparse_points_interest.ply,
                 sparse_txt/. Default: uguale a images_dir

ESEMPI SU COLAB:
    # Tutto nella stessa cartella
    python gen_cameras_with_masks.py /content/my_data /content/my_data/masks

    # Immagini, maschere e output COLMAP in cartelle separate
    python gen_cameras_with_masks.py \
        /content/my_data/images \
        /content/my_data/masks \
        /content/NeuS_thesis/public_data/my_object \
        /content/my_data/colmap_output
"""

import numpy as np
import trimesh
import cv2 as cv
import sys
import os
from pathlib import Path


def read_images_txt_names(path):
    """
    Legge sparse_txt/images.txt e restituisce i nomi file
    ordinati per nome — identico all'np.argsort di pose_utils.py.
    """
    names = []
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip() != '']
    for i in range(0, len(lines), 2):   # righe in coppia: posa + punti2D
        parts = lines[i].strip().split()
        names.append(parts[9])          # campo NAME
    return sorted(names)                # equivalente a np.argsort usato in pose_utils.py


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("USO: python gen_cameras_with_masks.py <images_dir> <masks_dir> [out_dir] [colmap_dir]")
        sys.exit(1)

    images_dir = sys.argv[1]
    masks_dir  = sys.argv[2]
    out_dir    = sys.argv[3] if len(sys.argv) > 3 else None
    colmap_dir = sys.argv[4] if len(sys.argv) > 4 else images_dir

    # out_dir default: <colmap_dir>/preprocessed
    if out_dir is None:
        out_dir = os.path.join(colmap_dir, 'preprocessed')

    print(f'images_dir : {images_dir}')
    print(f'masks_dir  : {masks_dir}')
    print(f'colmap_dir : {colmap_dir}')
    print(f'out_dir    : {out_dir}')

    # ── IDENTICO a gen_cameras.py originale ─────────────────────────────────
    poses_hwf = np.load(os.path.join(colmap_dir, 'poses.npy'))  # (N, 3, 5)
    poses_raw = poses_hwf[:, :, :4]
    hwf       = poses_hwf[:, :, 4]

    cam_dict = dict()
    n_images = len(poses_raw)

    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] =  1.0
    convert_mat[1, 0] =  1.0
    convert_mat[2, 2] = -1.0
    convert_mat[3, 3] =  1.0

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (w - 1) * 0.5
        intrinsic[1, 2] = (h - 1) * 0.5
        w2c       = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{}'.format(i)]     = intrinsic
        cam_dict['camera_mat_inv_{}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(i)]      = world_mat
        cam_dict['world_mat_inv_{}'.format(i)]  = np.linalg.inv(world_mat)

    pcd      = trimesh.load(os.path.join(colmap_dir, 'sparse_points_interest.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center   = (bbox_max + bbox_min) * 0.5
    radius   = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    for i in range(n_images):
        cam_dict['scale_mat_{}'.format(i)]     = scale_mat
        cam_dict['scale_mat_inv_{}'.format(i)] = np.linalg.inv(scale_mat)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'),  exist_ok=True)
    # ── fine parte identica all'originale ────────────────────────────────────

    # ── UNICA MODIFICA: immagini da images.txt invece di glob ────────────────
    images_txt   = os.path.join(colmap_dir, 'sparse_txt', 'images.txt')
    colmap_names = read_images_txt_names(images_txt)

    assert len(colmap_names) == n_images, \
        f"ERRORE: {len(colmap_names)} nomi in images.txt vs {n_images} pose in poses.npy!"

    missing_masks = []
    for i, fname in enumerate(colmap_names):
        # Immagine
        img = cv.imread(os.path.join(images_dir, fname))
        cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)

        # Maschera — stesso nome file dell'immagine (img000001.png → img000001.png)
        src_mask = os.path.join(masks_dir, fname)
        if os.path.exists(src_mask):
            mask = cv.imread(src_mask)
            cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), mask)
        else:
            missing_masks.append((i, fname))

    # Maschere mancanti → maschera bianca (tutto visibile, equivalente a womask)
    if missing_masks:
        sample = cv.imread(os.path.join(out_dir, 'image', '000.png'))
        white  = np.ones_like(sample) * 255
        for i, fname in missing_masks:
            cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), white)
            print(f'[WARN] maschera mancante per {fname} -> generata bianca')
    else:
        print(f'Tutte le {n_images} maschere trovate e copiate.')

    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    print('Process done!')
