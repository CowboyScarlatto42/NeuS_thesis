import argparse
import json
import shutil
from pathlib import Path
import numpy as np


def quat_wxyz_to_rot(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])


def rot_to_quat_xyzw(R):
    R = np.asarray(R, dtype=float)
    q = np.empty(4)
    t = np.trace(R)

    if t > 0:
        S = np.sqrt(t + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    return q


def relative_pose(q_cam_wxyz, t_cam, q_obj_wxyz, t_obj):
    """
    q_cam_wxyz: camera -> world
    q_obj_wxyz: object -> world

    ritorna:
    q_rel_xyzw: camera -> object
    t_rel: object origin espresso nel frame camera
    """
    R_wc = quat_wxyz_to_rot(q_cam_wxyz)
    R_wo = quat_wxyz_to_rot(q_obj_wxyz)

    R_co = R_wc.T @ R_wo
    t_co = R_wc.T @ (np.asarray(t_obj, dtype=float) - np.asarray(t_cam, dtype=float))

    q_rel_xyzw = rot_to_quat_xyzw(R_co)
    return q_rel_xyzw, t_co


def make_camera_json():
    fov_deg = 10
    Nu = 2048
    Nv = 2048
    fx = (Nu / 2) / np.tan(np.deg2rad(fov_deg / 2))
    fy = (Nv / 2) / np.tan(np.deg2rad(fov_deg / 2))
    ccx = Nu / 2
    ccy = Nv / 2

    camera_dict = {
        "Nu": Nu,
        "Nv": Nv,
        "fx": fx,
        "fy": fy,
        "ccx": ccx,
        "ccy": ccy,
        "cameraMatrix": [
            [float(fx), 0, float(ccx)],
            [0, float(fy), float(ccy)],
            [0, 0, 1]
        ],
        "distCoeffs": [0, 0, 0, 0, 0]
    }
    return camera_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path cartella immagini")
    parser.add_argument("--masks", required=True, help="Path cartella maschere")
    parser.add_argument("--geometry", required=True, help="Path geometry.json")
    parser.add_argument("--output", required=True, help="Path cartella output")
    args = parser.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    output_dir = Path(args.output)

    img_out = output_dir / "dawn_images"
    mask_out = output_dir / "dawn_masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_dir.iterdir() if p.is_file()])
    mask_files = sorted([p for p in masks_dir.iterdir() if p.is_file()])

    if len(image_files) != len(mask_files):
        raise ValueError(f"Numero immagini ({len(image_files)}) diverso da numero maschere ({len(mask_files)})")

    with open(args.geometry, "r") as f:
        geom = json.load(f)

    cam_pos = geom["camera"]["position"]
    cam_ori = geom["camera"]["orientation"]
    body_pos = geom["body"]["position"]
    body_ori = geom["body"]["orientation"]

    n = len(cam_pos)

    if not (len(cam_ori) == len(body_pos) == len(body_ori) == len(image_files) == len(mask_files) == n):
        raise ValueError(
            "Mismatch tra numero di immagini, maschere e pose in geometry.json: "
            f"images={len(image_files)}, masks={len(mask_files)}, "
            f"cam_pos={len(cam_pos)}, cam_ori={len(cam_ori)}, "
            f"body_pos={len(body_pos)}, body_ori={len(body_ori)}"
        )

    labels = []

    for i, (img_path, mask_path) in enumerate(zip(image_files, mask_files), start=1):
        out_name = f"img{i:06d}.png"
        out_stem = f"img{i:06d}"

        shutil.copy2(img_path, img_out / out_name)
        shutil.copy2(mask_path, mask_out / out_name)

        q_rel, t_rel = relative_pose(
            q_cam_wxyz=cam_ori[i - 1],
            t_cam=cam_pos[i - 1],
            q_obj_wxyz=body_ori[i - 1],
            t_obj=body_pos[i - 1],
        )

        labels.append({
            "filename": out_stem,
            "q_vbs2tango_true": q_rel.tolist(),
            "r_Vo2To_vbs_true": t_rel.tolist()
        })

    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    camera_dict = make_camera_json()
    with open(output_dir / "camera.json", "w") as f:
        json.dump(camera_dict, f, indent=2)

    print(f"Dataset creato in: {output_dir}")


if __name__ == "__main__":
    main()