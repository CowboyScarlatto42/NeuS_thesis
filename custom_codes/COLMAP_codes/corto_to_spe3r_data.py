import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# Blender/OpenGL camera frame -> CV/NeuS camera frame
# x stays the same, y and z are flipped.
CV_FROM_BLENDER = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def quat_xyzw_to_rot(q_xyzw):
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    return R.from_quat(q_xyzw).as_matrix().astype(np.float32)


def rot_to_quat_wxyz(R_mat):
    q_xyzw = R.from_matrix(np.asarray(R_mat, dtype=np.float64)).as_quat()
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def relative_pose_body_to_camera_cv(q_cam_xyzw, t_cam, q_obj_xyzw, t_obj):
    """
    Input from geometry.json:
      - q_cam_xyzw, t_cam: camera -> world
      - q_obj_xyzw, t_obj: object/body -> world

    Returns:
      - q_rel_wxyz: body -> camera quaternion in CV/NeuS convention, [w,x,y,z]
      - t_rel_cv: object origin expressed in camera frame, CV/NeuS convention

    Convention of the returned transform:
      X_cam = R * X_body + t
    """
    # world <- camera
    R_wc = quat_xyzw_to_rot(q_cam_xyzw)
    t_cam = np.asarray(t_cam, dtype=np.float32).reshape(3, 1)

    # world <- object/body
    R_wo = quat_xyzw_to_rot(q_obj_xyzw)
    t_obj = np.asarray(t_obj, dtype=np.float32).reshape(3, 1)

    # Blender/OpenGL relative pose: camera_blender <- object
    R_co_bl = R_wc.T @ R_wo
    t_co_bl = R_wc.T @ (t_obj - t_cam)

    # Convert camera frame Blender/OpenGL -> CV/NeuS
    R_co_cv = CV_FROM_BLENDER @ R_co_bl
    t_co_cv = CV_FROM_BLENDER @ t_co_bl

    q_rel_wxyz = rot_to_quat_wxyz(R_co_cv)
    return q_rel_wxyz, t_co_cv.reshape(3)


def make_camera_json():
    Nu = 1024
    Nv = 1024
    ppx = 2.74e-6
    ppy = 2.74e-6
    fx = 0.0035
    fy = 0.0035
    ccx = Nu / 2
    ccy = Nv / 2

    camera_dict = {
        "Nu": Nu,
        "Nv": Nv,
        "ppx": ppx,
        "ppy": ppy,
        "fx": fx,
        "fy": fy,
        "ccx": ccx,
        "ccy": ccy,
        "cameraMatrix": [
            [1277.37226, 0, float(ccx)],
            [0, 1277.37226, float(ccy)],
            [0, 0, 1],
        ],
        "distCoeffs": [0, 0, 0, 0, 0],
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
        raise ValueError(
            f"Numero immagini ({len(image_files)}) diverso da numero maschere ({len(mask_files)})"
        )

    with open(args.geometry, "r") as f:
        geom = json.load(f)

    cam_pos = geom["camera"]["position"]
    cam_ori = geom["camera"]["orientation"]  # geometry.json stores quaternions as [x,y,z,w]
    body_pos = geom["body"]["position"]
    body_ori = geom["body"]["orientation"]   # geometry.json stores quaternions as [x,y,z,w]

    n = len(cam_pos)

    if not (
        len(cam_ori) == len(body_pos) == len(body_ori) == len(image_files) == len(mask_files) == n
    ):
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

        q_rel_wxyz, t_rel_cv = relative_pose_body_to_camera_cv(
            q_cam_xyzw=cam_ori[i - 1],
            t_cam=cam_pos[i - 1],
            q_obj_xyzw=body_ori[i - 1],
            t_obj=body_pos[i - 1],
        )

        labels.append(
            {
                "filename": out_stem,
                # saved as [w,x,y,z]
                "q_vbs2tango_true": q_rel_wxyz.tolist(),
                # body/object origin expressed in camera frame (CV convention)
                "r_Vo2To_vbs_true": t_rel_cv.tolist(),
            }
        )

    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    camera_dict = make_camera_json()
    with open(output_dir / "camera.json", "w") as f:
        json.dump(camera_dict, f, indent=2)

    print(f"Dataset creato in: {output_dir}")
    print("q_vbs2tango_true salvato come [w,x,y,z]")
    print("r_Vo2To_vbs_true salvato in convenzione CV/NeuS")


if __name__ == "__main__":
    main()
