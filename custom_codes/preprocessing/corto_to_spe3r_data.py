import os
import shutil
import re
import json
import numpy as np
import argparse


def copy_and_rename_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    # ordina numericamente
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    for i, filename in enumerate(files, start=1):
        new_name = f"img{i:06d}.png"
        
        src = os.path.join(input_folder, filename)
        dst = os.path.join(output_folder, new_name)
        
        shutil.copy2(src, dst)

    print(f"Fatto: copiate {len(files)} immagini in {output_folder}")

def copy_and_rename_masks(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    def extract_index(filename):
        match = re.search(r"mask_(\d+)_\d+\.png", filename)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Nome file non valido: {filename}")

    # ordina usando il primo indice
    files.sort(key=extract_index)

    for i, filename in enumerate(files, start=1):
        new_name = f"img{i:06d}.png"

        src = os.path.join(input_folder, filename)
        dst = os.path.join(output_folder, new_name)

        shutil.copy2(src, dst)

    print(f"Fatto: copiate {len(files)} mask rinominate in {output_folder}")

def create_camera_json(
    output_folder,
    Nu,
    Nv,
    ppx,
    ppy,
    fx,
    fy,
    ccx,
    ccy,
    camera_matrix,
    dist_coeffs,
    filename="camera.json"
):
    os.makedirs(output_folder, exist_ok=True)

    camera_data = {
        "Nu": Nu,
        "Nv": Nv,
        "ppx": ppx,
        "ppy": ppy,
        "fx": fx,
        "fy": fy,
        "ccx": ccx,
        "ccy": ccy,
        "cameraMatrix": camera_matrix,
        "distCoeffs": dist_coeffs
    }

    output_path = os.path.join(output_folder, filename)

    with open(output_path, "w") as f:
        json.dump(camera_data, f, indent=2)

    print(f"Creato file: {output_path}")


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Quaternione nullo.")
    return (q / n).tolist()


def quat_conjugate(q):
    # q = [w, x, y, z]
    w, x, y, z = q
    return [w, -x, -y, -z]


def quat_multiply(q1, q2):
    # Hamilton product, input/output in [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def quat_rotate_vector(q, v):
    # ruota il vettore v con q, con q in [w, x, y, z]
    q = quat_normalize(q)
    vq = [0.0, v[0], v[1], v[2]]
    q_conj = quat_conjugate(q)
    out = quat_multiply(quat_multiply(q, vq), q_conj)
    return out[1:]


def reorder_quaternion(q_wxyz, output_order="xyzw"):
    w, x, y, z = q_wxyz

    if output_order == "wxyz":
        return [w, x, y, z]
    elif output_order == "xyzw":
        return [x, y, z, w]
    else:
        raise ValueError("output_order deve essere 'wxyz' o 'xyzw'")


def rotmat_to_quat_wxyz(R):
    """
    Conversione matrice di rotazione 3x3 -> quaternione [w, x, y, z]
    """
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return quat_normalize([w, x, y, z])

def generate_labels_from_geometry(
    geometry_json_path,
    output_labels_path,
    output_order="wxyz"
):
    """
    Genera labels.json a partire da geometry.json.

    Assunzioni:
    - camera.orientation = q_camera_to_world  [w, x, y, z] (Blender: -Y forward, +Z up)
    - body.orientation   = q_target_to_world  [w, x, y, z] (Blender: -Y forward, +Z up)

    Output (convenzione SPE3R / SPEED):
    - q_vbs2tango_true = orientazione target rispetto alla camera (target -> camera)
      tale che X_cam = R(q) * X_body + t
    - r_Vo2To_vbs_true = posizione target nel frame camera (CV)

    Fix di frame camera:
    Blender (-Y forward, +Z up)
    --> CV/NeuS (+Z forward, -Y up)
    """

    # Fix di frame camera: Blender (-Y forward, +Z up) -> CV/NeuS (+Z forward, -Y up)
    CAM_FRAME_FIX = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
    Q_FIX_WXYZ = rotmat_to_quat_wxyz(CAM_FRAME_FIX)

    with open(geometry_json_path, "r") as f:
        geometry = json.load(f)

    cam_pos = geometry["camera"]["position"]
    cam_quat = geometry["camera"]["orientation"]
    body_pos = geometry["body"]["position"]
    body_quat = geometry["body"]["orientation"]

    n = len(cam_pos)

    if not (len(cam_quat) == len(body_pos) == len(body_quat) == n):
        raise ValueError("Le liste in geometry.json non hanno la stessa lunghezza.")

    labels = []

    for i in range(n):
        p_c = np.asarray(cam_pos[i], dtype=float)
        q_cw = quat_normalize(cam_quat[i])   # camera -> world
        p_t = np.asarray(body_pos[i], dtype=float)
        q_tw = quat_normalize(body_quat[i])  # target -> world

        # =============================================
        # ROTAZIONE: target -> camera (frame CV)
        # =============================================
        # 1. world -> camera (in frame Blender)
        q_wc = quat_conjugate(q_cw)

        # 2. target -> world -> camera = target -> camera (Blender)
        q_tc_wxyz = quat_multiply(q_wc, q_tw)
        q_tc_wxyz = quat_normalize(q_tc_wxyz)

        # 3. Fix frame camera: Blender -> CV
        #    R_tc_cv = CAM_FIX @ R_tc_blender
        #    Moltiplicazione a SINISTRA perché il fix agisce sul
        #    frame di destinazione (camera), non sul frame sorgente (target)
        q_tc_wxyz = quat_multiply(Q_FIX_WXYZ, q_tc_wxyz)
        q_tc_wxyz = quat_normalize(q_tc_wxyz)

        # =============================================
        # TRASLAZIONE: posizione target in frame camera (CV)
        # =============================================
        # 1. Vettore target - camera in frame world
        dt_w = (p_t - p_c).tolist()

        # 2. Ruota nel frame camera Blender
        r_rel = quat_rotate_vector(q_wc, dt_w)

        # 3. Fix frame camera: Blender -> CV
        r_rel = (CAM_FRAME_FIX @ np.asarray(r_rel, dtype=float)).tolist()

        labels.append({
            "filename": f"img{i+1:06d}",
            "q_vbs2tango_true": reorder_quaternion(q_tc_wxyz, output_order),
            "r_Vo2To_vbs_true": r_rel
        })

    os.makedirs(os.path.dirname(output_labels_path) or ".", exist_ok=True)

    with open(output_labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Creato: {output_labels_path}")
    print(f"Numero frame: {n}")
    print(f"Quaternion output order: {output_order}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CORTO → SPE3R dataset pipeline")

    # Path principali
    parser.add_argument("--images_in", type=str, required=True)
    parser.add_argument("--masks_in", type=str, required=True)
    parser.add_argument("--geometry", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Camera params (puoi modificarli da Colab)
    parser.add_argument("--Nu", type=int, default=1024)
    parser.add_argument("--Nv", type=int, default=1024)
    parser.add_argument("--fx", type=float, default=2903.6963)
    parser.add_argument("--fy", type=float, default=2903.6963)
    parser.add_argument("--ccx", type=float, default=512)
    parser.add_argument("--ccy", type=float, default=512)

    args = parser.parse_args()

    # === Output structure ===
    images_out = os.path.join(args.output_dir, "images")
    masks_out  = os.path.join(args.output_dir, "masks")

    # 1. immagini
    copy_and_rename_images(args.images_in, images_out)

    # 2. masks
    copy_and_rename_masks(args.masks_in, masks_out)

    # 3. camera.json
    create_camera_json(
        output_folder=args.output_dir,
        Nu=args.Nu,
        Nv=args.Nv,
        ppx=1,
        ppy=1,
        fx=args.fx,
        fy=args.fy,
        ccx=args.ccx,
        ccy=args.ccy,
        camera_matrix=[
            [args.fx, 0, args.ccx],
            [0, args.fy, args.ccy],
            [0, 0, 1]
        ],
        dist_coeffs=[0, 0, 0, 0, 0]
    )

    # 4. labels.json
    generate_labels_from_geometry(
        geometry_json_path=args.geometry,
        output_labels_path=os.path.join(args.output_dir, "labels.json"),
        output_order="wxyz"
    )

    print("\n✔ Pipeline completata.")