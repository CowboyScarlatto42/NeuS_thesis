#!/usr/bin/env python3
"""
colmap_to_neus_npz.py

Single end-to-end wrapper:
SPE3R images+masks  -> COLMAP (CPU, masked) -> copy registered set ->
REORDER/RENAME images+maps to match images.bin order -> run NeuS preprocess
(imgs2poses.py + gen_cameras.py) WITHOUT modifying any NeuS/Colmap scripts.

Result:
OUT_DIR/preprocessed/cameras_sphere.npz   (correctly aligned to preprocessed/image order)
and all the usual NeuS preprocess artifacts.

Example (Colab):
!python /content/NeuS_thesis/custom_codes/dataset_code/colmap_to_neus_npz.py \
  --hst_root /content/drive/MyDrive/Tesi/neus/pipeline_test/data/hst_neus \
  --work_dir /content/drive/MyDrive/Tesi/neus/pipeline_test/data/colmap_subset/full_spe3r_500_black \
  --out_dir  /content/drive/MyDrive/Tesi/neus/pipeline_test/data/colmap_subset/labeled_data_500_black \
  --n_images 500 \
  --colmap_model SIMPLE_RADIAL \
  --single_camera 1
"""

import os
import re
import sys
import json
import shutil
import struct
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple


# ----------------------------
# helpers
# ----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def run(cmd: List[str], env=None, cwd=None):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def detect_dir(root: Path, candidates: List[str]) -> Path:
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    raise FileNotFoundError(f"None of these folders exist under {root}: {candidates}")


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in IMG_EXTS])


def copy_first_n_images_and_masks(img_root: Path, mask_root: Path, out_images: Path, out_masks: Path, n: int):
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    imgs = list_images(img_root)
    if len(imgs) < n:
        raise RuntimeError(f"Found only {len(imgs)} images in {img_root}, need {n}")

    imgs = imgs[:n]
    missing_masks = 0

    for p in imgs:
        name = p.name
        shutil.copy2(p, out_images / name)
        m = mask_root / name
        if m.exists():
            shutil.copy2(m, out_masks / name)
        else:
            # try same stem with common ext
            stem = p.stem
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                mm = mask_root / f"{stem}{ext}"
                if mm.exists():
                    shutil.copy2(mm, out_masks / mm.name)
                    found = True
                    break
            if not found:
                missing_masks += 1

    print(f"[COPY] images={len(imgs)} masks_missing={missing_masks}")


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


def read_c_string(fid):
    chars = []
    while True:
        c = fid.read(1)
        if c in (b"\x00", b""):
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8", errors="replace")


def read_images_bin_names(images_bin: Path) -> List[str]:
    names = []
    with open(images_bin, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            _image_id = read_next_bytes(fid, 4, "I")[0]
            fid.read(8 * 4)  # qvec
            fid.read(8 * 3)  # tvec
            fid.read(4)      # camera_id
            name = read_c_string(fid)
            n2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(n2d * (8 + 8 + 8), 1)
            names.append(name)
    return names


def parse_registered_names_from_images_txt(images_txt: Path) -> List[str]:
    # images.txt: header + 2 lines per image; image line ends with filename
    lines = images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        # first of 2 lines contains name as last token and has >= 9 tokens typically
        if ln.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append(ln.split()[-1])
    # unique, keep first occurrence order
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


def reorder_and_rename_by_images_bin(
    src_images_dir: Path,
    src_masks_dir: Path,
    dst_images_dir: Path,
    dst_masks_dir: Path,
    images_bin: Path,
):
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_masks_dir.mkdir(parents=True, exist_ok=True)

    order = read_images_bin_names(images_bin)
    print("[ORDER] images.bin count:", len(order))
    print("[ORDER] first 10:", order[:10])

    # clean destination
    for p in dst_images_dir.glob("*"):
        p.unlink()
    for p in dst_masks_dir.glob("*"):
        p.unlink()

    # copy+rename in that order: 000.png, 001.png, ...
    for i, name in enumerate(order):
        src = src_images_dir / name
        if not src.exists():
            raise FileNotFoundError(f"images.bin references {name} but missing in {src_images_dir}")
        dst = dst_images_dir / f"{i:03d}.png"
        shutil.copy2(src, dst)

        m = src_masks_dir / name
        if m.exists():
            shutil.copy2(m, dst_masks_dir / f"{i:03d}.png")
        else:
            # ok: some pipelines allow missing masks; but NeuS usually expects them
            # We keep empty if missing; user can decide.
            pass

    print(f"[RENAME] wrote {len(order)} images into {dst_images_dir} as 000..")


# ----------------------------
# main pipeline
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hst_root", type=Path, required=True, help="SPE3R root containing image(s) + mask(s)")
    ap.add_argument("--work_dir", type=Path, required=True, help="COLMAP working directory (will be overwritten)")
    ap.add_argument("--out_dir", type=Path, required=True, help="NeuS-ready filtered+ordered dataset output (will be overwritten)")
    ap.add_argument("--n_images", type=int, default=500, help="How many images to take from SPE3R (sorted)")
    ap.add_argument("--single_camera", type=int, default=1)
    ap.add_argument("--colmap_model", type=str, default="SIMPLE_RADIAL")
    ap.add_argument("--max_num_features", type=int, default=20000)
    ap.add_argument("--max_image_size", type=int, default=1024)
    ap.add_argument("--use_gpu", type=int, default=0)
    ap.add_argument("--neus_repo", type=Path, default=Path("/content/NeuS_thesis"))
    ap.add_argument("--match_type", type=str, default="exhaustive_matcher")
    args = ap.parse_args()

    # detect SPE3R folders
    img_root = detect_dir(args.hst_root, ["images", "image"])
    mask_root = detect_dir(args.hst_root, ["masks", "mask"])

    # reset work_dir
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    (args.work_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.work_dir / "masks").mkdir(parents=True, exist_ok=True)

    print("HST img_root :", img_root)
    print("HST mask_root:", mask_root)
    print("WORK_DIR     :", args.work_dir)
    print("OUT_DIR      :", args.out_dir)

    # 1) copy first N images+masks into work_dir/images + work_dir/masks
    copy_first_n_images_and_masks(
        img_root=img_root,
        mask_root=mask_root,
        out_images=args.work_dir / "images",
        out_masks=args.work_dir / "masks",
        n=args.n_images,
    )

    # 2) run COLMAP
    db = args.work_dir / "database.db"
    sparse = args.work_dir / "sparse"
    sparse_txt = args.work_dir / "sparse_txt"

    if db.exists():
        db.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    if sparse_txt.exists():
        shutil.rmtree(sparse_txt)
    sparse.mkdir(parents=True, exist_ok=True)
    sparse_txt.mkdir(parents=True, exist_ok=True)

    # headless env (Colab)
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DISPLAY"] = ""
    env["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["GALLIUM_DRIVER"] = "llvmpipe"
    os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)

    run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(args.work_dir / "images"),
        "--ImageReader.single_camera", str(args.single_camera),
        "--ImageReader.camera_model", args.colmap_model,
        "--ImageReader.mask_path", str(args.work_dir / "masks"),
        "--SiftExtraction.use_gpu", str(args.use_gpu),
        "--SiftExtraction.num_threads", "2",
        "--SiftExtraction.max_num_features", str(args.max_num_features),
        "--SiftExtraction.peak_threshold", "0.004",
        "--SiftExtraction.edge_threshold", "20",
        "--SiftExtraction.max_image_size", str(args.max_image_size),
    ], env=env)

    run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", str(args.use_gpu),
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_num_matches", "50000",
    ], env=env)

    run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(args.work_dir / "images"),
        "--output_path", str(sparse),
        "--Mapper.num_threads", "16",
        "--Mapper.multiple_models", "0",
        "--Mapper.extract_colors", "0",
        "--Mapper.init_min_num_inliers", "30",
        "--Mapper.abs_pose_min_num_inliers", "15",
        "--Mapper.abs_pose_min_inlier_ratio", "0.05",
        "--Mapper.min_num_matches", "15",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
    ], env=env)

    # sanity
    model0 = sparse / "0"
    for f in ["cameras.bin", "images.bin", "points3D.bin"]:
        if not (model0 / f).exists():
            raise RuntimeError(f"Missing {model0/f} (COLMAP failed or registered 0 images)")

    run([
        "colmap", "model_converter",
        "--input_path", str(model0),
        "--output_path", str(sparse_txt),
        "--output_type", "TXT"
    ], env=env)

    images_txt = sparse_txt / "images.txt"
    if not images_txt.exists():
        raise RuntimeError(f"Missing {images_txt}")

    reg_names = parse_registered_names_from_images_txt(images_txt)
    print("[COLMAP] registered count (from images.txt):", len(reg_names))
    if len(reg_names) == 0:
        raise RuntimeError("No registered images found")

    # 3) build OUT_DIR: copy full COLMAP outputs + ONLY registered images/masks
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # copy everything from work_dir except images/masks folders
    # (so OUT_DIR has sparse/0 + database.db + sparse_txt etc.)
    for item in args.work_dir.iterdir():
        if item.name in ("images", "masks"):
            continue
        dst = args.out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    # copy registered originals (names) into OUT_DIR/images and OUT_DIR/masks (still named original)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "masks").mkdir(parents=True, exist_ok=True)

    missing_imgs = 0
    for name in reg_names:
        s = args.work_dir / "images" / name
        if not s.exists():
            missing_imgs += 1
            continue
        shutil.copy2(s, args.out_dir / "images" / name)
        m = args.work_dir / "masks" / name
        if m.exists():
            shutil.copy2(m, args.out_dir / "masks" / name)
    if missing_imgs:
        print("[WARN] missing registered images while copying:", missing_imgs)

    # 4) **CRITICAL FIX**: reorder+rename OUT_DIR/images and OUT_DIR/masks to match images.bin order
    # This guarantees imgs2poses/gen_cameras produce aligned npz with NeuS order (000..)
    images_bin = args.out_dir / "sparse" / "0" / "images.bin"

    tmp_orig_images = args.out_dir / "_orig_images"
    tmp_orig_masks = args.out_dir / "_orig_masks"
    if tmp_orig_images.exists():
        shutil.rmtree(tmp_orig_images)
    if tmp_orig_masks.exists():
        shutil.rmtree(tmp_orig_masks)
    shutil.move(str(args.out_dir / "images"), str(tmp_orig_images))
    shutil.move(str(args.out_dir / "masks"), str(tmp_orig_masks))

    reorder_and_rename_by_images_bin(
        src_images_dir=tmp_orig_images,
        src_masks_dir=tmp_orig_masks,
        dst_images_dir=args.out_dir / "images",
        dst_masks_dir=args.out_dir / "masks",
        images_bin=images_bin,
    )

    # cleanup temp
    shutil.rmtree(tmp_orig_images)
    shutil.rmtree(tmp_orig_masks)

    # 5) run NeuS preprocess scripts unchanged
    colmap_prep_dir = args.neus_repo / "preprocess_custom_data" / "colmap_preprocess"
    imgs2poses = colmap_prep_dir / "imgs2poses.py"
    gen_cameras = colmap_prep_dir / "gen_cameras.py"

    if not imgs2poses.exists():
        raise FileNotFoundError(f"Missing {imgs2poses}")
    if not gen_cameras.exists():
        raise FileNotFoundError(f"Missing {gen_cameras}")

    run(["python", str(imgs2poses), str(args.out_dir), "--match_type", args.match_type], env=env, cwd=str(colmap_prep_dir))

    # gen_cameras expects sparse_points_interest.ply sometimes; create symlink if needed
    ply_raw = args.out_dir / "sparse_points.ply"
    ply_int = args.out_dir / "sparse_points_interest.ply"
    if ply_raw.exists() and not ply_int.exists():
        try:
            ply_int.symlink_to(ply_raw.name)
        except Exception:
            shutil.copy2(ply_raw, ply_int)

    run(["python", str(gen_cameras), str(args.out_dir)], env=env)

    # final checks
    npz = args.out_dir / "preprocessed" / "cameras_sphere.npz"
    if not npz.exists():
        raise RuntimeError("cameras_sphere.npz not produced")
    prep_img_dir = None
    if (args.out_dir / "preprocessed" / "image").exists():
        prep_img_dir = args.out_dir / "preprocessed" / "image"
    elif (args.out_dir / "preprocessed" / "images").exists():
        prep_img_dir = args.out_dir / "preprocessed" / "images"

    if prep_img_dir is None:
        raise RuntimeError("preprocessed/image(s) not found")

    # print quick consistency
    import numpy as np
    d = np.load(npz)
    wm = sorted([k for k in d.keys() if k.startswith("world_mat_")])
    imgs = sorted([p for p in prep_img_dir.glob("*.png")])
    print("\n✅ DONE")
    print("preprocessed images:", len(imgs), "| npz world_mats:", len(wm))
    print("NPZ:", npz)
    print("Example:", wm[0], d[wm[0]].shape)


if __name__ == "__main__":
    main()
