#!/usr/bin/env python3
"""
colmap2npz_clean.py  (pipeline "pulita")

ASSUNZIONE: hai GIÀ il dataset 500 black pronto, con:
  WORK_DIR/
    images/
    masks/

Questo script:
1) Esegue COLMAP in WORK_DIR (senza ricopiare nulla)
2) Crea OUT_DIR con:
   - sparse/0, sparse_txt, database.db, ecc. (dal WORK_DIR)
   - images/ + masks/ SOLO delle immagini registrate
3) FIX CRITICO: riordina+rinomina OUT_DIR/images e OUT_DIR/masks in ordine images.bin -> 000.png..
   + Scrive OUT_DIR/index_map.json e OUT_DIR/index_map.txt con la mappa (new -> orig)
4) Lancia gli script ORIGINALI NeuS preprocess (imgs2poses.py + gen_cameras.py) su OUT_DIR
   - Se serve, crea sparse_points_interest.ply COPIANDO sparse_points.ply (no symlink su Drive)
   -> genera OUT_DIR/preprocessed/cameras_sphere.npz coerente con preprocessed/image/000.png..

Uso (Colab):
!python /content/NeuS_thesis/custom_codes/colmap_preprocessing/colmap2npz_clean.py \
  --work_dir /content/drive/MyDrive/Tesi/neus/pipeline_test/data/colmap_subset/full_spe3r_500_black \
  --out_dir  /content/drive/MyDrive/Tesi/neus/pipeline_test/data/colmap_subset/labeled_data_500_black \
  --neus_repo /content/NeuS_thesis \
  --match_type exhaustive_matcher
"""

from __future__ import annotations

import os
import shutil
import struct
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Optional

IMG_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# ----------------------------
# system helpers
# ----------------------------
def run(cmd: List[str], env: Optional[dict] = None, cwd: Optional[str] = None):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def ensure_colmap_installed():
    if shutil.which("colmap") is not None:
        print("[OK] COLMAP found:", shutil.which("colmap"))
        return
    print("[INFO] COLMAP not found. Installing via apt-get ...")
    run(["apt-get", "update", "-y"])
    run(["apt-get", "install", "-y", "colmap"])
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP install failed (still not in PATH).")
    print("[OK] COLMAP installed:", shutil.which("colmap"))


def headless_env() -> dict:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DISPLAY"] = ""
    env["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["GALLIUM_DRIVER"] = "llvmpipe"
    os.makedirs(env["XDG_RUNTIME_DIR"], exist_ok=True)
    return env


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in IMG_EXTS])


# ----------------------------
# COLMAP parsing helpers
# ----------------------------
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
    """Return image names in the EXACT order stored in COLMAP images.bin."""
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
    """
    images.txt: header + 2 lines per image.
    The pose line ends with filename -> we take those.
    """
    lines = images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append(ln.split()[-1])

    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq


# ----------------------------
# CRITICAL FIX: reorder/rename by images.bin order
# + write mapping files
# ----------------------------
def reorder_and_rename_by_images_bin(
    src_images_dir: Path,
    src_masks_dir: Path,
    dst_images_dir: Path,
    dst_masks_dir: Path,
    images_bin: Path,
):
    order = read_images_bin_names(images_bin)
    print("[ORDER] images.bin count:", len(order))
    print("[ORDER] first 10:", order[:10])

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_masks_dir.mkdir(parents=True, exist_ok=True)

    # clear dst
    for p in dst_images_dir.glob("*"):
        if p.is_file():
            p.unlink()
    for p in dst_masks_dir.glob("*"):
        if p.is_file():
            p.unlink()

    mapping = []

    missing_masks = 0
    for i, name in enumerate(order):
        src = src_images_dir / name
        if not src.exists():
            raise FileNotFoundError(f"images.bin references {name} but missing in {src_images_dir}")

        new_name = f"{i:03d}.png"
        dst_img = dst_images_dir / new_name
        shutil.copy2(src, dst_img)

        row = {"new_index": i, "new_name": new_name, "orig_name": name}
        try:
            row["orig_index"] = int(Path(name).stem)
        except Exception:
            pass
        mapping.append(row)

        m = src_masks_dir / name
        if m.exists():
            shutil.copy2(m, dst_masks_dir / new_name)
        else:
            missing_masks += 1

    print(f"[RENAME] wrote {len(order)} images as 000.. | missing masks for {missing_masks} images")

    out_dir = dst_images_dir.parent
    (out_dir / "index_map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (out_dir / "index_map.txt").write_text(
        "\n".join([f'{m["new_name"]} {m["orig_name"]}' for m in mapping]) + "\n",
        encoding="utf-8",
    )
    print("[MAP] wrote:", out_dir / "index_map.json")
    print("[MAP] wrote:", out_dir / "index_map.txt")


# ----------------------------
# main pipeline
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=Path, required=True,
                    help="Directory that already contains images/ and masks/ (500 views)")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="NeuS-ready output directory (will be overwritten)")
    ap.add_argument("--neus_repo", type=Path, default=Path("/content/NeuS_thesis"))
    ap.add_argument("--match_type", type=str, default="exhaustive_matcher")

    # COLMAP knobs
    ap.add_argument("--single_camera", type=int, default=1)
    ap.add_argument("--camera_model", type=str, default="SIMPLE_RADIAL")
    ap.add_argument("--use_gpu", type=int, default=0)
    ap.add_argument("--max_num_features", type=int, default=20000)
    ap.add_argument("--max_image_size", type=int, default=1024)

    args = ap.parse_args()

    ensure_colmap_installed()
    env = headless_env()

    work_images = args.work_dir / "images"
    work_masks = args.work_dir / "masks"
    if not work_images.is_dir():
        raise FileNotFoundError(f"Missing {work_images} (expected WORK_DIR/images)")
    if not work_masks.is_dir():
        raise FileNotFoundError(f"Missing {work_masks} (expected WORK_DIR/masks)")

    n_imgs = len(list_images(work_images))
    n_msk = len(list_images(work_masks))
    print("WORK_DIR:", args.work_dir)
    print("Images :", n_imgs)
    print("Masks  :", n_msk)
    if n_imgs == 0:
        raise RuntimeError("WORK_DIR/images is empty.")

    # ------------------------------------------------------------------
    # 1) Run COLMAP in WORK_DIR (reset database + sparse)
    # ------------------------------------------------------------------
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

    run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(work_images),
        "--ImageReader.single_camera", str(args.single_camera),
        "--ImageReader.camera_model", args.camera_model,
        "--ImageReader.mask_path", str(work_masks),
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
        "--image_path", str(work_images),
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
    print("[COLMAP] registered images:", len(reg_names))
    if len(reg_names) == 0:
        raise RuntimeError("No registered images found.")

    # ------------------------------------------------------------------
    # 2) Build OUT_DIR: copy COLMAP outputs + subset registered images/masks
    # ------------------------------------------------------------------
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for item in args.work_dir.iterdir():
        if item.name in ("images", "masks"):
            continue
        dst = args.out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    out_images = args.out_dir / "images"
    out_masks = args.out_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    missing_imgs = 0
    missing_masks = 0
    for name in reg_names:
        src_i = work_images / name
        src_m = work_masks / name
        if not src_i.exists():
            missing_imgs += 1
            continue
        shutil.copy2(src_i, out_images / name)
        if src_m.exists():
            shutil.copy2(src_m, out_masks / name)
        else:
            missing_masks += 1

    print(f"[OUT COPY] copied registered images={len(reg_names)-missing_imgs} | missing_imgs={missing_imgs} | missing_masks={missing_masks}")

    # ------------------------------------------------------------------
    # 3) reorder+rename by OUT_DIR sparse/0/images.bin + write index_map
    # ------------------------------------------------------------------
    images_bin = args.out_dir / "sparse" / "0" / "images.bin"
    if not images_bin.exists():
        raise RuntimeError(f"Missing {images_bin} in OUT_DIR")

    tmp_images = args.out_dir / "_orig_images"
    tmp_masks = args.out_dir / "_orig_masks"
    if tmp_images.exists():
        shutil.rmtree(tmp_images)
    if tmp_masks.exists():
        shutil.rmtree(tmp_masks)

    shutil.move(str(out_images), str(tmp_images))
    shutil.move(str(out_masks), str(tmp_masks))

    reorder_and_rename_by_images_bin(
        src_images_dir=tmp_images,
        src_masks_dir=tmp_masks,
        dst_images_dir=args.out_dir / "images",
        dst_masks_dir=args.out_dir / "masks",
        images_bin=images_bin,
    )

    shutil.rmtree(tmp_images)
    shutil.rmtree(tmp_masks)

    # ------------------------------------------------------------------
    # 4) Run NeuS preprocess scripts (UNCHANGED) + your "second cell" logic
    # ------------------------------------------------------------------
    colmap_prep = args.neus_repo / "preprocess_custom_data" / "colmap_preprocess"
    imgs2poses = colmap_prep / "imgs2poses.py"
    gen_cameras = colmap_prep / "gen_cameras.py"

    if not imgs2poses.exists():
        raise FileNotFoundError(f"Missing {imgs2poses}")
    if not gen_cameras.exists():
        raise FileNotFoundError(f"Missing {gen_cameras}")

    run(["python", str(imgs2poses), str(args.out_dir), "--match_type", args.match_type],
        env=env, cwd=str(colmap_prep))

    # === integrate your second cell ===
    ply_raw = args.out_dir / "sparse_points.ply"
    ply_int = args.out_dir / "sparse_points_interest.ply"

    print("\n[POST] OUT_DIR  =", args.out_dir)
    print("[POST] PLY_RAW  =", ply_raw)
    print("[POST] PLY_INT  =", ply_int)

    if not ply_raw.exists():
        raise RuntimeError(f"❌ Missing {ply_raw}. imgs2poses did not write sparse_points.ply")

    # NO symlink on Drive: copy
    shutil.copy2(ply_raw, ply_int)
    print(f"✔ Created: {ply_int} (copied from sparse_points.ply)")

    # Now rerun gen_cameras
    run(["python", str(gen_cameras), str(args.out_dir)], env=env, cwd=str(colmap_prep))

    # ------------------------------------------------------------------
    # 5) Final checks
    # ------------------------------------------------------------------
    import numpy as np

    npz = args.out_dir / "preprocessed" / "cameras_sphere.npz"
    if not npz.exists():
        raise RuntimeError("cameras_sphere.npz not produced")

    prep_img_dir = None
    if (args.out_dir / "preprocessed" / "image").exists():
        prep_img_dir = args.out_dir / "preprocessed" / "image"
    elif (args.out_dir / "preprocessed" / "images").exists():
        prep_img_dir = args.out_dir / "preprocessed" / "images"
    else:
        raise RuntimeError("preprocessed/image(s) not found")

    d = np.load(npz)
    wm = sorted([k for k in d.keys() if k.startswith("world_mat_")])
    imgs = sorted(prep_img_dir.glob("*.png"))

    print("\n✅ DONE")
    print("OUT_DIR:", args.out_dir)
    print("preprocessed images:", len(imgs))
    print("npz world_mat_*    :", len(wm))
    print("NPZ path:", npz)
    print("Mapping files:", args.out_dir / "index_map.json", "and", args.out_dir / "index_map.txt")
    if imgs:
        print("First preprocessed image:", imgs[0].name)
    if wm:
        print("First world_mat key:", wm[0])

    if len(imgs) != len(wm):
        print("⚠️ WARNING: preprocessed images count != world_mat_* count (should match).")


if __name__ == "__main__":
    main()
