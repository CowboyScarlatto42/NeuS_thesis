#!/usr/bin/env python3
"""
Align two independent COLMAP reconstructions to the common CORTO/Tango frame.

STEP A ONLY: this script performs alignment and diagnostics. It does NOT build a
NeuS dataset and does NOT overwrite COLMAP outputs.

Inputs per orbit:
  - COLMAP sparse model directory or colmap_output root
  - filtered SPE3R-style labels.json
  - filtered geometry.json (recommended; used to validate label conventions)

Outputs per orbit:
  - similarity_all_fit.npz
  - similarity_ransac_fit.npz
  - aligned_poses_all_fit.json
  - aligned_poses_ransac_fit.json
  - alignment_report.csv
  - summary.json
  - aligned_sparse_points_all_fit.ply (when points3D.txt/bin is available)
  - aligned_sparse_points_ransac_fit.ply (when points3D.txt/bin is available)

Combined outputs:
  - run_summary.json
  - trajectories_all_fit.png
  - position_errors_all_fit.png
  - view_direction_errors_all_fit.png

The primary diagnostic transformation is the least-squares Umeyama fit using all
registered images. A RANSAC fit is also saved to identify isolated outliers. The
RANSAC fit is secondary: it should not be used to hide systematic SfM drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required. In Colab: pip install matplotlib") from exc


EPS = 1e-12


@dataclass
class ColmapPose:
    image_id: int
    image_name: str
    stem: str
    camera_id: int
    qvec_wxyz: np.ndarray
    tvec: np.ndarray
    R_wc: np.ndarray
    C_w: np.ndarray


@dataclass
class LabelPose:
    filename: str
    stem: str
    q_raw: np.ndarray
    r_camera_to_target_vbs: np.ndarray


@dataclass
class LabelConvention:
    quat_order: str       # wxyz or xyzw
    rotation_mode: str    # direct or inverse
    center_sign: int      # -1 or +1
    validation_rmse: Optional[float] = None

    @property
    def description(self) -> str:
        sign = "-" if self.center_sign < 0 else "+"
        rot = "R(q)" if self.rotation_mode == "direct" else "R(q)^T"
        return f"quat={self.quat_order}, center={sign}{rot}@r"


@dataclass
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply_points(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        return self.scale * (xyz @ self.rotation.T) + self.translation


@dataclass
class OrbitResult:
    tag: str
    output_dir: Path
    names: List[str]
    gt_centers: np.ndarray
    aligned_centers_all: np.ndarray
    aligned_centers_ransac: np.ndarray
    position_errors_all: np.ndarray
    position_errors_ransac: np.ndarray
    view_errors_all_deg: np.ndarray
    view_errors_ransac_deg: np.ndarray
    sparse_points_all: Optional[np.ndarray]
    sparse_points_ransac: Optional[np.ndarray]
    summary: dict


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < EPS:
        raise ValueError("Cannot normalize a zero-length vector")
    return v / norm


def quat_to_rotmat(q: Sequence[float], order: str = "wxyz") -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    if order == "xyzw":
        x, y, z, w = q
    elif order == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError(f"Unsupported quaternion order: {order}")

    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < EPS:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a scalar-first quaternion."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0:  # deterministic sign
        q *= -1
    return q


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def find_sparse_model_dir(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "sparse" / "0", root / "sparse"]
    for candidate in candidates:
        if (candidate / "images.txt").exists() or (candidate / "images.bin").exists():
            return candidate

    # Fallback: find sparse/0-like directories recursively, prefer shortest path.
    recursive = sorted(
        {p.parent for p in root.rglob("images.txt")} | {p.parent for p in root.rglob("images.bin")},
        key=lambda p: (len(p.parts), str(p)),
    )
    if not recursive:
        raise FileNotFoundError(
            f"No COLMAP sparse model found below {root}. Expected images.txt or images.bin."
        )
    return recursive[0]


def ensure_text_model(model_dir: Path, converted_dir: Path, colmap_exe: str) -> Path:
    """Return a model directory containing images.txt, converting BIN -> TXT if needed."""
    if (model_dir / "images.txt").exists():
        return model_dir
    if not (model_dir / "images.bin").exists():
        raise FileNotFoundError(f"Neither images.txt nor images.bin found in {model_dir}")

    converted_dir.mkdir(parents=True, exist_ok=True)
    command = [
        colmap_exe,
        "model_converter",
        "--input_path",
        str(model_dir),
        "--output_path",
        str(converted_dir),
        "--output_type",
        "TXT",
    ]
    print("[COLMAP] Converting sparse model BIN -> TXT:")
    print("         " + " ".join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"COLMAP executable not found: {colmap_exe}. "
            "Pass --colmap-exe or convert the model to TXT manually."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"COLMAP model_converter failed with exit code {exc.returncode}") from exc
    return converted_dir


def read_colmap_images_txt(path: Path) -> Dict[str, ColmapPose]:
    """Read registered image poses from COLMAP images.txt.

    Header lines contain exactly 10 tokens. POINTS2D lines are ignored.
    """
    poses: Dict[str, ColmapPose] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 10:
                continue  # POINTS2D line
            try:
                image_id = int(parts[0])
                q = np.asarray([float(v) for v in parts[1:5]], dtype=float)
                t = np.asarray([float(v) for v in parts[5:8]], dtype=float)
                camera_id = int(parts[8])
            except ValueError:
                continue
            image_name = parts[9]
            stem = Path(image_name).stem
            R_wc = quat_to_rotmat(q, order="wxyz")
            C_w = -R_wc.T @ t
            if stem in poses:
                raise ValueError(f"Duplicate COLMAP image stem in {path}: {stem}")
            poses[stem] = ColmapPose(
                image_id=image_id,
                image_name=image_name,
                stem=stem,
                camera_id=camera_id,
                qvec_wxyz=q,
                tvec=t,
                R_wc=R_wc,
                C_w=C_w,
            )
    if not poses:
        raise ValueError(f"No registered image poses parsed from {path}")
    return poses


def read_labels_json(path: Path) -> Tuple[Dict[str, LabelPose], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")

    labels: Dict[str, LabelPose] = {}
    ordered_stems: List[str] = []
    for index, record in enumerate(records):
        try:
            filename = str(record["filename"])
            stem = Path(filename).stem
            q = np.asarray(record["q_vbs2tango_true"], dtype=float).reshape(4)
            r = np.asarray(record["r_Vo2To_vbs_true"], dtype=float).reshape(3)
        except Exception as exc:
            raise ValueError(f"Invalid label record #{index} in {path}: {record}") from exc
        if stem in labels:
            raise ValueError(f"Duplicate label filename stem in {path}: {stem}")
        labels[stem] = LabelPose(
            filename=filename,
            stem=stem,
            q_raw=q,
            r_camera_to_target_vbs=r,
        )
        ordered_stems.append(stem)
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels, ordered_stems


def read_filtered_geometry_camera_positions(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        geometry = json.load(f)
    try:
        positions = np.asarray(geometry["camera"]["position"], dtype=float)
    except Exception as exc:
        raise ValueError(f"Expected geometry['camera']['position'] in {path}") from exc
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Invalid camera positions shape in {path}: {positions.shape}")
    return positions


def label_rotation(label: LabelPose, convention: LabelConvention) -> np.ndarray:
    R = quat_to_rotmat(label.q_raw, order=convention.quat_order)
    return R if convention.rotation_mode == "direct" else R.T


def label_camera_center(label: LabelPose, convention: LabelConvention) -> np.ndarray:
    R = label_rotation(label, convention)
    return convention.center_sign * (R @ label.r_camera_to_target_vbs)


def available_conventions() -> List[LabelConvention]:
    return [
        LabelConvention(order, mode, sign)
        for order in ("wxyz", "xyzw")
        for mode in ("direct", "inverse")
        for sign in (-1, +1)
    ]


def choose_label_convention(
    labels: Dict[str, LabelPose],
    ordered_stems: Sequence[str],
    geometry_positions: Optional[np.ndarray],
    quat_order: str,
    rotation_mode: str,
    center_sign: str,
    validation_tolerance: float,
    allow_geometry_mismatch: bool,
) -> Tuple[LabelConvention, List[dict]]:
    manual = quat_order != "auto" and rotation_mode != "auto" and center_sign != "auto"
    if manual:
        convention = LabelConvention(
            quat_order=quat_order,
            rotation_mode=rotation_mode,
            center_sign=-1 if center_sign == "minus" else +1,
        )
        table: List[dict] = []
        if geometry_positions is not None:
            if len(ordered_stems) != len(geometry_positions):
                raise ValueError(
                    "Filtered labels and geometry.json do not contain the same number of frames: "
                    f"labels={len(ordered_stems)}, geometry={len(geometry_positions)}"
                )
            pred = np.vstack([label_camera_center(labels[s], convention) for s in ordered_stems])
            rmse = float(np.sqrt(np.mean(np.sum((pred - geometry_positions) ** 2, axis=1))))
            convention.validation_rmse = rmse
        return convention, table

    if geometry_positions is None:
        # Documented SPE3R-like default. The script prints a warning because this
        # should be verified at least once using filtered geometry.json.
        convention = LabelConvention("wxyz", "direct", -1, validation_rmse=None)
        print(
            "[WARNING] geometry.json not provided: using default label convention "
            f"{convention.description}. Validate it on the first run."
        )
        return convention, []

    if len(ordered_stems) != len(geometry_positions):
        raise ValueError(
            "Automatic convention detection requires filtered labels.json and filtered geometry.json "
            "with the same number of frames and the same ordering. "
            f"Got labels={len(ordered_stems)}, geometry={len(geometry_positions)}."
        )

    candidates = available_conventions()
    table: List[dict] = []
    best: Optional[LabelConvention] = None
    for convention in candidates:
        pred = np.vstack([label_camera_center(labels[s], convention) for s in ordered_stems])
        errors = np.linalg.norm(pred - geometry_positions, axis=1)
        rmse = float(np.sqrt(np.mean(errors**2)))
        convention.validation_rmse = rmse
        row = {
            "description": convention.description,
            "rmse": rmse,
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
        }
        table.append(row)
        if best is None or rmse < float(best.validation_rmse):
            best = convention

    assert best is not None
    table.sort(key=lambda row: row["rmse"])
    print("[LABELS] Candidate conventions compared against filtered geometry.json:")
    for row in table:
        print(
            f"         {row['description']:<38} "
            f"RMSE={row['rmse']:.6g}, max={row['max_error']:.6g}"
        )
    print(f"[LABELS] Selected: {best.description}")

    if float(best.validation_rmse) > validation_tolerance:
        message = (
            "Best label-to-geometry validation RMSE exceeds tolerance: "
            f"{best.validation_rmse:.6g} > {validation_tolerance:.6g}. "
            "Check that geometry.json is filtered and ordered exactly like labels.json."
        )
        if not allow_geometry_mismatch:
            raise ValueError(message + " Pass --allow-geometry-mismatch only after inspecting the data.")
        print("[WARNING] " + message)
    return best, table


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Similarity:
    """Least-squares similarity dst ~= scale * R @ src + t."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError("At least 3 point correspondences are required for Sim(3) alignment")

    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    covariance = (dst_c.T @ src_c) / n
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1
    R = U @ correction @ Vt
    var_src = float(np.mean(np.sum(src_c**2, axis=1)))
    if var_src < EPS:
        raise ValueError("Degenerate source trajectory: near-zero variance")
    scale = float(np.sum(singular_values * np.diag(correction)) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return Similarity(scale=scale, rotation=R, translation=t)


def residuals(similarity: Similarity, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(similarity.apply_points(src) - dst, axis=1)


def automatic_ransac_threshold(initial_residuals: np.ndarray) -> float:
    initial_residuals = np.asarray(initial_residuals, dtype=float)
    median = float(np.median(initial_residuals))
    mad = float(np.median(np.abs(initial_residuals - median)))
    robust_sigma = 1.4826 * mad
    # Keep a small absolute floor in CORTO units while avoiding a zero threshold.
    return max(0.05, median + 3.0 * robust_sigma)


def ransac_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> Tuple[Similarity, np.ndarray]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = len(src)
    if n < 3:
        raise ValueError("RANSAC requires at least 3 correspondences")
    rng = np.random.default_rng(seed)
    best_inliers: Optional[np.ndarray] = None
    best_score: Optional[Tuple[int, float]] = None

    for _ in range(iterations):
        ids = rng.choice(n, size=3, replace=False)
        try:
            model = umeyama_similarity(src[ids], dst[ids])
        except (ValueError, np.linalg.LinAlgError):
            continue
        err = residuals(model, src, dst)
        inliers = err <= threshold
        count = int(np.sum(inliers))
        if count < 3:
            continue
        score = (count, -float(np.median(err[inliers])))
        if best_score is None or score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None:
        print("[WARNING] RANSAC did not find a valid model; falling back to all-point fit")
        model = umeyama_similarity(src, dst)
        return model, np.ones(n, dtype=bool)

    # Refine and reclassify until stable.
    inliers = best_inliers
    for _ in range(10):
        model = umeyama_similarity(src[inliers], dst[inliers])
        new_inliers = residuals(model, src, dst) <= threshold
        if int(np.sum(new_inliers)) < 3:
            break
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    model = umeyama_similarity(src[inliers], dst[inliers])
    return model, inliers


def transform_colmap_pose(pose: ColmapPose, similarity: Similarity) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned (R_wc, t_wc, C_w) in the CORTO/Tango frame."""
    C_aligned = similarity.apply_points(pose.C_w.reshape(1, 3))[0]
    R_cw_src = pose.R_wc.T
    R_cw_aligned = similarity.rotation @ R_cw_src
    R_wc_aligned = R_cw_aligned.T
    t_aligned = -R_wc_aligned @ C_aligned
    return R_wc_aligned, t_aligned, C_aligned


def read_points3d_txt(path: Path) -> np.ndarray:
    points: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
    return np.asarray(points, dtype=float).reshape(-1, 3)


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.10g} {y:.10g} {z:.10g}\n")


def summarize_errors(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def save_similarity(path: Path, similarity: Similarity, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scale": np.asarray(similarity.scale),
        "rotation": similarity.rotation,
        "translation": similarity.translation,
    }
    if extra:
        for key, value in extra.items():
            payload[key] = np.asarray(value)
    np.savez(path, **payload)


def pose_to_record(
    stem: str,
    pose: ColmapPose,
    gt_center: np.ndarray,
    similarity: Similarity,
    is_ransac_inlier: bool,
) -> dict:
    R_wc, t_wc, center = transform_colmap_pose(pose, similarity)
    forward = R_wc.T @ np.array([0.0, 0.0, 1.0])
    ideal_forward = -gt_center
    return {
        "filename": pose.image_name,
        "stem": stem,
        "image_id": int(pose.image_id),
        "camera_id": int(pose.camera_id),
        "q_wc_wxyz_aligned": rotmat_to_quat_wxyz(R_wc).tolist(),
        "t_wc_aligned": t_wc.tolist(),
        "camera_center_aligned": center.tolist(),
        "camera_center_gt": gt_center.tolist(),
        "position_error": float(np.linalg.norm(center - gt_center)),
        "view_direction_error_deg": float(angle_deg(forward, ideal_forward)),
        "ransac_inlier": bool(is_ransac_inlier),
    }


def align_one_orbit(
    tag: str,
    colmap_root: Path,
    labels_path: Path,
    geometry_path: Optional[Path],
    output_root: Path,
    colmap_exe: str,
    quat_order: str,
    rotation_mode: str,
    center_sign: str,
    geometry_validation_tolerance: float,
    allow_geometry_mismatch: bool,
    ransac_threshold: Optional[float],
    ransac_iterations: int,
    seed: int,
) -> OrbitResult:
    print("\n" + "=" * 78)
    print(f"ALIGNMENT: {tag}")
    print("=" * 78)

    orbit_out = output_root / tag
    orbit_out.mkdir(parents=True, exist_ok=True)

    model_dir = find_sparse_model_dir(colmap_root)
    text_model_dir = ensure_text_model(model_dir, output_root / "_converted_models" / tag, colmap_exe)
    images_txt = text_model_dir / "images.txt"
    points_txt = text_model_dir / "points3D.txt"

    print(f"[INPUT] COLMAP model: {model_dir}")
    print(f"[INPUT] Text model:   {text_model_dir}")
    print(f"[INPUT] Labels:       {labels_path}")
    if geometry_path is not None:
        print(f"[INPUT] Geometry:     {geometry_path}")

    colmap_poses = read_colmap_images_txt(images_txt)
    labels, ordered_label_stems = read_labels_json(labels_path)
    geometry_positions = (
        read_filtered_geometry_camera_positions(geometry_path) if geometry_path is not None else None
    )

    convention, convention_table = choose_label_convention(
        labels=labels,
        ordered_stems=ordered_label_stems,
        geometry_positions=geometry_positions,
        quat_order=quat_order,
        rotation_mode=rotation_mode,
        center_sign=center_sign,
        validation_tolerance=geometry_validation_tolerance,
        allow_geometry_mismatch=allow_geometry_mismatch,
    )

    common_stems = sorted(set(colmap_poses) & set(labels))
    registered_without_label = sorted(set(colmap_poses) - set(labels))
    labels_not_registered = sorted(set(labels) - set(colmap_poses))
    if registered_without_label:
        raise ValueError(
            f"{tag}: registered COLMAP images without labels: {registered_without_label[:20]}"
        )
    if len(common_stems) < 3:
        raise ValueError(f"{tag}: only {len(common_stems)} COLMAP-label correspondences found")

    src_centers = np.vstack([colmap_poses[s].C_w for s in common_stems])
    gt_centers = np.vstack([label_camera_center(labels[s], convention) for s in common_stems])

    similarity_all = umeyama_similarity(src_centers, gt_centers)
    initial_errors = residuals(similarity_all, src_centers, gt_centers)
    threshold = (
        float(ransac_threshold)
        if ransac_threshold is not None
        else automatic_ransac_threshold(initial_errors)
    )
    similarity_ransac, ransac_inliers = ransac_similarity(
        src=src_centers,
        dst=gt_centers,
        threshold=threshold,
        iterations=ransac_iterations,
        seed=seed,
    )

    records_all: List[dict] = []
    records_ransac: List[dict] = []
    for stem, inlier in zip(common_stems, ransac_inliers):
        records_all.append(
            pose_to_record(stem, colmap_poses[stem], label_camera_center(labels[stem], convention), similarity_all, bool(inlier))
        )
        records_ransac.append(
            pose_to_record(stem, colmap_poses[stem], label_camera_center(labels[stem], convention), similarity_ransac, bool(inlier))
        )

    errors_all = np.asarray([row["position_error"] for row in records_all], dtype=float)
    errors_ransac = np.asarray([row["position_error"] for row in records_ransac], dtype=float)
    view_all = np.asarray([row["view_direction_error_deg"] for row in records_all], dtype=float)
    view_ransac = np.asarray([row["view_direction_error_deg"] for row in records_ransac], dtype=float)
    aligned_centers_all = np.asarray([row["camera_center_aligned"] for row in records_all], dtype=float)
    aligned_centers_ransac = np.asarray([row["camera_center_aligned"] for row in records_ransac], dtype=float)

    save_similarity(
        orbit_out / "similarity_all_fit.npz",
        similarity_all,
        extra={"registered_names": np.asarray(common_stems)},
    )
    save_similarity(
        orbit_out / "similarity_ransac_fit.npz",
        similarity_ransac,
        extra={"registered_names": np.asarray(common_stems), "ransac_inliers": ransac_inliers},
    )
    save_json(orbit_out / "aligned_poses_all_fit.json", records_all)
    save_json(orbit_out / "aligned_poses_ransac_fit.json", records_ransac)

    with (orbit_out / "alignment_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stem",
                "image_name",
                "ransac_inlier",
                "position_error_all_fit",
                "position_error_ransac_fit",
                "view_direction_error_deg_all_fit",
                "view_direction_error_deg_ransac_fit",
                "gt_cx",
                "gt_cy",
                "gt_cz",
                "aligned_all_cx",
                "aligned_all_cy",
                "aligned_all_cz",
            ]
        )
        for i, stem in enumerate(common_stems):
            writer.writerow(
                [
                    stem,
                    colmap_poses[stem].image_name,
                    bool(ransac_inliers[i]),
                    errors_all[i],
                    errors_ransac[i],
                    view_all[i],
                    view_ransac[i],
                    *gt_centers[i].tolist(),
                    *aligned_centers_all[i].tolist(),
                ]
            )

    sparse_points_all: Optional[np.ndarray] = None
    sparse_points_ransac: Optional[np.ndarray] = None
    if points_txt.exists():
        sparse_src = read_points3d_txt(points_txt)
        if len(sparse_src):
            sparse_points_all = similarity_all.apply_points(sparse_src)
            sparse_points_ransac = similarity_ransac.apply_points(sparse_src)
            write_ascii_ply(orbit_out / "aligned_sparse_points_all_fit.ply", sparse_points_all)
            write_ascii_ply(orbit_out / "aligned_sparse_points_ransac_fit.ply", sparse_points_ransac)
            print(f"[POINTS] Transformed sparse points: {len(sparse_src)}")

    summary = {
        "tag": tag,
        "colmap_model_dir": str(model_dir),
        "text_model_dir": str(text_model_dir),
        "labels_path": str(labels_path),
        "geometry_path": str(geometry_path) if geometry_path is not None else None,
        "registered_colmap_images": len(colmap_poses),
        "label_records": len(labels),
        "matched_registered_images": len(common_stems),
        "registered_without_label": registered_without_label,
        "labels_not_registered": labels_not_registered,
        "selected_label_convention": {
            "quat_order": convention.quat_order,
            "rotation_mode": convention.rotation_mode,
            "center_sign": convention.center_sign,
            "description": convention.description,
            "geometry_validation_rmse": convention.validation_rmse,
        },
        "convention_candidates": convention_table,
        "all_fit_similarity": {
            "scale": similarity_all.scale,
            "rotation": similarity_all.rotation.tolist(),
            "translation": similarity_all.translation.tolist(),
        },
        "ransac_fit_similarity": {
            "scale": similarity_ransac.scale,
            "rotation": similarity_ransac.rotation.tolist(),
            "translation": similarity_ransac.translation.tolist(),
            "threshold": threshold,
            "iterations": ransac_iterations,
            "inliers": int(np.sum(ransac_inliers)),
            "outliers": int(np.sum(~ransac_inliers)),
            "outlier_stems": [s for s, ok in zip(common_stems, ransac_inliers) if not ok],
        },
        "position_error_all_fit": summarize_errors(errors_all),
        "position_error_ransac_fit_all_images": summarize_errors(errors_ransac),
        "position_error_ransac_fit_inliers_only": summarize_errors(errors_ransac[ransac_inliers]),
        "view_direction_error_deg_all_fit": summarize_errors(view_all),
        "view_direction_error_deg_ransac_fit": summarize_errors(view_ransac),
        "sparse_points_transformed": int(len(sparse_points_all)) if sparse_points_all is not None else 0,
    }
    save_json(orbit_out / "summary.json", summary)

    print(f"[COLMAP] Registered images:        {len(colmap_poses)}")
    print(f"[MATCH]  COLMAP-label matches:     {len(common_stems)}")
    print(f"[MATCH]  Labels not registered:    {len(labels_not_registered)}")
    print(f"[SIM3]   All-fit scale:            {similarity_all.scale:.10g}")
    print(f"[SIM3]   RANSAC threshold:         {threshold:.6g}")
    print(f"[SIM3]   RANSAC inliers:           {int(np.sum(ransac_inliers))}/{len(ransac_inliers)}")
    print(
        "[ERROR]  Position all-fit:          "
        f"mean={np.mean(errors_all):.6g}, median={np.median(errors_all):.6g}, max={np.max(errors_all):.6g}"
    )
    print(
        "[ERROR]  View direction all-fit:    "
        f"mean={np.mean(view_all):.6g} deg, median={np.median(view_all):.6g} deg, max={np.max(view_all):.6g} deg"
    )
    if labels_not_registered:
        print(f"[INFO]   Labels not registered: {labels_not_registered}")
    if int(np.sum(~ransac_inliers)):
        print(f"[INFO]   RANSAC candidate outliers: {[s for s, ok in zip(common_stems, ransac_inliers) if not ok]}")

    return OrbitResult(
        tag=tag,
        output_dir=orbit_out,
        names=common_stems,
        gt_centers=gt_centers,
        aligned_centers_all=aligned_centers_all,
        aligned_centers_ransac=aligned_centers_ransac,
        position_errors_all=errors_all,
        position_errors_ransac=errors_ransac,
        view_errors_all_deg=view_all,
        view_errors_ransac_deg=view_ransac,
        sparse_points_all=sparse_points_all,
        sparse_points_ransac=sparse_points_ransac,
        summary=summary,
    )


def downsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(points), size=max_points, replace=False)
    return points[ids]


def plot_trajectories(results: Sequence[OrbitResult], output: Path, max_sparse_points: int) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for result in results:
        ax.plot(
            result.gt_centers[:, 0],
            result.gt_centers[:, 1],
            result.gt_centers[:, 2],
            marker="o",
            markersize=2,
            linewidth=1,
            label=f"{result.tag} GT",
        )
        ax.plot(
            result.aligned_centers_all[:, 0],
            result.aligned_centers_all[:, 1],
            result.aligned_centers_all[:, 2],
            marker="x",
            markersize=3,
            linewidth=1,
            linestyle="--",
            label=f"{result.tag} COLMAP aligned (all fit)",
        )
        if result.sparse_points_all is not None and len(result.sparse_points_all):
            cloud = downsample(result.sparse_points_all, max_sparse_points)
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1, alpha=0.08, label=f"{result.tag} sparse")
    ax.scatter([0], [0], [0], marker="*", s=100, label="target origin")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("CORTO/Tango trajectories and aligned COLMAP reconstructions")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_error_curves(results: Sequence[OrbitResult], output: Path, kind: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for result in results:
        if kind == "position":
            values = result.position_errors_all
            ylabel = "camera-centre error [CORTO units]"
            title = "Camera-centre alignment residuals (all-point Sim(3) fit)"
        elif kind == "view":
            values = result.view_errors_all_deg
            ylabel = "view-direction error [deg]"
            title = "View-direction errors after alignment (all-point Sim(3) fit)"
        else:
            raise ValueError(kind)
        ax.plot(result.names, values, marker="o", markersize=3, linewidth=1, label=result.tag)
    ax.set_xlabel("registered image")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align two independent COLMAP reconstructions to the common CORTO/Tango frame (STEP A diagnostics only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--orbit1-colmap", type=Path, required=True, help="First orbit colmap_output root or sparse model directory")
    parser.add_argument("--orbit1-labels", type=Path, required=True, help="Filtered SPE3R-style labels.json for orbit 1")
    parser.add_argument("--orbit1-geometry", type=Path, default=None, help="Filtered geometry.json for orbit 1 (recommended)")
    parser.add_argument("--orbit2-colmap", type=Path, required=True, help="Second orbit colmap_output root or sparse model directory")
    parser.add_argument("--orbit2-labels", type=Path, required=True, help="Filtered SPE3R-style labels.json for orbit 2")
    parser.add_argument("--orbit2-geometry", type=Path, default=None, help="Filtered geometry.json for orbit 2 (recommended)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for alignment diagnostics")
    parser.add_argument("--colmap-exe", default="colmap", help="COLMAP executable used only when BIN -> TXT conversion is needed")

    parser.add_argument("--label-quat-order", choices=["auto", "wxyz", "xyzw"], default="auto")
    parser.add_argument("--label-rotation-mode", choices=["auto", "direct", "inverse"], default="auto")
    parser.add_argument("--label-center-sign", choices=["auto", "minus", "plus"], default="auto")
    parser.add_argument("--geometry-validation-tolerance", type=float, default=1e-4)
    parser.add_argument("--allow-geometry-mismatch", action="store_true", help="Continue even if labels and filtered geometry do not agree")

    parser.add_argument("--ransac-threshold", type=float, default=None, help="Candidate outlier threshold in CORTO units; default is estimated from all-fit residuals")
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sparse-points-in-plot", type=int, default=5000)
    return parser.parse_args()


def validate_path(path: Optional[Path], label: str) -> Optional[Path]:
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    orbit1_colmap = validate_path(args.orbit1_colmap, "orbit1 COLMAP root")
    orbit1_labels = validate_path(args.orbit1_labels, "orbit1 labels")
    orbit1_geometry = validate_path(args.orbit1_geometry, "orbit1 geometry")
    orbit2_colmap = validate_path(args.orbit2_colmap, "orbit2 COLMAP root")
    orbit2_labels = validate_path(args.orbit2_labels, "orbit2 labels")
    orbit2_geometry = validate_path(args.orbit2_geometry, "orbit2 geometry")
    assert orbit1_colmap and orbit1_labels and orbit2_colmap and orbit2_labels

    print("=" * 78)
    print("COLMAP ORBIT ALIGNMENT TO CORTO/TANGO - STEP A ONLY")
    print("=" * 78)
    print(f"Output: {output}")
    print("This script does not create a NeuS dataset and does not modify COLMAP outputs.")

    common_kwargs = dict(
        output_root=output,
        colmap_exe=args.colmap_exe,
        quat_order=args.label_quat_order,
        rotation_mode=args.label_rotation_mode,
        center_sign=args.label_center_sign,
        geometry_validation_tolerance=args.geometry_validation_tolerance,
        allow_geometry_mismatch=args.allow_geometry_mismatch,
        ransac_threshold=args.ransac_threshold,
        ransac_iterations=args.ransac_iterations,
        seed=args.seed,
    )

    orbit1 = align_one_orbit(
        tag="orbit1",
        colmap_root=orbit1_colmap,
        labels_path=orbit1_labels,
        geometry_path=orbit1_geometry,
        **common_kwargs,
    )
    orbit2 = align_one_orbit(
        tag="orbit2",
        colmap_root=orbit2_colmap,
        labels_path=orbit2_labels,
        geometry_path=orbit2_geometry,
        **common_kwargs,
    )

    plot_trajectories(
        [orbit1, orbit2],
        output / "trajectories_all_fit.png",
        max_sparse_points=args.max_sparse_points_in_plot,
    )
    plot_error_curves([orbit1, orbit2], output / "position_errors_all_fit.png", kind="position")
    plot_error_curves([orbit1, orbit2], output / "view_direction_errors_all_fit.png", kind="view")

    run_summary = {
        "step": "A_alignment_diagnostics_only",
        "output": str(output),
        "orbit1": orbit1.summary,
        "orbit2": orbit2.summary,
        "plots": {
            "trajectories": str(output / "trajectories_all_fit.png"),
            "position_errors": str(output / "position_errors_all_fit.png"),
            "view_direction_errors": str(output / "view_direction_errors_all_fit.png"),
        },
        "next_step": "Inspect diagnostics before building a combined NeuS dataset.",
    }
    save_json(output / "run_summary.json", run_summary)

    print("\n" + "=" * 78)
    print("COMPLETED: STEP A ALIGNMENT DIAGNOSTICS")
    print("=" * 78)
    print(f"Results: {output}")
    print("Inspect:")
    print(f"  - {output / 'run_summary.json'}")
    print(f"  - {output / 'trajectories_all_fit.png'}")
    print(f"  - {output / 'position_errors_all_fit.png'}")
    print(f"  - {output / 'view_direction_errors_all_fit.png'}")
    print("No NeuS dataset has been created.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
