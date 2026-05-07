#!/usr/bin/env python3
"""
filter_by_phase_angle.py

Calcola il sun phase angle per ogni frame di una sequenza CORTO e
restituisce i frame accettati (phase_angle <= threshold).

Il sole è un Blender sun light: la posizione è irrilevante, conta solo
l'orientazione. La direzione dei raggi è l'asse -Z dell'oggetto sole
ruotato nel world frame tramite il quaternione sun.orientation.

Il sun phase angle è l'angolo tra:
  - vettore verso la sorgente sole  (asse +Z sun ruotato nel world)
  - vettore camera → oggetto        (normalize(body_pos - cam_pos))

Output:
  - accepted_frames.npy  : array 1D con gli indici dei frame accettati
  - phase_angle_plot.png : grafico phase angle per frame con threshold

USO:
    python filter_by_phase_angle.py \\
        --geometry geometry.json \\
        --output_dir /path/to/output \\
        --threshold 90.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Geometria / quaternioni
# ─────────────────────────────────────────────

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Vettore quasi nullo: {v}")
    return v / n


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Quaternione [w, x, y, z] → matrice di rotazione 3x3."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1-2*(x**2+z**2),   2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x), 1-2*(x**2+y**2)],
    ])


def sun_toward_vector(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Restituisce il vettore unitario che punta VERSO il sole nel world frame.

    Blender sun light: i raggi viaggiano lungo l'asse -Z dell'oggetto sole.
    Il vettore 'verso il sole' è l'asse +Z ruotato nel world frame.
    """
    R = quat_wxyz_to_rotmat(q_wxyz)
    return normalize(R @ np.array([0.0, 0.0, 1.0]))


# ─────────────────────────────────────────────
# Calcolo phase angles
# ─────────────────────────────────────────────

def compute_phase_angles(geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcola il sun phase angle per ogni frame.

    Phase angle = angolo tra:
      - toward_sun  : vettore costante verso il sole (da quaternione)
      - cam_to_body : normalize(body_pos - cam_pos) per ogni frame

    Returns:
        phase_angles : array (N,) in gradi
        toward_sun   : vettore unitario (3,) verso il sole
    """
    # Sole fisso — usa il primo frame (tutti identici)
    q_sun = np.array(geometry["sun"]["orientation"][0], dtype=np.float64)
    toward_sun = sun_toward_vector(q_sun)

    camera_positions = np.array(geometry["camera"]["position"], dtype=np.float64)
    body_positions   = np.array(geometry["body"]["position"],   dtype=np.float64)

    n_frames = len(camera_positions)
    phase_angles = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        # body_to_cam: vettore oggetto→camera
        # Angolo piccolo = sole e camera dalla stessa parte = oggetto illuminato
        body_to_cam = normalize(camera_positions[i] - body_positions[i])
        cos_angle = np.clip(np.dot(toward_sun, body_to_cam), -1.0, 1.0)
        phase_angles[i] = np.degrees(np.arccos(cos_angle))

    return phase_angles, toward_sun


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────

def plot_phase_angles(
    phase_angles: np.ndarray,
    threshold: float,
    accepted: np.ndarray,
    rejected: np.ndarray,
    toward_sun: np.ndarray,
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 5))

    frame_indices = np.arange(len(phase_angles))

    ax.plot(frame_indices, phase_angles, color="steelblue", linewidth=1.4,
            zorder=3, label="Phase angle")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold}°)")

    ax.fill_between(frame_indices, phase_angles, threshold,
                    where=(phase_angles > threshold),
                    alpha=0.25, color="red", label=f"Rejected ({len(rejected)})")
    ax.fill_between(frame_indices, phase_angles, threshold,
                    where=(phase_angles <= threshold),
                    alpha=0.20, color="green", label=f"Accepted ({len(accepted)})")

    # Marker sui frame accettati/scartati sull'asse X
    ax.scatter(accepted, np.zeros(len(accepted)) + 2,
               color="green", s=12, zorder=4, linewidths=0)
    ax.scatter(rejected, np.zeros(len(rejected)) + 2,
               color="red", s=12, zorder=4, linewidths=0)

    ax.set_xlabel("Frame index", fontsize=11)
    ax.set_ylabel("Phase angle (degrees)", fontsize=11)
    ax.set_xlim(0, len(phase_angles) - 1)
    ax.set_ylim(0, 185)
    ax.set_yticks(range(0, 181, 30))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    elev = np.degrees(np.arcsin(np.clip(toward_sun[2], -1, 1)))
    azim = np.degrees(np.arctan2(toward_sun[1], toward_sun[0]))
    ax.set_title(
        f"Sun Phase Angle  |  threshold={threshold}°  |  "
        f"accepted={len(accepted)}/{len(phase_angles)} ({100*len(accepted)/len(phase_angles):.1f}%)  |  "
        f"sun elev={elev:.1f}° azim={azim:.1f}°",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"  Plot salvato: {output_path}")


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

def print_summary(phase_angles, accepted, rejected, threshold, toward_sun):
    elev = np.degrees(np.arcsin(np.clip(toward_sun[2], -1, 1)))
    azim = np.degrees(np.arctan2(toward_sun[1], toward_sun[0]))

    print("\n" + "=" * 60)
    print("RIEPILOGO PHASE ANGLE FILTERING")
    print("=" * 60)
    print(f"  Direzione verso sole:     {np.round(toward_sun, 4)}")
    print(f"  Elevazione sole:          {elev:.1f}°")
    print(f"  Azimuth sole:             {azim:.1f}°")
    print(f"  Threshold:                {threshold:.1f}°")
    print(f"\n  Frame totali:    {len(phase_angles)}")
    print(f"  Frame accettati: {len(accepted)} ({100*len(accepted)/len(phase_angles):.1f}%)")
    print(f"  Frame scartati:  {len(rejected)} ({100*len(rejected)/len(phase_angles):.1f}%)")
    print(f"\n  Phase angle (deg) — tutti i frame:")
    print(f"    min:    {phase_angles.min():.2f}°")
    print(f"    max:    {phase_angles.max():.2f}°")
    print(f"    mean:   {phase_angles.mean():.2f}°")
    print(f"    median: {np.median(phase_angles):.2f}°")
    if len(accepted) > 0:
        print(f"\n  Phase angle (deg) — frame accettati:")
        print(f"    min:  {phase_angles[accepted].min():.2f}°")
        print(f"    max:  {phase_angles[accepted].max():.2f}°")
        print(f"    mean: {phase_angles[accepted].mean():.2f}°")
    print("=" * 60)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calcola sun phase angle e salva indici frame accettati."
    )
    parser.add_argument("--geometry",   required=True, type=Path,
                        help="Path a geometry.json")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory output per accepted_frames.npy e plot")
    parser.add_argument("--threshold",  type=float, default=90.0,
                        help="Soglia in gradi (default: 90°). "
                             "Frame con phase_angle > threshold vengono scartati.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.geometry) as f:
        geometry = json.load(f)

    phase_angles, toward_sun = compute_phase_angles(geometry)

    accepted = np.where(phase_angles <= args.threshold)[0]
    rejected = np.where(phase_angles >  args.threshold)[0]

    print_summary(phase_angles, accepted, rejected, args.threshold, toward_sun)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Output 1: array indici accettati
    npy_path = args.output_dir / "accepted_frames.npy"
    np.save(npy_path, accepted)
    print(f"\n  accepted_frames.npy salvato: {npy_path}")
    print(f"  Indici: {accepted.tolist()}")

    # Output 2: plot
    plot_phase_angles(
        phase_angles, args.threshold, accepted, rejected,
        toward_sun, args.output_dir / "phase_angle_plot.png"
    )
