"""
orbit_selector.py

Seleziona subset di immagini da SPE3R simulando una traiettoria orbitale
realistica di un chaser satellite.

Input: labels.json (con pose ground truth)
Output: Lista immagini selezionate con overlap garantito

Uso:
    python orbit_selector.py \
        --labels labels.json \
        --num-images 100 \
        --max-angle 20 \
        --output selected_images.txt
"""

import json
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(q):
    """
    Converte quaternion in rotation matrix.
    
    Args:
        q: [qw, qx, qy, qz] (SPE3R format)
    
    Returns:
        R: 3x3 rotation matrix
    """
    # SPE3R usa [qw, qx, qy, qz], scipy usa [qx, qy, qz, qw]
    q_scipy = [q[1], q[2], q[3], q[0]]
    return Rotation.from_quat(q_scipy).as_matrix()


def compute_view_direction(q):
    """
    Calcola la direzione di vista della camera (Z axis nel sistema camera).
    
    Args:
        q: quaternion [qw, qx, qy, qz]
    
    Returns:
        view_dir: vettore 3D normalizzato
    """
    R = quaternion_to_rotation_matrix(q)
    # Camera guarda lungo -Z (convenzione OpenCV)
    view_dir = -R[:, 2]
    return view_dir / np.linalg.norm(view_dir)


def compute_angle_between_poses(pose1, pose2):
    """
    Calcola angolo tra due pose (in gradi).
    
    Args:
        pose1, pose2: dict con 'q_vbs2tango_true'
    
    Returns:
        angle: angolo in gradi tra direzioni di vista
    """
    dir1 = compute_view_direction(pose1['q_vbs2tango_true'])
    dir2 = compute_view_direction(pose2['q_vbs2tango_true'])
    
    # Cosine similarity
    cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_spatial_distance(pose1, pose2):
    """
    Calcola distanza euclidea tra posizioni camera.
    
    Args:
        pose1, pose2: dict con 'r_Vo2To_vbs_true'
    
    Returns:
        distance: distanza in metri
    """
    pos1 = np.array(pose1['r_Vo2To_vbs_true'])
    pos2 = np.array(pose2['r_Vo2To_vbs_true'])
    return np.linalg.norm(pos1 - pos2)


def compute_score(angle, baseline, max_angle_deg, min_baseline, w_angle=1.0, w_baseline=0.5):
    """
    Calcola score normalizzato e dimensionalmente corretto.
    
    Score più basso = candidato migliore
    
    Args:
        angle: angolo tra pose in gradi
        baseline: distanza tra posizioni in unità normalizzate
        max_angle_deg: angolo massimo accettabile (per normalizzazione)
        min_baseline: baseline minima accettabile (per normalizzazione)
        w_angle: peso per termine angolo (default: 1.0)
        w_baseline: peso per termine baseline (default: 0.5)
    
    Returns:
        score: valore normalizzato (più basso = migliore)
    
    Note:
        - Angolo normalizzato in [0, 1]: 0° → 0, max_angle → 1
        - Baseline penalty: baseline piccola → penalità alta
        - Entrambi i termini sono adimensionali
        - Pesi controllano trade-off overlap vs geometria
    """
    # Normalizza angolo a [0, 1]
    # angolo = 0° → 0 (perfetto)
    # angolo = max_angle → 1 (limite accettabile)
    angle_normalized = angle / max_angle_deg
    
    # Penalità per baseline piccola (normalizzata)
    # baseline = min_baseline → 1 (limite accettabile)
    # baseline → 0 → ∞ (pessimo)
    # baseline grande → <1 (ottimo)
    baseline_penalty = min_baseline / baseline
    
    # Score finale (più basso = migliore)
    score = w_angle * angle_normalized + w_baseline * baseline_penalty
    
    return score 
def greedy_orbit_selection(labels, num_images, max_angle_deg=20, min_baseline_m=0.05, 
                           start_idx=0, end_idx=500, w_angle=1.0, w_baseline=0.5):
    """
    Seleziona immagini simulando orbita con overlap garantito.
    
    NOTA: Le distanze in labels.json sono in UNITÀ NORMALIZZATE, non metri reali.
    Il satellite è normalizzato a unit-size e la distanza è scelta per riempire l'immagine.
    
    Algoritmo:
    1. Parte da immagine casuale nel range [start_idx, end_idx]
    2. Cerca prossima immagine con:
       - Angolo < max_angle (overlap garantito)
       - Baseline > min_baseline (non troppo vicine)
    3. Seleziona candidato con SCORE MINIMO (normalizzato)
    4. Ripeti fino a num_images
    
    Args:
        labels: lista pose da labels.json
        num_images: numero target di immagini
        max_angle_deg: angolo massimo tra pose consecutive (gradi)
        min_baseline_m: baseline minima tra pose (unità normalizzate, non metri!)
        start_idx: indice inizio range (default: 0, per prime 500 con sfondo nero)
        end_idx: indice fine range (default: 500, evita immagini 501-1000 con sfondo Terra)
        w_angle: peso termine angolo (default: 1.0)
        w_baseline: peso termine baseline (default: 0.5)
    
    Returns:
        selected_indices: lista indici immagini selezionate
    """
    # Limita la ricerca al range specificato (default: prime 500 immagini)
    available = set(range(start_idx, min(end_idx, len(labels))))
    selected = []
    
    # Start da immagine casuale nel range
    current_idx = np.random.choice(list(available))
    selected.append(current_idx)
    available.remove(current_idx)
    
    print(f"Starting from image {current_idx}: {labels[current_idx]['filename']}")
    print(f"Selection range: images {start_idx}-{end_idx-1} (black background)")
    print(f"Weights: angle={w_angle:.2f}, baseline={w_baseline:.2f}")
    
    while len(selected) < num_images and len(available) > 0:
        current_pose = labels[current_idx]
        
        # Trova candidati con overlap
        candidates = []
        for idx in available:
            candidate_pose = labels[idx]
            
            # Calcola angolo e baseline
            angle = compute_angle_between_poses(current_pose, candidate_pose)
            baseline = compute_spatial_distance(current_pose, candidate_pose)
            
            # Filtra per criteri
            if angle <= max_angle_deg and baseline >= min_baseline_m:
                # Score normalizzato (più basso = migliore)
                score = compute_score(angle, baseline, max_angle_deg, min_baseline_m, 
                                    w_angle, w_baseline)
                candidates.append((idx, angle, baseline, score))
        
        if not candidates:
            # Nessun candidato valido - rilassa criteri
            print(f"  ⚠️  No candidates with angle<{max_angle_deg}° from {current_idx}")
            
            # Fallback: prendi il più vicino in angolo
            best_idx = None
            best_angle = float('inf')
            for idx in available:
                angle = compute_angle_between_poses(current_pose, labels[idx])
                if angle < best_angle:
                    best_angle = angle
                    best_idx = idx
            
            if best_idx is not None:
                current_idx = best_idx
                selected.append(current_idx)
                available.remove(current_idx)
                print(f"  → Fallback: image {current_idx} (angle={best_angle:.1f}°)")
            else:
                print(f"  ❌ No more candidates available!")
                break
        else:
            # Prendi candidato con score migliore (più basso)
            candidates.sort(key=lambda x: x[3])  # Ordina per score
            best_idx, best_angle, best_baseline, best_score = candidates[0]
            
            current_idx = best_idx
            selected.append(current_idx)
            available.remove(current_idx)
            
            if len(selected) % 10 == 0:
                print(f"  Selected {len(selected)}/{num_images}: image {current_idx} "
                      f"(angle={best_angle:.1f}°, baseline={best_baseline:.3f}, score={best_score:.3f})")
    
    print(f"\n✅ Selected {len(selected)} images from range {start_idx}-{end_idx-1}")
    return selected


def analyze_selection(labels, selected_indices):
    """
    Analizza la selezione e mostra statistiche.
    """
    print("\n" + "="*70)
    print("ANALISI SELEZIONE")
    print("="*70)
    
    # Angoli tra consecutive
    angles = []
    baselines = []
    
    for i in range(len(selected_indices) - 1):
        idx1 = selected_indices[i]
        idx2 = selected_indices[i + 1]
        
        angle = compute_angle_between_poses(labels[idx1], labels[idx2])
        baseline = compute_spatial_distance(labels[idx1], labels[idx2])
        
        angles.append(angle)
        baselines.append(baseline)
    
    angles = np.array(angles)
    baselines = np.array(baselines)
    
    print(f"\n📊 Statistiche Angoli tra Consecutive:")
    print(f"   Media:   {np.mean(angles):.2f}°")
    print(f"   Mediana: {np.median(angles):.2f}°")
    print(f"   Min:     {np.min(angles):.2f}°")
    print(f"   Max:     {np.max(angles):.2f}°")
    
    print(f"\n📊 Statistiche Baseline:")
    print(f"   Media:   {np.mean(baselines):.4f} unità")
    print(f"   Mediana: {np.median(baselines):.4f} unità")
    print(f"   Min:     {np.min(baselines):.4f} unità")
    print(f"   Max:     {np.max(baselines):.4f} unità")
    
    # Coverage
    print(f"\n📊 Coverage Dataset:")
    print(f"   Immagini selezionate: {len(selected_indices)}")
    print(f"   Range indici: [{min(selected_indices)} - {max(selected_indices)}]")
    print(f"   Coverage: {100 * len(selected_indices) / len(labels):.1f}%")
    
    # Overlap estimate
    good_overlap = np.sum(angles <= 15)
    print(f"\n📊 Overlap Stimato:")
    print(f"   Coppie con overlap buono (<15°): {good_overlap}/{len(angles)} ({100*good_overlap/len(angles):.1f}%)")
    
    return {
        'angles': angles,
        'baselines': baselines,
        'mean_angle': np.mean(angles),
        'mean_baseline': np.mean(baselines),
    }


def save_selection(selected_indices, labels, output_path):
    """
    Salva selezione in formato utilizzabile.
    """
    output_path = Path(output_path)
    
    # 1. Lista immagini (per processing)
    with open(output_path, 'w') as f:
        for idx in selected_indices:
            filename = labels[idx]['filename']
            f.write(f"{filename}.jpg\n")  # Assumendo estensione .jpg
    
    print(f"\n✅ Lista immagini salvata in: {output_path}")
    
    # 2. Indici (per riferimento)
    indices_path = output_path.with_suffix('.indices.txt')
    with open(indices_path, 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")
    
    print(f"✅ Indici salvati in: {indices_path}")
    
    # 3. JSON completo con pose (per debug)
    json_path = output_path.with_suffix('.json')
    selected_data = [labels[idx] for idx in selected_indices]
    with open(json_path, 'w') as f:
        json.dump(selected_data, f, indent=2)
    
    print(f"✅ Pose selezionate salvate in: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Seleziona subset SPE3R simulando orbita chaser realistica"
    )
    
    parser.add_argument(
        '--labels',
        required=True,
        help="Path a labels.json"
    )
    
    parser.add_argument(
        '--num-images',
        type=int,
        default=100,
        help="Numero target immagini da selezionare (default: 100)"
    )
    
    parser.add_argument(
        '--max-angle',
        type=float,
        default=20.0,
        help="Angolo massimo tra pose consecutive in gradi (default: 20)"
    )
    
    parser.add_argument(
        '--min-baseline',
        type=float,
        default=0.05,
        help="Baseline minima tra pose in unità normalizzate (default: 0.05)"
    )
    
    parser.add_argument(
        '--output',
        default='selected_images.txt',
        help="File output con lista immagini (default: selected_images.txt)"
    )
    
    parser.add_argument(
        '--weight-angle',
        type=float,
        default=1.0,
        help="Peso per termine angolo nello score (default: 1.0)"
    )
    
    parser.add_argument(
        '--weight-baseline',
        type=float,
        default=0.5,
        help="Peso per termine baseline nello score (default: 0.5)"
    )
    
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help="Indice inizio range selezione (default: 0)"
    )
    
    parser.add_argument(
        '--end-idx',
        type=int,
        default=500,
        help="Indice fine range selezione (default: 500, solo immagini con sfondo nero)"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed per riproducibilità (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    print("="*70)
    print("SPE3R ORBIT SELECTOR")
    print("="*70)
    print(f"Labels: {args.labels}")
    print(f"Target immagini: {args.num_images}")
    print(f"Max angolo: {args.max_angle}°")
    print(f"Min baseline: {args.min_baseline} unità")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Carica labels
    print("\n📂 Caricamento labels.json...")
    with open(args.labels, 'r') as f:
        labels = json.load(f)
    
    print(f"✅ Caricate {len(labels)} pose")
    
    # Selezione
    print(f"\n🛰️  Selezione orbita (target: {args.num_images} immagini)...")
    print(f"Range: immagini {args.start_idx}-{args.end_idx-1}")
    print(f"(Prime 500 = sfondo nero, 501-1000 = sfondo Terra)\n")
    
    selected_indices = greedy_orbit_selection(
        labels,
        num_images=args.num_images,
        max_angle_deg=args.max_angle,
        min_baseline_m=args.min_baseline,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        w_angle=args.weight_angle,
        w_baseline=args.weight_baseline
    )
    
    # Analisi
    stats = analyze_selection(labels, selected_indices)
    
    # Salva
    save_selection(selected_indices, labels, args.output)
    
    print("\n" + "="*70)
    print("✅ COMPLETATO")
    print("="*70)


if __name__ == "__main__":
    main()
