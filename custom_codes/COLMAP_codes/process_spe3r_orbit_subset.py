"""
process_spe3r_orbit_subset.py

Versione modificata che usa subset selezionato da orbit_selector.py
invece di prime N immagini consecutive.

Uso:
    # 1. Prima genera subset
    python orbit_selector.py \
        --labels SPE3R/hst/labels.json \
        --num-images 100 \
        --output selected_orbit.txt
    
    # 2. Poi processa subset
    python process_spe3r_orbit_subset.py \
        --spe3r-path SPE3R/hst \
        --satellite hst \
        --selected-images selected_orbit.txt \
        --output output_orbit \
        --neus-path NeuS_thesis
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path


# ... (tutte le funzioni load_camera_params, run_colmap_with_intrinsics, generate_poses 
#      rimangono identiche a process_spe3r_single_sat.py)


def prepare_images_from_list(spe3r_path, satellite, selected_images_file, output_path):
    """
    Copia immagini specificate in selected_images_file invece di range numerico.
    
    Args:
        spe3r_path: Path al dataset SPE3R
        satellite: Nome satellite (hst, jwst, etc)
        selected_images_file: File .txt con lista immagini (output orbit_selector)
        output_path: Directory output
    
    Returns:
        num_images: Numero immagini copiate
    """
    spe3r_path = Path(spe3r_path)
    output_path = Path(output_path)
    
    # Directory immagini sorgente
    source_dir = spe3r_path / f"{satellite}_images"
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Directory immagini non trovata: {source_dir}")
    
    # Leggi lista immagini selezionate
    with open(selected_images_file, 'r') as f:
        selected_names = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Letta lista con {len(selected_names)} immagini da: {selected_images_file}")
    
    # Crea directory output
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copia immagini
    print(f"📋 Copia in corso...")
    copied = 0
    missing = []
    
    for img_name in selected_names:
        # Gestisci con/senza estensione
        if not img_name.endswith(('.jpg', '.png')):
            # Prova entrambe le estensioni
            img_file = None
            for ext in ['.jpg', '.png']:
                candidate = source_dir / f"{img_name}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break
        else:
            img_file = source_dir / img_name
        
        if img_file and img_file.exists():
            shutil.copy2(img_file, images_dir / img_file.name)
            copied += 1
        else:
            missing.append(img_name)
            print(f"  ⚠️  Non trovata: {img_name}")
    
    print(f"✅ Copiate {copied}/{len(selected_names)} immagini in: {images_dir}")
    
    if missing:
        print(f"⚠️  {len(missing)} immagini non trovate")
    
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Processa subset SPE3R selezionato da orbit_selector.py"
    )
    
    parser.add_argument(
        "--spe3r-path",
        required=True,
        help="Path alla directory del satellite (es: /path/to/SPE3R/hst)"
    )
    
    parser.add_argument(
        "--satellite",
        required=True,
        help="Nome satellite (es: hst, jwst, tango)"
    )
    
    parser.add_argument(
        "--selected-images",
        required=True,
        help="File .txt con lista immagini selezionate (output orbit_selector.py)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Directory output"
    )
    
    parser.add_argument(
        "--neus-path",
        required=True,
        help="Path a NeuS_thesis repository"
    )
    
    parser.add_argument(
        "--camera-model",
        default="PINHOLE",
        choices=["PINHOLE", "SIMPLE_PINHOLE"],
        help="Modello camera COLMAP (default: PINHOLE)"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Usa GPU per COLMAP (default: True)"
    )
    
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disabilita GPU, usa CPU"
    )
    
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Salta COLMAP, esegui solo generazione poses"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("="*70)
    print("PROCESSING SPE3R ORBIT SUBSET → SPARSE_POINTS.PLY")
    print("="*70)
    print(f"Dataset: {args.spe3r_path}")
    print(f"Satellite: {args.satellite}")
    print(f"Selected images: {args.selected_images}")
    print(f"Output: {args.output}")
    print(f"NeuS path: {args.neus_path}")
    print(f"GPU: {'ON' if args.use_gpu else 'OFF'}")
    print("="*70)
    
    # Step 0: Carica parametri camera
    print("\n" + "="*70)
    print("STEP 0: CARICAMENTO PARAMETRI CAMERA")
    print("="*70)
    camera_params = load_camera_params(args.spe3r_path)
    camera_params['model'] = args.camera_model
    
    print(f"\n📷 Parametri Camera Finali:")
    for key, val in camera_params.items():
        print(f"   {key}: {val}")
    
    # Step 1-2: COLMAP (se non skippato)
    if args.skip_colmap:
        print("\n" + "="*70)
        print("⏭️  SKIP: STEPS 1-2 (COLMAP già completato)")
        print("="*70)
        # Count images
        num_images = len(list((output_path / "images").glob("*.jpg"))) + \
                     len(list((output_path / "images").glob("*.png")))
    else:
        # Step 1: Prepara immagini da lista
        print("\n" + "="*70)
        print("STEP 1: PREPARAZIONE IMMAGINI (DA ORBIT SELECTION)")
        print("="*70)
        num_images = prepare_images_from_list(
            spe3r_path=args.spe3r_path,
            satellite=args.satellite,
            selected_images_file=args.selected_images,
            output_path=output_path
        )
        
        # Step 2: COLMAP
        print("\n" + "="*70)
        print("STEP 2: COLMAP (FEATURE EXTRACTION + MATCHING + SFM)")
        print("="*70)
        run_colmap_with_intrinsics(
            basedir=output_path,
            neus_path=args.neus_path,
            camera_params=camera_params,
            use_gpu=args.use_gpu
        )
    
    # Step 3: Genera poses
    print("\n" + "="*70)
    print("STEP 3: GENERAZIONE POSES E SPARSE POINTS")
    print("="*70)
    generate_poses(
        basedir=output_path,
        neus_path=args.neus_path
    )
    
    # Summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETATO - FERMATO A SPARSE_POINTS.PLY")
    print("="*70)
    print(f"📁 Output: {output_path}")
    print(f"📸 Immagini processate: {num_images}")
    
    print("\n📋 File generati:")
    files_to_check = [
        "database.db",
        "sparse/0/cameras.bin",
        "sparse/0/images.bin",
        "sparse/0/points3D.bin",
        "poses.npy",
        "sparse_points.ply",
    ]
    
    for file_path in files_to_check:
        full_path = output_path / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"   {status} {file_path}")
    
    print("\n" + "="*70)
    print("🎯 PROSSIMI PASSI:")
    print("="*70)
    print(f"1. Verifica success rate:")
    print(f"   cd {args.neus_path}/preprocess_custom_data/colmap_preprocess")
    print(f"   python -c \"from colmap_read_model import read_images_binary; print(len(read_images_binary('{output_path}/sparse/0/images.bin')))\"")
    print(f"2. Pulisci sparse_points.ply in MeshLab/CloudCompare")
    print(f"3. Salva come sparse_points_interest.ply")
    print(f"4. Lancia gen_cameras.py")
    print("="*70)


if __name__ == "__main__":
    main()
