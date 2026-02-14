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


def load_camera_params(spe3r_path):
    """
    Carica parametri camera da camera.json.
    
    Supporta due formati:
    
    1. Formato con cameraMatrix (SPE3R standard):
    {
        "Nu": 256,
        "Nv": 256,
        "cameraMatrix": [
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ]
    }
    
    2. Formato flat:
    {
        "fx": 3003.0,
        "fy": 3003.0,
        "cx": 128.0,
        "cy": 128.0
    }
    
    Returns:
        dict con fx, fy, cx, cy
    """
    camera_json_path = Path(spe3r_path) / "camera.json"
    
    if not camera_json_path.exists():
        raise FileNotFoundError(f"camera.json non trovato in: {spe3r_path}")
    
    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)
    
    # Estrai parametri
    if 'cameraMatrix' in camera_data:
        # Formato SPE3R con cameraMatrix (OpenCV style)
        K = camera_data['cameraMatrix']
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        
        width = camera_data.get('Nu', None)
        height = camera_data.get('Nv', None)
        
        print(f"📷 Caricati parametri da camera.json (cameraMatrix):")
        print(f"   fx: {fx}")
        print(f"   fy: {fy}")
        print(f"   cx: {cx}")
        print(f"   cy: {cy}")
        if width and height:
            print(f"   Risoluzione: {width}x{height}")
        
    elif 'fx' in camera_data:
        # Formato flat
        fx = camera_data['fx']
        fy = camera_data['fy']
        cx = camera_data['cx']
        cy = camera_data['cy']
        
        print(f"📷 Caricati parametri da camera.json (flat):")
        print(f"   fx: {fx}")
        print(f"   fy: {fy}")
        print(f"   cx: {cx}")
        print(f"   cy: {cy}")
        
    else:
        raise ValueError(
            "camera.json deve contenere 'cameraMatrix' o campi 'fx', 'fy', 'cx', 'cy'"
        )
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }


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


def run_colmap_with_intrinsics(basedir, neus_path, camera_params, use_gpu=True):
    """
    Lancia COLMAP usando il wrapper modificato di NeuS.
    """
    print("\n" + "="*70)
    print("ESECUZIONE COLMAP CON INTRINSECI IMPOSTI")
    print("="*70)
    
    # Aggiungi NeuS al path Python
    neus_preprocess_path = Path(neus_path) / "preprocess_custom_data" / "colmap_preprocess"
    sys.path.insert(0, str(neus_preprocess_path))
    
    # Importa il wrapper modificato
    from colmap_wrapper_with_intrinsics import run_colmap
    
    # Esegui COLMAP
    run_colmap(
        basedir=str(basedir),
        match_type='exhaustive_matcher',
        camera_params=camera_params,
        use_gpu=use_gpu
    )
    
    print("="*70)
    print("✅ COLMAP COMPLETATO")
    print("="*70)


def generate_poses(basedir, neus_path):
    """
    Genera poses.npy usando pose_utils.py di NeuS.
    Questo genera anche sparse_points.ply
    """
    print("\n" + "="*70)
    print("GENERAZIONE POSES.NPY e SPARSE_POINTS.PLY")
    print("="*70)
    
    neus_preprocess_path = Path(neus_path) / "preprocess_custom_data" / "colmap_preprocess"
    sys.path.insert(0, str(neus_preprocess_path))
    
    from pose_utils import load_colmap_data, save_poses
    
    # load_colmap_data può ritornare 3 o 4 valori a seconda della versione
    result = load_colmap_data(str(basedir))
    
    if len(result) == 4:
        poses, pts3d, perm, id_to_idx = result
        save_poses(str(basedir), poses, pts3d, perm, id_to_idx)
    elif len(result) == 3:
        poses, pts3d, perm = result
        save_poses(str(basedir), poses, pts3d, perm)
    else:
        raise ValueError(f"load_colmap_data ha ritornato {len(result)} valori, attesi 3 o 4")
    
    print(f"✅ poses.npy creato in: {basedir}")
    print(f"✅ sparse_points.ply creato in: {basedir}")
    print("="*70)


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
