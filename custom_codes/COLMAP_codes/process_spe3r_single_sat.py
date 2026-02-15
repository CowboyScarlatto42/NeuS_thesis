"""
process_spe3r_single_sat.py

Versione modificata per dataset SPE3R con SINGOLO satellite.
- Legge intrinseci da camera.json
- Cerca immagini in {satellite}_images/
- Default: prime 500 immagini (sfondo nero)

Struttura attesa:
    spe3r_path/
    ├── camera.json
    ├── {satellite}_images/
    │   ├── img000001.jpg
    │   ├── img000002.jpg
    │   └── ...
    ├── {satellite}_masks/
    └── labels.json

Uso:
    python process_spe3r_single_sat.py \
        --spe3r-path /path/to/SPE3R/hst \
        --satellite hst \
        --output /tmp/colmap_test \
        --neus-path /path/to/NeuS_thesis \
        --num-images 500
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


def prepare_images(spe3r_path, satellite, output_path, 
                   start_idx=1, num_images=500):
    """
    Copia immagini da {satellite}_images/.
    
    Args:
        start_idx: Indice iniziale (default: 1, per img000001)
        num_images: Numero di immagini da copiare (default: 500, solo sfondo nero)
    """
    spe3r_path = Path(spe3r_path)
    output_path = Path(output_path)
    
    # Directory immagini: {satellite}_images
    source_dir = spe3r_path / f"{satellite}_images"
    
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Directory immagini non trovata: {source_dir}\n"
            f"Assicurati che esista {satellite}_images/ nella directory SPE3R"
        )
    
    # Trova tutte le immagini
    all_image_files = sorted(
        list(source_dir.glob("*.png")) +
        list(source_dir.glob("*.jpg"))
    )
    
    if len(all_image_files) == 0:
        raise FileNotFoundError(f"Nessuna immagine in: {source_dir}")
    
    print(f"📁 Trovate {len(all_image_files)} immagini totali in {source_dir}")
    
    # Seleziona range
    # Converti start_idx (1-based) a 0-based per Python
    end_idx = start_idx - 1 + num_images
    selected_files = all_image_files[start_idx - 1:end_idx]
    
    if len(selected_files) == 0:
        raise ValueError(
            f"Nessuna immagine nel range [{start_idx}:{end_idx}]. "
            f"Totale disponibili: {len(all_image_files)}"
        )
    
    print(f"📸 Selezionate immagini da {start_idx} a {start_idx + len(selected_files) - 1}")
    print(f"   (totale: {len(selected_files)} immagini)")
    
    # Crea directory output
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copia mantenendo nomi originali
    print(f"📋 Copia in corso...")
    for img_file in selected_files:
        shutil.copy2(img_file, images_dir / img_file.name)
    
    print(f"✅ Immagini copiate in: {images_dir}")
    return len(selected_files)


def run_colmap_with_intrinsics(basedir, neus_path, camera_params, use_gpu=True, 
                                match_type='vocab_tree_matcher', vocab_tree_path=None):
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
    
    # Prepara colmap_extra_args con parametri aggressivi
    colmap_extra_args = {
        'feature_extractor': [
            '--SiftExtraction.max_num_features', '20000',
            '--SiftExtraction.peak_threshold', '0.002',
            '--SiftExtraction.edge_threshold', '10',
            '--SiftExtraction.first_octave', '-1',
            '--SiftExtraction.domain_size_pooling', '1',
            '--SiftExtraction.estimate_affine_shape', '1',
        ],
        'matcher': [
            '--SiftMatching.guided_matching', '1',
            '--SiftMatching.max_ratio', '0.9',
            '--SiftMatching.max_distance', '0.8',
            '--SiftMatching.cross_check', '1',
        ],
    }
    
    # Se usiamo vocab_tree_matcher, aggiungi vocab_tree_path
    if match_type == 'vocab_tree_matcher':
        if vocab_tree_path is None:
            print("\n⚠️  ERRORE: vocab_tree_matcher richiede --vocab-tree-path")
            print("Scarica il vocabulary tree con:")
            print("  python setup_vocab_tree.py --download --output vocab_tree.bin")
            print("\nPoi riavvia con:")
            print(f"  --vocab-tree-path vocab_tree.bin")
            print("\nOppure usa un matcher diverso:")
            print("  --match-type sequential_matcher  (consigliato per <1000 immagini)")
            print("  --match-type exhaustive_matcher  (lento ma accurato)")
            sys.exit(1)
        
        # Aggiungi vocab tree path e parametri specifici
        colmap_extra_args['matcher'].extend([
            '--VocabTreeMatching.vocab_tree_path', str(vocab_tree_path),
            '--VocabTreeMatching.num_images', '100',  # Match con top 100 immagini simili
            '--VocabTreeMatching.max_num_features', '-1',  # Usa tutte le features
        ])
        print(f"🌳 Vocabulary tree: {vocab_tree_path}")
    
    # Esegui COLMAP
    run_colmap(
        basedir=str(basedir),
        match_type=match_type,
        camera_params=camera_params,
        use_gpu=use_gpu,
        colmap_extra_args=colmap_extra_args
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
    
    poses, pts3d, perm, id_to_idx = load_colmap_data(str(basedir))
    save_poses(str(basedir), poses, pts3d, perm, id_to_idx)
    
    print(f"✅ poses.npy creato in: {basedir}")
    print(f"✅ sparse_points.ply creato in: {basedir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Processa SPE3R singolo satellite con COLMAP fino a sparse_points.ply"
    )
    
    parser.add_argument(
        "--spe3r-path",
        required=True,
        help="Path alla directory del satellite (es: /path/to/SPE3R/hst)"
    )
    
    parser.add_argument(
        "--satellite",
        required=True,
        help="Nome satellite (es: hst, jwst, tango). Usato per trovare {satellite}_images/"
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
        "--start-idx",
        type=int,
        default=1,
        help="Indice iniziale immagine (1-based, default: 1)"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
        help="Numero immagini da usare (default: 500, solo sfondo nero)"
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
        "--match-type",
        default="vocab_tree_matcher",
        choices=["exhaustive_matcher", "vocab_tree_matcher", "sequential_matcher", "spatial_matcher"],
        help="Tipo di matcher COLMAP (default: vocab_tree_matcher)"
    )
    
    parser.add_argument(
        "--vocab-tree-path",
        default=None,
        help="Path al vocabulary tree .bin (RICHIESTO per vocab_tree_matcher). "
             "Scarica con: python setup_vocab_tree.py --download --output vocab_tree.bin"
    )
    
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Salta COLMAP (Step 1-2), esegui solo generazione poses (Step 3). "
             "Usa se COLMAP è già completato e vuoi solo generare poses.npy"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("="*70)
    print("PROCESSING SPE3R SINGOLO SATELLITE → SPARSE_POINTS.PLY")
    print("="*70)
    print(f"Dataset: {args.spe3r_path}")
    print(f"Satellite: {args.satellite}")
    print(f"Output: {args.output}")
    print(f"NeuS path: {args.neus_path}")
    print(f"Range immagini: {args.start_idx} → {args.start_idx + args.num_images - 1}")
    print(f"GPU: {'ON' if args.use_gpu else 'OFF'}")
    print(f"Matcher: {args.match_type}")
    print("="*70)
    
    # Step 0: Carica parametri camera da camera.json
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
        print(f"✅ Usando risultati esistenti in: {output_path}")
        
        # Verifica che i file esistano
        required_files = [
            output_path / "database.db",
            output_path / "sparse" / "0" / "cameras.bin",
            output_path / "sparse" / "0" / "images.bin",
            output_path / "sparse" / "0" / "points3D.bin",
        ]
        
        missing = [f for f in required_files if not f.exists()]
        if missing:
            print("\n❌ ERRORE: File COLMAP mancanti:")
            for f in missing:
                print(f"   - {f}")
            print("\n💡 Rimuovi --skip-colmap per eseguire COLMAP da zero")
            return
        
        print("✅ Tutti i file COLMAP presenti")
        num_images = len(list((output_path / "images").glob("*.jpg"))) + \
                     len(list((output_path / "images").glob("*.png")))
    else:
        # Step 1: Prepara immagini
        print("\n" + "="*70)
        print("STEP 1: PREPARAZIONE IMMAGINI")
        print("="*70)
        num_images = prepare_images(
            spe3r_path=args.spe3r_path,
            satellite=args.satellite,
            output_path=output_path,
            start_idx=args.start_idx,
            num_images=args.num_images
        )
        
        # Step 2: COLMAP con intrinseci imposti
        print("\n" + "="*70)
        print("STEP 2: COLMAP (FEATURE EXTRACTION + MATCHING + SFM)")
        print("="*70)
        run_colmap_with_intrinsics(
            basedir=output_path,
            neus_path=args.neus_path,
            camera_params=camera_params,
            use_gpu=args.use_gpu,
            match_type=args.match_type,
            vocab_tree_path=args.vocab_tree_path
        )
    
    # Step 3: Genera poses.npy e sparse_points.ply
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
    print(f"   Range: img{args.start_idx:06d} → img{args.start_idx + num_images - 1:06d}")
    
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
    print(f"1. Apri {output_path}/sparse_points.ply in MeshLab/CloudCompare")
    print("2. Pulisci outliers manualmente")
    print("3. Salva come sparse_points_interest.ply")
    print("4. Lancia gen_cameras.py:")
    print(f"   cd {args.neus_path}/preprocess_custom_data/colmap_preprocess")
    print(f"   python gen_cameras.py {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
