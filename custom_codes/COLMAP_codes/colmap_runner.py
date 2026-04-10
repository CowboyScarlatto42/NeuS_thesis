
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
    1. Formato con cameraMatrix (SPE3R standard)
    2. Formato flat con fx, fy, cx, cy
    
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
        # Formato SPE3R con cameraMatrix
        K = camera_data['cameraMatrix']
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        
        width = camera_data.get('Nu', None)
        height = camera_data.get('Nv', None)
        
        print(f"📷 Parametri da camera.json (cameraMatrix):")
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
        
        print(f"📷 Parametri da camera.json (flat):")
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


def prepare_images(spe3r_path, satellite, output_path, start_idx=1, num_images=500):
    spe3r_path = Path(spe3r_path)
    output_path = Path(output_path)

    source_img_dir = spe3r_path / f"{satellite}_images"
    source_mask_dir = spe3r_path / f"{satellite}_masks"

    if not source_img_dir.exists():
        raise FileNotFoundError(f"Directory immagini non trovata: {source_img_dir}")
    if not source_mask_dir.exists():
        raise FileNotFoundError(f"Directory maschere non trovata: {source_mask_dir}")

    all_image_files = sorted(
        list(source_img_dir.glob("*.png")) + list(source_img_dir.glob("*.jpg"))
    )

    if len(all_image_files) == 0:
        raise FileNotFoundError(f"Nessuna immagine in: {source_img_dir}")

    print(f"📁 Trovate {len(all_image_files)} immagini totali in {source_img_dir}")

    end_idx = start_idx - 1 + num_images
    selected_files = all_image_files[start_idx - 1:end_idx]

    if len(selected_files) == 0:
        raise ValueError(
            f"Nessuna immagine nel range [{start_idx}:{end_idx}]. "
            f"Totale disponibili: {len(all_image_files)}"
        )

    print(f"📸 Selezionate immagini da {start_idx} a {start_idx + len(selected_files) - 1}")
    print(f"   (totale: {len(selected_files)} immagini)")

    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("📋 Copia immagini e maschere in corso...")
    for img_file in selected_files:
        shutil.copy2(img_file, images_dir / img_file.name)

        mask_file = source_mask_dir / img_file.name
        if not mask_file.exists():
            raise FileNotFoundError(f"Maschera mancante per {img_file.name}: {mask_file}")
        shutil.copy2(mask_file, masks_dir / mask_file.name)

    print(f"✅ Immagini copiate in: {images_dir}")
    print(f"✅ Maschere copiate in: {masks_dir}")
    return len(selected_files)

def run_colmap_simple(basedir, neus_path, camera_params, match_type='exhaustive_matcher', use_gpu=False):
    print("\n" + "="*70)
    print("ESECUZIONE COLMAP")
    print("="*70)

    neus_preprocess_path = Path(neus_path) / "preprocess_custom_data" / "colmap_preprocess"
    sys.path.insert(0, str(neus_preprocess_path))

    from colmap_wrapper_with_intrinsics import run_colmap

    run_colmap(
        basedir=str(basedir),
        match_type=match_type,
        camera_params=camera_params,
        use_gpu=use_gpu,
        mask_path=str(Path(basedir) / "masks")
    )

    print("\n✅ COLMAP completato")


def generate_poses(basedir, neus_path, match_type):
    import subprocess

    neus_preprocess_path = Path(neus_path) / "preprocess_custom_data" / "colmap_preprocess"
    imgs2poses_script = neus_preprocess_path / "imgs2poses.py"

    if not imgs2poses_script.exists():
        raise FileNotFoundError(f"Script non trovato: {imgs2poses_script}")

    print("📝 Eseguendo imgs2poses.py...")
    result = subprocess.run(
        ['python', str(imgs2poses_script), str(basedir), '--match_type', match_type],
        cwd=str(neus_preprocess_path),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"imgs2poses.py fallito con codice {result.returncode}")

    poses_file = Path(basedir) / "poses.npy"
    ply_file = Path(basedir) / "sparse_points.ply"

    if not poses_file.exists():
        raise RuntimeError(f"poses.npy non generato in: {basedir}")
    if not ply_file.exists():
        raise RuntimeError(f"sparse_points.ply non generato in: {basedir}")


def main():
    parser = argparse.ArgumentParser(
        description="COLMAP Runner per SPE3R-like datasets"
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
        help="Numero immagini da usare (default: 500)"
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
        help="Usa GPU per COLMAP (default: CPU - più stabile)"
    )
    
    parser.add_argument(
        "--skip-colmap",
        action="store_true",
        help="Salta COLMAP, esegui solo generazione poses (usa se COLMAP già completato)"
    )

    parser.add_argument(
    "--match-type",
    default="exhaustive_matcher",
    choices=["exhaustive_matcher", "sequential_matcher"],
    help="Tipo di matcher COLMAP (default: exhaustive_matcher)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    print("="*70)
    print("COLMAP RUNNER")
    print("="*70)
    print(f"Dataset: {args.spe3r_path}")
    print(f"Satellite: {args.satellite}")
    print(f"Output: {args.output}")
    print(f"NeuS path: {args.neus_path}")
    print(f"Range immagini: {args.start_idx} → {args.start_idx + args.num_images - 1}")
    print(f"GPU: {'ON' if args.use_gpu else 'OFF (CPU)'}")
    print(f"Matcher: {args.match_type}")
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
        print("⏭️  SKIP: COLMAP già completato")
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
        
        # Step 2: COLMAP
        print("\n" + "="*70)
        print("STEP 2: COLMAP (FEATURE EXTRACTION + MATCHING + SFM)")
        print("="*70)
        run_colmap_simple(
            basedir=output_path,
            neus_path=args.neus_path,
            camera_params=camera_params,
            match_type=args.match_type,
            use_gpu=args.use_gpu
        )
            
    # Step 3: Genera poses
    print("\n" + "="*70)
    print("STEP 3: GENERAZIONE POSES E SPARSE POINTS")
    print("="*70)
    generate_poses(
        basedir=output_path,
        neus_path=args.neus_path,
        match_type=args.match_type
    )
    
    # Summary
    print("\n" + "="*70)
    print("✅ PROCESSING COMPLETATO")
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
    
if __name__ == "__main__":
    main()
