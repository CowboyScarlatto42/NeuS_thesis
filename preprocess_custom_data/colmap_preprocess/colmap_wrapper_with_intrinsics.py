"""
colmap_wrapper_with_intrinsics.py

Versione modificata di colmap_wrapper.py di NeuS che permette di
IMPORRE gli intrinseci della camera invece di lasciarli stimare a COLMAP.

Per SPE3R: Impostiamo fx=1277.37, fy=1277.37, cx=128, cy=128

FIXED: Aggiunto QT_QPA_PLATFORM=offscreen per ambienti headless (Colab)
"""

import os
import subprocess


def run_colmap(basedir, match_type, camera_params=None, use_gpu=True):
    """
    Esegue COLMAP con possibilità di imporre parametri camera.
    
    Args:
        basedir: Directory con sottocartella images/
        match_type: 'exhaustive_matcher' o 'sequential_matcher'
        camera_params: Dict con parametri camera UGUALI PER TUTTE LE IMMAGINI:
            {
                'model': 'PINHOLE',  # o 'SIMPLE_PINHOLE'
                'fx': 3003.0,
                'fy': 3003.0,
                'cx': 128.0,
                'cy': 128.0
            }
            Se None, COLMAP stima automaticamente
        use_gpu: Se True, usa GPU per feature extraction (default: True)
    """
    
    # FIX per ambienti headless (Colab, server senza display)
    # COLMAP usa Qt che cerca un display anche quando non necessario
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    # Feature extraction
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
        '--database_path', os.path.join(basedir, 'database.db'), 
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',  # ← UNA SOLA CAMERA per tutte le immagini
    ]
    
    # GPU per feature extraction
    if use_gpu:
        feature_extractor_args.extend([
            '--SiftExtraction.use_gpu', '1',
        ])
        print("🎮 GPU abilitata per feature extraction")
    else:
        feature_extractor_args.extend([
            '--SiftExtraction.use_gpu', '0',
        ])
        print("💻 Uso CPU per feature extraction")
    
    # Imponi intrinseci se forniti (SPE3R case)
    if camera_params is not None:
        model = camera_params.get('model', 'PINHOLE')
        feature_extractor_args.extend([
            '--ImageReader.camera_model', model,
        ])
        
        # Costruisci stringa parametri (per SPE3R: fx=fy=1277.37, cx=cy=128)
        if model == 'PINHOLE':
            params_str = '{},{},{},{}'.format(
                camera_params['fx'],
                camera_params['fy'],
                camera_params['cx'],
                camera_params['cy']
            )
        elif model == 'SIMPLE_PINHOLE':
            # Se fx=fy possiamo usare SIMPLE_PINHOLE
            params_str = '{},{},{}'.format(
                camera_params.get('f', camera_params.get('fx')),
                camera_params['cx'],
                camera_params['cy']
            )
        else:
            raise ValueError(f"Modello {model} non supportato")
        
        feature_extractor_args.extend([
            '--ImageReader.camera_params', params_str
        ])
        
        print(f"📷 Intrinseci IMPOSTI (uguali per tutte le immagini):")
        print(f"   Modello: {model}")
        print(f"   Parametri: {params_str}")
    else:
        print("⚠️  Intrinseci NON imposti - COLMAP li stimerà")
    
    feat_output = subprocess.check_output(
        feature_extractor_args, 
        universal_newlines=True
    )
    logfile.write(feat_output)
    print('✅ Features extracted')

    # Feature matching
    exhaustive_matcher_args = [
        'colmap', match_type, 
        '--database_path', os.path.join(basedir, 'database.db'), 
    ]
    
    # GPU per matching
    if use_gpu:
        exhaustive_matcher_args.extend([
            '--SiftMatching.use_gpu', '1',
        ])
        print("🎮 GPU abilitata per feature matching")
    else:
        exhaustive_matcher_args.extend([
            '--SiftMatching.use_gpu', '0',
        ])
        print("💻 Uso CPU per feature matching")

    match_output = subprocess.check_output(
        exhaustive_matcher_args, 
        universal_newlines=True
    )
    logfile.write(match_output)
    print('✅ Features matched')
    
    # Crea directory sparse se non esiste
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    # Mapper (Structure from Motion)
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', os.path.join(basedir, 'sparse'),
        '--Mapper.num_threads', '16',
        '--Mapper.init_min_tri_angle', '4',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',
    ]

    map_output = subprocess.check_output(
        mapper_args, 
        universal_newlines=True
    )
    logfile.write(map_output)
    logfile.close()
    print('✅ Sparse map created')
    
    print(f'✅ Finished COLMAP, see {logfile_name} for logs')
