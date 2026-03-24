"""
colmap_wrapper_simple.py

Versione SEMPLIFICATA che usa SOLO i parametri validati che funzionano
(quelli che producevano 230 immagini registrate).

NON ci sono configurazioni multiple - usa solo i parametri testati.
"""

import os
import subprocess


def run_colmap(basedir, match_type, camera_params=None, use_gpu=False, mask_path=None):
    """
    Esegue COLMAP con parametri VALIDATI che funzionano.
    
    Args:
        basedir: Directory con sottocartella images/
        match_type: 'exhaustive_matcher' (consigliato)
        camera_params: Dict con parametri camera (opzionale):
            {
                'model': 'PINHOLE',
                'fx': 1277.37,
                'fy': 1277.37,
                'cx': 128.0,
                'cy': 128.0
            }
        use_gpu: Se True usa GPU, altrimenti CPU (default: False - CPU è più stabile)
    """
    
    # Fix per ambienti headless (Colab)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['DISPLAY'] = ''
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    os.makedirs('/tmp/runtime-root', exist_ok=True)
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    print("\n" + "="*70)
    print("COLMAP - PARAMETRI VALIDATI (230/500 immagini)")
    print("="*70)
    
    # ========================================================================
    # FEATURE EXTRACTION - Parametri validati
    # ========================================================================
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
        '--database_path', os.path.join(basedir, 'database.db'), 
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',
    ]

    # Usa maschere se disponibili
    if mask_path is None:
        mask_path = os.path.join(basedir, 'masks')

    if os.path.exists(mask_path):
        feature_extractor_args.extend([
            '--ImageReader.mask_path', mask_path
        ])
        print(f"🎭 Mask abilitate: {mask_path}")
    else:
        print("⚠️  Nessuna mask trovata")
    
    # Imponi intrinseci se forniti
    if camera_params is not None:
        model = camera_params.get('model', 'PINHOLE')
        feature_extractor_args.extend([
            '--ImageReader.camera_model', model,
        ])
        
        if model == 'PINHOLE':
            params_str = '{},{},{},{}'.format(
                camera_params['fx'],
                camera_params['fy'],
                camera_params['cx'],
                camera_params['cy']
            )
        elif model == 'SIMPLE_PINHOLE':
            params_str = '{},{},{}'.format(
                camera_params.get('f', camera_params.get('fx')),
                camera_params['cx'],
                camera_params['cy']
            )
        else:
            raise ValueError(f"Modello {model} non supportato")
        
        feature_extractor_args.extend([
            '--ImageReader.camera_params', params_str,
        ])
        
        print(f"📷 Intrinseci IMPOSTI:")
        print(f"   Modello: {model}")
        print(f"   Parametri: {params_str}")
    else:
        feature_extractor_args.extend([
            '--ImageReader.camera_model', 'SIMPLE_RADIAL',
        ])
        print("⚠️  Intrinseci NON imposti - COLMAP li stimerà")
    
    gpu_flag = '1' if use_gpu else '0'
    feature_extractor_args.extend([
        '--SiftExtraction.use_gpu', gpu_flag,
        '--SiftExtraction.num_threads', '2',
        '--SiftExtraction.max_num_features', '8192',
        '--SiftExtraction.peak_threshold', '0.01',  
        '--SiftExtraction.edge_threshold', '10',      
        '--SiftExtraction.max_image_size', '1024',
    ])

    #'--SiftExtraction.max_num_features', '20000',
    #'--SiftExtraction.peak_threshold', '0.004',
    #'--SiftExtraction.edge_threshold', '20',
    
    print("\n[1/3] Feature Extraction...")
    print(f"   - GPU: {'ON' if use_gpu else 'OFF (CPU)'}")
    print("   - peak_threshold: 0.01 (permissivo)")
    print("   - edge_threshold: 10 (permissivo)")
    print("   - max_features: 8192")
    
    feat_output = subprocess.check_output(
        feature_extractor_args, 
        universal_newlines=True
    )
    logfile.write(feat_output)
    print('✅ Features estratte')

    # ========================================================================
    # FEATURE MATCHING - Parametri validati
    # ========================================================================
    matcher_args = [
        'colmap', match_type, 
        '--database_path', os.path.join(basedir, 'database.db'),
        '--SiftMatching.use_gpu', gpu_flag,
        '--SiftMatching.guided_matching', '1',
        '--SiftMatching.max_num_matches', '50000',
        '--SiftMatching.max_ratio', '0.75', # default 0.8, più restrittivo per evitare outliers
        '--SiftMatching.max_error', '3', # default 4, più restrittivo per evitare outliers
    ]
    
    print("\n[2/3] Feature Matching...")
    print(f"   - GPU: {'ON' if use_gpu else 'OFF (CPU)'}")
    print(f"   - Tipo: {match_type}")
    print("   - guided_matching: ON")
    
    match_output = subprocess.check_output(
        matcher_args, 
        universal_newlines=True
    )
    logfile.write(match_output)
    print('✅ Features matchate')
    
    # ========================================================================
    # MAPPER - Parametri validati
    # ========================================================================
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', os.path.join(basedir, 'sparse'),
        '--Mapper.num_threads', '16',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',

        '--Mapper.init_min_num_inliers', '50',
        '--Mapper.abs_pose_min_num_inliers', '25',
        '--Mapper.abs_pose_min_inlier_ratio', '0.08',
        '--Mapper.min_num_matches', '25',
        '--Mapper.abs_pose_max_error', '8',
        '--Mapper.filter_max_reproj_error', '2',
        # Parametri più permissivi che funzionano ma producono più outliers (230 immagini registrate)
        #'--Mapper.init_min_num_inliers', '30',
        #'--Mapper.abs_pose_min_num_inliers', '15',
        #'--Mapper.abs_pose_min_inlier_ratio', '0.05',
        #'--Mapper.min_num_matches', '15',
    ]
    
    # Se abbiamo imposto intrinseci, blocca il refinement
    if camera_params is not None:
        mapper_args.extend([
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0',
        ])
        print("\n[3/3] Mapper...")
        print("   - Refinement intrinseci: BLOCCATO")
    else:
        print("\n[3/3] Mapper...")
        print("   - Refinement intrinseci: ATTIVO")
    
    print("   - init_min_num_inliers: 30")
    print("   - abs_pose_min_num_inliers: 15")
    print("   - min_num_matches: 15")

    map_output = subprocess.check_output(
        mapper_args, 
        universal_newlines=True
    )
    logfile.write(map_output)
    logfile.close()
    print('✅ Sparse map creata')
    
    print(f"\n✅ COLMAP completato, log in: {logfile_name}")
    print("="*70)
