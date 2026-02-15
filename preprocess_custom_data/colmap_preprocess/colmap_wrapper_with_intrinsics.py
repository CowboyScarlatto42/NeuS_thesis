"""
colmap_wrapper_with_intrinsics.py

Versione modificata di colmap_wrapper.py di NeuS che permette di
IMPORRE gli intrinseci della camera invece di lasciarli stimare a COLMAP.

Per SPE3R: Impostiamo fx=1277.37, fy=1277.37, cx=128, cy=128

FIXED: Aggiunto QT_QPA_PLATFORM=offscreen per ambienti headless (Colab)
"""

import os
import subprocess


def run_colmap(basedir, match_type, camera_params=None, use_gpu=True, colmap_extra_args=None, 
               gpu_matching_only=False):
    """
    Esegue COLMAP con possibilità di imporre parametri camera.
    
    Args:
        basedir: Directory con sottocartella images/
        match_type: 'exhaustive_matcher', 'vocab_tree_matcher', 'sequential_matcher', 'spatial_matcher'
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
        colmap_extra_args: Dict con flag aggiuntivi per step, es:
            {
                'feature_extractor': ['--Flag', 'value', ...],
                'matcher': ['--Flag', 'value', ...]
            }
            Se None, usa default aggressivi per SIFT.
        gpu_matching_only: Se True, usa CPU per extraction ma GPU per matching (default: False)
                          Utile in ambienti headless come Colab dove GPU extraction fallisce
    """
    
    # Default aggressivi per SIFT se colmap_extra_args non è fornito
    if colmap_extra_args is None:
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
        print("🔧 Usando impostazioni SIFT aggressive (default)")
    
    # FIX per ambienti headless (Colab, server senza display)
    # COLMAP usa Qt che cerca un display anche quando non necessario
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Determina uso GPU per feature extraction
    use_gpu_extraction = use_gpu and not gpu_matching_only
    use_gpu_match = use_gpu  # GPU matching funziona sempre (no OpenGL)
    
    if gpu_matching_only:
        print("🔄 Modalità GPU SELETTIVA:")
        print("   - Feature Extraction: CPU (evita errori OpenGL)")
        print("   - Feature Matching: GPU (accelerazione CUDA)")
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    # Feature extraction
    feature_extractor_args = [
        'colmap', 'feature_extractor', 
        '--database_path', os.path.join(basedir, 'database.db'), 
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.single_camera', '1',  # ← UNA SOLA CAMERA per tutte le immagini
    ]
    
    # GPU per feature extraction (solo se non in modalità gpu_matching_only)
    if use_gpu_extraction:
        feature_extractor_args.extend([
            '--SiftExtraction.use_gpu', '1',
        ])
        print("🎮 GPU abilitata per feature extraction")
    else:
        feature_extractor_args.extend([
            '--SiftExtraction.use_gpu', '0',
        ])
        if gpu_matching_only:
            print("💻 CPU usata per feature extraction (GPU matching only mode)")
        else:
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
            '--ImageReader.camera_params', params_str,
        ])
        
        print(f"📷 Intrinseci IMPOSTI (uguali per tutte le immagini):")
        print(f"   Modello: {model}")
        print(f"   Parametri: {params_str}")
        print(f"   ⚠️  NOTA: Con database nuovo, COLMAP userà questi come iniziali")
        print(f"   ⚠️  Durante mapper potrebbero essere leggermente raffinati")
    else:
        print("⚠️  Intrinseci NON imposti - COLMAP li stimerà")
    
    # Aggiungi extra args per feature_extractor
    if colmap_extra_args and 'feature_extractor' in colmap_extra_args:
        feature_extractor_args.extend(colmap_extra_args['feature_extractor'])
    
    feat_output = subprocess.check_output(
        feature_extractor_args, 
        universal_newlines=True
    )
    logfile.write(feat_output)
    print('✅ Features extracted')

    # Feature matching
    matcher_args = [
        'colmap', match_type, 
        '--database_path', os.path.join(basedir, 'database.db'), 
    ]
    
    # Per vocab_tree_matcher, verifica che vocab_tree_path sia presente
    if match_type == 'vocab_tree_matcher':
        # Cerca vocab_tree_path in colmap_extra_args['matcher']
        vocab_tree_path = None
        if colmap_extra_args and 'matcher' in colmap_extra_args:
            extra_matcher_args = colmap_extra_args['matcher']
            if '--VocabTreeMatching.vocab_tree_path' in extra_matcher_args:
                idx = extra_matcher_args.index('--VocabTreeMatching.vocab_tree_path')
                vocab_tree_path = extra_matcher_args[idx + 1]
        
        if vocab_tree_path:
            print(f"🌳 Usando vocabulary tree: {vocab_tree_path}")
        else:
            print("⚠️  ATTENZIONE: vocab_tree_matcher richiede --VocabTreeMatching.vocab_tree_path")
            print("   Scarica vocab tree con:")
            print("   python setup_vocab_tree.py --download --output vocab_tree.bin")
            print("   Poi passa nel colmap_extra_args['matcher']")
    
    # GPU per matching (funziona sempre, anche in Colab)
    if use_gpu_match:
        matcher_args.extend([
            '--SiftMatching.use_gpu', '1',
        ])
        if gpu_matching_only:
            print("🎮 GPU abilitata per feature matching (CUDA, no OpenGL)")
        else:
            print("🎮 GPU abilitata per feature matching")
    else:
        matcher_args.extend([
            '--SiftMatching.use_gpu', '0',
        ])
        print("💻 Uso CPU per feature matching")
    
    # Aggiungi extra args per matcher
    if colmap_extra_args and 'matcher' in colmap_extra_args:
        matcher_args.extend(colmap_extra_args['matcher'])

    match_output = subprocess.check_output(
        matcher_args, 
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
    ]
    
    # Aggiungi mapper settings da colmap_extra_args se presenti
    if colmap_extra_args and 'mapper' in colmap_extra_args:
        mapper_args.extend(colmap_extra_args['mapper'])
        print("🔧 Mapper: usando impostazioni custom")
    else:
        # Usa defaults ottimizzati per SPE3R
        mapper_args.extend([
            # === INITIALIZATION (permissivo per pose random) ===
            '--Mapper.init_min_tri_angle', '2.0',
            '--Mapper.init_max_forward_motion', '0.95',
            '--Mapper.init_min_num_inliers', '30',
            
            # === TRIANGULATION ===
            '--Mapper.tri_min_angle', '1.5',
            '--Mapper.tri_ignore_two_view_tracks', '0',
            '--Mapper.tri_complete_max_reproj_error', '4.0',
            
            # === FILTERING ===
            '--Mapper.filter_max_reproj_error', '4.0',
            '--Mapper.filter_min_tri_angle', '1.5',
            
            # === OTHER ===
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
            '--Mapper.min_num_matches', '15',
        ])
    
    # Se abbiamo imposto camera_params, blocca il refinement degli intrinseci
    # (sempre, indipendentemente da colmap_extra_args)
    if camera_params is not None:
        # Rimuovi eventuali flag di refinement già presenti
        mapper_args = [arg for arg in mapper_args if 'ba_refine' not in arg]
        
        mapper_args.extend([
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.ba_local_max_num_iterations', '50',
            '--Mapper.ba_global_max_num_iterations', '100',
        ])
        print("🔒 Mapper: refinement intrinseci DISABILITATO (parametri bloccati)")
    else:
        mapper_args.extend([
            '--Mapper.ba_local_max_num_iterations', '25',
            '--Mapper.ba_global_max_num_iterations', '50',
        ])

    map_output = subprocess.check_output(
        mapper_args, 
        universal_newlines=True
    )
    logfile.write(map_output)
    logfile.close()
    print('✅ Sparse map created')
    
    print(f'✅ Finished COLMAP, see {logfile_name} for logs')
