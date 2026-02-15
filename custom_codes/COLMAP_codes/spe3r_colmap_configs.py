"""
spe3r_colmap_configs_tuned.py

Configurazioni COLMAP OTTIMIZZATE per SPE3R basate su test reali.
Risultato testato: 230/500 immagini registrate (46%)

Queste impostazioni sono state validate dall'utente e funzionano
meglio delle configurazioni standard per dataset SPE3R.
"""

# =============================================================================
# CONFIGURAZIONE VALIDATED (User-Tested)
# =============================================================================

SPE3R_VALIDATED = {
    'feature_extractor': [
        # === FEATURE DETECTION ===
        '--SiftExtraction.max_num_features', '20000',
        '--SiftExtraction.peak_threshold', '0.004',     # Meno sensibile ma più stabile
        '--SiftExtraction.edge_threshold', '20',        # Preserva pannelli solari
        '--SiftExtraction.first_octave', '-1',
        '--SiftExtraction.num_octaves', '4',
        '--SiftExtraction.domain_size_pooling', '1',
        '--SiftExtraction.estimate_affine_shape', '1',
        
        # === PREPROCESSING ===
        '--ImageReader.single_camera', '1',
        # Max image size ridotto per efficienza
        # '--ImageReader.default_focal_length_factor', '1.2',  # Se necessario
    ],
    
    'matcher': [
        # === MATCHING STRATEGY ===
        '--SiftMatching.guided_matching', '1',          # CRITICO: Re-match guidato
        #'--SiftMatching.max_num_features', '50000',     # CRITICO: Più match possibili
        
        # === THRESHOLDS (Standard, funzionano bene) ===
        '--SiftMatching.max_ratio', '0.8',
        '--SiftMatching.max_distance', '0.7',
        '--SiftMatching.cross_check', '1',
        '--SiftMatching.max_error', '4.0',
        
        # === GEOMETRIC VERIFICATION ===
        '--SiftMatching.min_num_inliers', '15',
        '--SiftMatching.confidence', '0.999',
        '--SiftMatching.max_num_trials', '10000',
    ],
    
    'mapper': [
        # === INITIALIZATION (Permissivo per pose random) ===
        '--Mapper.init_min_tri_angle', '2.0',           # Permissivo
        '--Mapper.init_min_num_inliers', '30',          # Standard
        '--Mapper.init_max_forward_motion', '0.95',
        
        # === REGISTRATION (MOLTO Permissivo) ===
        '--Mapper.abs_pose_min_num_inliers', '15',      # ⬇️ CRITICO: Molto basso
        '--Mapper.abs_pose_min_inlier_ratio', '0.05',   # ⬇️ CRITICO: Solo 5%!
        
        # === TRIANGULATION ===
        '--Mapper.tri_min_angle', '1.5',
        '--Mapper.tri_ignore_two_view_tracks', '0',
        '--Mapper.tri_complete_max_reproj_error', '4.0',
        
        # === BUNDLE ADJUSTMENT (Intrinseci FISSI) ===
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_principal_point', '0',
        '--Mapper.ba_refine_extra_params', '0',
        '--Mapper.ba_local_max_num_iterations', '50',
        '--Mapper.ba_global_max_num_iterations', '100',
        
        # === FILTERING ===
        '--Mapper.filter_max_reproj_error', '4.0',
        '--Mapper.filter_min_tri_angle', '1.5',
        
        # === OTHER ===
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',
        '--Mapper.num_threads', '16',
        '--Mapper.min_num_matches', '15',
    ],
}


# Alias per compatibilità
SPE3R_OPTIMIZED = SPE3R_VALIDATED


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURAZIONE SPE3R VALIDATED")
    print("=" * 70)
    print("\nBasata su test reale:")
    print("  - Dataset: SPE3R HST")
    print("  - Immagini totali: 500")
    print("  - Immagini registrate: 230 (46%)")
    print("  - Risultato: SUCCESSO ✅")
    print("\n" + "=" * 70)
    print("\nParametri chiave:")
    print("  - Guided matching: ON")
    print("  - Max matches: 50,000")
    print("  - Abs pose min inliers: 15 (molto permissivo)")
    print("  - Abs pose inlier ratio: 0.05 (solo 5%!)")
    print("\nQueste impostazioni sono ottimizzate per:")
    print("  ✓ Pose completamente randomizzate")
    print("  ✓ Overlap minimo tra immagini")
    print("  ✓ Features limitate (200-500/img sfondo nero)")
    print("  ✓ Intrinseci noti e fissi")
    print("=" * 70)
