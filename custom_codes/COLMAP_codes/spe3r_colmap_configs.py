"""
spe3r_colmap_configs.py

Configurazioni COLMAP ottimizzate per dataset SPE3R.

Dataset SPE3R:
- Immagini sintetiche renderizzate in Unreal Engine
- Pose completamente randomizzate (NON sequenziali)
- Sfondo nero per prime 500 immagini
- Satelliti con pannelli solari (strutture sottili, ripetitive)
- Immagini 256x256, grayscale

Usa queste configurazioni nel file process_spe3r_single_sat.py
modificando la funzione run_colmap_with_intrinsics()
"""

# =============================================================================
# CONFIGURAZIONE 1: AGGRESSIVE (Consigliata per SPE3R)
# =============================================================================
# Massimizza il numero di features e matches per catturare strutture sottili
# come pannelli solari e antenne

SPE3R_AGGRESSIVE = {
    'feature_extractor': [
        # === FEATURE DETECTION ===
        '--SiftExtraction.max_num_features', '30000',  # ⬆️ Da 20k a 30k (più features)
        '--SiftExtraction.peak_threshold', '0.001',     # ⬇️ Da 0.002 a 0.001 (più sensibile)
        '--SiftExtraction.edge_threshold', '15',        # ⬆️ Da 10 a 15 (meno edge rejection)
        
        # === MULTI-SCALE ===
        '--SiftExtraction.first_octave', '-1',          # Inizia da risoluzione più alta
        '--SiftExtraction.num_octaves', '5',            # Più scale (default 4)
        # NOTA: octave_resolution potrebbe non essere supportato in COLMAP vecchio
        # '--SiftExtraction.octave_resolution', '4',    # Più immagini per octave (default 3)
        
        # === ADVANCED ===
        '--SiftExtraction.domain_size_pooling', '1',    # Migliora robustezza
        '--SiftExtraction.estimate_affine_shape', '1',  # Gestisce deformazioni prospettiche
        '--SiftExtraction.max_num_orientations', '2',   # Features con orientamenti multipli
        
        # === PREPROCESSING ===
        # NOTA: Questi parametri sono disponibili solo in COLMAP recente (>3.7)
        # '--ImageReader.camera_mask_path', '',         # No mask (sfondo già nero)
        # '--SiftExtraction.normalization', '1',        # Normalizza contrasto
    ],
    
    'matcher': [
        # === MATCHING STRATEGY ===
        '--SiftMatching.guided_matching', '1',          # Re-match con geometric verification
        # '--SiftMatching.multiple_models', '0',        # Non disponibile in tutte le versioni
        
        # === THRESHOLDS ===
        '--SiftMatching.max_ratio', '0.85',             # ⬇️ Da 0.9 a 0.85 (più selettivo)
        '--SiftMatching.max_distance', '0.75',          # ⬇️ Da 0.8 a 0.75 (più stringente)
        '--SiftMatching.cross_check', '1',              # Match bidirezionali
        '--SiftMatching.max_error', '4.0',              # Errore reproj in pixels
        
        # === GEOMETRIC VERIFICATION ===
        '--SiftMatching.min_num_inliers', '15',         # Minimo 15 inliers (default 15)
        '--SiftMatching.confidence', '0.999',           # Alta confidenza RANSAC
        '--SiftMatching.max_num_trials', '10000',       # Più iterazioni RANSAC
        '--SiftMatching.min_inlier_ratio', '0.25',      # Min 25% inliers
    ],
}


# =============================================================================
# CONFIGURAZIONE 2: BALANCED (Più veloce, comunque buona)
# =============================================================================
# Bilanciamento tra qualità e velocità

SPE3R_BALANCED = {
    'feature_extractor': [
        '--SiftExtraction.max_num_features', '20000',
        '--SiftExtraction.peak_threshold', '0.002',
        '--SiftExtraction.edge_threshold', '12',
        '--SiftExtraction.first_octave', '-1',
        '--SiftExtraction.num_octaves', '4',
        '--SiftExtraction.domain_size_pooling', '1',
        '--SiftExtraction.estimate_affine_shape', '1',
    ],
    
    'matcher': [
        '--SiftMatching.guided_matching', '1',
        '--SiftMatching.max_ratio', '0.8',
        '--SiftMatching.max_distance', '0.7',
        '--SiftMatching.cross_check', '1',
        '--SiftMatching.min_num_inliers', '15',
        '--SiftMatching.confidence', '0.999',
    ],
}


# =============================================================================
# CONFIGURAZIONE 3: CONSERVATIVE (Per strutture molto sottili/difficili)
# =============================================================================
# Massima detection, matching molto selettivo
# Usa se AGGRESSIVE non produce abbastanza match

SPE3R_CONSERVATIVE = {
    'feature_extractor': [
        # MASSIMA DETECTION
        '--SiftExtraction.max_num_features', '40000',   # 🔥 40k features!
        '--SiftExtraction.peak_threshold', '0.0008',    # 🔥 Super sensibile
        '--SiftExtraction.edge_threshold', '20',
        '--SiftExtraction.first_octave', '-1',
        '--SiftExtraction.num_octaves', '6',            # 6 scale
        # '--SiftExtraction.octave_resolution', '5',    # Non supportato in COLMAP vecchio
        '--SiftExtraction.domain_size_pooling', '1',
        '--SiftExtraction.estimate_affine_shape', '1',
        '--SiftExtraction.max_num_orientations', '3',
    ],
    
    'matcher': [
        # MATCHING MOLTO SELETTIVO
        '--SiftMatching.guided_matching', '1',
        '--SiftMatching.max_ratio', '0.75',             # 🔥 Molto stringente
        '--SiftMatching.max_distance', '0.65',
        '--SiftMatching.cross_check', '1',
        '--SiftMatching.min_num_inliers', '20',         # 🔥 Più inliers richiesti
        '--SiftMatching.confidence', '0.9999',          # 🔥 Confidenza altissima
        '--SiftMatching.max_num_trials', '20000',
        '--SiftMatching.min_inlier_ratio', '0.3',
    ],
}


# =============================================================================
# CONFIGURAZIONE 4: FAST (Per test rapidi)
# =============================================================================
# Meno features, più veloce - solo per validazione/debug

SPE3R_FAST = {
    'feature_extractor': [
        '--SiftExtraction.max_num_features', '10000',
        '--SiftExtraction.peak_threshold', '0.004',
        '--SiftExtraction.edge_threshold', '10',
        '--SiftExtraction.first_octave', '0',           # No upsampling
        '--SiftExtraction.num_octaves', '3',
    ],
    
    'matcher': [
        '--SiftMatching.guided_matching', '0',          # Skip guided matching
        '--SiftMatching.max_ratio', '0.8',
        '--SiftMatching.cross_check', '1',
    ],
}


# =============================================================================
# MAPPER SETTINGS (Bundle Adjustment)
# =============================================================================
# Ottimizzazioni per mapper con intrinseci FISSI

SPE3R_MAPPER_SETTINGS = [
    # === INITIALIZATION ===
    '--Mapper.init_min_tri_angle', '2.0',              # ⬇️ Da 4 a 2 (più permissivo per pose random)
    '--Mapper.init_max_forward_motion', '0.95',         # Previene inizializzazione degenere
    '--Mapper.init_min_num_inliers', '30',              # Min inliers per inizializzazione
    
    # === TRIANGULATION ===
    '--Mapper.tri_min_angle', '1.5',                    # ⬇️ Angolo minimo triangolazione
    '--Mapper.tri_ignore_two_view_tracks', '0',         # Usa anche tracks a 2 viste
    '--Mapper.tri_complete_max_reproj_error', '4.0',    # Errore reproj massimo
    
    # === BUNDLE ADJUSTMENT (con intrinseci FISSI) ===
    '--Mapper.ba_refine_focal_length', '0',             # 🔒 NON raffinare focal length
    '--Mapper.ba_refine_principal_point', '0',          # 🔒 NON raffinare principal point
    '--Mapper.ba_refine_extra_params', '0',             # 🔒 NON raffinare distorsione
    '--Mapper.ba_local_max_num_iterations', '50',       # Iterazioni BA locale
    '--Mapper.ba_global_max_num_iterations', '100',     # Iterazioni BA globale
    
    # === FILTERING ===
    '--Mapper.filter_max_reproj_error', '4.0',          # Rimuovi punti con errore > 4px
    '--Mapper.filter_min_tri_angle', '1.5',             # Rimuovi punti mal triangolati
    
    # === OTHER ===
    '--Mapper.multiple_models', '0',                    # Singolo modello
    '--Mapper.extract_colors', '0',                     # No colori (grayscale)
    '--Mapper.num_threads', '16',                       # Parallelizzazione
    '--Mapper.min_num_matches', '15',                   # Min match per registrare immagine
]


# =============================================================================
# VOCAB TREE MATCHER SETTINGS (se usi vocab_tree_matcher)
# =============================================================================

SPE3R_VOCAB_TREE_SETTINGS = [
    '--VocabTreeMatching.num_images', '100',            # Match con top 100 immagini simili
    '--VocabTreeMatching.num_nearest_neighbors', '5',   # 5 nearest neighbors per feature
    '--VocabTreeMatching.num_checks', '256',            # Checks durante search (default 256)
    '--VocabTreeMatching.max_num_features', '-1',       # Usa tutte le features
    
    # Poi aggiungi anche SiftMatching settings
    '--SiftMatching.guided_matching', '1',
    '--SiftMatching.max_ratio', '0.85',
    '--SiftMatching.max_distance', '0.75',
    '--SiftMatching.cross_check', '1',
]


# =============================================================================
# ESEMPI DI USO
# =============================================================================

"""
COME USARE QUESTE CONFIGURAZIONI:

1. Modifica process_spe3r_single_sat.py nella funzione run_colmap_with_intrinsics():

    from spe3r_colmap_configs import SPE3R_AGGRESSIVE, SPE3R_MAPPER_SETTINGS
    
    colmap_extra_args = {
        'feature_extractor': SPE3R_AGGRESSIVE['feature_extractor'],
        'matcher': SPE3R_AGGRESSIVE['matcher'],
    }
    
    # Poi aggiungi mapper settings nella chiamata a colmap mapper
    mapper_args.extend(SPE3R_MAPPER_SETTINGS)

2. Per vocab_tree_matcher:

    from spe3r_colmap_configs import SPE3R_AGGRESSIVE, SPE3R_VOCAB_TREE_SETTINGS
    
    colmap_extra_args = {
        'feature_extractor': SPE3R_AGGRESSIVE['feature_extractor'],
        'matcher': SPE3R_VOCAB_TREE_SETTINGS,  # ← Usa vocab tree settings
    }

3. Test progressivo:
   a) Prova FAST con 50 immagini (5 min)
   b) Se funziona, prova BALANCED con 100 immagini (30 min)
   c) Se va bene, usa AGGRESSIVE con 500 immagini (2-3 ore)
   d) Se hai ancora pochi match, prova CONSERVATIVE (4-5 ore)
"""


# =============================================================================
# DIAGNOSTICA: Come capire quale config usare
# =============================================================================

"""
DOPO AVER ESEGUITO COLMAP, controlla:

1. File: database.db
   Esegui: colmap database_stats --database_path database.db
   
   Guarda:
   - Features per image: se < 5000 → usa AGGRESSIVE o CONSERVATIVE
   - Matches per image pair: se < 100 → usa AGGRESSIVE
   
2. File: colmap_output.txt
   
   Cerca messaggi tipo:
   - "Registering image XXX" → Buono! Immagini registrate
   - "No matches" → PROBLEMA! Usa CONSERVATIVE
   - "Failed to register" → Prova AGGRESSIVE o cambia init_min_tri_angle

3. File: sparse/0/
   
   Controlla:
   - images.bin: quante immagini registrate su totale?
     Se < 80% → problemi con matching
   - points3D.bin: quanti punti 3D?
     Se < 1000 punti → usa CONSERVATIVE

REGOLA GENERALE:
- Se hai < 500 features/immagine → CONSERVATIVE
- Se hai 500-2000 features/img → AGGRESSIVE
- Se hai > 2000 features/img → BALANCED
- Se hai timeout → FAST (per debug)
"""


if __name__ == "__main__":
    print("=" * 70)
    print("CONFIGURAZIONI COLMAP PER SPE3R")
    print("=" * 70)
    print("\nConfigurations disponibili:")
    print("  1. SPE3R_AGGRESSIVE    - Consigliata (default)")
    print("  2. SPE3R_BALANCED      - Più veloce")
    print("  3. SPE3R_CONSERVATIVE  - Massima detection")
    print("  4. SPE3R_FAST          - Solo per test")
    print("\nMapper settings:")
    print("  - SPE3R_MAPPER_SETTINGS (sempre da usare)")
    print("\nVocab tree settings:")
    print("  - SPE3R_VOCAB_TREE_SETTINGS (se usi vocab_tree_matcher)")
    print("\n" + "=" * 70)
    print("\nVedi commenti nel file per dettagli e esempi d'uso.")
    print("=" * 70)
