"""
setup_vocab_tree.py

Script per configurare il vocabulary tree necessario per vocab_tree_matcher di COLMAP.

Opzioni:
1. Scarica vocab tree pre-addestrato (consigliato)
2. Genera vocab tree custom dal tuo dataset (lento, ma specifico)

Uso:
    # Opzione 1: Scarica pre-addestrato (CONSIGLIATO)
    python setup_vocab_tree.py --download --output /path/to/vocab_tree_flickr100K_words256K.bin
    
    # Opzione 2: Genera da dataset
    python setup_vocab_tree.py --generate --database /path/to/database.db --output custom_vocab.bin
"""

import os
import sys
import argparse
import subprocess
import urllib.request
from pathlib import Path


def download_pretrained_vocab_tree(output_path):
    """
    Scarica il vocabulary tree pre-addestrato da COLMAP.
    
    Questo è addestrato su 100K immagini Flickr ed è il default consigliato.
    """
    print("=" * 70)
    print("DOWNLOAD VOCABULARY TREE PRE-ADDESTRATO")
    print("=" * 70)
    
    # URL del vocab tree ufficiale COLMAP
    vocab_tree_url = "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Download in corso da:")
    print(f"   {vocab_tree_url}")
    print(f"📁 Destinazione:")
    print(f"   {output_path}")
    print(f"\n⏳ Attendere... (file ~600 MB)")
    
    try:
        # Download con progress bar
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r   Progresso: {percent}%")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(vocab_tree_url, output_path, reporthook)
        print("\n")
        
        # Verifica dimensione
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Download completato!")
        print(f"   Dimensione: {size_mb:.1f} MB")
        print(f"   Path: {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"\n❌ Errore durante il download: {e}")
        print("\n💡 Alternative:")
        print("1. Scarica manualmente da:")
        print("   https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin")
        print("2. Oppure genera un vocab tree custom (vedi --generate)")
        return None


def generate_custom_vocab_tree(database_path, output_path, vocab_tree_params=None):
    """
    Genera un vocabulary tree custom dal database COLMAP.
    
    NOTA: Questo è MOLTO lento (ore) ma può essere più accurato per dataset specifici.
    """
    print("=" * 70)
    print("GENERAZIONE VOCABULARY TREE CUSTOM")
    print("=" * 70)
    
    database_path = Path(database_path)
    output_path = Path(output_path)
    
    if not database_path.exists():
        print(f"❌ Database non trovato: {database_path}")
        print("💡 Esegui prima feature extraction per creare database.db")
        return None
    
    # Parametri default per vocab tree
    if vocab_tree_params is None:
        vocab_tree_params = {
            'num_words': 256000,      # Numero di visual words (default COLMAP)
            'branching': 128,          # Branching factor
            'num_iterations': 11,      # Iterazioni k-means
        }
    
    print(f"📊 Parametri vocab tree:")
    print(f"   Visual words: {vocab_tree_params['num_words']}")
    print(f"   Branching: {vocab_tree_params['branching']}")
    print(f"   Iterazioni: {vocab_tree_params['num_iterations']}")
    print(f"\n⚠️  ATTENZIONE: Questo processo può richiedere DIVERSE ORE")
    print(f"⚠️  Per dataset piccoli (<1000 immagini) usa sequential_matcher invece")
    
    response = input("\nContinuare? [y/N]: ")
    if response.lower() != 'y':
        print("Operazione annullata")
        return None
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FIX per ambienti headless
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    
    # Comando COLMAP
    vocab_tree_builder_args = [
        'colmap', 'vocab_tree_builder',
        '--database_path', str(database_path),
        '--vocab_tree_path', str(output_path),
        '--VocabTreeBuilder.num_words', str(vocab_tree_params['num_words']),
        '--VocabTreeBuilder.branching', str(vocab_tree_params['branching']),
        '--VocabTreeBuilder.num_iterations', str(vocab_tree_params['num_iterations']),
    ]
    
    print(f"\n🔧 Comando COLMAP:")
    print(f"   {' '.join(vocab_tree_builder_args)}")
    print(f"\n⏳ Generazione in corso...")
    
    try:
        result = subprocess.run(
            vocab_tree_builder_args,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Vocabulary tree generato!")
        print(f"   Output: {output_path}")
        
        # Verifica dimensione
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   Dimensione: {size_mb:.1f} MB")
        
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante la generazione:")
        print(e.stderr)
        return None


def update_colmap_wrapper(vocab_tree_path):
    """
    Mostra come usare il vocab tree nel wrapper.
    """
    print("\n" + "=" * 70)
    print("COME USARE IL VOCABULARY TREE")
    print("=" * 70)
    print("\nNel file colmap_wrapper_with_intrinsics.py, quando usi vocab_tree_matcher,")
    print("devi passare il path al vocab tree:")
    print("\n" + "-" * 70)
    print("# Nel matcher_args, aggiungi:")
    print(f"'--VocabTreeMatching.vocab_tree_path', '{vocab_tree_path}',")
    print("-" * 70)
    print("\nOppure modifica il default in colmap_extra_args:")
    print("\n" + "-" * 70)
    print("colmap_extra_args = {")
    print("    'matcher': [")
    print(f"        '--VocabTreeMatching.vocab_tree_path', '{vocab_tree_path}',")
    print("        '--VocabTreeMatching.num_images', '100',  # Match con top 100 immagini simili")
    print("        '--SiftMatching.guided_matching', '1',")
    print("        '--SiftMatching.max_ratio', '0.9',")
    print("    ]")
    print("}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Setup vocabulary tree per COLMAP vocab_tree_matcher"
    )
    
    # Opzioni mutuamente esclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--download",
        action="store_true",
        help="Scarica vocab tree pre-addestrato (CONSIGLIATO, ~600MB)"
    )
    group.add_argument(
        "--generate",
        action="store_true",
        help="Genera vocab tree custom dal database (MOLTO LENTO)"
    )
    
    parser.add_argument(
        "--output",
        required=True,
        help="Path output per vocab tree .bin"
    )
    
    # Solo per --generate
    parser.add_argument(
        "--database",
        help="Path a database.db COLMAP (richiesto per --generate)"
    )
    
    parser.add_argument(
        "--num-words",
        type=int,
        default=256000,
        help="Numero visual words (default: 256000, solo per --generate)"
    )
    
    parser.add_argument(
        "--branching",
        type=int,
        default=128,
        help="Branching factor (default: 128, solo per --generate)"
    )
    
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=11,
        help="Iterazioni k-means (default: 11, solo per --generate)"
    )
    
    args = parser.parse_args()
    
    # Download pre-addestrato
    if args.download:
        vocab_tree_path = download_pretrained_vocab_tree(args.output)
        if vocab_tree_path:
            update_colmap_wrapper(vocab_tree_path)
            print("\n✅ Setup completato!")
    
    # Genera custom
    elif args.generate:
        if not args.database:
            print("❌ Errore: --database richiesto per --generate")
            return
        
        vocab_tree_params = {
            'num_words': args.num_words,
            'branching': args.branching,
            'num_iterations': args.num_iterations,
        }
        
        vocab_tree_path = generate_custom_vocab_tree(
            args.database,
            args.output,
            vocab_tree_params
        )
        
        if vocab_tree_path:
            update_colmap_wrapper(vocab_tree_path)
            print("\n✅ Setup completato!")


if __name__ == "__main__":
    main()
