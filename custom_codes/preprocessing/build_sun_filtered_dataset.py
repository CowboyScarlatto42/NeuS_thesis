#!/usr/bin/env python3
"""
build_filtered_dataset.py

Costruisce un dataset filtrato a partire dagli indici accettati
prodotti da filter_by_phase_angle.py.

Output:
  <output_dir>/images/          immagini dei frame accettati
  <output_dir>/masks/           maschere dei frame accettati
  <output_dir>/geometry.json    geometry.json con solo i frame accettati

I nomi dei file vengono preservati (non rinominati sequenzialmente),
così restano compatibili con corto_to_spe3r_data.py e gli altri script.

USO:
    python build_filtered_dataset.py \\
        --accepted  accepted_frames.npy \\
        --images    /path/to/images \\
        --masks     /path/to/masks \\
        --geometry  geometry.json \\
        --output    /path/to/output
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Costruisce dataset filtrato da accepted_frames.npy."
    )
    parser.add_argument("--accepted",  required=True, type=Path,
                        help="Path a accepted_frames.npy")
    parser.add_argument("--images",    required=True, type=Path,
                        help="Directory immagini originali")
    parser.add_argument("--masks",     required=True, type=Path,
                        help="Directory maschere originali")
    parser.add_argument("--geometry",  required=True, type=Path,
                        help="Path a geometry.json originale")
    parser.add_argument("--output",    required=True, type=Path,
                        help="Directory output")
    return parser.parse_args()


def main():
    args = parse_args()

    # Carica indici accettati
    accepted = np.load(args.accepted).astype(int)
    print(f"Frame accettati: {len(accepted)}  {accepted.tolist()}")

    # Lista ordinata di tutte le immagini
    all_images = sorted(
        list(args.images.glob("*.png")) + list(args.images.glob("*.jpg"))
    )
    if len(all_images) == 0:
        raise FileNotFoundError(f"Nessuna immagine trovata in: {args.images}")

    print(f"Immagini totali disponibili: {len(all_images)}")

    if accepted.max() >= len(all_images):
        raise IndexError(
            f"Indice massimo accettato ({accepted.max()}) >= "
            f"numero immagini ({len(all_images)})"
        )

    # Crea cartelle output
    out_images = args.output / "images"
    out_masks  = args.output / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Copia immagini e maschere
    print("\nCopia immagini e maschere...")
    missing_masks = []
    for idx in accepted:
        src_img = all_images[idx]

        shutil.copy2(src_img, out_images / src_img.name)

        src_mask = args.masks / src_img.name
        if src_mask.exists():
            shutil.copy2(src_mask, out_masks / src_img.name)
        else:
            missing_masks.append(src_img.name)

    print(f"  Immagini copiate: {len(accepted)}")
    if missing_masks:
        print(f"  [WARN] Maschere mancanti ({len(missing_masks)}): {missing_masks}")
    else:
        print(f"  Maschere copiate: {len(accepted)}")

    # Costruisce geometry.json filtrato
    with open(args.geometry) as f:
        geometry = json.load(f)

    geo_filtered = {}
    for entity in geometry:
        geo_filtered[entity] = {}
        for field in geometry[entity]:
            geo_filtered[entity][field] = [
                geometry[entity][field][i] for i in accepted
            ]

    out_geo = args.output / "geometry.json"
    with open(out_geo, "w") as f:
        json.dump(geo_filtered, f, indent=2)
    print(f"\n  geometry.json salvato: {out_geo}")

    print("\n" + "=" * 50)
    print("FATTO")
    print("=" * 50)
    print(f"  Output:   {args.output}")
    print(f"  Immagini: {len(accepted)}")
    print(f"  Maschere: {len(accepted) - len(missing_masks)}")
    print(f"  Geometry: {len(accepted)} frame")


if __name__ == "__main__":
    main()
