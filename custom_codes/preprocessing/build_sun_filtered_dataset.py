#!/usr/bin/env python3
"""
build_filtered_dataset.py

Costruisce un dataset filtrato a partire dagli indici accettati
prodotti da filter_by_phase_angle.py.

Output:
    <output_dir>/img/             immagini dei frame accettati rinominate 000000.png, ...
    <output_dir>/masks/           maschere dei frame accettati rinominate mask_000000_0001.png, ...
    <output_dir>/geometry.json    geometry.json con solo i frame accettati

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
                        help="Directory maschere originali (mask_000000_0001.png, ...)")
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
    out_images = args.output / "img"
    out_masks  = args.output / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Copia immagini e maschere
    print("\nCopia immagini e maschere...")
    missing_masks = []
    for new_idx, src_idx in enumerate(accepted):
        src_img = all_images[int(src_idx)]
        dst_img_name = f"{new_idx:06d}.png"
        shutil.copy2(src_img, out_images / dst_img_name)

        # Le maschere in input seguono il pattern mask_{frame:06d}_{seq:04d}.png.
        # In output rinominiamo i frame filtrati usando il nuovo indice immagine.
        mask_glob = f"mask_{int(src_idx):06d}_*.png"
        src_masks = sorted(args.masks.glob(mask_glob))

        if not src_masks:
            missing_masks.append(dst_img_name)
            continue

        for mask_seq, src_mask in enumerate(src_masks, start=1):
            dst_mask_name = f"mask_{new_idx:06d}_{mask_seq:04d}.png"
            shutil.copy2(src_mask, out_masks / dst_mask_name)

    print(f"  Immagini copiate: {len(accepted)}")
    if missing_masks:
        print(f"  [WARN] Maschere mancanti ({len(missing_masks)}): {missing_masks}")
    else:
        print(f"  Maschere copiate: {len(list(out_masks.glob('*.png')))}")

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
