#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS

def find_images(root: Path, recursive: bool) -> list[Path]:
    if root.is_file() and is_image(root):
        return [root]
    if recursive:
        return [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    return [p for p in root.glob("*") if p.is_file() and is_image(p)]

def low_contrast_mask(L: np.ndarray, threshold: float) -> bool:
    # L expected in [0,255], uint8
    p2, p98 = np.percentile(L, (2, 98))
    return (p98 - p2) < threshold

def stretch_channel(L: np.ndarray, low: float, high: float) -> np.ndarray:
    L = L.astype(np.float32)
    denom = max(high - low, 1e-6)
    L = (L - low) * (255.0 / denom)
    return np.clip(L, 0, 255).astype(np.uint8)

def apply_contrast(img: np.ndarray, method: str, clip_limit: float, tile: int) -> np.ndarray:
    # Handle grayscale and color via LAB
    if img.ndim == 2 or img.shape[2] == 1:
        L = img if img.ndim == 2 else img[..., 0]
        if method in ("stretch", "both"):
            p2, p98 = np.percentile(L, (2, 98))
            L = stretch_channel(L, p2, p98)
        if method in ("clahe", "both"):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
            L = clahe.apply(L)
        return L
    # Color image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    if method in ("stretch", "both"):
        p2, p98 = np.percentile(L, (2, 98))
        L = stretch_channel(L, p2, p98)
    if method in ("clahe", "both"):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        L = clahe.apply(L)
    lab = cv2.merge([L, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def process_image(in_path: Path, out_root: Path, input_root: Path, args) -> tuple[bool, bool]:
    """Returns (saved, is_low_contrast)."""
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return (False, False)
    # Build output path preserving relative structure
    rel = in_path.relative_to(input_root) if in_path.is_relative_to(input_root) else in_path.name
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide low contrast using L channel or grayscale
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        L = img if img.ndim == 2 else img[..., 0]
    else:
        L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]

    is_low = low_contrast_mask(L, args.threshold)
    if args.only_low and not is_low:
        # Optionally copy without change
        if args.copy_if_skipped:
            cv2.imwrite(str(out_path), img)
            return (True, False)
        return (False, False)

    out = apply_contrast(img, args.method, args.clip_limit, args.tile)
    saved = cv2.imwrite(str(out_path), out)
    return (saved, is_low)

def main():
    parser = argparse.ArgumentParser(
        description="Aumenta il contrasto di immagini con CLAHE e/o contrast stretching."
    )
    parser.add_argument("--input", required=True, help="Cartella o file di input")
    parser.add_argument("--output", required=True, help="Cartella di output")
    parser.add_argument("--recursive", action="store_true", help="Scansione ricorsiva delle sottocartelle")
    parser.add_argument("--method", choices=["clahe", "stretch", "both"], default="both",
                        help="Tecnica di aumento del contrasto")
    parser.add_argument("--clip-limit", type=float, default=3.0, help="Clip limit per CLAHE")
    parser.add_argument("--tile", type=int, default=8, help="Dimensione della griglia CLAHE (tile x tile)")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Soglia in [0..255] per definire 'basso contrasto' sul canale L")
    parser.add_argument("--only-low", action="store_true",
                        help="Elabora solo immagini a basso contrasto secondo la soglia")
    parser.add_argument("--copy-if-skipped", action="store_true",
                        help="Copia le immagini non elaborate nella cartella di output")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    images = find_images(in_path, args.recursive)
    if not images:
        print("Nessuna immagine trovata.")
        return

    total = 0
    saved = 0
    low = 0
    for p in images:
        total += 1
        ok, is_low = process_image(p, out_root, in_path if in_path.is_dir() else p.parent, args)
        if ok:
            saved += 1
        if is_low:
            low += 1

    print(f"Immagini trovate: {total}")
    print(f"A basso contrasto (stimato): {low}")
    print(f"Salvate: {saved}")
    print("Fatto.")

if __name__ == "__main__":
    main()
