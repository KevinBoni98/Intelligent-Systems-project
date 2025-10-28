#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

# Mapping full → short
full_to_short = {
    "ace of clubs": "Ac", "ace of diamonds": "Ad", "ace of hearts": "Ah", "ace of spades": "As",
    "two of clubs": "2c", "two of diamonds": "2d", "two of hearts": "2h", "two of spades": "2s",
    "three of clubs": "3c", "three of diamonds": "3d", "three of hearts": "3h", "three of spades": "3s",
    "four of clubs": "4c", "four of diamonds": "4d", "four of hearts": "4h", "four of spades": "4s",
    "five of clubs": "5c", "five of diamonds": "5d", "five of hearts": "5h", "five of spades": "5s",
    "six of clubs": "6c", "six of diamonds": "6d", "six of hearts": "6h", "six of spades": "6s",
    "seven of clubs": "7c", "seven of diamonds": "7d", "seven of hearts": "7h", "seven of spades": "7s",
    "eight of clubs": "8c", "eight of diamonds": "8d", "eight of hearts": "8h", "eight of spades": "8s",
    "nine of clubs": "9c", "nine of diamonds": "9d", "nine of hearts": "9h", "nine of spades": "9s",
    "ten of clubs": "10c", "ten of diamonds": "10d", "ten of hearts": "10h", "ten of spades": "10s",
    "jack of clubs": "Jc", "jack of diamonds": "Jd", "jack of hearts": "Jh", "jack of spades": "Js",
    "queen of clubs": "Qc", "queen of diamonds": "Qd", "queen of hearts": "Qh", "queen of spades": "Qs",
    "king of clubs": "Kc", "king of diamonds": "Kd", "king of hearts": "Kh", "king of spades": "Ks",
}

# Fix comuni
typo_fixes = {"eigth": "eight"}
SUIT_BY_PREFIX = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}

def normalize_label(s: str) -> str:
    s = s.strip().lower()
    for bad, good in typo_fixes.items():
        s = s.replace(bad, good)
    return " ".join(s.split())

def map_full_to_short(label: str, filename: str) -> str | None:
    norm = normalize_label(label)
    if norm in full_to_short:
        return full_to_short[norm]
    # Ripara casi tipo “seven of seven” → deduci seme dal filename
    if norm == "seven of seven":
        prefix = Path(filename).stem[0].lower()
        suit = SUIT_BY_PREFIX.get(prefix)
        if suit:
            repaired = f"seven of {suit}"
            return full_to_short.get(repaired)
    return None

def get_image_stats(image_path):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        grayscale = np.array(im.convert("L"), dtype=np.float32)
        brightness = grayscale.mean() / 255.0
        arr = np.asarray(gray, dtype=np.float32)
        p2, p98 = np.percentile(arr, (2, 98))
        contrast = (p98 - p2) / 255.0
        return im.width, im.height, brightness, contrast

def convert_voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

def parse_voc_csv(csv_path: Path, base_dir: Path, convert_labels: bool = False) -> pd.DataFrame:
    df_csv = pd.read_csv(csv_path)
    all_rows = []
    skipped = 0

    for _, row in df_csv.iterrows():
        fname = str(row["filename"]).strip()
        label = str(row["class"]).strip()
        img_path = base_dir / fname
        short_label = label

        if convert_labels:
            short_label = map_full_to_short(label, fname)
            if not short_label:
                print(f"⚠️ Skipping unknown label '{label}' in {fname}")
                skipped += 1
                continue

        if not img_path.exists():
            print(f"⚠️ Missing image {img_path}")
            skipped += 1
            continue

        try:
            w, h, brightness, contrast = get_image_stats(img_path)
        except Exception as e:
            print(f"⚠️ Could not process {img_path}: {e}")
            w, h, brightness, contrast = None, None, None, None

        x_center, y_center, width, height = convert_voc_to_yolo(
            float(row["xmin"]), float(row["ymin"]),
            float(row["xmax"]), float(row["ymax"]),
            w or 1, h or 1
        )

        all_rows.append([
            str(img_path), short_label, x_center, y_center, width, height,
            w, h, brightness, contrast
        ])


    df = pd.DataFrame(all_rows, columns=[
        "image", "class_name", "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height",
        "image_width", "image_height", "brightness", "contrast"
    ])
    print(f"{csv_path}: {len(df)} valid annotations, skipped {skipped}.")
    return df

def main():
    parser = argparse.ArgumentParser(description="Process VOC-style playing card dataset.")
    parser.add_argument("--train_csv", required=True, help="Path to train_labels.csv")
    parser.add_argument("--test_csv", required=True, help="Path to test_labels.csv")
    parser.add_argument("--train_dir", required=True, help="Path to train image folder")
    parser.add_argument("--test_dir", required=True, help="Path to test image folder")
    parser.add_argument("--output_dir", required=True, help="Output directory for CSV")
    parser.add_argument("--convert_labels", required=False, action="store_true", help="Convert full labels to short labels")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = parse_voc_csv(Path(args.train_csv), Path(args.train_dir), convert_labels=args.convert_labels)
    test_df = parse_voc_csv(Path(args.test_csv), Path(args.test_dir), convert_labels=args.convert_labels)
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    out_path = output_dir / "dataset_converted.csv"
    all_df.to_csv(out_path, index=False)
    print(f"\n✅ All: {len(all_df)} rows → {out_path}")

if __name__ == "__main__":
    main()