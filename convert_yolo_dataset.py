import argparse
import os
from pathlib import Path
import yaml
import pandas as pd
from PIL import Image
import numpy as np

IMG_EXTS = [".jpg", ".jpeg", ".png"]

def find_image(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def get_image_stats(image_path):
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        grayscale = np.array(im.convert("L"), dtype=np.float32)
        brightness = grayscale.mean() / 255.0
        p2, p98 = np.percentile(grayscale, (2, 98))
        contrast = (p98 - p2) / 255.0
        return im.width, im.height, brightness, contrast

def load_yolo_split(base_dir: Path, class_names):
    labels_dir = base_dir / "labels"
    images_dir = base_dir / "images"
    if not labels_dir.exists():
        return pd.DataFrame()
    txt_files = sorted(labels_dir.glob("*.txt"))
    all_rows = []
    for txt_file in txt_files:
        stem = txt_file.stem
        image_path = find_image(images_dir, stem)
        if not image_path:
            continue
        try:
            w, h, brightness, contrast = get_image_stats(image_path)
        except Exception:
            w, h, brightness, contrast = None, None, None, None
        with open(txt_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, width, height = parts
                class_id = int(class_id)
                if class_id >= len(class_names):
                    continue
                class_name = class_names[class_id]
                all_rows.append([
                    str(image_path), class_name,
                    float(x_center), float(y_center), float(width), float(height),
                    w, h, brightness, contrast
                ])
    return pd.DataFrame(all_rows, columns=[
        "image", "class_name", "bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height",
        "image_width", "image_height", "brightness", "contrast"
    ])

def main():
    parser = argparse.ArgumentParser(description="Process YOLO dataset.")
    parser.add_argument("--dataset", required=True, help="Path to YOLO dataset base folder")
    parser.add_argument("--yaml", required=True, help="Path to YOLO data.yaml")
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        yolo_cfg = yaml.safe_load(f)
    class_names = yolo_cfg["names"]

    base_path = Path(args.dataset)
    splits = ["train", "valid", "test"]
    all_dfs = []
    for split in splits:
      df = load_yolo_split(base_path / split, class_names)
      if not df.empty:
        df['split'] = split
        all_dfs.append(df)
      print(f"{split}: {len(df)} annotations loaded")
    
    if all_dfs:
      combined_df = pd.concat(all_dfs, ignore_index=True)
      out_csv = base_path / "dataset_converted.csv"
      combined_df.to_csv(out_csv, index=False)
      print(f"Combined dataset: {len(combined_df)} annotations saved → {out_csv}")
    else:
      print("No data found in any split")

if __name__ == "__main__":
    main()
