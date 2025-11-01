import argparse
from pathlib import Path
import json
import pandas as pd
from PIL import Image
import numpy as np

COLUMNS = [
    "image","class_name",
    "bbox_x_center","bbox_y_center","bbox_width","bbox_height",
    "image_width","image_height","brightness","contrast"
]

SUIT_TO_LETTER = {
    "diamonds":"d","diamond":"d",
    "hearts":"h","heart":"h",
    "spades":"s","spade":"s",
    "clubs":"c","club":"c",
    "trefoils":"c","trefoil":"c"
}

def class_to_symbol(cat_name: str):
    s = cat_name.strip()
    if s.lower() in ("poker-cards","poker cards"):
        return None
    parts = s.replace("_"," ").split()
    if len(parts) < 2:
        return None
    rank, suit = parts[0], " ".join(parts[1:]).lower()
    suit_letter = None
    for k,v in SUIT_TO_LETTER.items():
        if k in suit:
            suit_letter = v
            break
    if suit_letter is None:
        return None
    rank_up = rank.upper()
    # A,J,Q,K or numbers 2..10
    if rank_up in {"A","J","Q","K"}:
        sym = f"{rank_up}{suit_letter}"
    else:
        sym = f"{rank}{suit_letter}"
    return sym

def image_stats(img_path: Path):
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.width, im.height
            gray = np.array(im.convert("L"), dtype=np.float32)
            brightness = float(gray.mean() / 255.0)
            p2, p98 = np.percentile(gray, (2,98))
            contrast = float((p98 - p2) / 255.0)
            return w, h, brightness, contrast
    except Exception:
        return None, None, None, None

def coco_to_rows(coco_path: Path, split_dir: Path):
    with open(coco_path, "r") as f:
        coco = json.load(f)
    # map id→image info
    imgs = {im["id"]: im for im in coco.get("images", [])}
    # map cat id→symbol
    cat_sym = {}
    for c in coco.get("categories", []):
        sym = class_to_symbol(c.get("name",""))
        if sym:  # save only valid classes
            cat_sym[c["id"]] = sym
    rows = []
    for ann in coco.get("annotations", []):
        cid = ann.get("category_id")
        if cid not in cat_sym:
            continue
        img = imgs.get(ann["image_id"])
        if not img:
            continue
        file_name = img["file_name"]
        img_path = split_dir / file_name
        w, h, brightness, contrast = image_stats(img_path)
        if not w or not h:
            continue
        # COCO bbox: [x_min, y_min, width, height]
        x, y, bw, bh = ann["bbox"]
        x_center = (x + bw/2.0) / w
        y_center = (y + bh/2.0) / h
        bw_n = bw / w
        bh_n = bh / h
        rows.append([
            str(img_path),
            cat_sym[cid],
            float(x_center), float(y_center), float(bw_n), float(bh_n),
            int(w), int(h), brightness, contrast
        ])
    return rows

def main():
    ap = argparse.ArgumentParser(description="Convert COCO dataset from three splits into dataset_converted.csv")
    ap.add_argument("--dataset", required=True, help="Folder containing train/, test/, valid/")
    args = ap.parse_args()
    base = Path(args.dataset)
    splits = ["train","test","valid"]
    all_rows = []
    for s in splits:
        split_dir = base / s
        coco_path = split_dir / "_annotations.coco.json"
        if coco_path.exists():
            all_rows.extend(coco_to_rows(coco_path, split_dir))
        else:
            print(f"WARNING: missing {coco_path}")
    if not all_rows:
        print("No annotations found")
        return
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    out_csv = base / "dataset_converted.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
