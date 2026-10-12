"""
ccpd_to_yolo.py

Converts the CCPD dataset (which encodes bounding boxes in the filename,
not in separate annotation files) into standard YOLO-format labels, and
splits into train/val folders ready for `yolo detect train`.

CCPD filename format (fields separated by '-'):
    area-tilt-bbox-vertices-platenumber-brightness-blur.jpg
Field 2 (0-indexed) is the axis-aligned bounding box:
    "x1&y1_x2&y2"  (top-left and bottom-right corners, in pixels)

Usage:
    python ccpd_to_yolo.py --source /path/to/CCPD2019/ccpd_base \
                            --dest /path/to/ccpd_yolo \
                            --val-split 0.1 \
                            --limit 10000

--limit is optional and highly recommended for a student project: CCPD has
250k+ images. A random subset of 5,000-15,000 images trains a perfectly
usable detector in a fraction of the time on a free-tier GPU.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Please run: pip install pillow")


def parse_ccpd_bbox(filename: str):
    """Extract (x1, y1, x2, y2) pixel bbox from a CCPD filename.
    Returns None if the filename doesn't match the expected pattern."""
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    try:
        bbox_str = parts[2]
        tl, br = bbox_str.split("_")
        x1, y1 = map(int, tl.split("&"))
        x2, y2 = map(int, br.split("&"))
        return x1, y1, x2, y2
    except (ValueError, IndexError):
        return None


def to_yolo_line(x1, y1, x2, y2, img_w, img_h, class_id=0):
    """Convert pixel bbox to normalised YOLO format: class cx cy w h."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def convert(source_dir, dest_dir, val_split=0.1, limit=None, seed=42):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    image_files = [f for f in source_dir.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    print(f"Found {len(image_files)} images in {source_dir}")

    random.seed(seed)
    random.shuffle(image_files)
    if limit:
        image_files = image_files[:limit]
        print(f"Limiting to {len(image_files)} images (--limit)")

    split_idx = int(len(image_files) * (1 - val_split))
    splits = {"train": image_files[:split_idx], "val": image_files[split_idx:]}

    for split_name, files in splits.items():
        img_out = dest_dir / "images" / split_name
        lbl_out = dest_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        skipped = 0
        for f in files:
            bbox = parse_ccpd_bbox(f.name)
            if bbox is None:
                skipped += 1
                continue
            try:
                with Image.open(f) as im:
                    img_w, img_h = im.size
            except Exception:
                skipped += 1
                continue

            x1, y1, x2, y2 = bbox
            # basic sanity check on the parsed box
            if not (0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h):
                skipped += 1
                continue

            yolo_line = to_yolo_line(x1, y1, x2, y2, img_w, img_h)

            shutil.copy(f, img_out / f.name)
            (lbl_out / f"{f.stem}.txt").write_text(yolo_line + "\n")

        print(f"[{split_name}] wrote {len(files) - skipped} labels, skipped {skipped}")

    print(f"\nDone. Dataset ready at: {dest_dir}")
    print("Now create a data.yaml pointing at this folder (see data.yaml.template).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Folder of raw CCPD images")
    ap.add_argument("--dest", required=True, help="Output folder for YOLO dataset")
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None,
                     help="Optional cap on number of images (recommended: 5000-15000)")
    args = ap.parse_args()
    convert(args.source, args.dest, args.val_split, args.limit)
