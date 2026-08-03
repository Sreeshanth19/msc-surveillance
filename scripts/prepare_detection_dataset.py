"""Convert a detection-format mask dataset into classification crops.

BAFMD (and most "in the wild" mask datasets) ship as full photos plus
bounding-box annotations. The baseline MobileNetV2 classifier, however, expects
*cropped face images* sorted into ``with_mask/`` and ``without_mask/`` folders.
This script bridges the two: it reads each annotation, cuts out every labelled
face, and saves it into the right class folder so the result drops straight into
``scripts/evaluate_mask.py``.

Supports both annotation formats BAFMD provides:
  * Pascal VOC  (.xml)  -- preferred: class is given by NAME, so no guessing.
  * YOLO        (.txt)  -- class is a NUMBER; supply --class-map to name them.

Examples
--------
    # VOC annotations (recommended)
    python -m scripts.prepare_detection_dataset \
        --dir bafmd/test_set --ann-format voc --out data/bafmd_crops

    # YOLO annotations, telling it which class id is which
    python -m scripts.prepare_detection_dataset \
        --dir bafmd/test_set --ann-format yolo \
        --class-map "0:with_mask,1:without_mask" --out data/bafmd_crops
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# How raw label names map to our two classes. Lower-cased, matched whole.
DEFAULT_NAME_MAP: Dict[str, str] = {
    "with_mask": "with_mask", "mask": "with_mask", "masked": "with_mask",
    "face_mask": "with_mask", "good": "with_mask", "with mask": "with_mask",
    "without_mask": "without_mask", "no_mask": "without_mask", "nomask": "without_mask",
    "no-mask": "without_mask", "unmasked": "without_mask", "none": "without_mask",
    "without mask": "without_mask", "face_no_mask": "without_mask", "bad": "without_mask",
}


def _clamp(v, lo, hi):
    return max(lo, min(int(v), hi))


def _expand(x1, y1, x2, y2, W, H, margin):
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin; x2 += bw * margin
    y1 -= bh * margin; y2 += bh * margin
    return _clamp(x1, 0, W - 1), _clamp(y1, 0, H - 1), _clamp(x2, 0, W - 1), _clamp(y2, 0, H - 1)


def _parse_voc(xml_path: Path) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    root = ET.parse(xml_path).getroot()
    out = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        box = (int(float(bb.findtext("xmin"))), int(float(bb.findtext("ymin"))),
               int(float(bb.findtext("xmax"))), int(float(bb.findtext("ymax"))))
        out.append((name, box))
    return out


def _parse_yolo(txt_path: Path, W: int, H: int,
                class_map: Dict[int, str]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    out = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:5])
        x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
        name = class_map.get(cid, f"class_{cid}")
        out.append((name, (int(x1), int(y1), int(x2), int(y2))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop a detection dataset into mask/no-mask folders")
    ap.add_argument("--dir", required=True, help="folder with images AND their annotations")
    ap.add_argument("--ann-format", choices=["voc", "yolo"], default="voc")
    ap.add_argument("--out", required=True, help="output folder (with_mask/ and without_mask/ created inside)")
    ap.add_argument("--class-map", default="", help="YOLO only, e.g. '0:with_mask,1:without_mask'")
    ap.add_argument("--name-map", default="", help="extra VOC name aliases, e.g. 'mask:with_mask'")
    ap.add_argument("--margin", type=float, default=0.15, help="expand each box by this fraction")
    ap.add_argument("--min-size", type=int, default=24, help="skip crops smaller than this (px)")
    args = ap.parse_args()

    class_map = {}
    for pair in filter(None, args.class_map.split(",")):
        k, v = pair.split(":"); class_map[int(k)] = v.strip()

    name_map = dict(DEFAULT_NAME_MAP)
    for pair in filter(None, args.name_map.split(",")):
        k, v = pair.split(":"); name_map[k.strip().lower()] = v.strip()

    src = Path(args.dir)
    out = Path(args.out)
    (out / "with_mask").mkdir(parents=True, exist_ok=True)
    (out / "without_mask").mkdir(parents=True, exist_ok=True)

    images = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    saved = Counter()
    raw_names = Counter()
    skipped_unknown = Counter()

    for img_path in images:
        ann = img_path.with_suffix(".xml" if args.ann_format == "voc" else ".txt")
        if not ann.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        H, W = image.shape[:2]
        objs = (_parse_voc(ann) if args.ann_format == "voc"
                else _parse_yolo(ann, W, H, class_map))

        for i, (raw, (x1, y1, x2, y2)) in enumerate(objs):
            raw_names[raw] += 1
            cls = name_map.get(raw.strip().lower())
            if cls not in ("with_mask", "without_mask"):
                skipped_unknown[raw] += 1
                continue
            x1, y1, x2, y2 = _expand(x1, y1, x2, y2, W, H, args.margin)
            if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cv2.imwrite(str(out / cls / f"{img_path.stem}_{i}.jpg"), crop)
            saved[cls] += 1

    print("=== label names found in annotations ===")
    for name, n in raw_names.most_common():
        print(f"  {name!r}: {n}")
    if skipped_unknown:
        print("\n!! UNMAPPED names (skipped) — add them with --name-map if needed:")
        for name, n in skipped_unknown.most_common():
            print(f"  {name!r}: {n}")
    print(f"\nSaved crops -> {out}")
    print(f"  with_mask:    {saved['with_mask']}")
    print(f"  without_mask: {saved['without_mask']}")
    print("\nNext: python -m scripts.evaluate_mask --dataset", args.out)


if __name__ == "__main__":
    main()
