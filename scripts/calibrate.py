"""Produce the ground-plane homography used for metric distance estimation.

You provide four image points that form a rectangle on the ground and that
rectangle's real-world size in metres. Example: a 1.0 m x 1.0 m floor marker
whose corners appear at the given pixels.

    python -m scripts.calibrate \
        --image-points 320,540 980,540 1100,720 200,720 \
        --width-m 3.0 --height-m 2.0 \
        --out calibration/homography.npy

The four image points must be given in the order:
    top-left, top-right, bottom-right, bottom-left.
The world rectangle is placed with its top-left at the origin.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import compute_homography, save_homography  # noqa: E402


def _parse_pair(s: str):
    x, y = s.split(",")
    return (float(x), float(y))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ground-plane homography")
    ap.add_argument("--image-points", nargs=4, required=True, metavar="x,y",
                    help="four image points: TL TR BR BL (pixels)")
    ap.add_argument("--width-m", type=float, required=True, help="rectangle width in metres")
    ap.add_argument("--height-m", type=float, required=True, help="rectangle height in metres")
    ap.add_argument("--out", default="calibration/homography.npy")
    args = ap.parse_args()

    img_pts = [_parse_pair(p) for p in args.image_points]
    w, h = args.width_m, args.height_m
    # world rectangle (metres): TL, TR, BR, BL
    world_pts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    H = compute_homography(img_pts, world_pts)
    save_homography(H, args.out)
    print(f"Saved homography to {args.out}")
    print(H)


if __name__ == "__main__":
    main()
