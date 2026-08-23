"""Interactive calibration: click four ground points to create the homography.

Produces ``calibration/homography.npy``, which is what allows the pipeline to
report distance in metres rather than pixels. A frame is opened from the source
video; four points forming a rectangle on the ground are clicked in the order
top-left, top-right, bottom-right, bottom-left, and that rectangle's real size
in metres is then entered. A display is required, so this cannot be run on a
headless machine.

    python -m scripts.calibrate_interactive --source mm/test4.mp4 --frame 15

The rectangle should be one whose size is known rather than guessed - floor
tiles, a parking bay, paving slabs, or two markers a measured distance apart.
An estimated size does not invalidate the method, but it does propagate
directly into every distance the system subsequently reports, so an estimate
must be recorded as such.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.calibration import compute_homography, save_homography, pick_points_interactive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive ground-plane calibration")
    ap.add_argument("--source", required=True, help="video file to calibrate against")
    ap.add_argument("--frame", type=int, default=10, help="which frame to show")
    ap.add_argument("--out", default="calibration/homography.npy")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    for _ in range(max(0, args.frame)):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame; try a smaller --frame value")

    print("\nClick FOUR points on a flat rectangle on the GROUND, in this order:")
    print("  1) top-left   2) top-right   3) bottom-right   4) bottom-left")
    print("Then press any key.\n")
    pts = pick_points_interactive(frame, 4)
    if len(pts) != 4:
        raise SystemExit("Need exactly 4 points.")

    w = float(input("Real-world WIDTH  (point 1 -> point 2) in metres: "))
    h = float(input("Real-world HEIGHT (point 1 -> point 4) in metres: "))
    world = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    H = compute_homography(pts, world)
    out = args.out if Path(args.out).is_absolute() else str(ROOT / args.out)
    save_homography(H, out)
    print(f"\nSaved homography -> {out}")
    print("You can now run the full pipeline; distances will be in metres.")


if __name__ == "__main__":
    main()
