"""Interactive calibration: click four ground points to create the homography.

This is the beginner-friendly way to produce ``calibration/homography.npy`` so
the pipeline can report distance in real metres. It opens a frame from your
video, you click four points that form a rectangle on the ground (in the order
top-left, top-right, bottom-right, bottom-left), then type that rectangle's real
size in metres. Needs a screen (run it on your own machine, not a headless server).

    python -m scripts.calibrate_interactive --source mm/test4.mp4 --frame 15

Tip: pick a rectangle you can estimate the size of - floor tiles, a parking bay,
paving slabs, or two markers a known distance apart. If you are only estimating
the size, say so in your report (the distance is then approximate but the method
is still sound).
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
