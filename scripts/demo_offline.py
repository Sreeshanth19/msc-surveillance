"""Offline demo: annotate a video on CPU with no model downloads.

Wires the legacy YOLOv3 detector into the *real* distance and visualisation
modules to prove the pipeline end to end without a GPU or network. Distance
runs in uncalibrated (pixel) mode here; metric output requires a calibration
for the camera concerned, produced by scripts/calibrate.py.

    python -m scripts.demo_offline --source mm/test4.mp4 \
        --yolo-dir m/yolo --output output/annotated_demo.mp4 --frames 40
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.detection_legacy import LegacyPersonDetector  # noqa: E402
from src.distance import DistanceEstimator              # noqa: E402
from src import visualize                               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline CPU demo")
    ap.add_argument("--source", required=True)
    ap.add_argument("--yolo-dir", required=True, help="folder with yolov3.cfg/.weights/coco.names")
    ap.add_argument("--output", default="output/annotated_demo.mp4")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--pixel-threshold", type=float, default=90.0)
    args = ap.parse_args()

    y = Path(args.yolo_dir)
    detector = LegacyPersonDetector(str(y / "yolov3.cfg"), str(y / "yolov3.weights"),
                                    str(y / "coco.names"))
    estimator = DistanceEstimator(None, fallback_pixel_distance=args.pixel_threshold)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")

    writer = None
    n = 0
    t_start = time.time()
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        scale = args.width / frame.shape[1]
        frame = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))

        t0 = time.time()
        tracks = detector.track(frame)
        foot = [t.foot_point for t in tracks]
        offenders, pairs = estimator.violations(foot)
        fps = 1.0 / max(time.time() - t0, 1e-6)

        visualize.draw_people(frame, tracks, offenders)
        visualize.draw_violation_links(frame, tracks, pairs, estimator.is_calibrated)
        visualize.draw_hud(frame, len(tracks), len(offenders), 0, fps, estimator.is_calibrated)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (w, h))
        writer.write(frame)
        n += 1
        print(f"frame {n:3d}: people={len(tracks):2d} violators={len(offenders):2d} "
              f"({fps:.2f} fps cpu)")

    cap.release()
    if writer:
        writer.release()
    elapsed = time.time() - t_start
    print(f"\nProcessed {n} frames in {elapsed:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
