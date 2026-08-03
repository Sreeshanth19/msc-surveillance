"""Live webcam monitoring in an OpenCV window (the display that works on your Mac).

Runs the full four-state pipeline on your FaceTime camera. Distance is shown in
pixels (the saved homography was for the test video, not your webcam).

    python -m scripts.live_camera                 # live view
    python -m scripts.live_camera --save live.mp4 # also save the processed video
    python -m scripts.live_camera --no-mask       # detector + distance only (faster)

Press the 'q' key in the video window to quit.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.config import Config
from src.pipeline import MonitoringPipeline
from src.distance import DistanceEstimator


def main() -> None:
    ap = argparse.ArgumentParser(description="Live webcam mask & distance monitor")
    ap.add_argument("--camera", type=int, default=0, help="camera index (0 = built-in)")
    ap.add_argument("--save", default=None, help="optional path to save the processed video")
    ap.add_argument("--no-mask", action="store_true", help="skip mask classification (faster)")
    args = ap.parse_args()

    cfg = Config().resolve(ROOT)
    cfg.use_gpu = False
    print("Loading detection models… (first time can take ~20-30s, please wait)", flush=True)
    pipe = MonitoringPipeline(cfg, enable_mask=not args.no_mask)
    # webcam runs in pixel-distance mode (the saved calibration was for the test video)
    pipe.distance = DistanceEstimator(None, cfg.min_safe_distance_m, cfg.fallback_pixel_distance)
    print("Models loaded. Opening camera…", flush=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check macOS camera permission for Terminal.")

    win = "Live - Mask & Distance Monitor (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 620)
    cv2.moveWindow(win, 60, 60)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    writer = None
    first = True
    print("Live! A window should now be open. Press 'q' in it to quit.", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame from camera — stopping.")
            break
        annotated, _ = pipe.process_frame(frame)
        if args.save:
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
            writer.write(annotated)
        cv2.imshow(win, annotated)
        if first:
            cv2.moveWindow(win, 60, 60)
            try:
                cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
            print("  (window opened - if you cannot see it, press Cmd+Tab to the Python icon)", flush=True)
            first = False
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
