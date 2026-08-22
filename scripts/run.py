"""Run the monitoring pipeline.

Examples
--------
    # webcam, live window
    python -m scripts.run --source 0 --display

    # process a recorded clip and save an annotated copy
    python -m scripts.run --source mm/test4.mp4 --output output/annotated.mp4

    # turn on privacy blurring and disable mask classification
    python -m scripts.run --source mm/test4.mp4 --output output/anon.mp4 --blur --no-mask
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config            # noqa: E402
from src.pipeline import MonitoringPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time mask & social-distance monitoring")
    ap.add_argument("--source", default="0", help="webcam index, video file, or stream URL")
    ap.add_argument("--output", default=None, help="path to write an annotated video")
    ap.add_argument("--config", default=None, help="optional YAML config file")
    ap.add_argument("--display", action="store_true", help="show a live window (needs a GUI)")
    ap.add_argument("--blur", action="store_true", help="pixelate faces for privacy")
    ap.add_argument("--no-mask", action="store_true", help="skip mask classification")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--frame-log", default=None,
                    help="write one CSV row per frame (frame, people, "
                         "distance_offenders, no_mask, fps) so session totals "
                         "can be traced back to the frames that produced them")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg = cfg.resolve(ROOT)
    if args.blur:
        cfg.privacy_blur = True
    if args.cpu:
        cfg.use_gpu = False

    pipe = MonitoringPipeline(cfg, enable_mask=not args.no_mask)
    stats = pipe.run(args.source, output=args.output, display=args.display,
                     frame_log=args.frame_log)
    print(json.dumps(stats.summary(), indent=2))


if __name__ == "__main__":
    main()
