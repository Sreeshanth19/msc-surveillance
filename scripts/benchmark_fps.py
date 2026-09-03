"""Benchmark processing speed (FPS / per-frame latency).

Throughput is one of the properties this investigation measures, so it has to
be measured rather than estimated. This script runs a fixed number of frames through the
full pipeline and reports the rate reproducibly. The hardware must be recorded
alongside the figures: a frames-per-second number means nothing without the
machine that produced it.

    python -m scripts.benchmark_fps --source mm/test4.mp4 --frames 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config            # noqa: E402
from src.pipeline import MonitoringPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark pipeline FPS")
    ap.add_argument("--source", default="0")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--detector",
                    help="override the detector weights, e.g. yolov8s.pt")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import cv2
    cfg = Config().resolve(ROOT)
    if args.cpu:
        cfg.use_gpu = False
    if args.detector:
        cfg.detector_model = args.detector
    print(f"detector: {cfg.detector_model}  gpu: {cfg.use_gpu}")
    pipe = MonitoringPipeline(cfg, enable_mask=not args.no_mask)

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process_frame(frame)
    latencies = []
    n = 0
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        pipe.process_frame(frame)
        latencies.append(time.time() - t0)
        n += 1
    cap.release()

    if not latencies:
        print("No frames processed.")
        return
    import statistics
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    print(f"Frames: {n}")
    print(f"Mean latency: {mean * 1000:.1f} ms/frame")
    print(f"Mean FPS:     {1.0 / mean:.2f}")
    print(f"Median latency: {median * 1000:.1f} ms/frame")
    print(f"Median FPS:     {1.0 / median:.2f}")
    print(f"p95 latency:  {sorted(latencies)[int(0.95 * len(latencies)) - 1] * 1000:.1f} ms")


if __name__ == "__main__":
    main()
