"""Validate a saved homography against an INDEPENDENT physical reference."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.calibration import pick_points_interactive  # noqa: E402
from src.distance import DistanceEstimator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate homography vs independent reference")
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame", type=int, default=10)
    ap.add_argument("--homography", default="calibration/homography.npy")
    ap.add_argument("--out", default="results/calibration/homography_validation_report.txt")
    args = ap.parse_args()

    hpath = ROOT / args.homography
    if not hpath.exists():
        raise SystemExit(f"No homography at {hpath}. Run scripts.calibrate_interactive first.")
    H = np.load(hpath)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.source}")
    for _ in range(max(0, args.frame)):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read a frame; try a smaller --frame value")

    print("\nClick the TWO endpoints of your INDEPENDENT reference object.")
    print("This must NOT be the rectangle you calibrated on.")
    print("Then press any key.\n")
    pts = pick_points_interactive(frame, 2)
    if len(pts) != 2:
        raise SystemExit("Need exactly 2 points.")

    true_m = float(input("True measured length of that reference, in metres: "))

    est = DistanceEstimator(H)
    if not est.is_calibrated:
        raise SystemExit("Estimator reports uncalibrated - check the homography file.")

    D = est.pairwise([tuple(map(float, p)) for p in pts])
    measured = float(D[0, 1])
    err = measured - true_m
    pct = (err / true_m) * 100.0 if true_m else float("nan")

    lines = [
        "Homography validation against an independent physical reference",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source video: {args.source} (frame {args.frame})",
        f"Homography: {args.homography}",
        "",
        f"Reference image points: {pts[0]} -> {pts[1]}",
        f"True length:      {true_m:.4f} m",
        f"Measured length:  {measured:.4f} m",
        f"Absolute error:   {err:+.4f} m",
        f"Relative error:   {pct:+.2f} %",
        "",
        "Note: the reference measured here was not used to compute the homography,",
        "so this is an independent check rather than a restatement of the calibration.",
        "",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
