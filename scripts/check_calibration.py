"""TODO: write the module docstring yourself.

Explain: why unit tests cannot catch this (they verify the transform maths,
which was never wrong - the defect is in the calibration DATA), and what a
degenerate homography does (near-zero perspective terms -> effectively affine
-> one constant metres-per-pixel scale -> arithmetically a pixel threshold).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402


# Your data: degenerate calibrations gave 0.44% and 2.69%; working ones gave
# 56.53%, 125.9%, 159.8%. A4 at 7.3% coverage carried +6.82% error; your final
# at 63.8% coverage carried -0.81%.
# Thresholds set from the calibrations measured in this project.
# Perspective: the two degenerate calibrations measured 0.44 % and 2.69 %;
# every calibration that produced usable metric distances exceeded 56 %. 10 %
# sits in the gap with wide margin on both sides.
PERSPECTIVE_WARN_PCT = 10.0
# Scale ratio: degenerate calibrations gave 1.00-1.01 (uniform scale across the
# frame); working ones gave 1.45 and above. 1.15 separates them.
SCALE_RATIO_WARN = 1.15
# Coverage: this is the threshold with direct error evidence behind it. A
# calibration covering 7.3 % of the frame carried +6.82 % error against an
# independent reference, while one covering 63.8 % carried -0.81 %. 10 % is set
# just above the coverage that was demonstrably too small.
COVERAGE_WARN_FRAC = 0.10


def perspective_strength(H, W, Ht):
    """Variation of the projective denominator across the frame, as a percentage.
    ...
    """
    corners = [(0, 0), (W, 0), (W, Ht), (0, Ht)]
    ws = [H[2, 0] * x + H[2, 1] * y + H[2, 2] for x, y in corners]
    variation = (max(ws) - min(ws)) / abs(sum(ws) / len(ws)) * 100
    return variation, ws


def scale_profile(H, W, Ht, rows=5):
    """Metres-per-pixel implied by H, sampled down the frame."""

    x0, x1 = W * 0.45, W * 0.55
    span_px = x1 - x0
    profile = []
    for frac in np.linspace(0.15, 0.9, rows):
        y = Ht * frac
        pts = np.array([[[x0, y]], [[x1, y]]], dtype=np.float64)
        world = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        metres = float(np.linalg.norm(world[0] - world[1]))
        profile.append((int(y), metres / span_px))
    return profile


def calibration_coverage(H, rect_w, rect_h, W, Ht):
    """Recover the clicked points and report what fraction of the frame they span.

    TODO: you ran this today. Invert H, apply to the world rectangle corners
    (0,0), (w,0), (w,h), (0,h). Shoelace formula for the area. Return
    (recovered_points, area_fraction).
    """
    world = np.array([[[0.0, 0.0]], [[rect_w, 0.0]],
                      [[rect_w, rect_h]], [[0.0, rect_h]]], dtype=np.float64)
    quad = cv2.perspectiveTransform(world, np.linalg.inv(H)).reshape(-1, 2)
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return quad, area / float(W * Ht)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose a saved homography")
    ap.add_argument("--homography", default="calibration/homography.npy")
    ap.add_argument("--source", help="video file, read only for its frame size")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--rect", nargs=2, type=float, metavar=("W_M", "H_M"))
    ap.add_argument("--out", default="results/calibration/calibration_check_report.txt")
    args = ap.parse_args()

    hpath = ROOT / args.homography
    if not hpath.exists():
        raise SystemExit(f"No homography at {hpath}")
    H = np.load(hpath)

    W, Ht = args.width, args.height
    if args.source:
        cap = cv2.VideoCapture(args.source)
        if cap.isOpened():
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or W
            Ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or Ht
        cap.release()

    lines, warnings = [], []

    emit = lines.append

    emit("Calibration diagnostic report")
    emit(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"Homography: {args.homography}")
    emit(f"Frame size: {W} x {Ht}")
    emit("")

    pct, ws = perspective_strength(H, W, Ht)
    emit("1. Perspective strength")
    emit(f"   bottom row of H: [{H[2,0]:.6e}, {H[2,1]:.6e}, {H[2,2]:.4f}]")
    emit(f"   projective denominator at corners: {[round(float(v), 4) for v in ws]}")
    emit(f"   variation across frame: {pct:.2f} %")
    if pct < PERSPECTIVE_WARN_PCT:
        warnings.append(
            f"DEGENERATE: perspective variation is only {pct:.2f} %, below the "
            f"{PERSPECTIVE_WARN_PCT:.0f} % threshold. The projective terms are near "
            "zero, so this homography is effectively affine: it applies one constant "
            "metres-per-pixel scale everywhere in the frame and performs no "
            "perspective correction. Distances from it are a scaled pixel threshold "
            "reported in metres.")
        emit("   VERDICT: degenerate")
    else:
        emit("   VERDICT: perspective correction present")
    emit("")

    prof = scale_profile(H, W, Ht)
    vals = [m for _, m in prof]
    ratio = max(vals) / min(vals)
    emit("2. Scale profile down the frame")
    for y, mpp in prof:
        emit(f"   image row y={y:5d}:  {mpp:.6f} m/px")
    emit(f"   far/near ratio: {ratio:.2f}x")
    if ratio < SCALE_RATIO_WARN:
        warnings.append(
            f"UNIFORM SCALE: metres-per-pixel varies by only {(ratio - 1) * 100:.1f} % "
            "between the far and near parts of the frame. A genuine oblique view of a "
            "floor shows a much larger variation, so distances are not being "
            "perspective-corrected.")
        emit("   VERDICT: uniform - no depth-dependent scaling")
    else:
        emit("   VERDICT: scale varies with depth, as expected")
    emit("")

    if args.rect:
        quad, frac = calibration_coverage(H, args.rect[0], args.rect[1], W, Ht)
        emit("3. Recovered calibration region")
        emit(f"   calibration rectangle: {args.rect[0]} m x {args.rect[1]} m")
        emit(f"   implied clicked points: {np.round(quad, 1).tolist()}")
        emit(f"   covers {frac * 100:.1f} % of the frame")
        if frac < COVERAGE_WARN_FRAC:
            warnings.append(
                f"SMALL CALIBRATION REGION: the rectangle covers only {frac * 100:.1f} % "
                f"of the frame, below the {COVERAGE_WARN_FRAC * 100:.0f} % threshold. "
                "Distances measured outside it are extrapolations, and scale error grows "
                "with distance from the calibrated region.")
            emit("   VERDICT: too small - most of the frame is extrapolated")
        else:
            emit("   VERDICT: adequate coverage")
        emit("")

    emit("Summary")
    if warnings:
        for i, w in enumerate(warnings, 1):
            emit(f"   [{i}] {w}")
        emit("")
        emit("   This calibration should not be used for reported metric distances")
        emit("   until it is redone against a measured rectangle on the ground plane")
        emit("   spanning the region where people are actually tracked.")
    else:
        emit("   No problems detected.")
    emit("")

    report = "\n".join(lines)
    print(report)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Saved -> {out}")
    raise SystemExit(1 if warnings else 0)


if __name__ == "__main__":
    main()
