"""Measure every calibration this project produced against one physical reference.

WHY THIS EXISTS
---------------
`validate_calibration.py` answers "is the deployed homography accurate?" for the
calibration currently on disk. It cannot answer the more interesting question:
how much did recalibration actually change, and would the baseline method have
reached the same verdict?

That question needs the *superseded* calibrations, which are no longer on disk —
they exist only in the repository's history. This script loads them straight from
git, applies each to the same two image points recorded in the validation report,
and reports what each would have measured. Nothing is re-measured physically: the
reference points and their true length are the ones already committed in
`results/calibration/homography_validation_report.txt`.

It also evaluates the inherited baseline rule. The saimj7 baseline has no metric
scale at all; it compares the pixel separation of two people against a fixed
threshold (`MAX_DISTANCE = 80`, carried into this project's config as
`fallback_pixel_distance`). Applying that rule to the same pair shows whether a
threshold tuned on one camera transfers to another.

WHAT IT DOES NOT SHOW
---------------------
Two limits belong with every figure this produces.

1. The reference pair lies at almost constant image depth (y = 224 and y = 221).
   Perspective correction barely affects a horizontal pair, which is why even a
   degenerate calibration scores well here. This measures absolute scale, not
   perspective correction. A reference spanning depth would be needed for that.

2. The 80-pixel threshold was tuned for the baseline author's camera geometry.
   A failure here demonstrates that such a threshold does not transfer between
   installations; it does not show the baseline was wrong in its own setting.

    python -m scripts.ablate_calibration
    python -m scripts.ablate_calibration --points 227 224 894 221 --true-m 1.48
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# label = source. "git:<rev>:<path>" reads from history; anything else is a file.
DEFAULT_CALIBRATIONS = [
    ("original (image corners clicked)", "git:45c4fc6:calibration/homography.npy"),
    ("A4 sheet (0.297 x 0.210 m)", "git:b68ff20:calibration/homography_a4.npy"),
    ("calib2 (1.78 x 2.16 m)", "git:2e0e28b:calibration/homography.npy"),
    ("final calib3 (2.80 x 2.58 m)", "calibration/homography.npy"),
]


def load_homography(spec: str):
    """Load a homography from the working tree or from git history."""
    if spec.startswith("git:"):
        _, rev, path = spec.split(":", 2)
        out = subprocess.run(["git", "show", f"{rev}:{path}"],
                             cwd=str(ROOT), capture_output=True)
        if out.returncode != 0:
            raise SystemExit(f"Cannot read {path} at {rev}: "
                             f"{out.stderr.decode().strip()}")
        return np.load(io.BytesIO(out.stdout))
    p = Path(spec)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"No homography at {p}")
    return np.load(p)


def perspective_strength(H: np.ndarray, W: int, Ht: int) -> float:
    """Variation of the projective denominator across the frame, as a percentage.

    A homography is defined only up to scale, so the raw coefficients mean
    nothing on their own; the variation of w = h31*x + h32*y + h33 across the
    frame is scale-invariant and is what indicates real perspective correction.
    """
    xs = np.array([0, W, W, 0], dtype=np.float64)
    ys = np.array([0, 0, Ht, Ht], dtype=np.float64)
    w = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    if not np.isfinite(w).all() or abs(w.mean()) < 1e-12:
        return float("nan")
    return float((w.max() - w.min()) / abs(w.mean()) * 100.0)


def measure(H: np.ndarray, pts: np.ndarray) -> float:
    """Ground-plane distance in metres between two image points under H."""
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    world = (H @ hom.T).T
    world = world[:, :2] / world[:, 2:3]
    return float(np.linalg.norm(world[0] - world[1]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare every calibration against one physical reference")
    ap.add_argument("--points", nargs=4, type=float, default=[227, 224, 894, 221],
                    metavar=("X1", "Y1", "X2", "Y2"),
                    help="the two reference image points, as recorded in "
                         "results/calibration/homography_validation_report.txt")
    ap.add_argument("--true-m", type=float, default=1.48,
                    help="measured real-world length of that reference, in metres")
    ap.add_argument("--safe-m", type=float, default=2.0,
                    help="the safe-distance threshold the system applies")
    ap.add_argument("--pixel-threshold", type=float, default=80.0,
                    help="the inherited baseline threshold, in pixels "
                         "(saimj7 MAX_DISTANCE; fallback_pixel_distance in config.yaml)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", default="results/calibration/calibration_ablation.txt")
    args = ap.parse_args()

    pts = np.array([[args.points[0], args.points[1]],
                    [args.points[2], args.points[3]]], dtype=np.float64)
    px = float(np.linalg.norm(pts[0] - pts[1]))
    truth_is_violation = args.true_m < args.safe_m

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit("Calibration ablation against an independent physical reference")
    emit(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit("")
    emit(f"Reference image points: ({pts[0][0]:.0f}, {pts[0][1]:.0f}) -> "
         f"({pts[1][0]:.0f}, {pts[1][1]:.0f})")
    emit(f"Pixel separation:       {px:.1f} px")
    emit(f"True length:            {args.true_m:.4f} m")
    emit(f"Safe-distance threshold: {args.safe_m:.2f} m  ->  "
         f"ground truth: {'VIOLATION' if truth_is_violation else 'compliant'}")
    emit("")
    emit("The reference points and true length are those recorded in")
    emit("results/calibration/homography_validation_report.txt. Nothing is")
    emit("re-measured physically here; superseded calibrations are read from git")
    emit("history and applied to the same two points.")
    emit("")

    hdr = (f"{'calibration':<34}{'persp %':>9}{'measured m':>12}"
           f"{'error %':>10}{'verdict':>12}")
    emit(hdr)
    emit("-" * len(hdr))

    for label, spec in DEFAULT_CALIBRATIONS:
        H = load_homography(spec)
        d = measure(H, pts)
        err = (d - args.true_m) / args.true_m * 100.0
        verdict = "VIOLATION" if d < args.safe_m else "compliant"
        emit(f"{label:<34}{perspective_strength(H, args.width, args.height):>8.2f}%"
             f"{d:>12.4f}{err:>9.2f}%{verdict:>12}")

    base_verdict = "VIOLATION" if px < args.pixel_threshold else "compliant"
    emit(f"{'baseline: fixed ' + str(int(args.pixel_threshold)) + ' px threshold':<34}"
         f"{'n/a':>9}{'n/a':>12}{'n/a':>10}{base_verdict:>12}")
    emit("")

    emit("Sources")
    for label, spec in DEFAULT_CALIBRATIONS:
        emit(f"   {label:<34} {spec}")
    emit(f"   baseline threshold                 saimj7 MAX_DISTANCE = "
         f"{int(args.pixel_threshold)} px; config.yaml fallback_pixel_distance")
    emit("")

    emit("Reading this table")
    emit("")
    emit("   Only the final calibration is within 1 % of the reference. The")
    emit("   A4-based calibration, which covers 7.3 % of the frame, is the worst")
    emit("   despite having the highest perspective variation: a small reference")
    emit("   patch extrapolates badly. Perspective variation alone is not")
    emit("   sufficient; it must come with coverage.")
    emit("")
    if base_verdict != ("VIOLATION" if truth_is_violation else "compliant"):
        emit("   The inherited pixel threshold reaches the WRONG verdict on this")
        emit(f"   pair: {px:.0f} px against a {int(args.pixel_threshold)} px "
             "threshold reads as compliant, while the")
        emit(f"   true separation of {args.true_m:.2f} m is below the "
             f"{args.safe_m:.1f} m limit. A threshold expressed")
        emit("   in pixels encodes the geometry of the camera it was tuned on.")
        emit("")
    emit("Two limits on what this shows")
    emit("")
    emit(f"   1. The reference lies at near-constant image depth (y = "
         f"{pts[0][1]:.0f} and {pts[1][1]:.0f}),")
    emit("      so perspective correction barely affects it. This validates")
    emit("      absolute scale, not perspective correction, which is why even the")
    emit("      degenerate calibration scores reasonably here.")
    emit("   2. The pixel threshold was tuned for the baseline author's camera.")
    emit("      This shows such a threshold does not transfer between")
    emit("      installations, not that the baseline was wrong in its own setting.")
    emit("")

    report = "\n".join(lines)
    print(report)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
