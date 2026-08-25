"""Measure how far the camera moved between the calibration and the analysed video.

Why this exists
---------------
``calibration/homography.npy`` was fitted on one recording and is applied to
another. A homography encodes the geometry of one camera at one viewpoint, so
that reuse is only legitimate if the camera did not move between the two
recordings. Chapter 5 needs that stated as a measured bound rather than as an
assumption from the file timestamps: "recorded two minutes apart, so it probably
did not move" is not evidence to the standard the rest of the project is held to.

What it measures
----------------
An affine warp is estimated between a reference frame of each video, using ECC
alignment on the greyscale frames. The warp is then applied to the four corners
of the *calibration region itself* - recovered from the homography the same way
``check_calibration.py`` recovers it - and the largest corner displacement is
reported. Displacing the calibration region rather than the whole frame is the
point: a rotation about the frame centre moves the corners of the frame a long
way and the calibrated area very little, and it is the calibrated area that the
distances depend on.

Pixel displacement is converted to millimetres using the metres-per-pixel scale
the homography itself implies, sampled down the frame. Because that scale varies
with depth the conversion is reported as a range, near-scale to far-scale.

The number to compare against is the calibration's own validated error:
-0.81 % on a 1.48 m reference is 12 mm (see
results/calibration/homography_validation_report.txt). A camera displacement
smaller than that is below the calibration's own measurement uncertainty, and
the reuse is justified. A displacement larger than it is not.

Usage
-----
    python -m scripts.measure_camera_shift \\
        --calib-video mm/calib3.mp4 --calib-frame 15 \\
        --video mm/demo3.mp4 --frame 15 \\
        --drift 407 813 \\
        --rect 2.80 2.58 \\
        --out results/calibration/camera_shift_report.txt

``--drift`` additionally measures movement *within* the analysed video, between
its first frame and each frame index given, which bounds drift over the run.
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

# The calibration's own validated error, from
# results/calibration/homography_validation_report.txt: -0.81 % of 1.48 m.
CALIBRATION_UNCERTAINTY_MM = 0.0081 * 1.48 * 1000.0


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {index} of {path}")
    return frame


def calibration_quad(H: np.ndarray, rect_w: float, rect_h: float) -> np.ndarray:
    """The four image points the homography was fitted to.

    Recovered exactly as scripts/check_calibration.py recovers them: the
    inverse transform maps the corners of the known world rectangle back into
    image coordinates.
    """
    world = np.array([[[0.0, 0.0]], [[rect_w, 0.0]],
                      [[rect_w, rect_h]], [[0.0, rect_h]]], dtype=np.float64)
    return cv2.perspectiveTransform(world, np.linalg.inv(H)).reshape(-1, 2)


def scale_range_m_per_px(H: np.ndarray, W: int, Ht: int, rows: int = 5):
    """Metres-per-pixel implied by H, sampled down the frame (near, far)."""
    x0, x1 = W * 0.45, W * 0.55
    vals = []
    for k in range(rows):
        y = Ht * (k + 0.5) / rows
        pts = np.array([[[x0, y]], [[x1, y]]], dtype=np.float64)
        w = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        metres = float(np.linalg.norm(w[1] - w[0]))
        vals.append(metres / (x1 - x0))
    return min(vals), max(vals)


def estimate_affine(ref: np.ndarray, mov: np.ndarray, iterations: int = 400,
                    eps: float = 1e-7) -> np.ndarray:
    """Affine warp mapping `mov` onto `ref`, by ECC on the greyscale frames.

    ECC is used rather than feature matching because the two frames are of the
    same static scene under near-identical conditions, which is the regime ECC
    is strongest in and where sparse features would be dominated by whatever
    moved in the scene (a person walking through) rather than by the camera.
    """
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        _, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error as exc:
        raise SystemExit(
            "ECC alignment did not converge: " + str(exc) +
            "\nThe two frames may be of different scenes, which is itself the answer: "
            "the calibration does not belong to this footage.")
    return warp.astype(np.float64)


def displace(quad: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Per-corner displacement, in pixels, produced by the affine warp."""
    ones = np.ones((quad.shape[0], 1))
    moved = (np.hstack([quad, ones]) @ warp.T)
    return np.linalg.norm(moved - quad, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure camera displacement between the calibration video and another")
    ap.add_argument("--calib-video", required=True, help="the video the homography was fitted on")
    ap.add_argument("--calib-frame", type=int, default=15)
    ap.add_argument("--video", required=True, help="the video the homography is applied to")
    ap.add_argument("--frame", type=int, default=15)
    ap.add_argument("--drift", type=int, nargs="*", default=[],
                    help="also measure frame 0 -> each of these indices, within --video")
    ap.add_argument("--homography", default="calibration/homography.npy")
    ap.add_argument("--rect", nargs=2, type=float, required=True, metavar=("W_M", "H_M"),
                    help="the calibration rectangle in metres, as recorded by check_calibration")
    ap.add_argument("--out", default="results/calibration/camera_shift_report.txt")
    args = ap.parse_args()

    hpath = ROOT / args.homography
    if not hpath.exists():
        raise SystemExit(f"No homography at {hpath}")
    H = np.load(hpath)

    calib_frame = read_frame(Path(args.calib_video), args.calib_frame)
    Ht, W = calib_frame.shape[:2]

    quad = calibration_quad(H, args.rect[0], args.rect[1])
    near, far = scale_range_m_per_px(H, W, Ht)

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("Camera displacement between the calibration recording and the analysed video")
    emit(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    emit(f"Homography: {args.homography}")
    emit(f"Calibration rectangle: {args.rect[0]} m x {args.rect[1]} m")
    emit(f"Frame size: {W} x {Ht}")
    emit("")
    emit("Method: affine warp estimated by ECC on greyscale frames, then applied to")
    emit("        the four corners of the calibration region recovered from H.")
    emit("        Pixels are converted to millimetres with the metres-per-pixel scale")
    emit(f"        H implies, which ranges {near:.6f} to {far:.6f} m/px across the frame.")
    emit("")

    results = []

    target = read_frame(Path(args.video), args.frame)
    if target.shape[:2] != (Ht, W):
        raise SystemExit(
            f"Frame sizes differ: {args.calib_video} is {W}x{Ht}, "
            f"{args.video} is {target.shape[1]}x{target.shape[0]}. "
            "A homography cannot be reused across different frame sizes.")
    warp = estimate_affine(calib_frame, target)
    d = displace(quad, warp)
    results.append((f"{Path(args.calib_video).name} frame {args.calib_frame} -> "
                    f"{Path(args.video).name} frame {args.frame}", d, warp))

    if args.drift:
        base = read_frame(Path(args.video), 0)
        for idx in args.drift:
            f = read_frame(Path(args.video), idx)
            w = estimate_affine(base, f)
            results.append((f"{Path(args.video).name} frame 0 -> frame {idx}",
                            displace(quad, w), w))

    emit("Corner displacement of the calibration region")
    emit("")
    emit(f"   {'comparison':<52} {'max px':>8} {'mm (near-far)':>18}")
    emit("   " + "-" * 80)
    worst_mm = 0.0
    for label, d, _ in results:
        mx = float(d.max())
        lo, hi = mx * near * 1000.0, mx * far * 1000.0
        worst_mm = max(worst_mm, hi)
        emit(f"   {label:<52} {mx:>8.2f} {f'{lo:.1f} - {hi:.1f}':>18}")
    emit("")

    emit("Estimated affine warps")
    for label, _, w in results:
        emit(f"   {label}")
        emit(f"      scale     {w[0,0]:.6f} / {w[1,1]:.6f}")
        emit(f"      rotation  {w[0,1]:+.6f} / {w[1,0]:+.6f}  "
             f"({np.degrees(abs(np.arctan2(w[1,0], w[0,0]))):.3f} deg)")
        emit(f"      translate {w[0,2]:+.2f} / {w[1,2]:+.2f} px")
    emit("")

    emit("Verdict")
    emit(f"   The calibration's own validated error is -0.81 % on a 1.48 m reference,")
    emit(f"   which is {CALIBRATION_UNCERTAINTY_MM:.0f} mm "
         f"(results/calibration/homography_validation_report.txt).")
    emit(f"   The largest camera displacement measured here is {worst_mm:.1f} mm.")
    emit("")
    if worst_mm < CALIBRATION_UNCERTAINTY_MM:
        emit("   The camera moved less than the calibration's own measurement uncertainty.")
        emit("   Reusing the calibration across these recordings is therefore justified,")
        emit("   as a measured bound rather than as an assumption.")
        status = 0
    else:
        emit("   The camera moved by MORE than the calibration's own uncertainty.")
        emit("   Distances computed on this footage inherit that displacement as error,")
        emit("   and the calibration should be refitted for this viewpoint.")
        status = 1
    emit("")
    emit("Limit of this check")
    emit("   An affine warp captures translation, rotation, scale and shear. It does")
    emit("   not capture a change in the camera's distance from, or angle to, the")
    emit("   ground plane, which would alter perspective rather than the image")
    emit("   similarity. A near-zero result here means the two views are close in")
    emit("   image terms; it is strong evidence for, but not proof of, an unmoved camera.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    sys.exit(status)


if __name__ == "__main__":
    main()
