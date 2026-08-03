"""Ground-plane calibration.

Computes the homography that the :class:`DistanceEstimator` needs. The user
identifies four points in the image that form a rectangle of known size on the
ground (for example floor tiles, a parking bay, or markers placed during
recording) and supplies the corresponding real-world coordinates in metres.
A homography is then fitted mapping image pixels to that metric ground plane.

Two ways to obtain the four image points are provided:
  * ``compute_homography`` — for when you already have the coordinates;
  * ``pick_points_interactive`` — click them on a still frame (needs a display).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def compute_homography(image_pts: Sequence[Tuple[float, float]],
                       world_pts: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Fit a 3x3 homography from >=4 image/world point correspondences.

    Parameters
    ----------
    image_pts : pixel coordinates of the reference points, in order.
    world_pts : the same points' real-world ground coordinates in metres.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV is required to compute a homography")
    if len(image_pts) < 4 or len(world_pts) < 4:
        raise ValueError("Need at least 4 point correspondences")
    if len(image_pts) != len(world_pts):
        raise ValueError("image_pts and world_pts must be the same length")
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(world_pts, dtype=np.float64)
    if len(image_pts) == 4:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    else:
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
    return H.astype(np.float64)


def save_homography(H: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, H)


def pick_points_interactive(frame: np.ndarray, n: int = 4) -> List[Tuple[int, int]]:  # pragma: no cover
    """Let the user click ``n`` reference points on a frame. Requires a display."""
    if cv2 is None:
        raise ImportError("OpenCV is required for interactive calibration")
    pts: List[Tuple[int, int]] = []

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))

    win = "Calibration - click reference points, then press any key"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_click)
    while True:
        disp = frame.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (x + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        if cv2.waitKey(20) != -1 and len(pts) == n:
            break
    cv2.destroyWindow(win)
    return pts
