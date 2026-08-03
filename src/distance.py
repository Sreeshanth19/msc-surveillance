"""Distance estimation between people.

This module is the core methodological contribution over the baseline.

The baseline measured distance as the raw Euclidean distance **in pixels**
between bounding-box centroids and flagged a violation when that fell below a
fixed pixel threshold. That is fundamentally unsound: because of perspective,
the same pixel gap means very different real-world distances at the top of the
frame (far away) versus the bottom (close), so a single pixel threshold is wrong
almost everywhere in the scene.

The principled fix is to calibrate the camera to the ground plane once, via a
homography ``H`` that maps image points to a metric top-down ("bird's-eye")
coordinate system. People's foot points are projected through ``H`` and the
distance between them is then computed in metres, so a single real-world
threshold (e.g. 2 m) is valid everywhere in the frame.

A naive pixel-based fallback is retained *only* so the inferior baseline
behaviour can be reproduced for a controlled comparison in the evaluation
chapter.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # the metric path needs cv2; the pure-pixel fallback does not


class DistanceEstimator:
    """Estimate pairwise distances and flag close-contact violations.

    Parameters
    ----------
    homography:
        3x3 matrix mapping image pixel coordinates to a metric ground plane.
        If ``None`` the estimator falls back to (inferior) pixel distances.
    safe_distance_m:
        Minimum permitted real-world separation in metres (metric mode).
    fallback_pixel_distance:
        Threshold used only in pixel-fallback mode.
    """

    def __init__(self, homography: Optional[np.ndarray] = None,
                 safe_distance_m: float = 2.0,
                 fallback_pixel_distance: float = 80.0):
        self.H = None if homography is None else np.asarray(homography, dtype=np.float64)
        self.safe_distance_m = float(safe_distance_m)
        self.fallback_pixel_distance = float(fallback_pixel_distance)

    # -- public API ---------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        return self.H is not None

    def to_world(self, points: Sequence[Tuple[float, float]]) -> np.ndarray:
        """Project image points (px) to metric ground-plane coordinates."""
        if self.H is None:
            raise RuntimeError("to_world() requires a homography; estimator is uncalibrated")
        if cv2 is None:  # pragma: no cover
            raise ImportError("OpenCV is required for the metric (homography) path")
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        world = cv2.perspectiveTransform(pts, self.H)
        return world.reshape(-1, 2)

    def pairwise(self, points: Sequence[Tuple[float, float]]) -> np.ndarray:
        """Symmetric matrix of pairwise distances.

        Units are metres in calibrated (metric) mode and pixels otherwise.
        """
        coords = self.to_world(points) if self.is_calibrated else np.asarray(points, dtype=np.float64)
        if len(coords) < 2:
            return np.zeros((len(coords), len(coords)), dtype=np.float64)
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    def violations(self, points: Sequence[Tuple[float, float]]
                   ) -> Tuple[Set[int], List[Tuple[int, int, float]]]:
        """Return violating point indices and the offending pairs.

        Returns
        -------
        offenders : set of indices that violate the threshold
        pairs     : list of (i, j, distance) for each violating pair
        """
        n = len(points)
        offenders: Set[int] = set()
        pairs: List[Tuple[int, int, float]] = []
        if n < 2:
            return offenders, pairs
        D = self.pairwise(points)
        threshold = self.safe_distance_m if self.is_calibrated else self.fallback_pixel_distance
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] < threshold:
                    offenders.add(i)
                    offenders.add(j)
                    pairs.append((i, j, float(D[i, j])))
        return offenders, pairs

    # -- persistence --------------------------------------------------------
    @classmethod
    def from_file(cls, path: str, safe_distance_m: float = 2.0,
                  fallback_pixel_distance: float = 80.0) -> "DistanceEstimator":
        H = np.load(path)
        return cls(H, safe_distance_m, fallback_pixel_distance)
