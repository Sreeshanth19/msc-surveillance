"""Unit tests for the metric distance estimator.

These verify the core geometric claim of the project: that projecting foot
points through a calibrated homography yields correct *real-world* distances,
and that the naive pixel mode does not. Run with:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import compute_homography
from src.distance import DistanceEstimator


def _synthetic_homography():
    """A known image->world mapping for a 10m x 10m ground patch.

    We pretend the camera sees a 10 m square as a trapezoid in the image
    (near edge wider than far edge, as perspective dictates).
    """
    image_pts = [(100, 100), (540, 100), (640, 480), (0, 480)]   # TL TR BR BL (px)
    world_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]  # metres
    return compute_homography(image_pts, world_pts), image_pts, world_pts


def test_corners_map_to_known_world_points():
    H, image_pts, world_pts = _synthetic_homography()
    est = DistanceEstimator(H, safe_distance_m=2.0)
    world = est.to_world(image_pts)
    assert np.allclose(world, np.array(world_pts), atol=1e-6)


def test_metric_distance_matches_ground_truth():
    H, _, _ = _synthetic_homography()
    est = DistanceEstimator(H, safe_distance_m=2.0)
    # two image points that we know map to (0,0) and (10,0) -> 10 m apart
    pts = [(100, 100), (540, 100)]
    D = est.pairwise(pts)
    assert abs(D[0, 1] - 10.0) < 1e-3


def test_violation_flagged_in_metres():
    H, _, _ = _synthetic_homography()
    est = DistanceEstimator(H, safe_distance_m=2.0)
    # map of (0,0) and (1,0) -> 1 m apart -> violation (< 2 m)
    # invert H to find image pixels for those world points
    import cv2
    Hinv = np.linalg.inv(H)
    world = np.array([[[0.0, 0.0]], [[1.0, 0.0]]], dtype=np.float64)
    img = cv2.perspectiveTransform(world, Hinv).reshape(-1, 2)
    offenders, pairs = est.violations([tuple(p) for p in img])
    assert offenders == {0, 1}
    assert len(pairs) == 1
    assert abs(pairs[0][2] - 1.0) < 1e-3


def test_uncalibrated_falls_back_to_pixels():
    est = DistanceEstimator(None, fallback_pixel_distance=50.0)
    assert not est.is_calibrated
    # 30 px apart -> below 50 px threshold -> violation
    offenders, pairs = est.violations([(0, 0), (30, 0)])
    assert offenders == {0, 1}
    # 80 px apart -> no violation
    offenders, _ = est.violations([(0, 0), (80, 0)])
    assert offenders == set()
