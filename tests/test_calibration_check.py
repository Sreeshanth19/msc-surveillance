"""Unit tests for the calibration diagnostic.

These pin the failure mode the diagnostic exists to catch: a homography fitted
to the four corners of the image rather than to a rectangle on the ground plane.
Such a homography is numerically valid and passes tests/test_distance.py, yet
performs no perspective correction.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import compute_homography
from scripts.check_calibration import (
    COVERAGE_WARN_FRAC, PERSPECTIVE_WARN_PCT, SCALE_RATIO_WARN,
    calibration_coverage, perspective_strength, scale_profile,
)

W, HT = 1920, 1080


def _corner_clicked():
    """The failure mode: image corners declared to be a room-sized rectangle."""
    return compute_homography(
        [(0, 0), (W, 0), (W, HT), (0, HT)],
        [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)])


def _oblique_floor():
    """A genuine oblique view: far edge much narrower than the near edge."""
    return compute_homography(
        [(500, 200), (1400, 200), (1800, 1000), (100, 1000)],
        [(0.0, 0.0), (2.8, 0.0), (2.8, 2.6), (0.0, 2.6)])


def test_corner_clicked_is_flagged_degenerate():
    pct, _ = perspective_strength(_corner_clicked(), W, HT)
    assert pct < PERSPECTIVE_WARN_PCT


def test_oblique_shows_perspective():
    pct, _ = perspective_strength(_oblique_floor(), W, HT)
    assert pct > PERSPECTIVE_WARN_PCT


def test_corner_clicked_scale_is_uniform():
    vals = [m for _, m in scale_profile(_corner_clicked(), W, HT)]
    assert max(vals) / min(vals) < SCALE_RATIO_WARN


def test_oblique_scale_varies_with_depth():
    vals = [m for _, m in scale_profile(_oblique_floor(), W, HT)]
    assert max(vals) / min(vals) > SCALE_RATIO_WARN


def test_coverage_alone_does_not_catch_corner_clicking():
    """Coverage is near total here - only the perspective checks catch it."""
    _, frac = calibration_coverage(_corner_clicked(), 4.0, 3.0, W, HT)
    assert frac > COVERAGE_WARN_FRAC
