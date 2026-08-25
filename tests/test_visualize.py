"""Regression test: the annotated output must carry the track identifier.

FR-04's acceptance criterion is "Mask status appears against the correct
track identifier in the annotated output", and FR-11 requires the annotated
video to show "bounding boxes, track identifiers, distances and risk
states". ``draw_status_boxes`` previously rendered only the risk-state label
in each chip, with no track identifier anywhere in the frame - the tracker
still assigned ``track_id`` and it was used internally (session statistics,
violation-history keys), but it never reached the pixels written to disk.

This pins the corrected behaviour: the chip text passed to the drawing
primitive includes the track's ``track_id``.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("cv2")

import numpy as np

from src import visualize
from src.detection import Track


def test_status_chip_includes_track_id(monkeypatch):
    calls = []

    def fake_chip(frame, text, org, **kwargs):
        calls.append(text)

    monkeypatch.setattr(visualize, "_chip", fake_chip)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracks = [
        Track(track_id=7, bbox=(1, 1, 20, 20), confidence=0.9),
        Track(track_id=42, bbox=(30, 30, 50, 50), confidence=0.8),
    ]
    visualize.draw_status_boxes(frame, tracks, ["safe", "nomask"])

    assert len(calls) == 2
    assert "7" in calls[0] and "Safe" in calls[0]
    assert "42" in calls[1] and "No Mask" in calls[1]


def test_status_chip_distinguishes_same_state_different_tracks(monkeypatch):
    """Two tracks in the same risk state must not render identical chips."""
    calls = []
    monkeypatch.setattr(visualize, "_chip", lambda frame, text, org, **kw: calls.append(text))

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracks = [
        Track(track_id=1, bbox=(1, 1, 20, 20), confidence=0.9),
        Track(track_id=2, bbox=(30, 30, 50, 50), confidence=0.9),
    ]
    visualize.draw_status_boxes(frame, tracks, ["safe", "safe"])

    assert calls[0] != calls[1]
