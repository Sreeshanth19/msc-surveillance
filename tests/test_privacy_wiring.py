"""Regression tests: turning off mask classification must not turn off privacy.

Detected faces must be obscured in displayed and saved output, and no
identifiable facial detail may appear there. Neither behaviour is conditional on
mask classification being enabled.

Anonymisation was previously gated on the mask classifier having produced
results, so building the pipeline with ``enable_mask=False`` left the result
list empty and ``privacy_blur`` had no effect - silently, with the flag still
set. These tests pin the corrected wiring: face detection and classification are
separate, privacy depends only on ``privacy_blur``, and disabling classification
avoids loading the classifier without avoiding anonymisation.

The models are not committed, so the two networks are stubbed. What is under
test is the wiring, which is where the defect was.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("cv2")
pytest.importorskip("imutils")

from src import pipeline as pipeline_mod          # noqa: E402
from src.config import Config                     # noqa: E402
from src.mask_classifier import MaskResult        # noqa: E402

FACE = (10, 10, 30, 30)


class _FakeDetector:
    def __init__(self, *a, **k):
        pass

    def track(self, frame):
        return []


class _FakeMaskClassifier:
    """Records which of the two paths the pipeline took."""

    def __init__(self, *a, **k):
        self.detect_calls = 0
        self.detect_faces_calls = 0

    def detect(self, frame):
        self.detect_calls += 1
        return [MaskResult(FACE, "No Mask", 0.9)]

    def detect_faces(self, frame):
        self.detect_faces_calls += 1
        return [FACE]


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "PersonDetector", _FakeDetector)
    monkeypatch.setattr(pipeline_mod, "MaskClassifier", _FakeMaskClassifier)


@pytest.fixture
def anonymised(monkeypatch):
    """Replace anonymise_faces with a recorder; returns the list of box lists."""
    seen = []

    def _record(frame, boxes, blocks):
        seen.append(list(boxes))
        return frame

    monkeypatch.setattr(pipeline_mod.privacy, "anonymise_faces", _record)
    return seen


def _cfg(**over):
    cfg = Config()
    cfg.homography_file = None      # keep the estimator uncalibrated and offline
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _frame():
    return np.zeros((120, 200, 3), dtype=np.uint8)


def test_face_detector_is_built_when_privacy_on_and_classification_off(stubbed):
    p = pipeline_mod.MonitoringPipeline(_cfg(privacy_blur=True), enable_mask=False)
    assert p.mask is not None
    assert p.classify_masks is False


def test_nothing_is_built_when_privacy_and_classification_are_both_off(stubbed):
    p = pipeline_mod.MonitoringPipeline(_cfg(privacy_blur=False), enable_mask=False)
    assert p.mask is None


def test_faces_are_anonymised_with_classification_disabled(stubbed, anonymised):
    p = pipeline_mod.MonitoringPipeline(_cfg(privacy_blur=True), enable_mask=False)
    p.process_frame(_frame())
    assert anonymised == [[FACE]]           # this is the assertion that used to fail
    assert p.mask.detect_calls == 0         # the classifier was never run
    assert p.mask.detect_faces_calls == 1   # only face detection was


def test_faces_are_anonymised_with_classification_enabled(stubbed, anonymised):
    p = pipeline_mod.MonitoringPipeline(_cfg(privacy_blur=True), enable_mask=True)
    p.process_frame(_frame())
    assert anonymised == [[FACE]]
    assert p.classify_masks is True
    assert p.mask.detect_calls == 1
    assert p.mask.detect_faces_calls == 0


def test_privacy_blur_false_is_the_only_way_to_disable_anonymisation(stubbed, anonymised):
    p = pipeline_mod.MonitoringPipeline(_cfg(privacy_blur=False), enable_mask=True)
    p.process_frame(_frame())
    assert anonymised == []
