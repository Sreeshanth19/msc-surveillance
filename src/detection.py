"""Person detection with persistent multi-object tracking.

Improvement over the baseline
------------------------------
The baseline ran a raw darknet ``YOLOv3`` detector independently on every frame,
so it had no notion of identity across time: a "violation" was only ever a
per-frame event. Here we use a modern Ultralytics detector with a built-in
tracker (ByteTrack), which assigns each person a stable ``track_id``. That
single change enables per-person violation histories, dwell-time analysis, and
far more meaningful evaluation than per-frame counting.

The heavy ``ultralytics`` import is deferred so that the rest of the pipeline
(and the unit tests for the geometry code) can be used without it installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Track:
    """A single tracked person in one frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def foot_point(self) -> Tuple[int, int]:
        """Bottom-centre of the box.

        For ground-plane distance this is the right reference point: it is where
        the person contacts the floor, so it maps cleanly through a homography to
        a top-down position. Using the centroid (as the baseline implicitly did)
        biases distance with a person's height and camera angle.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


class PersonDetector:
    """Wraps an Ultralytics model in tracking mode."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.3,
                 iou: float = 0.5, person_class_id: int = 0,
                 tracker_cfg: str = "bytetrack.yaml", use_gpu: bool = True):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Person detection requires ultralytics: pip install ultralytics"
            ) from exc
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.person_class_id = person_class_id
        self.tracker_cfg = tracker_cfg
        self.device = "cuda" if use_gpu else "cpu"

    def track(self, frame: np.ndarray) -> List[Track]:
        """Detect and track people in a single frame."""
        results = self.model.track(
            frame,
            persist=True,                  # keep IDs across consecutive frames
            classes=[self.person_class_id],
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker_cfg,
            device=self.device,
            verbose=False,
        )
        tracks: List[Track] = []
        if not results:
            return tracks
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return tracks
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
            tracks.append(Track(int(tid), (int(x1), int(y1), int(x2), int(y2)), float(c)))
        return tracks
