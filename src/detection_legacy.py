"""Offline fallback person detector (no GPU, no downloads).

``PersonDetector`` in ``detection.py`` uses a modern Ultralytics model with
ByteTrack — the right choice for the real system on a GPU. But Ultralytics
downloads its weights on first use and benefits from a GPU, neither of which is
always available (e.g. a CPU-only laptop, or an offline/locked-down machine).

This class implements the *same* ``track(frame) -> List[Track]`` interface using
OpenCV's DNN module and the YOLOv3 weights that ship with the baseline, plus a
minimal greedy nearest-centroid tracker so IDs are still reasonably stable. It
is intentionally the weaker option; it exists so the pipeline is runnable
anywhere and so the two detectors can be compared like for like in evaluation.
"""
# ---------------------------------------------------------------------------
# ATTRIBUTION
# Uses the YOLOv3 configuration and weights bundled with
#   saimj7/Social-Distancing-Detection-in-Real-Time
#   MIT License, Copyright (c) 2020 Sai Subhakar T
#   https://github.com/saimj7/Social-Distancing-Detection-in-Real-Time
# The YOLOv3 model itself originates from J. Redmon's Darknet project.
# Reused: the Darknet YOLOv3 person-detection approach and its model assets.
#   The inference body of track() - blobFromImage preprocessing, the
#   class-score confidence loop, centre-to-corner box conversion, and
#   NMSBoxes filtering - follows the standard OpenCV Darknet pattern as
#   used by the baseline project.
# Written by the author: the LegacyPersonDetector class, the Track-compatible
#   interface, and the greedy nearest-centroid tracker.
# See THIRD_PARTY_LICENSES.md.
# ---------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .detection import Track


class LegacyPersonDetector:
    def __init__(self, cfg_path: str, weights_path: str, names_path: str,
                 conf: float = 0.3, nms: float = 0.3, input_size: int = 416,
                 max_match_dist: int = 80):
        if cv2 is None:  # pragma: no cover
            raise ImportError("OpenCV is required")
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        layers = self.net.getLayerNames()
        self.out_layers = [layers[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        names = Path(names_path).read_text().strip().split("\n")
        self.person_idx = names.index("person")
        self.conf = conf
        self.nms = nms
        self.input_size = input_size
        self.max_match_dist = max_match_dist
        self._prev: List[Tuple[int, Tuple[int, int]]] = []  # (id, centroid)
        self._next_id = 1

    def _assign_ids(self, boxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """Greedy nearest-centroid matching to the previous frame."""
        centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in boxes]
        ids: List[int] = [-1] * len(centroids)
        used_prev = set()
        for i, c in enumerate(centroids):
            best, best_d = -1, self.max_match_dist
            for pid, pc in self._prev:
                if pid in used_prev:
                    continue
                d = ((c[0] - pc[0]) ** 2 + (c[1] - pc[1]) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = pid, d
            if best != -1:
                ids[i] = best
                used_prev.add(best)
            else:
                ids[i] = self._next_id
                self._next_id += 1
        self._prev = list(zip(ids, centroids))
        return ids

    def track(self, frame: np.ndarray) -> List[Track]:
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.out_layers)

        boxes, confidences = [], []
        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if class_id == self.person_idx and confidence > self.conf:
                    cx, cy, w, h = (det[0:4] * np.array([W, H, W, H])).astype("int")
                    x = int(cx - w / 2)
                    y = int(cy - h / 2)
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(confidence)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.nms)
        kept = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                kept.append(((x, y, x + w, y + h), confidences[i]))

        ids = self._assign_ids([b for b, _ in kept])
        return [Track(tid, bbox, conf) for tid, (bbox, conf) in zip(ids, kept)]
