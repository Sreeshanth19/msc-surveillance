"""Face detection and mask classification.

This stage is deliberately kept compatible with the baseline assets: the
OpenCV SSD face detector (``deploy.prototxt`` + ``res10`` caffemodel) and the
trained MobileNetV2 mask classifier (``mask_detector.model``). Reusing the
baseline classifier lets the rest of the system run end to end immediately and
keeps the classifier fixed while the components around it change. Retraining
and independent evaluation are handled separately by ``scripts/train_mask.py``
and ``scripts/evaluate_mask.py``.

TensorFlow is imported lazily so the pipeline's geometry/detection parts and the
unit tests do not require it.
"""
# ---------------------------------------------------------------------------
# ATTRIBUTION
# Adapted from chandrikadeb7/Face-Mask-Detection
#   MIT License, Copyright (c) 2021 chandrikadeb7
#   https://github.com/chandrikadeb7/Face-Mask-Detection
# Reused: the face-detection and mask-classification procedure (300x300 SSD
#   blob with (104, 177, 123) mean subtraction, the detection-confidence loop,
#   and the 224x224 MobileNetV2 crop pipeline); the trained mask_detector.model
#   and res10 face-detector assets, used unmodified.
# Written by the author: the MaskClassifier class structure, lazy model
#   loading, typed MaskResult records, configurable thresholds, and the
#   TF_USE_LEGACY_KERAS handling for Keras 3 compatibility.
# The face-detector assets originate from OpenCV samples/dnn/face_detector.
# See THIRD_PARTY_LICENSES.md.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
# The baseline mask_detector.model was saved with Keras 2. Modern TensorFlow
# (>=2.16) ships Keras 3 and cannot load it directly. Routing tf.keras to the
# legacy Keras 2 API (provided by the tf-keras package) makes the old model
# load cleanly. Must be set before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class MaskResult:
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    label: str                        # "Mask" or "No Mask"
    confidence: float


class MaskClassifier:
    def __init__(self, face_proto: str, face_weights: str, mask_model: str,
                 face_conf: float = 0.5, input_size: int = 224):
        if cv2 is None:  # pragma: no cover
            raise ImportError("OpenCV is required for face detection")
        self.face_net = cv2.dnn.readNet(face_proto, face_weights)
        self.face_conf = face_conf
        self.input_size = input_size
        self._mask_model_path = mask_model
        self._mask_net = None  # lazily loaded keras model

    @property
    def mask_net(self):
        if self._mask_net is None:
            try:
                from tensorflow.keras.models import load_model
            except ImportError as exc:  # pragma: no cover
                raise ImportError("Mask classification requires tensorflow") from exc
            self._mask_net = load_model(self._mask_model_path)
        return self._mask_net

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Locate faces and return their boxes. No classification, no TensorFlow.

        Anonymisation needs face boxes whether or not masks are being
        classified, so this half of the work is separated from ``detect``. It
        runs one SSD forward pass and imports nothing from TensorFlow, which is
        what allows privacy to remain in force when classification is disabled
        without loading the classifier.
        """
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        locs: List[Tuple[int, int, int, int]] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.face_conf:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            locs.append((int(x1), int(y1), int(x2), int(y2)))
        return locs

    def detect(self, frame: np.ndarray) -> List[MaskResult]:
        """Locate faces and classify each as masked or unmasked."""
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array

        faces, locs = [], []
        for (x1, y1, x2, y2) in self.detect_faces(frame):
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (self.input_size, self.input_size))
            face = preprocess_input(img_to_array(face))
            faces.append(face)
            locs.append((x1, y1, x2, y2))

        results: List[MaskResult] = []
        if faces:
            preds = self.mask_net.predict(np.array(faces, dtype="float32"), verbose=0)
            for box, pred in zip(locs, preds):
                mask, without_mask = float(pred[0]), float(pred[1])
                label = "Mask" if mask > without_mask else "No Mask"
                results.append(MaskResult(box, label, max(mask, without_mask)))
        return results
