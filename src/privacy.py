"""Privacy-preserving face anonymisation.

The baseline report *discussed* privacy but never implemented it. Because your
project's ethics appendix commits to data minimisation and anonymisation of
saved outputs, this module actually delivers it: detected faces are mosaicked
(pixelated) before a frame is displayed or written to disk, so the monitoring
overlay can be retained for review without storing identifiable imagery.

Mosaicking is preferred over a Gaussian blur because it is harder to reverse.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def pixelate(region: np.ndarray, blocks: int = 12) -> np.ndarray:
    """Return a mosaicked copy of an image region."""
    if cv2 is None:  # pragma: no cover
        raise ImportError("OpenCV is required for pixelation")
    h, w = region.shape[:2]
    if h == 0 or w == 0:
        return region
    blocks = max(1, min(blocks, h, w))
    temp = cv2.resize(region, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def anonymise_faces(frame: np.ndarray,
                    face_boxes: Iterable[Tuple[int, int, int, int]],
                    blocks: int = 12) -> np.ndarray:
    """Pixelate every supplied face box in-place and return the frame."""
    for (x1, y1, x2, y2) in face_boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        frame[y1:y2, x1:x2] = pixelate(frame[y1:y2, x1:x2], blocks)
    return frame
