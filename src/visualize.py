"""Overlays for the five-state risk model.

Per-person status combines mask + distance:
    GREEN  Safe                 (mask worn  + safe distance)
    YELLOW Distance Violation   (mask worn  + too close)
    ORANGE No Mask              (no mask    + safe distance)
    RED    High-Risk           (no mask    + too close)
    GREY   Mask N/A            (no face detected -> mask unknown)
"""
from __future__ import annotations

from typing import Sequence, Set, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

GREEN = (0, 170, 0)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)
RED = (0, 0, 220)
GREY = (150, 150, 150)
DARK = (28, 28, 28)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX if cv2 else 0

# status name -> (colour, label, text colour)
STATUS = {
    "safe":      (GREEN,  "Safe",               WHITE),
    "distance":  (YELLOW, "Distance Violation", BLACK),
    "nomask":    (ORANGE, "No Mask",            BLACK),
    "highrisk":  (RED,    "High-Risk",          WHITE),
    "unknown":   (GREY,   "Mask N/A",           WHITE),
}


def classify(masked, too_close) -> str:
    """masked is True / False / None (unknown); too_close is bool."""
    if masked is None:
        return "unknown"
    if masked and not too_close:
        return "safe"
    if masked and too_close:
        return "distance"
    if not masked and not too_close:
        return "nomask"
    return "highrisk"


def _chip(frame, text, org, fg=WHITE, bg=RED, scale=0.5, thick=1, pad=4):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    x = max(x, 2); y = max(y, th + pad + 2)
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, fg, thick, cv2.LINE_AA)


def draw_status_boxes(frame: np.ndarray, tracks, statuses: Sequence[str]) -> None:
    for t, s in zip(tracks, statuses):
        colour, label, fg = STATUS[s]
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        _chip(frame, f"#{t.track_id} {label}", (x1 + 2, y1 - 4), fg=fg, bg=colour, scale=0.5, thick=1)


def draw_violation_links(frame: np.ndarray, tracks,
                         pairs: Sequence[Tuple[int, int, float]], calibrated: bool) -> None:
    for i, j, dist in pairs:
        p1, p2 = tracks[i].foot_point, tracks[j].foot_point
        cv2.line(frame, p1, p2, RED, 1, cv2.LINE_AA)
    for i, j, dist in pairs:
        p1, p2 = tracks[i].foot_point, tracks[j].foot_point
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        text = f"{dist:.1f} m" if calibrated else f"{dist:.0f} px"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.55, 2)
        _chip(frame, text, (mid[0] - tw // 2, mid[1]), fg=WHITE, bg=RED, scale=0.55, thick=2)


def _legend(frame, x, y):
    items = [("safe", "Safe"), ("distance", "Distance"), ("nomask", "No Mask"), ("highrisk", "High-Risk")]
    col_x = [x, x + 150]
    for k, (name, _lbl) in enumerate(items):
        colour = STATUS[name][0]
        cx = col_x[k % 2]; cy = y + (k // 2) * 22
        cv2.rectangle(frame, (cx, cy - 10), (cx + 16, cy + 4), colour, -1)
        cv2.putText(frame, STATUS[name][1], (cx + 22, cy + 2), FONT, 0.45, WHITE, 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, total: int, mask_count: int, no_mask_count: int,
             violation_count: int, fps: float, calibrated: bool = True) -> None:
    lines = [
        ("Total Persons", str(total)),
        ("Mask", str(mask_count)),
        ("No Mask", str(no_mask_count)),
        ("Violations", str(violation_count)),
        ("FPS", f"{fps:.1f}"),
    ]
    pad, lh = 12, 27
    w, h = 330, pad * 2 + lh * len(lines) + 58
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + w, 8 + h), DARK, -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    y = 8 + pad + 16
    for label, val in lines:
        cv2.putText(frame, f"{label}:", (18, y), FONT, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, val, (210, y), FONT, 0.62, WHITE, 2, cv2.LINE_AA)
        y += lh
    cv2.putText(frame, f"distance in {'metres' if calibrated else 'pixels'}", (18, y - 2),
                FONT, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    _legend(frame, 18, y + 20)
