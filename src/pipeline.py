"""End-to-end monitoring pipeline.

Per frame: detect & track people -> classify masks on detected faces ->
estimate metric distances and flag close contacts -> anonymise faces -> draw
overlays. Because people are tracked, the pipeline also keeps a per-identity
violation history, enabling dwell-style statistics that the per-frame baseline
could not produce.

Anonymisation is ordered after classification and before any drawing, so the
classifier sees the unobscured face and no consumer of the frame does. It is
governed by ``privacy_blur`` alone: disabling mask classification reduces what
the system reports, never what it discloses.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from .config import Config
from .detection import PersonDetector
from .distance import DistanceEstimator
from .mask_classifier import MaskClassifier
from . import privacy, visualize


@dataclass
class FrameStats:
    frame_index: int
    n_people: int
    n_distance_offenders: int
    n_no_mask: int
    fps: float


@dataclass
class SessionStats:
    frames: int = 0
    total_no_mask: int = 0
    total_distance_offenders: int = 0
    per_track_violation_frames: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def summary(self) -> dict:
        return {
            "frames_processed": self.frames,
            "cumulative_no_mask_detections": self.total_no_mask,
            "cumulative_distance_offender_detections": self.total_distance_offenders,
            "unique_tracks_ever_in_violation": len(self.per_track_violation_frames),
        }


class MonitoringPipeline:
    def __init__(self, cfg: Config, enable_mask: bool = True):
        if cv2 is None:  # pragma: no cover
            raise ImportError("OpenCV is required to run the pipeline")
        self.cfg = cfg
        self.detector = PersonDetector(
            cfg.detector_model, cfg.detector_conf, cfg.detector_iou,
            cfg.person_class_id, cfg.tracker_cfg, cfg.use_gpu,
        )
        # distance estimator: metric if a homography is available, else pixel fallback
        H = None
        if cfg.homography_file:
            try:
                H = np.load(cfg.homography_file)
            except (FileNotFoundError, OSError):
                H = None
        self.distance = DistanceEstimator(H, cfg.min_safe_distance_m, cfg.fallback_pixel_distance)

        # MaskClassifier owns the face detector, and anonymisation needs face
        # boxes whether or not masks are being classified. Building it for
        # either purpose is what stops disabling classification from also
        # disabling privacy: anonymisation is independent of classification.
        # When classification is off, only detect_faces() is called, so no
        # TensorFlow is imported and the classifier is never loaded.
        self.classify_masks = enable_mask
        self.mask: Optional[MaskClassifier] = None
        if enable_mask or cfg.privacy_blur:
            self.mask = MaskClassifier(
                cfg.face_proto, cfg.face_weights, cfg.mask_model,
                cfg.face_conf, cfg.mask_input_size,
            )
        self.stats = SessionStats()

    def process_frame(self, frame: np.ndarray):
        """Process a single BGR frame; return (annotated_frame, FrameStats)."""
        t0 = time.time()
        import imutils  # lightweight resize; optional
        frame = imutils.resize(frame, width=self.cfg.process_width)

        tracks = self.detector.track(frame)
        foot_points = [t.foot_point for t in tracks]
        offenders, pairs = self.distance.violations(foot_points)

        if self.mask is not None and self.classify_masks:
            mask_results = self.mask.detect(frame)
            face_boxes = [r.bbox for r in mask_results]
        elif self.mask is not None:
            mask_results = []                      # no verdicts: none were asked for
            face_boxes = self.mask.detect_faces(frame)
        else:
            mask_results, face_boxes = [], []

        # link each detected face's mask verdict to the person that contains it
        person_masked = [None] * len(tracks)  # True / False / None(unknown)
        for r in mask_results:
            cx = (r.bbox[0] + r.bbox[2]) // 2
            cy = (r.bbox[1] + r.bbox[3]) // 2
            for ti, t in enumerate(tracks):
                x1, y1, x2, y2 = t.bbox
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    person_masked[ti] = (r.label == "Mask")
                    break

        statuses = [visualize.classify(person_masked[i], i in offenders)
                    for i in range(len(tracks))]

        total = len(tracks)
        mask_count = sum(1 for m in person_masked if m is True)
        no_mask_count = sum(1 for m in person_masked if m is False)
        violation_count = sum(1 for i in range(total)
                              if (i in offenders) or (person_masked[i] is False))

        if self.cfg.privacy_blur and face_boxes:
            privacy.anonymise_faces(frame, face_boxes, self.cfg.privacy_blocks)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        visualize.draw_violation_links(frame, tracks, pairs, self.distance.is_calibrated)
        visualize.draw_status_boxes(frame, tracks, statuses)
        visualize.draw_hud(frame, total, mask_count, no_mask_count, violation_count, fps,
                           self.distance.is_calibrated)

        # update session statistics
        self.stats.frames += 1
        self.stats.total_no_mask += no_mask_count
        self.stats.total_distance_offenders += len(offenders)
        for idx in offenders:
            self.stats.per_track_violation_frames[tracks[idx].track_id] += 1

        return frame, FrameStats(self.stats.frames, total, len(offenders), no_mask_count, fps)

    def run(self, source, output: Optional[str] = None, display: bool = False,
            frame_log: Optional[str] = None) -> SessionStats:
        """Process a webcam index, video path, or stream URL to completion.

        ``frame_log`` writes one CSV row per frame. Session totals alone cannot
        show *where* a count came from: a run reporting several people in
        footage known to contain one is reporting false detections, and only a
        per-frame record makes that visible and checkable afterwards.
        """
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        writer = None
        log_fh = None
        if frame_log:
            lp = Path(frame_log)
            lp.parent.mkdir(parents=True, exist_ok=True)
            log_fh = lp.open("w")
            log_fh.write("frame,people,distance_offenders,no_mask,fps\n")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, fstats = self.process_frame(frame)
                if log_fh is not None:
                    log_fh.write(f"{fstats.frame_index},{fstats.n_people},"
                                 f"{fstats.n_distance_offenders},{fstats.n_no_mask},"
                                 f"{fstats.fps:.2f}\n")
                if output and writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output, fourcc, 20.0, (w, h))
                if writer is not None:
                    writer.write(annotated)
                if display:
                    cv2.imshow("Monitoring", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if log_fh is not None:
                log_fh.close()
            if display:
                cv2.destroyAllWindows()
        return self.stats
