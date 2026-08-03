"""Central configuration for the monitoring pipeline.

Keeping every tunable parameter in one place (rather than scattered as magic
numbers through the code, as in the baseline) makes the system easier to test,
to document in the dissertation, and to vary systematically during evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ---- person detection (modern detector, replaces darknet YOLOv3) ----
    detector_model: str = "yolov8n.pt"   # any Ultralytics model; n=nano (fast), swap for s/m/l
    detector_conf: float = 0.30          # min confidence for a person detection
    detector_iou: float = 0.50           # NMS IoU threshold
    person_class_id: int = 0             # COCO 'person'
    tracker_cfg: str = "bytetrack.yaml"  # built-in multi-object tracker config

    # ---- face + mask classification (reuses the baseline trained model) ----
    face_proto: str = "models/deploy.prototxt"
    face_weights: str = "models/res10_300x300_ssd_iter_140000.caffemodel"
    mask_model: str = "models/mask_detector.model"
    face_conf: float = 0.50              # min confidence for a face detection
    mask_input_size: int = 224           # MobileNetV2 input

    # ---- distance estimation (metric, via homography) ----
    homography_file: Optional[str] = "calibration/homography.npy"
    min_safe_distance_m: float = 2.0     # real-world safe distance in METRES
    # fallback only: naive pixel threshold, used when no homography is supplied.
    # Kept solely so the inferior baseline behaviour can be reproduced for the
    # ablation/comparison in the evaluation chapter.
    fallback_pixel_distance: int = 80

    # ---- privacy ----
    privacy_blur: bool = False           # pixelate faces in the output stream
    privacy_blocks: int = 12             # mosaic granularity (smaller = more anonymised)

    # ---- runtime ----
    process_width: int = 700             # frames resized to this width before processing
    use_gpu: bool = True                 # passed to the detector ('cuda' vs 'cpu')

    def resolve(self, root: Path) -> "Config":
        """Resolve relative asset paths against the project root."""
        for attr in ("face_proto", "face_weights", "mask_model", "homography_file"):
            val = getattr(self, attr)
            if val and not Path(val).is_absolute():
                setattr(self, attr, str((root / val)))
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install pyyaml to load YAML configs: pip install pyyaml") from exc
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        known = {f for f in cls().__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
