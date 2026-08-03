#!/usr/bin/env bash
# Stage the reused baseline assets into ./models so the pipeline can find them.
# Run from the project root, pointing at where you unzipped the baseline (the
# folder that contains face_detector/, mask_detector.model and yolo/).
#
#   bash setup_models.sh /path/to/baseline/m
#
set -euo pipefail
SRC="${1:?Usage: bash setup_models.sh /path/to/baseline/m}"
mkdir -p models
cp "$SRC/face_detector/deploy.prototxt"                         models/
cp "$SRC/face_detector/res10_300x300_ssd_iter_140000.caffemodel" models/
cp "$SRC/mask_detector.model"                                   models/
echo "Staged: deploy.prototxt, res10 caffemodel, mask_detector.model -> models/"
echo "Person detection uses Ultralytics weights downloaded automatically on first run."
