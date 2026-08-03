# Real-Time Mask & Social-Distance Monitoring

A modular re-implementation for the MSc project *"Deep Learning Based Real-Time
Face Mask Detection and Social Distance Analysis in Surveillance Environments."*
It keeps the parts of the open-source baseline that work (the trained MobileNetV2
mask classifier and OpenCV face detector) and replaces the parts that don't with
methodologically sound components.

## How this improves on the baseline

| Aspect | Baseline | This implementation |
|---|---|---|
| Person detection | darknet **YOLOv3**, per-frame | modern Ultralytics detector (YOLOv8/11) |
| Identity over time | none (per-frame only) | **ByteTrack** tracking → stable `track_id` |
| Distance | pixel gap vs. fixed pixel threshold | **homography → metric distance** in m |
| Reference point | box centroid | **foot point** (correct for ground plane) |
| Privacy | discussed, never built | **face mosaicking** actually implemented |
| Evaluation | one number on training data | **harness** w/ holdout + leakage caveat, FPS |
| Code | magic numbers, monolithic | typed config, modular, unit-tested |

The single most important change is **metric distance estimation**: a fixed pixel
threshold is wrong almost everywhere in a perspective view, so the baseline's
distance flags are unreliable. Calibrating to the ground plane fixes this and is
the main contribution to defend.

## Project structure

```
msc_surveillance/
├── src/
│   ├── config.py          # one typed Config for every parameter
│   ├── detection.py       # PersonDetector: modern detector + tracking
│   ├── distance.py        # DistanceEstimator: metric (homography) + pixel fallback
│   ├── calibration.py     # fit the ground-plane homography
│   ├── mask_classifier.py # face detection + mask classification (reuses baseline model)
│   ├── privacy.py         # face mosaicking
│   ├── visualize.py       # overlays + HUD
│   └── pipeline.py        # end-to-end orchestration + per-track stats
├── scripts/
│   ├── run.py             # run on webcam / video / stream
│   ├── calibrate.py       # produce calibration/homography.npy
│   ├── evaluate_mask.py   # precision/recall/F1 + confusion matrix
│   └── benchmark_fps.py   # FPS / latency numbers
├── tests/test_distance.py # geometry unit tests (pytest)
├── config.yaml            # documented defaults
├── requirements.txt
└── setup_models.sh        # copy reused assets into models/
```

## Quick start

```bash
pip install -r requirements.txt          # install torch's CUDA build for GPU
bash setup_models.sh /path/to/baseline/m # stage the reused models

# 1. calibrate to your camera's ground plane (one-off per camera)
python -m scripts.calibrate \
    --image-points 100,400 600,400 700,720 50,720 \
    --width-m 3.0 --height-m 2.0 \
    --out calibration/homography.npy

# 2. run on a recorded clip and save an annotated copy
python -m scripts.run --source /path/to/baseline/mm/test4.mp4 \
    --output output/annotated.mp4

# 3. get reportable numbers
python -m scripts.evaluate_mask --dataset /path/to/baseline/m/dataset --holdout 0.2
python -m scripts.benchmark_fps --source /path/to/baseline/mm/test4.mp4 --frames 200

# 4. verify the core geometry
python -m pytest -q
```

Without a homography the system still runs, but falls back to the inferior pixel
threshold — useful precisely because it lets you produce a **before/after
comparison** (pixel vs. metric) for the evaluation chapter.

## Mapping to the dissertation

- **Chapter 3 (design):** `src/` module breakdown + the comparison table above.
- **Chapter 4 (implementation):** the detector/tracker integration, the
  calibration + homography maths, the privacy stage.
- **Chapter 5 (testing/validation):** `evaluate_mask.py`, `benchmark_fps.py`,
  `tests/`, and the pixel-vs-metric ablation.

## Cross-dataset evaluation (BAFMD)
To test generalisation on a demographically diverse set (the honest "fresh exam"):
1. Request BAFMD: https://github.com/Alpkant/BAFMD (fill the form, await approval).
2. Download images with BAFMD's own `fetch_dataset.py` (you receive it on approval).
3. Convert its detection annotations into mask/no-mask crops:
   `python -m scripts.prepare_detection_dataset --dir bafmd/test_set --ann-format voc --out data/bafmd_crops`
4. Score the baseline model on it:
   `python -m scripts.evaluate_mask --dataset data/bafmd_crops`
The gap between this number and the in-dataset number is a reportable finding.

## Train your own mask model
Stronger provenance than reusing a pre-trained model — train on a dataset you chose:
```
python -m scripts.train_mask --dataset /path/to/data --out models/my_mask_model.keras --epochs 15
```
Streams images from disk (memory-safe). Then evaluate and compare against the baseline model
with `scripts/evaluate_mask.py`. Document the dataset, split, and hyperparameters in your report.

## Compare multiple models (comparative study)
Train and evaluate several backbones on the same data, like-for-like:
```
python -m scripts.compare_models --dataset /path/to/data \
    --models mobilenetv2,resnet50,vgg16,inceptionv3,efficientnetb0 --epochs 12 \
    --out output/model_comparison
```
Outputs a comparison table (accuracy, precision, recall, F1, inference ms/image, params)
plus a bar chart. Needs a GPU and internet (ImageNet weights) on first run.

## What is still yours to do
This is a working scaffold, not a finished dissertation. You need to: run it on
your GPU, calibrate against your own footage, run the evaluation (ideally on a
*second* mask dataset to avoid the training-data leakage flagged in
`evaluate_mask.py`), tune thresholds, and write up the analysis and limitations.

## Attribution
Built as an independent re-implementation that reuses two MIT-licensed projects,
which **must be cited** in the dissertation:
- Mask classifier + dataset: *chandrikadeb7/Face-Mask-Detection* (MIT).
- Social-distancing baseline structure: *saimj7/Social-Distancing-Detection-in-Real-Time* (MIT).

Person detection/tracking uses Ultralytics YOLO + ByteTrack; mask classification
uses TensorFlow/Keras; image handling uses OpenCV. Cite these libraries too.
```
