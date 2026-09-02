# Real-Time Mask & Social-Distance Monitoring

The experimental platform for the MSc project *"Deep Learning Based Real-Time
Face Mask Detection and Social Distance Analysis in Surveillance Environments."*

The dissertation is an **empirical investigation** into the performance and
reliability of deep-learning mask classification and calibrated interpersonal
distance estimation. This prototype is the apparatus through which those
questions are measured — it is not itself the primary contribution. It keeps the
parts of the open-source baseline that work (the trained MobileNetV2 mask
classifier and the OpenCV res10 face detector, both reused unchanged) and
replaces the parts that would make measurement unsound.

## How this improves on the baseline

| Aspect | Baseline | This implementation |
|---|---|---|
| Person detection | darknet **YOLOv3**, per-frame | **YOLOv8n** via Ultralytics (v8s / 11s also benchmarked) |
| Identity over time | none (per-frame only) | **ByteTrack** tracking → stable `track_id` |
| Distance | pixel gap vs. fixed 80 px threshold | **homography → metric distance** in m |
| Reference point | box centroid | **foot point** (correct for ground plane) |
| Privacy | discussed, never built | **face mosaicking** actually implemented |
| Evaluation | one number on training data | three evaluation sets of differing independence, verified by content hashing |
| Calibration | none | fitted, **validated against a measured physical reference**, and checked for degeneracy |
| Code | magic numbers, monolithic | typed config, modular, unit-tested |

The single most important change is **metric distance estimation**: a fixed pixel
threshold is wrong almost everywhere in a perspective view, so the baseline's
distance flags cannot be validated against a physical reference at all.
Calibrating to the ground plane is what makes the distance-accuracy strand of
the investigation measurable — it turns an unfalsifiable flag into a quantity
that can be checked against a measured 1.48 m reference.

## Project structure

```
msc_surveillance/
├── src/
│   ├── config.py             # one typed Config for every parameter
│   ├── detection.py          # PersonDetector: YOLOv8n + ByteTrack
│   ├── detection_legacy.py   # inherited YOLOv3 path, retained for comparison
│   ├── distance.py           # DistanceEstimator: metric (homography) + pixel fallback
│   ├── calibration.py        # fit the ground-plane homography
│   ├── mask_classifier.py    # face detection + mask classification (reuses baseline model)
│   ├── privacy.py            # face mosaicking
│   ├── visualize.py          # overlays, HUD, five-state risk model
│   └── pipeline.py           # end-to-end orchestration + per-track stats
├── scripts/
│   ├── run.py                     # run on webcam / video / stream (--frame-log for per-frame CSV)
│   ├── desktop_app.py             # Tkinter application for recorded video
│   ├── live_camera.py             # live webcam mode
│   ├── calibrate.py               # produce calibration/homography.npy
│   ├── calibrate_interactive.py   # click reference points on a frame
│   ├── validate_calibration.py    # measure a calibration against a physical reference
│   ├── check_calibration.py       # degeneracy diagnostic
│   ├── ablate_calibration.py      # compare every calibration in git history
│   ├── evaluate_mask.py           # accuracy / precision / recall / F1 + confusion matrix
│   ├── compute_roc_auc.py         # ROC curve and AUC
│   ├── check_dataset_overlap.py   # byte-identical overlap between two datasets
│   ├── compare_models.py          # seeded five-backbone comparative study
│   ├── benchmark_fps.py           # FPS / latency
│   ├── prepare_detection_dataset.py
│   └── train_mask.py
├── tests/
│   ├── test_distance.py           # geometry unit tests (pytest)
│   └── test_calibration_check.py  # degeneracy diagnostic tests
├── results/                  # every reported figure, as the artefact that produced it
├── config.yaml               # documented defaults
├── requirements.txt
├── LICENSE                   # MIT (this project's own code)
├── THIRD_PARTY_LICENSES.md   # reused components — licences are NOT uniform, read this
└── setup_models.sh           # copy reused assets into models/
```

## Quick start

```bash
pip install -r requirements.txt
bash setup_models.sh /path/to/baseline/m # stage the reused models

# 1. calibrate to your camera's ground plane (one-off per camera)
python -m scripts.calibrate \
    --image-points 100,400 600,400 700,720 50,720 \
    --width-m 3.0 --height-m 2.0 \
    --out calibration/homography.npy

# 2. check the calibration is not degenerate BEFORE trusting its metres
python -m scripts.check_calibration

# 3. run on a recorded clip and save an annotated copy
python -m scripts.run --source /path/to/baseline/mm/test4.mp4 \
    --output output/annotated.mp4 --frame-log results/demo/frame_log.csv

# 4. reproduce the reported numbers
python -m scripts.evaluate_mask --dataset /path/to/data --relation independent \
    --exclude-from /path/to/baseline/m/dataset
python -m scripts.check_dataset_overlap --baseline m/dataset --evaluation data
python -m scripts.benchmark_fps --source /path/to/baseline/mm/test4.mp4 --frames 200

# 5. verify the core geometry
python -m pytest -q
```

A GPU is **not** required, and on the development machine was not faster — see
Results below. Without a homography the system still runs but falls back to the
pixel threshold, which is useful precisely because it lets you produce a
**before/after comparison** (pixel vs. metric) for the evaluation chapter.

## Mapping to the dissertation

- **Chapter 3 — Datasets and Experimental Design:** the three image datasets and
  their differing independence, `config.yaml` as the recorded parameter set, the
  experimental protocols, and `scripts/` as the procedures that implement them.
- **Chapter 4 — Results of the Empirical Investigation:** everything under
  `results/` — the classifier evaluations, the five-backbone comparison, the
  throughput benchmarks and the calibration measurements.
- **Chapter 5 — Validation of Results:** `results/dataset/` for the independence
  screening, `results/calibration/calibration_ablation.txt` for the calibration
  ablation, `results/model_comparison/colab_t4_crosscheck/` for the second-machine
  reproduction, and `results/reproducibility_check_2026-08-27/` for the final
  seven-group check.

`src/` is the platform those procedures run on; it is described where the
pipeline needs explaining rather than as a deliverable in its own right.

## Results

Every figure below is reproduced by a script in `scripts/` and stored as an
artefact under `results/`. Where a figure was later found to be invalid, it is
kept here with the reason, because the correction is itself a reported finding.

### Mask classification (deployed classifier, reused unchanged)

| Evaluation set | Images | Accuracy | Status |
|---|---|---|---|
| In-distribution holdout | 613 | 97.88% | Training distribution — optimistic by construction |
| Second dataset ("cross-dataset") | 3,021 | 98.34% | **INVALID** — see below |
| Same set, duplicates removed | 3,792 | 98.60% | Valid generalisation estimate (AUC 0.9991) |
| BAFMD (demographically diverse) | 1,682 | **80.56%** | Independent; 0 duplicates found (AUC 0.9601) |

**The invalid figure is the important one.** Content hashing showed that 92.1% of
the baseline dataset's unique images also appear in the "independent" second
dataset, and that 49.9% of the holdout drawn from it was training data. The
98.34% was therefore not a generalisation estimate. `check_dataset_overlap.py`
reproduces the finding; the deduplicated row above is the corrected result.

On BAFMD the direction of error reverses: `with_mask` recall falls to 77.25% and
`without_mask` precision to 50.08% (macro F1 0.7594). The classifier stops
missing breaches and starts flagging compliant people instead. The drop cannot be
attributed to demographic composition alone — image domain, crop provenance and
mask-type diversity are confounded with it.

### Five-backbone comparison (seeded, identical conditions, CPU)

| Backbone | Accuracy | ms/image |
|---|---|---|
| InceptionV3 | 0.9963 | 22.1 |
| ResNet50 | 0.9927 | 16.0 |
| EfficientNetB0 | 0.9927 | 5.1 |
| VGG16 | 0.9890 | 56.3 |
| MobileNetV2 | 0.9817 | 3.0 |

EfficientNetB0 gives the best in-distribution accuracy-per-millisecond of the
five. That ranking is **in-distribution only**: on BAFMD, InceptionV3 leads
clearly (0.9423 accuracy, macro F1 0.9072) and VGG16 collapses (0.8573, F1
0.6874). These five instances are freshly initialised backbones with a common
head and a uniform 12-epoch budget — none of them is the deployed classifier.
A clean-checkout re-run on different hardware (Colab T4) reproduced four of the
five results exactly; MobileNetV2 differed by one image.

### Detector throughput (200 frames each)

| Detector | CPU median FPS | GPU median FPS |
|---|---|---|
| YOLOv8n | **9.15** | 7.26 |
| YOLOv8s | 8.02 | 6.74 |
| YOLO11s | 7.96 | 7.00 |

All three exceed the 5 fps threshold adopted for this investigation. Every one
ran **faster on the CPU than on
the GPU backend** available on the development machine, which is the opposite of
what is usually assumed at these model sizes.

### Calibration

Validated against an independently measured 1.48 m ground-plane reference:

| Calibration | Error |
|---|---|
| Original (image corners clicked) | −3.90% |
| A4 sheet reference | −43.01% |
| Second attempt | −3.51% |
| **Deployed** | **−0.81%** |
| Inherited 80 px threshold | reaches the **wrong verdict** on this pair |

The original calibration was **degenerate**: its perspective terms were near
zero, so it applied one constant scale across the frame and was arithmetically
equivalent to a pixel threshold while still reporting metres.
`check_calibration.py` exists to catch that condition before its output is read
as metric. `ablate_calibration.py` reproduces the whole table by loading each
superseded homography out of git history.

## Status

**Done:**
- Modular pipeline: detection, tracking, homography-based metric distance, mask classification, privacy mosaicking, five-state risk model
- Three-tier mask evaluation with independence verified by content hashing, including the invalidation of the original cross-dataset result
- BAFMD demographically diverse evaluation (access granted 21 August 2026)
- Seeded five-backbone comparison, reproduced on separate hardware
- Detector throughput benchmarked on both compute backends
- Calibration validated against a measured physical reference; degeneracy diagnostic and full calibration ablation
- Threshold recalibration measured on BAFMD: macro-F1 recovers from 0.7594 to
  0.8656 cross-fitted, 92% of the oracle ceiling, with no retraining
- Final reproducibility check across seven test groups, all values reproducing
  (`results/reproducibility_check_2026-08-27/`)
- Every reported figure traceable to the script, command and commit that produced it
- Version control and this GitHub repository

**Still open:**
- Detection and tracking accuracy were measured on a single recorded sequence only
- Perspective correction is validated for absolute scale but not across depth: the reference pair lies at near-constant image depth, so a depth-spanning reference remains future work
- Test coverage is partial: 16 tests over four modules cover the geometry, the
  calibration diagnostic, the privacy wiring and the overlay, and the privacy
  tests do exercise the pipeline end to end with stubbed networks. The detector,
  the legacy detector and the mask classifier are not unit-tested, because their
  weights are not committed
- Dissertation write-up in progress

## Cross-dataset evaluation (BAFMD)

Completed. To repeat it:

1. Request BAFMD: https://github.com/Alpkant/BAFMD (fill the form, await approval).
2. Download images with BAFMD's own `fetch_dataset.py` (you receive it on approval).
   Note that 2,055 of 6,264 URLs were no longer retrievable when this project
   gathered the data, so the set you obtain will differ.
3. Convert its detection annotations into mask/no-mask crops:
   `python -m scripts.prepare_detection_dataset --dir bafmd/test_set --ann-format voc --out data/bafmd_crops`
4. Score the deployed model on it:
   `python -m scripts.evaluate_mask --dataset data/bafmd_crops --relation independent --exclude-from m/dataset`

The gap between this number and the in-distribution number is the project's
headline finding: 97.88% against 80.56%.

## Train your own mask model

Stronger provenance than reusing a pre-trained model — train on a dataset you chose:

```
python -m scripts.train_mask --dataset /path/to/data --out models/my_mask_model.keras --epochs 15
```

Streams images from disk (memory-safe). Then evaluate and compare against the
baseline model with `scripts/evaluate_mask.py`. Document the dataset, split and
hyperparameters in your report. **Note:** the deployed classifier in this project
is the inherited one, retained deliberately so that the evaluation measures an
existing model rather than a newly trained one.

## Compare multiple models (comparative study)

Train and evaluate several backbones on the same data, like-for-like:

```
python -m scripts.compare_models --dataset /path/to/data \
    --models mobilenetv2,resnet50,vgg16,inceptionv3,efficientnetb0 --epochs 12 \
    --seed 42 --deterministic --out output/model_comparison
```

Outputs a comparison table (accuracy, precision, recall, F1, inference ms/image,
params) plus a bar chart, with a provenance header recording the seed, the device
and the library versions. Needs internet on first run for the ImageNet weights.
A GPU is optional; the reported figures were produced on CPU.

## Attribution and licensing

This project's own code is MIT-licensed (`LICENSE`). **The components it reuses
are not uniformly licensed** — read `THIRD_PARTY_LICENSES.md` before reusing
anything here. In summary:

- Mask classifier, dataset and res10 face-detector assets: *chandrikadeb7/Face-Mask-Detection* (MIT).
- Social-distancing baseline structure and YOLOv3 assets: *saimj7/Social-Distancing-Detection-in-Real-Time* (MIT).
- Person detection and tracking: **Ultralytics YOLO — AGPL-3.0**, a copyleft
  licence. Academic research and coursework fall within its free tier; a closed
  or commercial deployment would need Ultralytics' commercial licence or a
  permissively licensed detector.
- BAFMD dataset: the repository code is MIT, but **the images are for
  non-commercial research only** and remain subject to the terms of the platform
  they were collected from. They are not redistributed here.
- The res10 `.caffemodel` weights carry **no licence statement upstream**; the
  accompanying `deploy.prototxt` comes from the OpenCV repository.

Mask classification uses TensorFlow/Keras and image handling uses OpenCV. Cite
these libraries too.
