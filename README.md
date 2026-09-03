# Real-Time Mask & Social-Distance Monitoring

This repository includes the experimental platform for the MSc project *"Deep
Learning Based Real-Time Face Mask Detection and Social Distance Analysis in
Surveillance Environments."*

The dissertation is an empirical investigation into the performance and
reliability of the deep-learning mask classification and calibrated interpersonal
distance estimation. The prototype is therefore used as the apparatus through
which those questions are measured, rather than as the primary contribution in
its own right. The parts of the open-source baseline that remain sound are
reused unchanged, specifically the trained MobileNetV2 mask classifier and the
OpenCV res10 face detector. The parts that would make measurement unsound have
been replaced.

## Changes relative to the baseline

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

The most substantial of these changes is metric distance estimation. A fixed
pixel threshold is incorrect almost everywhere in a perspective view, and the
distance flags produced by the baseline therefore cannot be validated against a
physical reference at all. Calibration to the ground plane is what makes the
distance-accuracy strand of the investigation measurable, because an
unfalsifiable flag is converted into a quantity that can be checked against a
measured 1.48 m reference.

## Project structure

```
msc_surveillance/
├── src/
│   ├── config.py             # one typed Config for every parameter
│   ├── detection.py          # PersonDetector: YOLOv8n + ByteTrack
│   ├── detection_legacy.py   # inherited YOLOv3 path, used only by demo_offline.py
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
│   ├── demo_offline.py            # legacy YOLOv3 path, CPU-only, no downloads
│   ├── record_calibration_clip.py # capture a short calibration clip from the camera
│   ├── calibrate.py               # produce calibration/homography.npy
│   ├── calibrate_interactive.py   # click reference points on a frame
│   ├── validate_calibration.py    # measure a calibration against a physical reference
│   ├── check_calibration.py       # degeneracy diagnostic
│   ├── ablate_calibration.py      # compare every calibration in git history
│   ├── evaluate_mask.py           # accuracy / precision / recall / F1 + confusion matrix
│   ├── compute_roc_auc.py         # ROC curve and AUC
│   ├── threshold_sweep.py         # deployed / oracle / cross-fitted decision threshold
│   ├── check_dataset_overlap.py   # byte-identical overlap between two datasets
│   ├── check_split_overlap.py     # overlap between the holdout and the comparison split
│   ├── compare_models.py          # seeded five-backbone comparative study
│   ├── benchmark_fps.py           # FPS / latency
│   ├── measure_camera_shift.py    # ECC displacement of the calibrated view
│   ├── make_figures.py            # build Figures 4.1, 4.2 and 5.1 from results/
│   ├── prepare_detection_dataset.py
│   └── train_mask.py
├── tests/
│   ├── test_distance.py           # geometry unit tests (pytest)
│   ├── test_calibration_check.py  # degeneracy diagnostic tests
│   ├── test_privacy_wiring.py     # pipeline-level privacy tests
│   └── test_visualize.py          # overlay tests
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

A GPU is not required, and on the development machine the GPU backend was not
faster, as the throughput results below record. Where no homography is supplied
the system still runs, but distance is reported using the inherited pixel
threshold. That fallback is retained deliberately, because it provides the
before-and-after comparison between pixel and metric distance on which the
evaluation depends.

## Mapping to the dissertation

- **Chapter 3 — Datasets and Experimental Design:** the three image datasets and
  their differing independence, `config.yaml` as the recorded parameter set, the
  experimental protocols, and `scripts/` as the procedures that implement them.
- **Chapter 4 — Results of the Empirical Investigation:** everything under
  `results/`, comprising the classifier evaluations, the five-backbone
  comparison, the throughput benchmarks and the calibration measurements.
- **Chapter 5 — Validation of Results:** `results/dataset/` for the independence
  screening, `results/calibration/calibration_ablation.txt` for the calibration
  ablation, `results/model_comparison/colab_t4_crosscheck/` for the second-machine
  reproduction, and `results/reproducibility_check_2026-08-27/` for the final
  seven-group check.

`src/` is the platform on which those procedures run. It is therefore described
where the pipeline needs explaining, rather than as a deliverable in its own
right.

## Results

Every figure reported below is reproduced by a script in `scripts/` and is
stored as an artefact under `results/`. Where a figure was subsequently found to
be invalid it is retained here together with the reason, because the correction
is itself a reported finding.

### Mask classification (deployed classifier, reused unchanged)

| Evaluation set | Images | Accuracy | Status |
|---|---|---|---|
| In-distribution holdout | 613 | 97.88% | Training distribution — optimistic by construction |
| Second dataset ("cross-dataset") | 3,021 | 98.34% | **INVALID** — see below |
| Same set, duplicates removed | 3,792 | 98.60% | Valid generalisation estimate (AUC 0.9991) |
| BAFMD (demographically diverse) | 1,682 | **80.56%** | Independent; 0 duplicates found (AUC 0.9601) |

The invalid figure is the more informative of the two. Content hashing
established that 92.1% of the unique images in the baseline dataset also appear
in the second dataset that had been treated as independent, and that 49.9% of
the holdout drawn from it was training data. The 98.34% therefore does not
constitute a generalisation estimate. The finding is reproduced by
`check_dataset_overlap.py`, and the deduplicated row above reports the corrected
result.

On BAFMD the direction of the error reverses. Recall for `with_mask` falls to
77.25% and precision for `without_mask` falls to 50.08%, giving a macro F1 of
0.7594. Breaches are therefore no longer missed, and compliant people are
flagged instead. The drop cannot be attributed to demographic composition alone,
because image domain, crop provenance and mask-type diversity are confounded
with it. Recalibrating the decision threshold on BAFMD alone raises the macro F1
from 0.7594 to 0.8656 under cross-fitting. That recovers 92% of the gain the
oracle threshold achieves, and it requires no retraining.

### Five-backbone comparison (seeded, identical conditions, CPU)

| Backbone | Accuracy | ms/image |
|---|---|---|
| InceptionV3 | 0.9963 | 22.1 |
| ResNet50 | 0.9927 | 16.0 |
| EfficientNetB0 | 0.9927 | 5.1 |
| VGG16 | 0.9890 | 56.3 |
| MobileNetV2 | 0.9817 | 3.0 |

EfficientNetB0 provides the best in-distribution accuracy per millisecond of the
five backbones. That ranking holds in distribution only. On BAFMD, InceptionV3
leads clearly with 0.9423 accuracy and a macro F1 of 0.9072, whereas VGG16 falls
to 0.8573 and 0.6874 respectively. These five instances are freshly initialised
backbones sharing a common head and a uniform 12-epoch budget, and none of them
is the deployed classifier. A clean-checkout re-run on separate hardware (Colab
T4) reproduced four of the five results exactly, with MobileNetV2 differing by a
single image.

### Detector throughput (200 frames each)

| Detector | CPU median FPS | GPU median FPS |
|---|---|---|
| YOLOv8n | **9.15** | 7.26 |
| YOLOv8s | 8.02 | 6.74 |
| YOLO11s | 7.96 | 7.00 |

All three detectors exceed the 5 fps threshold adopted for this investigation.
Each of them ran faster on the CPU than on the GPU backend available on the
development machine, which is the opposite of the behaviour usually assumed at
these model sizes.

### Calibration

Validated against an independently measured 1.48 m ground-plane reference:

| Calibration | Error |
|---|---|
| Original (image corners clicked) | −3.90% |
| A4 sheet reference | −43.01% |
| Second attempt | −3.51% |
| **Deployed** | **−0.81%** |
| Inherited 80 px threshold | reaches the **wrong verdict** on this pair |

The original calibration was degenerate. Its perspective terms were close to
zero, so a single constant scale was applied across the whole frame, and the
result was arithmetically equivalent to a pixel threshold while still being
reported in metres. `check_calibration.py` was written in order to detect that
condition before the output is read as metric. The table as a whole is
reproduced by `ablate_calibration.py`, which loads each superseded homography
out of the git history.

## Status

**Done:**
- Modular pipeline: detection, tracking, homography-based metric distance, mask classification, privacy mosaicking, five-state risk model
- Three-tier mask evaluation with independence verified by content hashing, including the invalidation of the original cross-dataset result
- BAFMD demographically diverse evaluation (access granted 21 August 2026)
- Seeded five-backbone comparison, reproduced on separate hardware
- Detector throughput benchmarked on both compute backends
- Calibration validated against a measured physical reference, with a degeneracy diagnostic and a full calibration ablation
- Threshold recalibration measured on BAFMD, where the macro F1 recovers from 0.7594 to 0.8656 under cross-fitting, recovering 92% of the gain the oracle threshold achieves and requiring no retraining
- Final reproducibility check across seven test groups, in which all values reproduced (`results/reproducibility_check_2026-08-27/`)
- Every reported figure traceable to the script, command and commit that produced it
- Version control and this GitHub repository

**Still open:**
- Detection and tracking accuracy were measured on a single recorded sequence only
- Perspective correction is validated for absolute scale but not across depth, because the reference pair lies at near-constant image depth. A depth-spanning reference therefore remains future work
- Test coverage is partial. The 16 tests across four modules cover the geometry, the calibration diagnostic, the privacy wiring and the overlay, and the privacy tests do exercise the pipeline end to end with stubbed networks. The detector, the legacy detector and the mask classifier are not unit-tested, because their weights are not committed
- Dissertation write-up in progress

## Cross-dataset evaluation (BAFMD)

This evaluation is complete. It is repeated as follows:

1. Request BAFMD: https://github.com/Alpkant/BAFMD (fill the form, await approval).
2. Download the images using BAFMD's own `fetch_dataset.py`, which is supplied on
   approval. 2,055 of the 6,264 URLs were no longer retrievable when the data for
   this project was gathered, and any set obtained later will therefore differ.
3. Convert its detection annotations into mask/no-mask crops:
   `python -m scripts.prepare_detection_dataset --dir bafmd/test_set --ann-format voc --out data/bafmd_crops`
4. Score the deployed model on it:
   `python -m scripts.evaluate_mask --dataset data/bafmd_crops --relation independent --exclude-from m/dataset`

The gap between this figure and the in-distribution figure is the principal
finding of the investigation, at 97.88% against 80.56%.

## Train your own mask model

Training on a dataset chosen for the purpose provides stronger provenance than
reusing a pre-trained model:

```
python -m scripts.train_mask --dataset /path/to/data --out models/my_mask_model.keras --epochs 15
```

Images are streamed from disk, so memory use remains bounded. The resulting
model is then evaluated and compared against the baseline model using
`scripts/evaluate_mask.py`, and the dataset, the split and the hyperparameters
should be recorded alongside the result. The classifier deployed in this project
is nevertheless the inherited one, retained deliberately so that the evaluation
measures an existing model rather than a newly trained one.

## Compare multiple models (comparative study)

Several backbones are trained and evaluated on the same data under identical
conditions:

```
python -m scripts.compare_models --dataset /path/to/data \
    --models mobilenetv2,resnet50,vgg16,inceptionv3,efficientnetb0 --epochs 12 \
    --seed 42 --deterministic --out output/model_comparison
```

The script produces a comparison table reporting accuracy, precision, recall,
F1, inference milliseconds per image and parameter count, together with a bar
chart and a provenance header recording the seed, the determinism setting, the
training configuration and the TensorFlow version. Network access is required on the first run in order to obtain the
ImageNet weights. A GPU is optional, and the reported figures were produced on
CPU.

## Attribution and licensing

The code written for this project is MIT-licensed (`LICENSE`). The components it
reuses are not uniformly licensed, and `THIRD_PARTY_LICENSES.md` should
therefore be read before any part of this repository is reused. In summary:

- Mask classifier, dataset and res10 face-detector assets: *chandrikadeb7/Face-Mask-Detection* (MIT).
- Social-distancing baseline structure and YOLOv3 assets: *saimj7/Social-Distancing-Detection-in-Real-Time* (MIT).
- Person detection and tracking: Ultralytics YOLO, AGPL-3.0, which is a copyleft
  licence. Academic research and coursework fall within its free tier, whereas a
  closed or commercial deployment would require either Ultralytics' commercial
  licence or a permissively licensed detector.
- BAFMD dataset: the repository code is MIT, but the images are licensed for
  non-commercial research only and remain subject to the terms of the platform
  from which they were collected. They are not redistributed here.
- The res10 `.caffemodel` weights carry no licence statement upstream, and the
  accompanying `deploy.prototxt` is taken from the OpenCV repository.

Mask classification is implemented in TensorFlow/Keras and image handling uses
OpenCV. Both libraries should also be cited.
