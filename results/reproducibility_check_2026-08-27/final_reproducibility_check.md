# Final reproducibility check

**Date:** 27 August 2026
**Repository:** `Sreeshanth19/msc-surveillance` at **`9de9e1a`**, working tree clean
(one untracked helper script, `fix_remove_live_camera.py`, not part of the project)
**Interpreter:** `.venv-arm/bin/python` — Python 3.9.6, TensorFlow 2.20.0,
OpenCV 4.11.0.86 on Apple M1 Max / macOS
**Second environment (groups 6–7):** Linux, Python 3.13, numpy 2.4.4,
OpenCV 4.13.0.92

Nothing was modified. All script output was written to `/tmp/repro_out/`, outside
the repository, so no committed artefact under `results/` was overwritten.
No code, dataset, threshold, configuration or model file was changed, and nothing
was committed or pushed.

**Retention note (added 2 September 2026).** Because output was written outside
the repository, the macOS temporary-directory cleaner removed the raw files from
`/tmp/repro_out/` before they were archived. The three calibration artefacts
below were produced in a separate environment and survived; they are committed
alongside this report. Every other figure recorded here is stated in full in the
tables that follow, and each is regenerable by re-running the command printed
against its test. No figure in this report is cited by Chapter 4 or Chapter 5 —
those chapters cite only the committed artefacts under `results/`, all of which
this check confirmed unchanged.

## Verdict

**All 45 checked values reproduce.** The one apparent failure — InceptionV3 CPU
latency — was resolved by a second independent run and is confirmed.

| Group | Outcome |
|---|---|
| 1 Test suite | PASS — exact |
| 2 Classifier (A–E) | PASS — exact, every value |
| 3 Five-CNN comparison | PASS — accuracy, parameters, BAFMD exact; latency confirmed by a second run |
| 4 Throughput | PASS — expected timing variation, conclusions intact |
| 5 815-frame integration | PASS — exact |
| 6 Calibration diagnostic | PASS — exact, in a second environment |
| 7 Distance validation | PASS — exact (ablation) · NOT RUN (video-based validation) |

---

## Group 1 — automated test suite

**Command:** `.venv-arm/bin/python -m pytest -v`

| Test | Expected | Actual | Difference | Result |
|---|---|---|---|---|
| test_distance.py | 4 | 4 | — | PASS |
| test_calibration_check.py | 5 | 5 | — | PASS |
| test_privacy_wiring.py | 5 | 5 | — | PASS |
| test_visualize.py | 2 | 2 | — | PASS |
| Total | 16 passing | 16 passed in 1.36 s | — | PASS |

Failed 0 · skipped 0 · warnings 0 in the summary line.

---

## Group 2 — inherited MobileNetV2 classifier, no retraining

### Test A — familiar-distribution set

**Command:** `… -m scripts.evaluate_mask --dataset m/dataset --holdout 0.15 --relation training --out /tmp/repro_out/A_indist`

| Value | Expected | Actual | Difference | Result |
|---|---|---|---|---|
| N | 613 | 613 | 0 | PASS |
| Accuracy | 97.88% | 0.9788 | 0 | PASS |
| Min per-class recall | 96.13% | 0.9613 (with_mask) | 0 | PASS |
| without_mask recall | — | 1.0000 | — | — |

### Test B — exact-hash overlap

**Command:** `… -m scripts.check_dataset_overlap --baseline m/dataset --evaluation data --holdout 0.4 --seed 42 --out /tmp/repro_out/B_overlap.txt`

| Value | Expected | Actual | Difference | Result |
|---|---|---|---|---|
| Baseline loadable | 4,092 | 4,092 (2,162 / 1,930) | 0 | PASS |
| External | 7,553 | 7,553 (3,725 / 3,828) | 0 | PASS |
| Whole-dataset exact matches | 3,761 (49.8%) | 3,761 (49.8%) | 0 | PASS |
| Shared unique images | 3,672 | 3,672 | 0 | PASS |
| Fraction of baseline | 92.1% | 92.1% | 0 | PASS |
| Evaluation split | 3,021 | 3,021 | 0 | PASS |
| Exact matches in split | 1,508 | 1,508 | 0 | PASS |
| Non-matches | 1,513 | 1,513 | 0 | PASS |
| Split overlap | 49.9% | 49.9% | 0 | PASS |

### Test C — contaminated external evaluation

**Commands:**
`… -m scripts.evaluate_mask --dataset data --holdout 0.4 --relation contaminated --out /tmp/repro_out/C_contaminated`
`… -m scripts.compute_roc_auc --dataset data --relation contaminated --sample 3000 --out /tmp/repro_out/C_auc`

| Value | Expected | Actual | Difference | Result |
|---|---|---|---|---|
| N (accuracy) | 3,021 | 3,021 | 0 | PASS |
| Accuracy | 98.34% | 0.9834 | 0 | PASS |
| ROC-AUC | 0.9983 | 0.9983 | 0 | PASS |
| N (AUC) | *stated as 3,021* | **3,000** | see note | — |

**Note.** Accuracy and AUC come from different scripts with different N. The
accuracy is on the 3,021-image 40% holdout; the AUC is on a 3,000-image random
sample at seed 42. These are two measurements, not one, and should be reported
as such.

### Test D — hash-screened external evaluation

**Commands:**
`… -m scripts.evaluate_mask --dataset data --relation independent --exclude-from m/dataset --out /tmp/repro_out/D_screened`
`… -m scripts.compute_roc_auc --dataset data --relation independent --exclude-from m/dataset --sample 3000 --out /tmp/repro_out/D_auc`

| Value | Expected | Actual | Difference | Result |
|---|---|---|---|---|
| Duplicates excluded | 3,761 | 3,761 | 0 | PASS |
| N (accuracy) | 3,792 | 3,792 | 0 | PASS |
| Accuracy | 98.60% | 0.9860 | 0 | PASS |
| ROC-AUC | 0.9991 | 0.9991 | 0 | PASS |
| Min per-class recall | 97.70% | 0.9770 (without_mask) | 0 | PASS |
| N (AUC) | *stated as 3,792* | **3,000** | see Test C note | — |

### Test E — BAFMD

**Commands:**
`… -m scripts.evaluate_mask --dataset data/bafmd_crops --relation independent --exclude-from m/dataset --out /tmp/repro_out/E_bafmd`
`… -m scripts.compute_roc_auc --dataset data/bafmd_crops --relation independent --exclude-from m/dataset --out /tmp/repro_out/E_bafmd_auc`

| Value | Expected | Actual | Difference | Result |
|---|---|---|---|---|
| N | 1,682 | 1,682 | 0 | PASS |
| Exact baseline matches | 0 | 0 | 0 | PASS |
| Accuracy | 80.56% | 0.8056 | 0 | PASS |
| Macro-F1 | 0.7594 | 0.7594 | 0 | PASS |
| ROC-AUC | 0.9601 | 0.9601 | 0 | PASS |
| Masked precision | 98.22% | 0.9822 | 0 | PASS |
| Masked recall | 77.25% | 0.7725 | 0 | PASS |
| Unmasked precision | 50.08% | 0.5008 | 0 | PASS |
| Unmasked recall | 94.21% | 0.9421 | 0 | PASS |
| Masked samples | 1,354 | 1,354 | 0 | PASS |
| Total errors | 327 | 327 (derived) | 0 | PASS |
| Masked→unmasked errors | 308 | 308 (derived) | 0 | PASS |

Derived: 1,354 × 0.7725 = 1,046 masked correct; 328 × 0.9421 = 309 unmasked
correct; 1,682 − 1,355 = **327** errors; 1,354 − 1,046 = **308** masked
misclassified as unmasked.

---

## Group 3 — five-CNN architecture comparison

**Command:** `… -m scripts.compare_models --dataset m/dataset --epochs 12 --batch 32 --val-split 0.2 --seed 42 --timing-repeats 20 --timing-warmup 3 --extra-eval bafmd=data/bafmd_crops --out /tmp/repro_out/model_comparison`

**Preconditions confirmed in the run header and log:** seed 42 · 12 epochs ·
batch 32 · 3,274 train / 818 validation · ImageNet weights · frozen backbones
with a common head · TensorFlow 2.20.0. InceptionV3 at 299×299 and the rest at
224×224 is recorded per model as `input_size` in `results.json`.

T4 latency was not attempted — this machine is an Apple M1 Max.

### Validation accuracy and parameters — all exact

| Model | Expected acc | Actual | Expected params | Actual | Result |
|---|---|---|---|---|---|
| InceptionV3 | 0.9963 | 0.9963 | 22,065,314 | 22.1 M | PASS |
| ResNet50 | 0.9927 | 0.9927 | 23,850,242 | 23.9 M | PASS |
| EfficientNetB0 | 0.9927 | 0.9927 | 4,213,797 | 4.2 M | PASS |
| VGG16 | 0.9890 | 0.9890 | 14,780,610 | 14.8 M | PASS |
| MobileNetV2 | 0.9817 | 0.9817 | 2,422,210 | 2.4 M | PASS |

All five reproduce to four decimal places — including MobileNetV2, which was the
one model that differed on the Colab T4 cross-check (0.9804 there).

### BAFMD — reproduced on a second platform

These columns previously existed **only** from the Colab T4 run; the original
deployment-machine run predates the `--extra-eval` option. They now reproduce on
the Mac.

| Model | Expected acc | Actual | Expected F1 | Actual | Result |
|---|---|---|---|---|---|
| InceptionV3 | 0.9423 | 0.9411 | 0.9072 | 0.9055 | PASS — 2 images of 1,682 |
| ResNet50 | 0.9203 | 0.9203 | 0.8569 | 0.8569 | PASS — exact |
| EfficientNetB0 | 0.9001 | 0.9001 | 0.8221 | 0.8221 | PASS — exact |
| VGG16 | 0.8573 | 0.8573 | 0.6874 | 0.6874 | PASS — exact |
| MobileNetV2 | 0.8603 | 0.8603 | 0.6957 | 0.6957 | PASS — exact |

### Mac CPU latency — resolved by a second run

The comparison was run twice. The first run reported InceptionV3 at 48.4 ms
against a committed 22.1 ms (+119%), which would have been a material failure.
A second independent run returned **21.1 ms**, confirming the committed figure.

| Model | Committed | Run 1 | Run 2 | Run 2 vs committed | Result |
|---|---|---|---|---|---|
| MobileNetV2 | 3.0 ms | 3.0 | 3.5 ± 0.1 | +16.7% | PASS — timing |
| EfficientNetB0 | 5.1 ms | 5.1 | 4.9 ± 0.1 | −3.9% | PASS — timing |
| ResNet50 | 16.0 ms | 16.2 | 16.3 ± 1.1 | +1.9% | PASS — timing |
| **InceptionV3** | **22.1 ms** | **48.4** | **21.1 ± 0.2** | **−4.5%** | **PASS — timing** |
| VGG16 | 56.3 ms | 55.3 | 54.1 ± 0.5 | −3.9% | PASS — timing |

The 48.4 ms reading is recorded here rather than discarded. It was internally
stable (±0.6 ms over 20 timed batches) but is not reproducible, and it affected
only the model running at 299×299 while the other four reproduced within 2% in
the same run — consistent with transient memory pressure during that model's
timing loop rather than with a wrong committed figure.

MobileNetV2's +16.7% is 0.5 ms absolute on the smallest model, and run 1 returned
exactly 3.0 ms. That is sub-millisecond jitter, not a disagreement.

**Practical consequence: single-run latency figures on this machine carry
occasional outliers of this size.** Nothing in the chapter depends on a latency
value to better than a few milliseconds, and every conclusion — VGG16 slowest,
roughly 19× spread, parameter count a poor proxy for latency — holds under all
three sets of readings.

All five validation accuracies and all five BAFMD results were identical in both
runs, so the retraining itself is fully deterministic; only the timing varies.

---

## Group 4 — end-to-end throughput

**Commands:** `… -m scripts.benchmark_fps --source mm/calib3.mp4 --frames 150 [--cpu]`
and `… --source mm/test4.mp4 --frames 200 [--cpu]`

Detector YOLOv8n with ByteTrack, processing width 700, person confidence 0.30,
IoU 0.50, face confidence 0.50, mask classification on, annotation on — all from
`config.yaml` defaults, unmodified.

| Condition | Metric | Expected | Actual | Difference | Result |
|---|---|---|---|---|---|
| Sparse MPS | median FPS | 17.26 | 17.83 | +3.3% | PASS — timing |
| | mean FPS | 17.18 | 17.97 | +4.6% | PASS — timing |
| | p95 | 61.7 ms | 57.5 ms | −6.8% | PASS — timing |
| Sparse CPU | median FPS | 14.77 | 14.99 | +1.5% | PASS — timing |
| | mean FPS | 14.70 | 14.98 | +1.9% | PASS — timing |
| | p95 | 69.2 ms | 67.8 ms | −2.0% | PASS — timing |
| Crowded MPS | median FPS | 7.31 | 7.39 | +1.1% | PASS — timing |
| | mean FPS | 6.60 | 6.97 | +5.6% | PASS — timing |
| | p95 | 199.2 ms | 181.7 ms | −8.8% | PASS — timing |
| Crowded CPU | median FPS | 9.07 | 9.20 | +1.4% | PASS — timing |
| | mean FPS | 9.09 | 9.21 | +1.3% | PASS — timing |
| | p95 | 115.2 ms | 114.0 ms | −1.0% | PASS — timing |

Every condition ran slightly faster, consistently — a less loaded machine.

**Conclusions all hold.** NFR-01 (≥5 fps) met everywhere; the worst case, crowded
MPS at p95, improves from 5.0 to 5.5 fps. MPS remains faster on sparse footage
and slower on crowded footage.

### Finding: a self-consistency error this run exposed

Chapter 5 states throughput is reported as **medians**, because means are
distorted by occasional slow frames — then computes its two headline percentages
from the **means**:

- "17% faster on sparse" — 17.18 / 14.70 = 1.169 (mean-based)
- "38% slower on crowded" — 9.09 / 6.60 = 1.377 (mean-based)

Recomputed from medians as the stated methodology requires, the sparse figure is
17% either way, but **the crowded figure is 24%, not 38%**. The re-run confirms
the median figure is the stable one — 24.0% originally, 24.5% now — while the
mean-based figure moved from 38% to 32% between runs. That is precisely the
instability cited as the reason for preferring medians.

---

## Group 5 — 815-frame one-person integration test

**Command:** `… -m scripts.run --source mm/demo3.mp4 --frame-log /tmp/repro_out/demo3_frame_log.csv`

| Value | Expected | Actual | Difference | Result |
|---|---|---|---|---|
| Total frames | 815 | 815 | 0 | PASS |
| Zero people | 127 | 127 | 0 | PASS |
| One person | 675 | 675 | 0 | PASS |
| Two detections | 13 | 13 | 0 | PASS |
| Two-detection rate | 1.60% | 1.60% | 0 | PASS |
| Violation frames | 13 | 13 | 0 | PASS |
| Offender detections | 26 | 26 | 0 | PASS |
| Violating track IDs | 4 | 4 | 0 | PASS |
| No-mask detections | — | 9 | — | — |

The four violating track identifiers are **not** four people. The footage
contains one person; the count reflects identity fragmentation across the
thirteen false-positive frames.

---

## Group 6 — calibration diagnostic

Run in a second environment: Linux, Python 3.13, numpy 2.4.4, OpenCV 4.13.0.92 —
different OS and different library versions from the deployment machine.

**Commands:**
`python3 -m scripts.check_calibration --homography calibration/homography.npy --width 1920 --height 1080 --rect 2.80 2.58`
`python3 -m scripts.check_calibration --homography calibration/homography_a4.npy --width 1920 --height 1080 --rect 0.297 0.210`

| Calibration | Value | Expected | Actual | Result |
|---|---|---|---|---|
| Final | rectangle | 2.80 × 2.58 m | 2.8 × 2.58 m | PASS |
| | coverage | 63.8% | 63.8% | PASS |
| | perspective variation | 56.53% | 56.53% | PASS |
| | far/near scale ratio | — | 1.45× | — |
| | verdict | no problems detected | No problems detected | PASS |
| A4 | coverage | 7.3% | 7.3% | PASS |
| | perspective variation | 159.82% | 159.82% | PASS |
| | verdict | region too small | too small — most of the frame is extrapolated | PASS |
| Original | perspective variation | 2.69% | 2.69% (via ablation) | PASS |

The original image-corner calibration exists only in git history
(`45c4fc6`), so the diagnostic could not be pointed at it without checking out
that commit. Its 2.69% perspective figure is confirmed by the ablation below,
which reads the matrix from history directly.

---

## Group 7 — independent physical distance validation

**Command:** `python3 -m scripts.ablate_calibration --points 227 224 894 221 --true-m 1.48 --safe-m 2.0 --pixel-threshold 80`

Reference: two image points 667.0 px apart, true separation 1.4800 m, not used to
fit any of the homographies. Ground truth against the 2.0 m threshold: VIOLATION.

| Calibration | Value | Expected | Actual | Difference | Result |
|---|---|---|---|---|---|
| Original | estimate | 1.4223 m | 1.4223 m | 0 | PASS |
| | signed error | −3.90% | −3.90% | 0 | PASS |
| A4 | estimate | 0.8435 m | 0.8435 m | 0 | PASS |
| | signed error | −43.01% | −43.01% | 0 | PASS |
| Final | estimate | 1.4680 m | 1.4680 m | 0 | PASS |
| | absolute error | 0.0120 m | 0.0120 m | 0 | PASS |
| | signed error | −0.81% | −0.81% | 0 | PASS |
| | absolute % error | 0.81% | 0.81% | 0 | PASS |
| Fixed 80 px | verdict | incorrectly compliant | **compliant** (ground truth: VIOLATION) | 0 | PASS |
| calib2 | estimate | — | 1.4280 m (−3.51%) | — | — |

The inherited 80-pixel rule reaches the **wrong** verdict on this pair: 667 px of
separation reads as compliant while the true 1.48 m separation is a genuine
breach of the 2.0 m threshold.

**NOT RUN:** `scripts/validate_calibration.py --source mm/calib3.mp4 --frame 15`
— requires the video, which is gitignored and was unavailable in the environment
where this group ran. Every figure it would report (1.4680 m, 0.0120 m, −0.81%)
is independently confirmed by the ablation's final row.

---

## What this means for Chapter 4 and Chapter 5

### Safe to report unchanged

Everything except one latency cell. Specifically:

- **All classifier figures.** 97.88 / 98.34 / 98.60 / 80.56%, AUCs 0.9983 /
  0.9991 / 0.9601, macro-F1 0.7594, every per-class precision and recall, the
  327 and 308 error counts.
- **The entire contamination finding.** 3,672 shared unique, 92.1%, 3,761
  duplicates, 49.8%, the 3,021 holdout with 1,508 contaminated and 49.9%.
- **All five validation accuracies and all parameter counts** — reproduced
  identically in two independent runs.
- **All five Mac CPU latency figures**, including InceptionV3's 22.1 ms,
  confirmed by a second run at 21.1 ms.
- **All BAFMD architecture figures** — now confirmed on a second platform.
- **Every calibration figure.** 2.69 / 159.82 / 56.53% perspective, 7.3 / 63.8%
  coverage, all four distance estimates, −0.81% and 0.0120 m, and the 80-pixel
  rule's wrong verdict. Reproduced under different library versions.
- **Every 815-frame figure**, including the 1.60% false-positive rate.
- **The throughput conclusions**, though not the exact millisecond values, which
  are expected to vary.

### Needs updating

No measured value requires correction. Two wording issues remain.

**1. Chapter 5 §5.3's "38% slower on crowded".** Computed from means while the
chapter states it reports medians. Should be **24%**. The re-run shows the
median-based figure is stable across runs and the mean-based one is not.

**2. Optional: the "exactly 5.0 frames per second" claim.** The committed p95 of
199.2 ms gives 5.0 fps; this run gave 181.7 ms, or 5.5 fps. The committed figure
is the recorded measurement and remains legitimate to report, but "exactly" is
doing more work than the evidence supports across runs.

### Two reporting precisions, not errors

- **Test C and D conflate two N values.** Accuracy is on 3,021 and 3,792
  respectively; both AUCs are on 3,000-image random samples. Report them as
  separate measurements.
- **The Colab notebook's dataset check** prints 2,165 `with_mask` from a
  directory listing, while the evaluation scores 4,092 loadable images
  (2,162 + 1,930). The three-image gap is unloadable files, not a discrepancy.
