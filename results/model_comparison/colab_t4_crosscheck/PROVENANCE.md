# Model comparison — cross-check on Google Colab (NVIDIA T4)

An independent re-run of the model comparison on different hardware, and the
source of the BAFMD architecture evaluation.

| | |
|---|---|
| Produced by | `notebooks/model_comparison_crosscheck_colab.ipynb` |
| Script | `scripts/compare_models.py` at commit **`3655ea1`**, cloned from GitHub inside the notebook |
| Hardware | Google Colab, NVIDIA T4 GPU |
| Python / TensorFlow | Python 3.12 / TensorFlow **2.20.0** (CUDA build) |
| GPU reported | Tesla T4, 13,757 MB, compute capability 7.5, XLA enabled |
| Completed | 2026-08-21 13:57:43 |
| Dataset | `m/dataset` — 2,165 `with_mask`, 1,930 `without_mask`; 3,274 train / 818 validation |
| Additional evaluation | `data/bafmd_crops` — 1,354 / 328, 1,682 crops |
| Flags | `--epochs 12 --seed 42 --extra-eval bafmd=…` (identical to the deployment-machine run, plus the BAFMD evaluation) |
| Date | 21 August 2026 |

The notebook clones the repository rather than embedding the code, so the script
that ran here is provably the committed one. That also demonstrates the
repository is runnable from a clean environment.

## Why this run exists

**To test reproducibility.** The comparison is seeded, so a second machine should
reach the same accuracies. Without a second run there is no evidence the figures
are anything more than one machine's behaviour.

**To separate hardware from method.** An earlier Colab run
(`../colab_t4/`) differed from the deployment-machine run in hardware, dataset
*and* timing protocol simultaneously, so nothing could be attributed to anything.
Here only the hardware differs.

**To make the comparison discriminating.** On curated data the five architectures
are separated by 3 to 16 misclassified images out of 818, which cannot support a
ranking. BAFMD produces 97 to 240 errors, where differences are measurable. The
trained models are never saved, so this evaluation had to happen inside the run;
adding it later would have meant retraining all five.

## Result 1 — accuracy reproduced

| model | deployment machine (CPU) | this run (T4) |
|---|---|---|
| inceptionv3 | 0.9963 | 0.9963 |
| resnet50 | 0.9927 | 0.9927 |
| efficientnetb0 | 0.9927 | 0.9927 |
| vgg16 | 0.9890 | 0.9890 |
| mobilenetv2 | 0.9817 | 0.9804 |

Four of five reproduce to four decimal places. MobileNetV2 differs by one image
out of 818.

State the environment claim precisely: **different hardware and different Python
versions, on the same TensorFlow version (2.20.0) built for different
platforms** — CUDA/Linux here, macOS CPU there. This is not a claim about
different framework versions.

## Result 2 — latency is hardware-dependent

| model | deployment machine (ms) | this run (ms) |
|---|---|---|
| mobilenetv2 | 3.0 | 3.8 |
| efficientnetb0 | 5.1 | 5.0 |
| resnet50 | 16.0 | 5.3 |
| vgg16 | 56.3 | 7.1 |
| inceptionv3 | 22.1 | 8.2 |

The spread collapses from 19× to 2.2×, and the ranking inverts at the top: VGG16
is clearly slowest on CPU and only second-slowest on GPU, where InceptionV3 takes
that position. A GPU parallelises away most of VGG16's arithmetic disadvantage.

**The figures reported in the dissertation are the deployment-machine ones.**
Latency must be measured on the hardware the system runs on, and every other
performance figure in this project — the pipeline throughput benchmarks, the
MPS-versus-CPU detector comparison — comes from that machine. This column is the
hardware comparison, not a replacement.

It also explains the earlier Colab table: its latencies were approximately right
*for a T4* and wrong as a description of the deployment machine. The defect was
hardware attribution rather than arithmetic.

## Result 3 — BAFMD

The BAFMD architecture evaluation exists **only in this run**; the
deployment-machine run predates the `--extra-eval` option. Accuracy is
hardware-independent, so this is sound, but the provenance should be stated
rather than blurred: in-distribution accuracy comes from both machines, latency
from the deployment machine, BAFMD from here.

See `comparison_table.txt` for the figures. The headline is that all five
retrained architectures exceed the deployed classifier's 80.56 %, and that
recall on the unmasked class — the metric a compliance system depends on —
ranges from 0.8384 down to 0.3079 across architectures that are separated by
less than two accuracy points in distribution.

**These models are not the deployed classifier.** They use frozen backbones with
a freshly initialised head and a uniform 12-epoch budget; `mask_detector.model`
was trained separately by the baseline author. The comparison is between the
inherited model as shipped and architectures retrained here under one procedure.

## Files

| File | Contents |
|---|---|
| `comparison_table.txt` | both tables, with the provenance header |
| `results.json` | full metrics including the `run` block recording seed, flags, versions |
| `comparison_accuracy.png` | accuracy chart |
| `run_log_colab.txt` | complete console output, including per-epoch training history |
