# Dissertation figures

The three data figures in the dissertation are built by `scripts/make_figures.py`
from the committed artefacts under `results/`. No value in any of them is typed
in by hand; each is parsed from the report that produced it, so a figure cannot
drift away from the measurement it depicts.

Regenerate all three with:

    python -m scripts.make_figures

Each figure is written as both PNG (300 dpi, for the document) and PDF (vector,
for printing).

## Figure 4.1 - Confusion matrices

Sources:

  - `results/mask_eval/mask_eval_deduplicated_report.txt` (screened cross-dataset)
  - `results/mask_eval/mask_eval_bafmd_report.txt` (BAFMD)

The matrices are reconstructed from per-class recall and support. For two
classes those two quantities determine the matrix exactly: recall times support
is the diagonal entry, and the remainder of that true-class row is the single
off-diagonal entry. The reconstruction reproduces the counts stated in the text
of Chapter 4 (44 and 9 for the screened evaluation, 19 and 308 for BAFMD).

Shading is normalised within each true-class row, because the two evaluations
differ in size by more than a factor of two and the comparison is between error
rates rather than counts. The raw counts are printed in every cell, so the
normalisation conceals nothing.

## Figure 4.2 - BAFMD macro-F1 against Mac CPU latency

Sources:

  - `results/model_comparison/comparison_table.txt`
    (latency, measured on the Mac CPU)
  - `results/model_comparison/colab_t4_crosscheck/comparison_table.txt`
    (BAFMD macro-F1, from the additional-evaluation block)

The two axes come from different runs deliberately. Latency is a property of the
machine the system is intended to run on, and is therefore taken from the Mac
measurement. BAFMD macro-F1 is a property of the trained model rather than of
the host, and is taken from the cross-check run, in which four of the five
architectures reproduced exactly and MobileNetV2 differed by a single image.

Identity is carried by the point labels rather than by colour, so the figure
remains readable in greyscale.

## Figure 5.1 - Threshold recalibration on BAFMD

Source: `results/mask_eval/threshold_sweep_bafmd_report.txt`

Two bars are drawn: the macro-F1 at the deployed threshold of 0.500, and the
cross-fitted macro-F1 obtained by selecting a threshold on one random half of
BAFMD and scoring it on the other, over 40 tune/test splits. The error bar is
the standard deviation of macro-F1 across those splits.

The oracle value is drawn as a reference line rather than as a third bar. It is
the macro-F1 of a threshold chosen on the very data it is then scored on, which
no deployment can achieve, and giving it a bar would invite it to be read as an
achievable result.
