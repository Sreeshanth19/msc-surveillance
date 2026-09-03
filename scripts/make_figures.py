"""Regenerate the three data figures reported in the dissertation.

Every figure this project reports is meant to be traceable to the artefact that
produced it. Figures 4.1, 4.2 and 5.1 were the exception: they were drawn once,
by hand, and no script in the repository could rebuild them. This script closes
that gap.

Nothing here is a new measurement. Every value is parsed out of a committed
report under ``results/``; no figure is hard-coded, so a change to any artefact
changes the figure and a disagreement between the two is impossible to hide.

    Figure 4.1  Confusion matrices, screened cross-dataset and BAFMD
                sources: results/mask_eval/mask_eval_deduplicated_report.txt
                         results/mask_eval/mask_eval_bafmd_report.txt
                The matrices are reconstructed from per-class recall and
                support, which determine them exactly for two classes.

    Figure 4.2  BAFMD macro-F1 against Mac CPU inference latency
                sources: results/model_comparison/comparison_table.txt
                             (latency, measured on the Mac CPU)
                         results/model_comparison/colab_t4_crosscheck/
                             comparison_table.txt  (BAFMD macro-F1)

    Figure 5.1  BAFMD macro-F1 at the deployed threshold and after
                cross-fitted threshold selection
                source: results/mask_eval/threshold_sweep_bafmd_report.txt

Run from the project root:

    python -m scripts.make_figures

Figures are written to results/figures/ at 300 dpi. The console prints every
value it parsed, so the numbers can be checked against the dissertation without
opening the images.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"

# Restrained, print-first styling: the dissertation is read on paper and may be
# photocopied, so identity is carried by position and text rather than by hue.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
FILL = "#3d6b8e"
FILL_LIGHT = "#c9d8e4"
ACCENT = "#b0623a"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def read(rel: str) -> str:
    p = RESULTS / rel
    if not p.exists():
        sys.exit(f"Missing artefact: {p}\nRun this from the project root.")
    return p.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_class_report(text: str) -> dict:
    """Pull per-class precision, recall and support out of an evaluation report.

    The reports are sklearn classification_report output, so each class line is
    name, precision, recall, f1, support.
    """
    out = {}
    for name in ("without_mask", "with_mask"):
        m = re.search(
            rf"^\s*{name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
            text, re.M)
        if not m:
            sys.exit(f"Could not read the {name} row from a report.")
        out[name] = dict(precision=float(m.group(1)), recall=float(m.group(2)),
                         f1=float(m.group(3)), support=int(m.group(4)))
    n = re.search(r"^Images scored:\s*(\d+)", text, re.M)
    out["n"] = int(n.group(1)) if n else None
    acc = re.search(r"^\s*accuracy\s+([\d.]+)\s+\d+\s*$", text, re.M)
    out["accuracy"] = float(acc.group(1)) if acc else None
    mac = re.search(r"^\s*macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text, re.M)
    out["macro_f1"] = float(mac.group(1)) if mac else None
    return out


def confusion_from_report(rep: dict) -> np.ndarray:
    """Recover the 2x2 confusion matrix from recall and support.

    For two classes the pair determines the matrix exactly: recall times support
    is the diagonal, and each off-diagonal is what remains in that true-class
    row. Rows are the true class, columns the predicted class, ordered
    (without_mask, with_mask).
    """
    m = np.zeros((2, 2), dtype=int)
    for i, name in enumerate(("without_mask", "with_mask")):
        n = rep[name]["support"]
        hit = int(round(rep[name]["recall"] * n))
        m[i, i] = hit
        m[i, 1 - i] = n - hit
    return m


def parse_comparison_table(text: str) -> dict:
    """Main table: model -> accuracy and ms/image."""
    out = {}
    for m in re.finditer(
            r"^([a-z0-9]+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+"
            r"([\d.]+)\s+[\d.]+\s+([\d.]+)\s*$", text, re.M):
        out[m.group(1)] = dict(accuracy=float(m.group(2)), f1=float(m.group(3)),
                               ms=float(m.group(4)), params_m=float(m.group(5)))
    if not out:
        sys.exit("Could not read the model comparison table.")
    return out


def parse_bafmd_block(text: str) -> dict:
    """Additional-evaluation block: model -> BAFMD accuracy and macro-F1."""
    block = text.split("Additional evaluation set: bafmd", 1)
    if len(block) < 2:
        sys.exit("No BAFMD block in the cross-check comparison table.")
    out = {}
    for m in re.finditer(r"^([a-z0-9]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
                         block[1], re.M):
        out[m.group(1)] = dict(n=int(m.group(2)), accuracy=float(m.group(3)),
                               f1=float(m.group(4)))
    if not out:
        sys.exit("Could not read the BAFMD rows.")
    return out


def parse_sweep(text: str) -> dict:
    """Threshold sweep: the deployed, cross-fitted and oracle macro-F1."""
    def row(label):
        m = re.search(rf"^\s*{label}[^\n]*?([\d.]+e?-?\d*)\s+([\d.]+)\s*$",
                      text, re.M)
        if not m:
            sys.exit(f"Could not read the '{label}' row from the sweep report.")
        return float(m.group(1)), float(m.group(2))
    dep_t, dep_f1 = row("deployed, as shipped")
    cf_t, cf_f1 = row(r"recalibrated, cross-fitted")
    or_t, or_f1 = row(r"recalibrated, oracle")
    sd = re.search(r"macro F1 sd ([\d.]+)", text)
    auc = re.search(r"^AUC:\s*([\d.]+)", text, re.M)
    n = re.search(r"^Images scored:\s*(\d+)", text, re.M)
    return dict(deployed=(dep_t, dep_f1), crossfit=(cf_t, cf_f1),
                oracle=(or_t, or_f1),
                sd=float(sd.group(1)) if sd else None,
                auc=float(auc.group(1)) if auc else None,
                n=int(n.group(1)) if n else None)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def figure_4_1(screened: dict, bafmd: dict) -> None:
    """Paired confusion matrices, each normalised within its true-class row.

    Counts differ by more than a factor of two between the two evaluations, so
    the shading is row-normalised: it encodes the error RATE, which is what the
    comparison is about, and the raw counts are printed on every cell so the
    normalisation hides nothing.
    """
    labels = ["without\nmask", "with\nmask"]
    panels = [("Screened cross-dataset", screened),
              ("BAFMD", bafmd)]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for ax, (title, rep) in zip(axes, panels):
        cm = confusion_from_report(rep)
        rates = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{rates[i, j]*100:.1f}%",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if rates[i, j] > 0.55 else INK)
        # set_xticks(ticks, labels) needs matplotlib >= 3.5; the two-call form
        # below behaves identically and works on every version
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("predicted", fontsize=8.5)
        ax.set_ylabel("actual", fontsize=8.5)
        ax.set_title(f"{title}\nn = {rep['n']:,}   accuracy "
                     f"{rep['accuracy']*100:.2f}%   macro F1 {rep['macro_f1']:.4f}",
                     fontsize=8.5, pad=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    save(fig, "Figure_4_1_confusion_matrices")


def figure_4_2(mac: dict, baf: dict) -> None:
    """Macro-F1 against latency. Identity is carried by the label, not by hue,
    so the figure survives greyscale printing."""
    order = ["mobilenetv2", "efficientnetb0", "resnet50", "inceptionv3", "vgg16"]
    pretty = {"mobilenetv2": "MobileNetV2", "efficientnetb0": "EfficientNetB0",
              "resnet50": "ResNet50", "inceptionv3": "InceptionV3",
              "vgg16": "VGG16"}
    # label placement chosen per point so nothing collides with a neighbour
    offsets = {"mobilenetv2": (6, -12), "efficientnetb0": (7, 4),
               "resnet50": (7, 4), "inceptionv3": (-8, 8), "vgg16": (-8, 8)}
    align = {"inceptionv3": "right", "vgg16": "right"}

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    xs = [mac[k]["ms"] for k in order]
    ys = [baf[k]["f1"] for k in order]
    ax.scatter(xs, ys, s=42, color=FILL, edgecolor="white", linewidth=0.8,
               zorder=3)
    for k in order:
        ax.annotate(pretty[k], (mac[k]["ms"], baf[k]["f1"]),
                    textcoords="offset points", xytext=offsets[k],
                    fontsize=8.5, color=INK,
                    ha=align.get(k, "left"), zorder=4)
    ax.set_xlabel("Mac CPU inference latency (ms per image)")
    ax.set_ylabel("BAFMD macro-F1")
    ax.grid(axis="both", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(xs) * 1.18)
    ax.set_ylim(min(ys) - 0.06, max(ys) + 0.06)
    fig.tight_layout()
    save(fig, "Figure_4_2_BAFMD_macroF1_vs_Mac_CPU_latency")


def figure_5_1(sw: dict) -> None:
    """Deployed against cross-fitted macro-F1, with the oracle drawn as a
    reference line rather than a third bar: it is an upper bound no deployment
    can reach, and a bar would invite reading it as an achievable result."""
    names = ["Deployed threshold\n(0.500)", "Cross-fitted\nthreshold selection"]
    vals = [sw["deployed"][1], sw["crossfit"][1]]

    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    bars = ax.bar(names, vals, width=0.5,
                  color=[FILL_LIGHT, FILL], edgecolor="white", linewidth=1.5,
                  zorder=3)
    if sw["sd"]:
        ax.errorbar([1], [vals[1]], yerr=[sw["sd"]], fmt="none",
                    ecolor=MUTED, elinewidth=1, capsize=4, zorder=4)
    # values sit inside the bars: the cross-fitted bar rises to within 0.01 of
    # the oracle line, so a label above it would collide with that annotation
    for b, v, colour in zip(bars, vals, (INK, "white")):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.045, f"{v:.4f}",
                ha="center", va="top", fontsize=9.5, color=colour, zorder=5)

    oracle = sw["oracle"][1]
    ax.axhline(oracle, color=ACCENT, linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=2)
    ax.text(-0.42, oracle + 0.014,
            f"oracle ceiling {oracle:.4f} (not achievable)",
            ha="left", va="bottom", fontsize=7.8, color=ACCENT, zorder=5)

    ax.set_ylabel("BAFMD macro-F1")
    ax.set_ylim(0, 1.06)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.5, length=0)
    ax.tick_params(axis="x", colors=INK)
    fig.tight_layout()
    save(fig, "Figure_5_1_BAFMD_Threshold_Validation")


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote results/figures/{stem}.png (and .pdf)")


# --------------------------------------------------------------------------

def main() -> None:
    screened = parse_class_report(read("mask_eval/mask_eval_deduplicated_report.txt"))
    bafmd = parse_class_report(read("mask_eval/mask_eval_bafmd_report.txt"))
    mac = parse_comparison_table(read("model_comparison/comparison_table.txt"))
    baf = parse_bafmd_block(read("model_comparison/colab_t4_crosscheck/comparison_table.txt"))
    sweep = parse_sweep(read("mask_eval/threshold_sweep_bafmd_report.txt"))

    print("Values parsed from the committed artefacts\n")
    for label, rep in (("Screened cross-dataset", screened), ("BAFMD", bafmd)):
        cm = confusion_from_report(rep)
        print(f"  {label}: n={rep['n']:,}  accuracy={rep['accuracy']:.4f}  "
              f"macro F1={rep['macro_f1']:.4f}")
        print(f"      confusion [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]"
              "  (rows actual, cols predicted, order without_mask/with_mask)")
    print()
    for k in ("inceptionv3", "resnet50", "efficientnetb0", "vgg16", "mobilenetv2"):
        print(f"  {k:<16} Mac {mac[k]['ms']:>5.1f} ms   "
              f"BAFMD macro F1 {baf[k]['f1']:.4f}")
    print()
    print(f"  threshold sweep: deployed {sweep['deployed'][1]:.4f} at "
          f"{sweep['deployed'][0]:.3f}, cross-fitted {sweep['crossfit'][1]:.4f} "
          f"at {sweep['crossfit'][0]:.3e}, oracle {sweep['oracle'][1]:.4f}")
    print(f"  cross-fitted spread: macro F1 sd {sweep['sd']}")
    print()

    figure_4_1(screened, bafmd)
    figure_4_2(mac, baf)
    figure_5_1(sweep)
    print("\nDone. Figures written to results/figures/.")


if __name__ == "__main__":
    main()
