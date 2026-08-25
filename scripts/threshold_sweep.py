"""Sweep the mask classifier's decision threshold and report what recalibration recovers.

WHY THIS EXISTS
---------------
On BAFMD the deployed classifier scores AUC 0.9601 while macro F1 falls to
0.7594. A high AUC beside a low macro F1 is *consistent with* the decision
threshold sitting in the wrong place for that population rather than with the
model having lost the ability to separate the classes - but consistent with is
not the same as demonstrated. AUC is threshold-free: it says the ranking is
good, not that any particular threshold recovers performance. Claiming that
recalibration "would recover much of the loss" without sweeping the threshold
is an inference presented as a result.

This script measures it.

THE TRAP IT AVOIDS
------------------
Choosing the threshold that maximises macro F1 on a dataset and then reporting
that macro F1 on the same dataset is fitting to the test set. The number is real
but it is an upper bound that no deployment could achieve, because in
deployment the threshold must be chosen without having seen the labels it will
be judged on.

So two figures are reported and they mean different things:

  ORACLE       best macro F1 over the sweep, threshold chosen on the same data.
               An upper bound on what threshold recalibration could ever give.
               NOT an achievable deployment figure.

  CROSS-FITTED the dataset is split in half at random; the threshold is chosen
               on one half and evaluated on the other, both ways round, repeated
               over several seeds. This is what recalibration would actually be
               worth, because the threshold never sees the data it is scored on.

Report the cross-fitted figure as the result. Report the oracle beside it as the
ceiling.

The deployed threshold is 0.50: evaluate_mask.py predicts with_mask when
preds[:,0] > preds[:,1], and the two columns are a softmax pair, so that is
exactly a 0.50 cut on the with_mask score. Image loading, preprocessing and the
score convention are identical to compute_roc_auc.py so the figures are directly
comparable.

USAGE
-----
    python -m scripts.threshold_sweep --dataset data/bafmd_crops \\
        --relation independent --exclude-from m/dataset \\
        --out results/mask_eval/threshold_sweep_bafmd

Produces:
  {out}_report.txt   macro F1 at 0.50, the oracle bound, the cross-fitted
                     estimate, and per-class figures at both thresholds
  {out}_curve.png    macro F1 against threshold, with both marked
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # load the Keras-2 baseline model
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = ["without_mask", "with_mask"]  # 0, 1 - matches evaluate_mask.py exactly
DEPLOYED_THRESHOLD = 0.50


def _load_paths(dataset: Path):
    items = []
    for label, name in enumerate(CLASSES):
        folder = dataset / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected class folder: {folder}")
        for p in folder.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((p, label))
    return items


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of the two per-class F1 scores."""
    f1s = []
    for c in (0, 1):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def per_class(y_true: np.ndarray, y_pred: np.ndarray):
    out = {}
    for c, name in enumerate(CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = (prec, rec, f1, int((y_true == c).sum()))
    return out


def predict(y_score: np.ndarray, t: float) -> np.ndarray:
    """with_mask (1) when the with_mask score is at or above the threshold."""
    return (y_score >= t).astype(int)


def candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    """Every threshold that can produce a distinct classification of these scores.

    Macro F1 as a function of the threshold is a step function: it changes only
    when the threshold crosses an observed score. The midpoints between
    consecutive distinct scores therefore enumerate every reachable outcome,
    exhaustively and without a grid to truncate. This is the same construction
    roc_curve uses internally, which is why a sweep built this way cannot report
    an optimum sitting on the edge of its own search range.
    """
    u = np.unique(y_score)
    if u.size < 2:
        return np.array([float(u[0])], dtype=float)
    mids = (u[:-1] + u[1:]) / 2.0
    # bracket the range so "predict everything" and "predict nothing" are reachable
    lo = float(np.nextafter(u[0], -np.inf))
    hi = float(np.nextafter(u[-1], np.inf))
    return np.unique(np.concatenate([[lo], mids, [hi]]))


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, grid=None):
    """Best macro F1 over candidates drawn from y_score itself.

    ``grid`` is accepted and ignored; it remains only so the call sites read the
    same. Passing a fixed grid is what produced the truncated result this
    replaced.
    """
    cands = candidate_thresholds(y_score)
    scores = np.array([macro_f1(y_true, predict(y_score, t)) for t in cands])
    i = int(scores.argmax())
    return float(cands[i]), float(scores[i]), (cands, scores)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep the classifier's decision threshold and report what recalibration recovers")
    ap.add_argument("--dataset", required=True, help="dir containing with_mask/ and without_mask/")
    ap.add_argument("--model", default="models/mask_detector.model")
    ap.add_argument("--exclude-from", help="drop images byte-identical to those in this dataset")
    ap.add_argument("--relation", choices=["training", "independent", "contaminated", "unknown"],
                    default="unknown",
                    help="relationship of --dataset to the training data, recorded in the report")
    ap.add_argument("--grid", type=int, default=999, help="threshold grid resolution")
    ap.add_argument("--repeats", type=int, default=20,
                    help="random split-half repeats for the cross-fitted estimate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output/threshold_sweep")
    args = ap.parse_args()

    import cv2
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_path = args.model if Path(args.model).is_absolute() else str(ROOT / args.model)
    model = load_model(model_path)

    items = _load_paths(Path(args.dataset))
    excluded = 0
    if args.exclude_from:
        import hashlib
        seen = {hashlib.md5(q.read_bytes()).hexdigest()
                for q, _ in _load_paths(Path(args.exclude_from))}
        before = len(items)
        items = [(q, lab) for q, lab in items
                 if hashlib.md5(q.read_bytes()).hexdigest() not in seen]
        excluded = before - len(items)
        print(f"Excluded {excluded} images also present in {args.exclude_from}")

    X, y_true = [], []
    for path, label in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        X.append(preprocess_input(img_to_array(img)))
        y_true.append(label)
    X = np.array(X, dtype="float32")
    y_true = np.array(y_true)

    preds = model.predict(X, batch_size=32, verbose=1)
    y_score = preds[:, 0]          # with_mask score, same convention as compute_roc_auc.py
    auc_val = float(roc_auc_score(y_true, y_score))


    # deployed
    dep_pred = predict(y_score, DEPLOYED_THRESHOLD)
    dep_f1 = macro_f1(y_true, dep_pred)
    dep_cls = per_class(y_true, dep_pred)

    # oracle: threshold chosen on the same data it is scored on
    orc_t, orc_f1, (curve_x, curve_y) = best_threshold(y_true, y_score)
    orc_cls = per_class(y_true, predict(y_score, orc_t))

    # cross-fitted: threshold chosen on one half, scored on the other, both ways
    rng = np.random.default_rng(args.seed)
    n = len(y_true)
    cf_scores, cf_thresholds = [], []
    for _ in range(args.repeats):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        for tune, test in ((a, b), (b, a)):
            t, _, _ = best_threshold(y_true[tune], y_score[tune])
            cf_scores.append(macro_f1(y_true[test], predict(y_score[test], t)))
            cf_thresholds.append(t)
    cf_mean, cf_sd = float(np.mean(cf_scores)), float(np.std(cf_scores))
    cf_t_mean, cf_t_sd = float(np.mean(cf_thresholds)), float(np.std(cf_thresholds))

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    relation_note = {
        "training": "drawn from the model's own training distribution (optimistic)",
        "independent": "from a dataset independent of the training data",
        "contaminated": "from a dataset that shares images with the training data",
        "unknown": "relationship to the training data not declared",
    }

    emit("Decision-threshold sweep for the deployed mask classifier")
    emit(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}")
    emit(f"Dataset: {args.dataset} - {relation_note[args.relation]}")
    if args.exclude_from:
        emit(f"   {excluded} byte-identical duplicates excluded via --exclude-from {args.exclude_from}")
    emit(f"Images scored: {n} ({int((y_true == 1).sum())} with_mask, "
         f"{int((y_true == 0).sum())} without_mask)")
    emit(f"AUC: {auc_val:.4f}   (threshold-free: the quality of the ranking)")
    emit(f"Threshold candidates: {len(curve_x)} midpoints between consecutive "
         f"distinct scores (exhaustive; no fixed grid, so no truncation)")
    emit("")

    emit("Result")
    emit("")
    emit(f"   {'':<44}{'threshold':>11}{'macro F1':>11}")
    emit("   " + "-" * 66)
    emit(f"   {'deployed, as shipped':<44}{DEPLOYED_THRESHOLD:>11.3f}{dep_f1:>11.4f}")
    emit(f"   {'recalibrated, cross-fitted (ACHIEVABLE)':<44}"
         f"{cf_t_mean:>11.3e}{cf_mean:>11.4f}")
    emit(f"   {'recalibrated, oracle (UPPER BOUND ONLY)':<44}{orc_t:>11.3e}{orc_f1:>11.4f}")
    emit("")
    emit(f"   Cross-fitted spread over {len(cf_scores)} tune/test folds: "
         f"macro F1 sd {cf_sd:.4f}, threshold sd {cf_t_sd:.3e}")
    emit(f"   Recovery attributable to recalibration alone: "
         f"{cf_mean - dep_f1:+.4f} macro F1 "
         f"({(cf_mean - dep_f1) / max(orc_f1 - dep_f1, 1e-9) * 100:.0f}% of the oracle gain)")
    emit("")

    emit("Per class")
    emit("")
    emit(f"   {'class':<16}{'threshold':>10}{'precision':>11}{'recall':>9}{'F1':>9}{'n':>7}")
    emit("   " + "-" * 66)
    for name in CLASSES:
        p, r, f, sup = dep_cls[name]
        emit(f"   {name:<16}{DEPLOYED_THRESHOLD:>10.3f}{p:>11.4f}{r:>9.4f}{f:>9.4f}{sup:>7}")
    emit("")
    for name in CLASSES:
        p, r, f, sup = orc_cls[name]
        emit(f"   {name:<16}{orc_t:>10.3e}{p:>11.4f}{r:>9.4f}{f:>9.4f}{sup:>7}")
    emit("")

    emit("How to read this")
    emit("")
    emit("   The cross-fitted figure is the result. The threshold is chosen on one")
    emit("   random half of the data and scored on the other, twenty times in both")
    emit("   directions, so it never sees the labels it is judged on. That is what")
    emit("   recalibration would actually be worth on unseen data from this")
    emit("   population.")
    emit("")
    emit("   The oracle figure is NOT an achievable result. It picks the threshold")
    emit("   that maximises macro F1 on the very data it then reports, which no")
    emit("   deployment can do. It is included only as the ceiling: no threshold")
    emit("   choice on this data can beat it.")
    emit("")
    emit("   If the cross-fitted figure sits close to the oracle, the threshold is")
    emit("   stable and recalibration transfers. If it sits far below, the best")
    emit("   threshold is an artefact of whichever half it was fitted on and")
    emit("   recalibration is not the answer.")
    emit("")
    _edge = (orc_t <= curve_x[1]) or (orc_t >= curve_x[-2])
    if _edge:
        emit("   WARNING: the optimum sits at the extreme of the observed score")
        emit("   range, which means one class is being predicted for (almost) every")
        emit("   image. Check the per-class rows before reading this as a threshold")
        emit("   effect.")
        emit("")
    emit("Limits")
    emit("")
    emit("   A threshold is fitted and evaluated on halves of the SAME dataset, so")
    emit("   this measures transfer across a random split, not across populations.")
    emit("   It does not show that a threshold tuned here would suit a different")
    emit("   deployment; it shows how much of the deployed loss on THIS population")
    emit("   is attributable to threshold placement rather than to ranking.")
    emit("")
    emit("   Recalibration cannot exceed what the ranking supports. The AUC above")
    emit("   bounds every row in this table.")

    out = Path(ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (Path(str(out) + "_report.txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve_x, curve_y, lw=1.4, color="#c8672a", label="macro F1")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.axvline(DEPLOYED_THRESHOLD, ls="--", lw=1.1, color="#444",
               label=f"deployed 0.50 (F1 {dep_f1:.4f})")
    ax.axvline(orc_t, ls=":", lw=1.3, color="#1f6f8b",
               label=f"oracle {orc_t:.3f} (F1 {orc_f1:.4f})")
    ax.axhline(cf_mean, ls="-.", lw=1.1, color="#2a7d4f",
               label=f"cross-fitted {cf_mean:.4f}")
    ax.set_xlabel("decision threshold on the with_mask score")
    ax.set_ylabel("macro F1")
    ax.set_title("Macro F1 against decision threshold")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(str(out) + "_curve.png", dpi=150)

    print(f"\nWrote {out}_report.txt and {out}_curve.png")


if __name__ == "__main__":
    main()
