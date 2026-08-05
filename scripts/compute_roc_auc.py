"""Compute ROC curve and AUC for the mask classifier.

Reuses the same image-loading, preprocessing, and model-output convention as
evaluate_mask.py, so results are directly comparable and methodologically
consistent with the rest of this project's evaluation. Unlike evaluate_mask.py,
this script scores the model's raw probability output rather than just the hard
0/1 prediction, which is what a ROC curve and AUC actually need.

METHODOLOGICAL NOTE (same caveat as evaluate_mask.py applies here)
--------------------------------------------------------------------
Point --dataset at an INDEPENDENT dataset the model never trained on for an
honest AUC — e.g. the same cross-dataset evaluation that produced the 98.57%
accuracy figure. Scoring on the training dataset itself will look optimistic.

    python -m scripts.compute_roc_auc --dataset /path/to/cross_dataset --sample 3000 \
        --out output/roc_auc

Produces:
  - {out}_report.txt    : AUC value + evaluation mode + image count
  - {out}_roc_curve.png : the actual ROC curve plot
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # load the Keras-2 baseline model
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = ["without_mask", "with_mask"]  # 0, 1 — matches evaluate_mask.py exactly


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ROC curve and AUC for the mask classifier")
    ap.add_argument("--dataset", required=True, help="dir containing with_mask/ and without_mask/")
    ap.add_argument("--model", default="models/mask_detector.model")
    ap.add_argument("--sample", type=int, default=0,
                    help="evaluate only this many randomly sampled images (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output/roc_auc")
    args = ap.parse_args()

    import cv2
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from sklearn.metrics import roc_curve, auc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_path = args.model if Path(args.model).is_absolute() else str(ROOT / args.model)
    model = load_model(model_path)

    items = _load_paths(Path(args.dataset))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(items)
    mode = f"ALL images ({len(items)})"
    if args.sample and 0 < args.sample < len(items):
        items = items[:args.sample]
        mode = f"random sample of {len(items)} images (seed={args.seed})"

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
    # model output columns are [with_mask, without_mask] (same convention as evaluate_mask.py);
    # ROC/AUC need a continuous score for the POSITIVE class, which here is "with_mask" (label 1)
    y_score = preds[:, 0]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    out = Path(ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + "_report.txt", "w") as fh:
        fh.write(f"Evaluation mode: {mode}\nImages scored: {len(y_true)}\nAUC: {roc_auc:.4f}\n")
    print(f"Evaluation mode: {mode}")
    print(f"Images scored: {len(y_true)}")
    print(f"AUC: {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Mask Classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(out) + "_roc_curve.png", dpi=150)
    print(f"Saved report and ROC curve under {out}_*")


if __name__ == "__main__":
    main()