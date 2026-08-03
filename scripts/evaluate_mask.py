"""Evaluate the mask classifier and produce reportable metrics.

Outputs precision, recall, F1 and a confusion matrix figure for the dissertation
results chapter.

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
The supplied ``mask_detector.model`` was trained on the very dataset shipped with
the baseline. Evaluating on that same data measures *training* performance and
will look optimistic (the baseline's headline ~93% is exactly this kind of
number). For an honest result you should either:
  (a) point ``--dataset`` at a *different* mask dataset the model never saw, or
  (b) use ``--holdout`` to score only a held-out fraction (still weaker than a
      truly independent set, but better than scoring all training data).
The script prints which mode it used so the caveat can be stated in the report.

    python -m scripts.evaluate_mask --dataset m/dataset --holdout 0.2 \
        --out output/mask_eval
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # load the Keras-2 baseline model
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = ["without_mask", "with_mask"]  # 0, 1


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
    ap = argparse.ArgumentParser(description="Evaluate the mask classifier")
    ap.add_argument("--dataset", required=True, help="dir containing with_mask/ and without_mask/")
    ap.add_argument("--model", default="models/mask_detector.model")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="evaluate only this random fraction (0 = all images)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output/mask_eval")
    args = ap.parse_args()

    import cv2
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_path = args.model if Path(args.model).is_absolute() else str(ROOT / args.model)
    model = load_model(model_path)

    items = _load_paths(Path(args.dataset))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(items)
    mode = "ALL images (training performance — optimistic)"
    if args.holdout and 0 < args.holdout < 1:
        k = int(len(items) * args.holdout)
        items = items[:k]
        mode = f"held-out {args.holdout:.0%} ({len(items)} images)"

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
    preds = model.predict(X, batch_size=32, verbose=1)
    # model output columns are [with_mask, without_mask]; map to our 0/1 scheme
    y_pred = (preds[:, 0] > preds[:, 1]).astype(int)
    y_true = np.array(y_true)

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    out = Path(ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + "_report.txt", "w") as fh:
        fh.write(f"Evaluation mode: {mode}\nImages scored: {len(y_true)}\n\n{report}\n")
    print(f"Evaluation mode: {mode}")
    print(report)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(str(out) + "_confusion_matrix.png", dpi=150)
    print(f"Saved report and confusion matrix under {out}_*")


if __name__ == "__main__":
    main()
