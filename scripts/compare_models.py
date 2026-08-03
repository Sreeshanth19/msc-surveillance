"""Compare multiple CNN backbones for mask classification.

Trains and evaluates several transfer-learning models on the same dataset and
produces a like-for-like comparison table + chart. This is a proper comparative
study: same data, same head, same training budget — only the backbone changes.

Reported per model: test accuracy, macro precision/recall/F1, trainable+total
parameter count, and mean inference time per image (a speed/accuracy trade-off,
which matters for a *real-time* surveillance system).

Requires a GPU and internet for ImageNet weights on the first run. Example:

    python -m scripts.compare_models --dataset /path/to/data \
        --models mobilenetv2,resnet50,vgg16,inceptionv3,efficientnetb0 \
        --epochs 12 --out output/model_comparison

Dataset layout: <dataset>/with_mask/*  and  <dataset>/without_mask/*
"""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_registry():
    """name -> (constructor, preprocess_input, input_size)."""
    from tensorflow.keras import applications as A
    from tensorflow.keras.applications import (mobilenet_v2, resnet, vgg16,
                                               inception_v3, efficientnet, densenet)
    return {
        "mobilenetv2":   (A.MobileNetV2,   mobilenet_v2.preprocess_input, 224),
        "resnet50":      (A.ResNet50,      resnet.preprocess_input,       224),
        "vgg16":         (A.VGG16,         vgg16.preprocess_input,        224),
        "inceptionv3":   (A.InceptionV3,   inception_v3.preprocess_input, 299),
        "efficientnetb0":(A.EfficientNetB0, efficientnet.preprocess_input,224),
        "densenet121":   (A.DenseNet121,   densenet.preprocess_input,     224),
    }


def make_model(constructor, size, imagenet):
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
    from tensorflow.keras.models import Model
    base = constructor(weights="imagenet" if imagenet else None,
                       include_top=False, input_tensor=Input(shape=(size, size, 3)))
    for layer in base.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(2, activation="softmax")(x)
    return Model(base.input, out)


def loaders(dataset, preprocess, size, batch, val_split):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_aug = ImageDataGenerator(preprocessing_function=preprocess, validation_split=val_split,
                                   rotation_range=20, zoom_range=0.15, horizontal_flip=True,
                                   width_shift_range=0.2, height_shift_range=0.2, fill_mode="nearest")
    plain = ImageDataGenerator(preprocessing_function=preprocess, validation_split=val_split)
    common = dict(target_size=(size, size), batch_size=batch, class_mode="categorical",
                  classes=["with_mask", "without_mask"])
    train = train_aug.flow_from_directory(dataset, subset="training", shuffle=True, **common)
    val = plain.flow_from_directory(dataset, subset="validation", shuffle=False, **common)
    return train, val


def evaluate(model, val):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    val.reset()
    preds = model.predict(val, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val.classes[:len(y_pred)]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    # inference timing on one batch
    xb = next(iter(val))[0]
    t0 = time.time(); model.predict(xb, verbose=0); dt = (time.time() - t0) / len(xb)
    return dict(accuracy=acc, precision=p, recall=r, f1=f, ms_per_image=dt * 1000)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare CNN backbones for mask classification")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="mobilenetv2,resnet50,vgg16,inceptionv3,efficientnetb0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--out", default="output/model_comparison")
    ap.add_argument("--no-imagenet", dest="imagenet", action="store_false", default=True)
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps/epoch (smoke test only)")
    args = ap.parse_args()

    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf

    registry = build_registry()
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise SystemExit(f"Unknown models: {unknown}. Available: {list(registry)}")

    out = Path(ROOT / args.out); out.mkdir(parents=True, exist_ok=True)
    results = {}
    steps = args.max_steps or None

    for name in names:
        print(f"\n{'='*54}\n  {name}\n{'='*54}")
        constructor, preprocess, size = registry[name]
        train, val = loaders(args.dataset, preprocess, size, args.batch, args.val_split)
        model = make_model(constructor, size, args.imagenet)
        model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=args.epochs,
                  steps_per_epoch=steps, validation_steps=steps, verbose=2)
        metrics = evaluate(model, val)
        total = model.count_params()
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        metrics.update(total_params=int(total), trainable_params=trainable, input_size=size)
        results[name] = metrics
        print(f"  {name}: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
              f"{metrics['ms_per_image']:.1f} ms/img")
        tf.keras.backend.clear_session()

    # save + print comparison table
    (out / "results.json").write_text(json.dumps(results, indent=2))
    hdr = f"{'model':<16}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}{'ms/img':>9}{'params(M)':>11}"
    lines = [hdr, "-" * len(hdr)]
    for n, m in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]):
        lines.append(f"{n:<16}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}"
                     f"{m['f1']:>8.4f}{m['ms_per_image']:>9.1f}{m['total_params']/1e6:>11.1f}")
    table = "\n".join(lines)
    (out / "comparison_table.txt").write_text(table)
    print("\n" + table)

    # bar chart of accuracy
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ns = list(results); accs = [results[n]["accuracy"] for n in ns]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(ns, accs, color="#1C6A8F")
        ax.set_ylim(min(accs) - 0.05 if accs else 0, 1.0)
        ax.set_ylabel("Test accuracy"); ax.set_title("Mask classifier — model comparison")
        for i, a in enumerate(accs):
            ax.text(i, a + 0.003, f"{a:.3f}", ha="center", fontsize=9)
        plt.xticks(rotation=20); fig.tight_layout()
        fig.savefig(out / "comparison_accuracy.png", dpi=150)
        print(f"\nSaved: {out}/results.json, comparison_table.txt, comparison_accuracy.png")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
