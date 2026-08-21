"""Compare multiple CNN backbones for mask classification.

Trains and evaluates several transfer-learning models on the same dataset and
produces a like-for-like comparison table + chart. This is a proper comparative
study: same data, same head, same training budget — only the backbone changes.

Reported per model: test accuracy, macro precision/recall/F1, trainable+total
parameter count, and median inference time per image (a speed/accuracy trade-off,
which matters for a *real-time* surveillance system).

REPRODUCIBILITY
---------------
Every source of randomness that the script controls is seeded from ``--seed``:
Python's ``random``, NumPy, TensorFlow, and the directory iterators that decide
the train/validation split and the shuffling order. Without this the comparison
is a single unrepeatable run, and a reader cannot check any figure in the table.
Residual nondeterminism can remain in GPU/Metal kernels; ``--deterministic``
requests op-level determinism from TensorFlow, at some cost in speed and with the
risk that an unsupported op raises, so it is off by default and the residual is
stated rather than hidden.

ADDITIONAL EVALUATION SETS
--------------------------
``--extra-eval NAME=PATH`` scores each trained model on a further dataset before
the session is cleared. This exists because on curated mask data every
architecture saturates: the accuracy ranking produced here rests on between three
and fifteen misclassified images out of 818, which is not enough to separate the
middle of the field. A harder, independent set produces error counts in the
hundreds, where differences between architectures are actually measurable.

The models trained here are NOT the deployed classifier. They use frozen
backbones with a freshly initialised head, trained for a fixed budget on
``m/dataset``. The deployed ``mask_detector.model`` was trained separately by the
baseline author. A score obtained here for MobileNetV2 therefore corroborates the
deployed model's behaviour only loosely; a large divergence reflects the training
regime, not the architecture.

TIMING
------
Inference time is the median of ``--timing-repeats`` timed batches taken after
``--timing-warmup`` untimed ones. A single timed batch is dominated by first-call
cost — graph tracing and kernel allocation — which differs per architecture and
therefore distorts exactly the comparison this table exists to make. The same
defect was found and fixed in ``scripts/benchmark_fps.py``, where it understated
throughput by roughly 32%.

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
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_seeds(seed: int) -> None:
    """Seed every generator this script controls, before a model is built."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


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


def loaders(dataset, preprocess, size, batch, val_split, seed):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_aug = ImageDataGenerator(preprocessing_function=preprocess, validation_split=val_split,
                                   rotation_range=20, zoom_range=0.15, horizontal_flip=True,
                                   width_shift_range=0.2, height_shift_range=0.2, fill_mode="nearest")
    plain = ImageDataGenerator(preprocessing_function=preprocess, validation_split=val_split)
    common = dict(target_size=(size, size), batch_size=batch, class_mode="categorical",
                  classes=["with_mask", "without_mask"], seed=seed)
    # The seed fixes both the train/validation partition and the shuffling order,
    # so every backbone is trained and scored on exactly the same images.
    train = train_aug.flow_from_directory(dataset, subset="training", shuffle=True, **common)
    val = plain.flow_from_directory(dataset, subset="validation", shuffle=False, **common)
    return train, val


def time_inference(model, xb, repeats: int, warmup: int):
    """Median and spread of per-image inference time, in milliseconds.

    The first predict() call pays for graph tracing and kernel allocation. That
    cost varies by architecture, so timing a single batch measures start-up as
    much as it measures inference. Warm-up runs absorb it; the median over
    repeated runs resists the occasional scheduling outlier.
    """
    for _ in range(warmup):
        model.predict(xb, verbose=0)
    per_image = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(xb, verbose=0)
        per_image.append((time.perf_counter() - t0) / len(xb) * 1000.0)
    spread = statistics.stdev(per_image) if len(per_image) > 1 else 0.0
    return statistics.median(per_image), spread


def evaluate_extra(model, dataset, preprocess, size, batch):
    """Score an already-trained model on a further dataset.

    Uses the same class order as training (``with_mask`` = 0), the architecture's
    own preprocessing and input size, and no shuffling, so the labels line up
    with the predictions. Per-class recall and precision are reported as well as
    accuracy, because on an imbalanced set accuracy alone hides which direction
    the errors run in.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    flow = ImageDataGenerator(preprocessing_function=preprocess).flow_from_directory(
        dataset, target_size=(size, size), batch_size=batch, class_mode="categorical",
        classes=["with_mask", "without_mask"], shuffle=False)
    preds = model.predict(flow, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = flow.classes[:len(y_pred)]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cp, cr, _, sup = precision_recall_fscore_support(y_true, y_pred, average=None,
                                                     zero_division=0, labels=[0, 1])
    return dict(images=int(len(y_true)), accuracy=float(acc), precision=float(p),
                recall=float(r), f1=float(f),
                with_mask_recall=float(cr[0]), with_mask_precision=float(cp[0]),
                without_mask_recall=float(cr[1]), without_mask_precision=float(cp[1]),
                support_with_mask=int(sup[0]), support_without_mask=int(sup[1]))


def evaluate(model, val, repeats: int, warmup: int):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    val.reset()
    preds = model.predict(val, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val.classes[:len(y_pred)]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    val.reset()
    xb = next(iter(val))[0]
    ms, ms_sd = time_inference(model, xb, repeats, warmup)
    return dict(accuracy=acc, precision=p, recall=r, f1=f,
                ms_per_image=ms, ms_per_image_sd=ms_sd,
                timing_repeats=repeats, timing_warmup=warmup,
                timing_batch=int(len(xb)))


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
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds Python, NumPy, TensorFlow and the directory iterators")
    ap.add_argument("--timing-repeats", type=int, default=20,
                    help="timed inference batches per model (median is reported)")
    ap.add_argument("--timing-warmup", type=int, default=3,
                    help="untimed batches run first, to absorb graph tracing and "
                         "kernel allocation")
    ap.add_argument("--extra-eval", nargs="*", default=[], metavar="NAME=PATH",
                    help="further evaluation sets, e.g. bafmd=/content/data/bafmd_crops. "
                         "Each trained model is scored on these before its session is "
                         "cleared, since the models are not saved")
    ap.add_argument("--deterministic", action="store_true",
                    help="request op-level determinism from TensorFlow; slower, and "
                         "raises if an op in the graph has no deterministic kernel")
    args = ap.parse_args()

    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf

    determinism = "seeded"
    if args.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            determinism = "seeded + op determinism"
        except Exception as e:  # unsupported on this build or device
            determinism = f"seeded (op determinism unavailable: {e})"
    print(f"Reproducibility: {determinism}, seed={args.seed}")

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
        # Re-seed before each backbone so the comparison is like-for-like: every
        # model sees the same split, the same shuffle and the same head
        # initialisation, and is not affected by the models run before it.
        set_seeds(args.seed)
        constructor, preprocess, size = registry[name]
        train, val = loaders(args.dataset, preprocess, size, args.batch,
                             args.val_split, args.seed)
        model = make_model(constructor, size, args.imagenet)
        model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=args.epochs,
                  steps_per_epoch=steps, validation_steps=steps, verbose=2)
        metrics = evaluate(model, val, args.timing_repeats, args.timing_warmup)
        extra = {}
        for spec in args.extra_eval:
            if "=" not in spec:
                raise SystemExit(f"--extra-eval expects NAME=PATH, got {spec!r}")
            ename, epath = spec.split("=", 1)
            print(f"  extra evaluation '{ename}' <- {epath}")
            extra[ename] = evaluate_extra(model, epath, preprocess, size, args.batch)
            e = extra[ename]
            print(f"    {ename}: n={e['images']}  acc={e['accuracy']:.4f}  "
                  f"f1={e['f1']:.4f}  with_mask recall={e['with_mask_recall']:.4f}  "
                  f"without_mask precision={e['without_mask_precision']:.4f}")
        if extra:
            metrics["extra"] = extra
        total = model.count_params()
        trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
        metrics.update(total_params=int(total), trainable_params=trainable, input_size=size)
        results[name] = metrics
        print(f"  {name}: acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
              f"{metrics['ms_per_image']:.1f} +/- {metrics['ms_per_image_sd']:.1f} ms/img")
        tf.keras.backend.clear_session()

    # save + print comparison table
    run = dict(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               seed=args.seed, reproducibility=determinism,
               epochs=args.epochs, batch=args.batch, val_split=args.val_split,
               imagenet_weights=args.imagenet,
               timing_repeats=args.timing_repeats, timing_warmup=args.timing_warmup,
               tensorflow=tf.__version__, dataset=str(args.dataset),
               extra_eval=list(args.extra_eval))
    (out / "results.json").write_text(json.dumps({"run": run, "models": results}, indent=2))

    hdr = (f"{'model':<16}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}"
           f"{'ms/img':>9}{'+/-':>7}{'params(M)':>11}")
    preamble = [
        "Model comparison - mask classification",
        f"Date: {run['date']}   TensorFlow: {run['tensorflow']}",
        f"Reproducibility: {determinism}, seed={args.seed}",
        f"Training: {args.epochs} epochs, batch {args.batch}, "
        f"validation split {args.val_split:.2f}, "
        f"ImageNet weights: {'yes' if args.imagenet else 'no'}",
        f"Inference timing: median of {args.timing_repeats} timed batches after "
        f"{args.timing_warmup} warm-up batches; +/- is the standard deviation",
        "",
    ]
    lines = preamble + [hdr, "-" * len(hdr)]
    for n, m in sorted(results.items(), key=lambda kv: -kv[1]["accuracy"]):
        lines.append(f"{n:<16}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}"
                     f"{m['f1']:>8.4f}{m['ms_per_image']:>9.1f}{m['ms_per_image_sd']:>7.1f}"
                     f"{m['total_params']/1e6:>11.1f}")
    # one further table per additional evaluation set
    extra_names = sorted({k for m in results.values() for k in m.get("extra", {})})
    for ename in extra_names:
        lines.append("")
        lines.append(f"Additional evaluation set: {ename}")
        lines.append("These models are not the deployed classifier; they use frozen backbones")
        lines.append("with a freshly initialised head trained on the dataset above.")
        h2 = (f"{'model':<16}{'images':>8}{'acc':>8}{'f1':>8}"
              f"{'mask rec':>10}{'nomask rec':>12}{'nomask prec':>13}")
        lines.append(h2)
        lines.append("-" * len(h2))
        ranked = sorted(results.items(),
                        key=lambda kv: -kv[1].get("extra", {}).get(ename, {}).get("accuracy", -1))
        for n, m in ranked:
            e = m.get("extra", {}).get(ename)
            if not e:
                continue
            lines.append(f"{n:<16}{e['images']:>8}{e['accuracy']:>8.4f}{e['f1']:>8.4f}"
                         f"{e['with_mask_recall']:>10.4f}{e['without_mask_recall']:>12.4f}"
                         f"{e['without_mask_precision']:>13.4f}")

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
