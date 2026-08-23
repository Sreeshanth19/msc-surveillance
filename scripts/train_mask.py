"""Train a mask classifier from scratch on a nominated dataset.

Why this exists
---------------
The baseline author's pre-trained ``mask_detector.model`` is enough to get the
pipeline running end to end, but it carries no record of what it was trained on
or how. Training a classifier here instead makes the dataset, the backbone and
the hyperparameters explicit and reproducible, and allows different backbones
and datasets to be compared as a controlled experiment rather than assumed.

This trains a MobileNetV2 transfer-learning classifier (head only, base frozen)
and streams images from disk (no loading the whole set into memory), so it runs
without the out-of-memory problems a naive in-memory loader hits on large sets.

Dataset layout expected (class-per-folder):
    <dataset>/with_mask/*.jpg
    <dataset>/without_mask/*.jpg

Example (on a GPU machine):
    python -m scripts.train_mask --dataset /path/to/data --out models/my_mask_model.keras --epochs 15
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a mask classifier")
    ap.add_argument("--dataset", required=True, help="folder with with_mask/ and without_mask/")
    ap.add_argument("--out", default="models/my_mask_model.keras")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--imagenet", dest="imagenet", action="store_true", default=True,
                    help="initialise MobileNetV2 with ImageNet weights (default)")
    ap.add_argument("--no-imagenet", dest="imagenet", action="store_false",
                    help="random init (used only for a quick pipeline smoke test)")
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps/epoch (smoke test only)")
    args = ap.parse_args()

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    size = args.img_size
    # streaming loaders with light augmentation on the training split
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input, validation_split=args.val_split,
        rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.15, horizontal_flip=True, fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=args.val_split)

    common = dict(target_size=(size, size), batch_size=args.batch,
                  class_mode="categorical", classes=["with_mask", "without_mask"])
    train = train_gen.flow_from_directory(args.dataset, subset="training", shuffle=True, **common)
    val = val_gen.flow_from_directory(args.dataset, subset="validation", shuffle=False, **common)

    base = MobileNetV2(weights="imagenet" if args.imagenet else None,
                       include_top=False, input_tensor=Input(shape=(size, size, 3)))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(2, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    steps = args.max_steps if args.max_steps else None
    model.fit(train, validation_data=val, epochs=args.epochs,
              steps_per_epoch=steps, validation_steps=steps)

    out_path = args.out if Path(args.out).is_absolute() else str(ROOT / args.out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"\nSaved trained model -> {out_path}")
    print("Note: class order is [with_mask, without_mask], matching the pipeline.")
    print("Next: evaluate it with scripts/evaluate_mask.py, and compare against the baseline model.")


if __name__ == "__main__":
    main()
