import sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
from scripts.evaluate_mask import _load_paths

DATASET = "m/dataset"   # adjust if your local baseline dataset folder differs

# reproduce the 613-image in-distribution holdout exactly
items = _load_paths(Path(DATASET))
rng = np.random.default_rng(42)
rng.shuffle(items)
k = int(len(items) * 0.15)
indist = {str(p.resolve()) for p, _ in items[:k]}
print(f"in-distribution holdout: {len(indist)} paths")

# reproduce the 818-image comparison validation split exactly
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

gen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
val = gen.flow_from_directory(DATASET, subset="validation", shuffle=False,
                               target_size=(224, 224), batch_size=32,
                               class_mode="categorical", classes=["with_mask", "without_mask"])
compare = {str(Path(p).resolve()) for p in val.filepaths}
print(f"compare_models validation split: {len(compare)} paths")

overlap = indist & compare
print(f"overlap: {len(overlap)} images ({len(overlap)/len(indist):.1%} of the holdout)")
