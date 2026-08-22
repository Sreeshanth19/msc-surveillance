"""Measure byte-identical overlap between two image datasets.

WHY THIS EXISTS
---------------
This project evaluated the inherited mask classifier on a second, separately
obtained dataset and reported the result as cross-dataset generalisation. That
claim was wrong. The two datasets share thousands of byte-identical images,
almost certainly because both were assembled from overlapping upstream sources,
so a large part of the "unseen" evaluation set was training data.

A different dataset is not the same thing as an unseen dataset, and the
difference is not visible by inspection — the file names differ, the folder
structures differ, and the images look like different collections. Only content
hashing shows it.

This script produces the artefact behind that finding: how many images the two
datasets share, what fraction of the baseline that represents, and how
contaminated a given holdout fraction of the evaluation set actually was.

    python -m scripts.check_dataset_overlap --baseline m/dataset --evaluation data
    python -m scripts.check_dataset_overlap --holdout 0.4
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_mask import _load_paths, CLASSES  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Byte-identical overlap between two datasets")
    ap.add_argument("--baseline", default="m/dataset",
                    help="the dataset the deployed model was trained on")
    ap.add_argument("--evaluation", default="data",
                    help="the dataset used as an independent evaluation set")
    ap.add_argument("--holdout", type=float, default=0.4,
                    help="reproduce the contamination of this holdout fraction")
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the seed used by evaluate_mask.py")
    ap.add_argument("--out", default="results/dataset/dataset_overlap_report.txt")
    args = ap.parse_args()

    base_items = _load_paths(Path(args.baseline))
    eval_items = _load_paths(Path(args.evaluation))

    base_hashes = {}
    for p, lab in base_items:
        base_hashes.setdefault(digest(p), []).append((p, lab))
    eval_hashed = [(p, lab, digest(p)) for p, lab in eval_items]

    shared = {h for _, _, h in eval_hashed if h in base_hashes}
    n_shared_eval = sum(1 for _, _, h in eval_hashed if h in base_hashes)

    # class balance, for context
    base_by_class = Counter(CLASSES[lab] for _, lab in base_items)
    eval_by_class = Counter(CLASSES[lab] for _, lab in eval_items)

    # reproduce the holdout exactly as evaluate_mask.py forms it
    rng = np.random.default_rng(args.seed)
    order = list(eval_hashed)
    rng.shuffle(order)
    k = int(len(order) * args.holdout) if 0 < args.holdout < 1 else 0
    holdout = order[:k]
    n_contaminated = sum(1 for _, _, h in holdout if h in base_hashes)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    emit("Byte-identical overlap between the training and evaluation datasets")
    emit(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"Method: MD5 of file contents. Identical hash means identical bytes,")
    emit(f"        regardless of filename or folder.")
    emit("")
    emit(f"Baseline dataset:   {args.baseline}")
    emit(f"   images scored:   {len(base_items)}  "
         f"({base_by_class['with_mask']} with_mask, "
         f"{base_by_class['without_mask']} without_mask)")
    emit(f"   unique hashes:   {len(base_hashes)}")
    emit("")
    emit(f"Evaluation dataset: {args.evaluation}")
    emit(f"   images scored:   {len(eval_items)}  "
         f"({eval_by_class['with_mask']} with_mask, "
         f"{eval_by_class['without_mask']} without_mask)")
    emit("")
    emit("Overlap")
    emit(f"   shared unique hashes:                 {len(shared)}")
    emit(f"   as a fraction of the baseline:        "
         f"{len(shared) / max(1, len(base_hashes)) * 100:.1f} %")
    emit(f"   evaluation images that are duplicates: {n_shared_eval} "
         f"({n_shared_eval / max(1, len(eval_items)) * 100:.1f} % of the evaluation set)")
    emit("")

    if k:
        emit(f"Contamination of a {args.holdout:.0%} holdout (seed {args.seed}, "
             f"formed exactly as evaluate_mask.py forms it)")
        emit(f"   holdout size:            {len(holdout)}")
        emit(f"   of which were training data: {n_contaminated} "
             f"({n_contaminated / max(1, len(holdout)) * 100:.1f} %)")
        emit(f"   genuinely unseen:        {len(holdout) - n_contaminated}")
        emit("")
        emit("   An accuracy figure computed on that holdout is therefore not a")
        emit("   generalisation estimate. Use --exclude-from in evaluate_mask.py to")
        emit("   drop duplicates before scoring.")
        emit("")

    emit("Note")
    emit("   Overlap of this size is unlikely to be plagiarism between dataset")
    emit("   authors; the more probable explanation is that both collections drew")
    emit("   on the same upstream sources. The consequence for evaluation is the")
    emit("   same either way, which is why the check is worth running on any pair")
    emit("   of datasets before one is described as independent of the other.")

    report = "\n".join(lines)
    print(report)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
