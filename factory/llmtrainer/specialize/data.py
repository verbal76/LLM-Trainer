"""Training-data access with hard test-split protection.

The trainer sees only train/validation rows. The test split is read by the
evaluation suite alone, and any attempt to load it through this module raises.
"""

from __future__ import annotations

from pathlib import Path

from ..datasets import load_split
from ..schemas import DatasetManifest


class TestSplitAccessError(RuntimeError):
    __test__ = False  # not a pytest class


TRAINING_SPLITS = ("train", "validation")


def load_training_split(root: Path, manifest: DatasetManifest, split: str) -> list[dict]:
    if split not in TRAINING_SPLITS:
        raise TestSplitAccessError(f"split {split!r} must not be read during training (only {TRAINING_SPLITS})")
    return load_split(root, manifest, split)


def load_training_data(root: Path, manifest: DatasetManifest) -> tuple[list[dict], list[dict]]:
    lr = manifest.leakage_report
    if lr.group_overlap_pairs or lr.residual_cross_split_near_duplicates:
        raise ValueError("dataset manifest reports unresolved train/eval leakage; refusing to train")
    return load_training_split(root, manifest, "train"), load_training_split(root, manifest, "validation")


def format_example(row: dict) -> str:
    """Plain instruction format; tokenizer chat templates are intentionally not assumed."""
    return f"### Instruction:\n{row['prompt']}\n\n### Response:\n{row['response']}"


def approx_tokens(rows: list[dict]) -> int:
    """Rough token count (~1.3 tokens per whitespace word) for resource estimation only."""
    return int(sum(len(format_example(r).split()) for r in rows) * 1.3)
