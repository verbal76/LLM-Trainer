"""Training interface, experiment config, resource estimator and the stub trainer."""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from . import splits as sp
from .evaluation import BASE_VOCAB, StubSubject
from .hashing import hash_file, hash_obj
from .schemas import Base, ResourceEstimate


class ExperimentConfig(Base):
    """Everything material to a training run. Hashed into the TrainingRun identity."""

    base_model_id: str
    method: Literal["stub", "lora", "qlora", "full"] = "stub"
    seed: int = 1234
    context_tokens: int = 2048
    epochs: int = 1
    learning_rate: float = 2e-4
    lora_rank: int = 16
    micro_batch_size: int = 4
    base_params_b: float = 1.0
    gpu_tflops_fp16: float = 100.0
    assumed_mfu: float = 0.3
    gradient_checkpointing: bool = True


def estimate_resources(cfg: ExperimentConfig, train_tokens: int) -> ResourceEstimate:
    """Order-of-magnitude estimate; deliberately simple and conservative-ish. Replace with measurement."""
    p = cfg.base_params_b * 1e9
    bytes_per = {"full": 2.0, "lora": 2.0, "qlora": 0.55, "stub": 0.0}[cfg.method]
    trainable = p if cfg.method == "full" else p * cfg.lora_rank * 3.6e-5
    if cfg.method == "stub":
        trainable = 0.0
    weights = p * bytes_per
    optim = trainable * (16 if cfg.method == "full" else 14)
    act = (1.5 * 2**30) * (cfg.base_params_b / 7) * (cfg.context_tokens / 2048) * cfg.micro_batch_size
    if not cfg.gradient_checkpointing:
        act *= 4
    gib = (weights + optim + act) / 2**30 * 1.2 if cfg.method != "stub" else 0.0
    flop_per_token = (6 if cfg.method == "full" else 4) * p
    total_flops = flop_per_token * train_tokens * cfg.epochs
    hours = 0.0 if cfg.method == "stub" else total_flops / (cfg.gpu_tflops_fp16 * 1e12 * cfg.assumed_mfu) / 3600
    notes = ["order-of-magnitude estimate: ~20% memory headroom included; not measured"]
    if cfg.method == "stub":
        notes = ["stub trainer: runs on CPU in well under a second; no GPU needed"]
    return ResourceEstimate(
        basis=f"{cfg.method} params_b={cfg.base_params_b} rank={cfg.lora_rank} ctx={cfg.context_tokens}",
        trainable_params_m=round(trainable / 1e6, 3),
        gpu_memory_gib=round(gib, 2),
        train_tokens=train_tokens,
        gpu_hours=round(hours, 4),
        notes=notes,
    )


@dataclass
class TrainerResult:
    metrics: dict[str, float]
    artifacts: dict[str, Path]  # name -> file path
    subject: Any = None


@runtime_checkable
class Trainer(Protocol):
    name: str
    version: str
    is_stub: bool

    def train(self, cfg: ExperimentConfig, train_rows: list[dict], val_rows: list[dict], out_dir: Path) -> TrainerResult: ...


class StubTrainer:
    """PIPELINE-VALIDATION STUB. Builds a unigram count table from the training rows.

    It is deterministic, needs no GPU and does not fine-tune a language model.
    Results must never be presented as evidence of real specialist improvement.
    """

    name = "unigram_stub_trainer"
    version = "1"
    is_stub = True

    def train(self, cfg, train_rows, val_rows, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        counts: Counter[str] = Counter()
        for r in train_rows:
            counts.update(sp.words(r["prompt"] + " " + r["response"]))
        vocab = {w: 1 for w in BASE_VOCAB}
        for w, c in counts.items():
            vocab[w] = vocab.get(w, 0) + c
        adapter = out_dir / "stub_adapter.json"
        adapter.write_text(
            json.dumps({"stub": True, "seed": cfg.seed, "vocab": dict(sorted(vocab.items()))}, sort_keys=True, indent=1),
            encoding="utf-8",
        )
        val_vocab = set()
        for r in val_rows:
            val_vocab.update(sp.words(r["response"]))
        val_cov = sum(w in vocab for w in val_vocab) / len(val_vocab) if val_vocab else 0.0
        return TrainerResult(
            metrics={"stub_train_tokens": float(sum(counts.values())), "stub_val_vocab_coverage": round(val_cov, 6)},
            artifacts={"adapter": adapter},
            subject=StubSubject("specialist", "stub-specialist", vocab),
        )


def environment_info() -> dict[str, str]:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "trainer": "stub"}


def run_id_for(cfg: ExperimentConfig, dataset_hash: str) -> str:
    return "run-" + hash_obj([cfg.model_dump(mode="json"), dataset_hash]).removeprefix("sha256:")[:12]


def hash_artifacts(artifacts: dict[str, Path]) -> dict[str, str]:
    return {k: hash_file(v) for k, v in sorted(artifacts.items())}
