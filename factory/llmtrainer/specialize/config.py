"""User-facing training config (JSON) for the real LoRA/QLoRA path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..hashing import hash_obj
from ..schemas import Base
from ..training import ExperimentConfig


class SpecializeConfig(Base):
    project: str  # workspace directory
    base_model_id: str  # license registry id, e.g. "Qwen3@Qwen3-1.7B"
    base_model_path: str  # local directory (preferred) or Hugging Face repo id
    base_params_b: float = Field(gt=0)
    method: Literal["lora", "qlora"] = "lora"
    registry_path: str | None = None  # extra registry file/dir; workspace registry/ is always loaded
    dataset_id: str | None = None  # default: latest dataset for the current source manifest
    seed: int = 1234
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    micro_batch_size: int = Field(default=4, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    max_seq_len: int = Field(default=1024, ge=16)
    target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    gradient_checkpointing: bool = True
    bf16: bool = True
    gpu_tflops_fp16: float = 100.0
    assumed_mfu: float = 0.3

    @model_validator(mode="after")
    def _sane(self):
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if self.method == "qlora" and not self.bf16:
            raise ValueError("qlora here assumes bf16 compute; set bf16=true")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "SpecializeConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def config_hash(self) -> str:
        """Hash of everything material; excludes machine-local paths (project, registry_path)."""
        return hash_obj(self.model_dump(mode="json", exclude={"project", "registry_path"}))

    def to_experiment(self) -> ExperimentConfig:
        return ExperimentConfig(
            base_model_id=self.base_model_id, method=self.method, seed=self.seed, context_tokens=self.max_seq_len,
            epochs=self.epochs, learning_rate=self.learning_rate, lora_rank=self.lora_rank,
            micro_batch_size=self.micro_batch_size, base_params_b=self.base_params_b,
            gpu_tflops_fp16=self.gpu_tflops_fp16, assumed_mfu=self.assumed_mfu,
            gradient_checkpointing=self.gradient_checkpointing, lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, max_seq_len=self.max_seq_len, grad_accum_steps=self.grad_accum_steps,
            target_modules=list(self.target_modules), warmup_ratio=self.warmup_ratio, bf16=self.bf16,
        )
