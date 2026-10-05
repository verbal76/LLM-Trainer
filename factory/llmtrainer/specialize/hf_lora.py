"""Hugging Face transformers + peft + trl LoRA/QLoRA trainer backend.

HONESTY NOTE: this module has been unit-tested only for everything that does not need the
heavy libraries (dependency detection, error paths, settings translation). The actual
``train`` body has NOT been executed in the development sandbox (no torch/GPU/model weights).
Library APIs (notably trl's ``SFTConfig``/``SFTTrainer``) change between releases; the code
adapts to the two most common signatures but must be smoke-tested on the target machine
(see docs/specialization/RUNBOOK.md).

All heavy imports are lazy: importing this module costs nothing and works without them.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

from ..training import ExperimentConfig, TrainerResult
from .data import format_example

CORE_LIBS = ("torch", "transformers", "peft", "trl", "datasets", "accelerate")
QLORA_LIBS = ("bitsandbytes",)
INSTALL_HINT = "pip install 'llmtrainer-factory[train]'   # torch must match your CUDA/ROCm setup"


class MissingTrainingDependencies(RuntimeError):
    pass


def missing_dependencies(method: str) -> list[str]:
    libs = CORE_LIBS + (QLORA_LIBS if method == "qlora" else ())
    return [m for m in libs if importlib.util.find_spec(m) is None]


def require_dependencies(method: str) -> None:
    missing = missing_dependencies(method)
    if missing:
        raise MissingTrainingDependencies(
            f"cannot run real {method} training: missing {', '.join(missing)}. Install with: {INSTALL_HINT}"
        )


def library_versions() -> dict[str, str]:
    out = {}
    for m in CORE_LIBS + QLORA_LIBS:
        try:
            out[f"lib:{m}"] = metadata.version(m)
        except metadata.PackageNotFoundError:
            out[f"lib:{m}"] = "not-installed"
    return out


def environment_snapshot() -> dict[str, str]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
        "trainer": "hf_lora",
        **library_versions(),
    }
    try:  # hardware: only if torch is importable
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = props.name
            env["gpu_memory_gib"] = f"{props.total_memory / 2**30:.1f}"
            env["gpu_count"] = str(torch.cuda.device_count())
        else:
            env["gpu"] = "none (cuda unavailable)"
    except Exception:  # noqa: BLE001
        env["gpu"] = "unknown (torch not importable)"
    return env


def lora_target_modules(cfg: ExperimentConfig):
    """peft wants the string 'all-linear' rather than a one-element list."""
    return cfg.target_modules[0] if len(cfg.target_modules) == 1 else list(cfg.target_modules)


class HFLoraTrainer:
    name = "hf_lora_sft"
    version = "1"
    is_stub = False

    def __init__(self, base_model_path: str, *, allow_download: bool = False):
        self.base_model_path = base_model_path
        self.allow_download = allow_download

    def train(self, cfg: ExperimentConfig, train_rows, val_rows, out_dir: Path) -> TrainerResult:
        require_dependencies(cfg.method)
        import dataclasses
        import inspect

        import torch
        import transformers
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer

        local_only = not self.allow_download
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        transformers.set_seed(cfg.seed)

        tok = AutoTokenizer.from_pretrained(self.base_model_path, local_files_only=local_only)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        quant = None
        if cfg.method == "qlora":
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path, quantization_config=quant, torch_dtype=dtype, local_files_only=local_only,
            device_map="auto" if cfg.method == "qlora" else None,
        )
        if cfg.method == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
        t_load = time.perf_counter()

        lora = LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
            target_modules=lora_target_modules(cfg), task_type="CAUSAL_LM",
        )
        train_ds = Dataset.from_list([{"text": format_example(r) + tok.eos_token} for r in train_rows])
        val_ds = Dataset.from_list([{"text": format_example(r) + tok.eos_token} for r in val_rows])

        sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}
        kw = dict(
            output_dir=str(out_dir / "work"), num_train_epochs=cfg.epochs, per_device_train_batch_size=cfg.micro_batch_size,
            per_device_eval_batch_size=cfg.micro_batch_size, gradient_accumulation_steps=cfg.grad_accum_steps,
            learning_rate=cfg.learning_rate, lr_scheduler_type="cosine", warmup_ratio=cfg.warmup_ratio, bf16=cfg.bf16,
            fp16=not cfg.bf16, gradient_checkpointing=cfg.gradient_checkpointing, seed=cfg.seed, data_seed=cfg.seed,
            logging_steps=10, save_strategy="no", report_to=[], dataset_text_field="text", packing=False,
        )
        kw["max_length" if "max_length" in sft_fields else "max_seq_length"] = cfg.max_seq_len
        kw["eval_strategy" if "eval_strategy" in sft_fields else "evaluation_strategy"] = "epoch"
        args = SFTConfig(**{k: v for k, v in kw.items() if k in sft_fields})

        params = inspect.signature(SFTTrainer.__init__).parameters
        tok_kw = {"processing_class": tok} if "processing_class" in params else {"tokenizer": tok}
        trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, peft_config=lora, **tok_kw)
        out = trainer.train()
        t_train = time.perf_counter()
        ev = trainer.evaluate()

        adapter_dir = out_dir / "adapter"
        tok_dir = out_dir / "tokenizer"
        trainer.model.save_pretrained(str(adapter_dir))  # peft: adapter weights + adapter_config.json only
        tok.save_pretrained(str(tok_dir))
        artifacts = {
            f"{d.name}/{f.name}": f for d in (adapter_dir, tok_dir) for f in sorted(d.iterdir()) if f.is_file()
        }
        metrics = {
            "train_loss": float(out.metrics.get("train_loss", float("nan"))),
            "eval_loss": float(ev.get("eval_loss", float("nan"))),
            "timing_load_s": round(t_load - t0, 2),
            "timing_train_s": round(t_train - t_load, 2),
            "timing_total_s": round(time.perf_counter() - t0, 2),
        }
        return TrainerResult(metrics={k: v for k, v in metrics.items() if v == v}, artifacts=artifacts, subject=None)
