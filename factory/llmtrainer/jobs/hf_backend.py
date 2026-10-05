"""Hugging Face generation backend for ``llmtrainer evaluate --backend hf``.

transformers/torch/peft are imported lazily and only when an evaluation is really executed. Weights are loaded with
``local_files_only=True`` unless the caller explicitly allows downloads. NOT exercised by CI (needs weights and, usually, a GPU);
the surrounding plan/dry-run logic is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROMPT = "### Instruction:\n{prompt}\n\n### Response:\n"  # same plain format the trainer used (specialize.data.format_example)


def missing_dependencies(adapter: bool = True) -> list[str]:
    need = ["torch", "transformers"] + (["peft"] if adapter else [])
    return [m for m in need if importlib.util.find_spec(m) is None]


class HFPair:
    """One loaded base model; ``base`` generates with the adapter disabled, ``specialist`` with it enabled."""

    def __init__(self, base_model_path: str, adapter_path: str | None, *, allow_download: bool = False, max_new_tokens: int = 160):
        missing = missing_dependencies(adapter_path is not None)
        if missing:
            raise RuntimeError(f"missing evaluation dependencies {missing}; install the 'train' extra (pip install 'llmtrainer-factory[train]')")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_only = not allow_download
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base_model_path, local_files_only=local_only)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", local_files_only=local_only, device_map="auto")
        self.peft = adapter_path is not None
        if adapter_path is not None:
            from peft import PeftModel

            if not Path(adapter_path).is_dir():
                raise FileNotFoundError(f"adapter directory not found: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens

    def _gen(self, prompt: str) -> str:
        enc = self.tok(PROMPT.format(prompt=prompt), return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def base(self, prompt: str) -> str:
        if self.peft:
            with self.model.disable_adapter():
                return self._gen(prompt)
        return self._gen(prompt)

    def specialist(self, prompt: str) -> str:
        return self._gen(prompt)
