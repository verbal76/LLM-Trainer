"""Real specialization path: license-gated LoRA/QLoRA training behind lazy imports.

Nothing here imports torch/transformers/peft/trl at module import time; the heavy
libraries are only touched inside ``hf_lora`` when a run is explicitly executed.
"""
