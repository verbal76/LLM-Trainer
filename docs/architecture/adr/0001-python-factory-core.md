# ADR 0001: Python factory core as CLI-first training backend

- Status: accepted (fixed by lead; recorded, not re-litigated)
- Date: 2026-10-05

## Decision
The factory core (ingestion, rights, corpus, dataset, training, evaluation, retrieval build, registry, quantization, export, reference qualifier) is a Python package that runs on workstation, cloud GPU hosts and CI. It is CLI-first (`llmt ...`), a library with typed interfaces, and has no dependency on any UI.

## Why (supporting evidence)
- The maintained fine-tuning stack is Python: transformers 5.x (PyPI, Apache-2.0), PEFT 0.21.x (Apache-2.0), TRL 1.x (Apache-2.0), bitsandbytes 0.50.x (MIT). Versions as read from PyPI JSON on 2026-10-05 (https://pypi.org/pypi/transformers/json, /peft/, /trl/, /bitsandbytes/).
- llama.cpp's conversion and quantization tools are Python scripts plus C++ binaries (https://github.com/ggml-org/llama.cpp), orchestrated best from Python.
- Phones and GitHub-hosted CI runners cannot train 7B-8B models; GPU hosts are needed regardless.

## Alternatives considered and rejected
- Kotlin/JVM backend: no equivalent training/document-parsing ecosystem; would reimplement or shell out to Python anyway.
- Rust/Go core: attractive for packaging, but every ML dependency would still be Python; doubles the language count with no gain in v1.
- On-phone training: out of scope by ADR 0002 and by compute (7B-8B).

## Consequences
Contracts are files (JSON/SQLite/Parquet), so any other language can consume them. Python is pinned via lock file; GPU extras are optional dependency groups so CI can run CPU-only.
