# ADR 0007: Evaluation harness = thin in-house domain harness + lm-evaluation-harness for regression

- Status: proposed
- Date: 2026-10-05

## Decision
Domain evaluation is an in-house, config-driven harness (module 7) because domain metrics (unsupported-claim rate, citation/source accuracy, retrieval grounding, procedure correctness) are not covered by generic leaderboards. It runs the same frozen unseen test set against: base, base+retrieval, specialist, specialist+retrieval, and each quantized export, through one generation-backend interface (HF/vLLM on workstation; llama.cpp for GGUF so the quantized artifact itself is what is measured).

General-capability regression uses EleutherAI `lm-eval` (0.4.13, MIT, PyPI 2026-08-31). Optional adapters: Hugging Face `lighteval` (0.13.0, MIT, last PyPI upload 2025-11-24) and UK AISI `inspect-ai` (MIT).

## Judging policy
Exact-match/regex/unit-tested checkers where possible (numbers, citations against the reference package); LLM-as-judge only as a secondary signal, run with a local judge by default, with judge identity and prompt hashed in the Evaluation Run. No single aggregate score (`CLAUDE.md`). A claim of improvement requires paired comparison with confidence intervals on the unseen set.

## Alternatives
- Only lm-eval: no domain/citation metrics; rejected as sole harness.
- Only inspect-ai: strong agent/eval framework; kept as optional back-end rather than hard dependency to limit surface; revisit.
- lighteval: less recent release observed on PyPI; optional.
- Hosted eval services: would transmit data externally; rejected by default.

## Uncertainty
Not benchmarked in this round; version freshness read from PyPI JSON only.
