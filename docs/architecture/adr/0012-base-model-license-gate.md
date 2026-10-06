# ADR 0012: Machine-readable base-model registry with a blocking license gate

- Status: proposed
- Date: 2026-10-05

## Decision
`registry/base-models/<family-version>.json` is the source of truth for model identity, license and permissions. Each entry is schema_version 2 (v1 entries are rejected by the loader, not upgraded). Each entry carries `verification.state` (`VERIFIED | UNVERIFIED | DISALLOWED`) and `verification.evidence_level` (`primary_license_text_read | primary_model_card_read | secondary_source_only | none`). The permission fields (`yes | no | conditional | unverified`) are recorded *claims*: the gate ignores them unless the state is `VERIFIED`. `VERIFIED` requires a primary evidence level, `license_text_sha256` of the text actually inspected, `verified_on`, `source_urls` and `verified_scope`; `secondary_source_only`/`none` evidence must be `UNVERIFIED`. The factory fails closed: download-for-training, merge, quantized export and package export are refused for `UNVERIFIED` and `DISALLOWED` entries with machine-readable reason codes (`LICENSE_UNVERIFIED`, `LICENSE_DISALLOWED`, `LICENSE_EVIDENCE_INVALID`, `PERMISSION_*`, `FORMAT_NOT_SUPPORTED`) and a remediation message. For a `VERIFIED` entry the requested permissions must be `yes`; `conditional` still blocks until a human resolves the conditions. The gate compares the license file hash found at download with the one recorded (future work; needs Hub access).

## Findings that motivate it
- Licenses differ materially in the same size class. Apache-2.0 (Qwen3/3.5, SmolLM3, Olmo 3, Gemma 4 reported) vs the Llama 3.x Community Licenses (name prefix "Llama", "Built with Llama" notice, 700M-MAU clause, Acceptable Use Policy; EU restriction for multimodal 3.2 per AUP summary). Gemma <=3 used a custom Terms of Use; Gemma 4 is reported as Apache-2.0 (needs primary confirmation). Mistral has shipped custom licenses for some models (MNPL/MRL per mistral-inference README) while Ministral 3 is reported Apache-2.0 (unverified).
- Open licensing of weights is separate from pretraining-data rights, and from rights to the *user's* source corpus (tracked by the rights registry).

## Alternatives rejected
- Hard-coded allowlist in code: not auditable/diffable.
- Trusting Hub metadata (`license:` tag): can be wrong or coarse; we record the license file and verification evidence.
- Allow-all with a warning: violates `CLAUDE.md` ("never silently package or redistribute").

## Consequences
Adding a model = adding a reviewed JSON file with `verification.source_urls` and `uncertainties`. No legal conclusions are asserted; "conditional" means conditions exist that a human must accept.
