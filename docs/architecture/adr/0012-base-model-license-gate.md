# ADR 0012: Machine-readable base-model registry with a blocking license gate

- Status: proposed
- Date: 2026-10-05

## Decision
`registry/base-models/<family-version>.json` is the source of truth for model identity, license and permissions. Each permission is one of `yes | no | conditional | unverified`. The factory refuses download-for-training, merge, quantized export and package export unless required permissions are `yes`, or `conditional` with conditions generated into the attribution/license bundle and acknowledged by the user. `unverified` always blocks export. The gate compares the license file hash found at download with the one recorded (future work; needs Hub access).

## Findings that motivate it
- Licenses differ materially in the same size class. Apache-2.0 (Qwen3/3.5, SmolLM3, Olmo 3, Gemma 4 reported) vs the Llama 3.x Community Licenses (name prefix "Llama", "Built with Llama" notice, 700M-MAU clause, Acceptable Use Policy; EU restriction for multimodal 3.2 per AUP summary). Gemma <=3 used a custom Terms of Use; Gemma 4 is reported as Apache-2.0 (needs primary confirmation). Mistral has shipped custom licenses for some models (MNPL/MRL per mistral-inference README) while Ministral 3 is reported Apache-2.0 (unverified).
- Open licensing of weights is separate from pretraining-data rights, and from rights to the *user's* source corpus (tracked by the rights registry).

## Alternatives rejected
- Hard-coded allowlist in code: not auditable/diffable.
- Trusting Hub metadata (`license:` tag): can be wrong or coarse; we record the license file and verification evidence.
- Allow-all with a warning: violates `CLAUDE.md` ("never silently package or redistribute").

## Consequences
Adding a model = adding a reviewed JSON file with `verification.source_urls` and `uncertainties`. No legal conclusions are asserted; "conditional" means conditions exist that a human must accept.
