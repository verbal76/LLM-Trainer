# LLM Trainer Architecture (v1 proposal)

Status: proposal for owner/lead review. Research date: 2026-10-05. Evidence and uncertainty live in `docs/research/TECH_SURVEY.md` and `docs/research/BASE_MODELS.md`; decisions live in `docs/architecture/adr/`. This document does not re-plan the product (see `CLAUDE.md`).

## 1. Fixed decisions (recorded as ADRs, not re-litigated)

| ADR | Decision |
|---|---|
| 0001 | Python factory core is the training/data/eval backend. It runs on workstation, cloud or CI, is CLI-first and independently testable. No UI dependency. |
| 0002 | Android app is native Kotlin (host APK + OTA-updatable Kotlin bundle, see lead's `docs/OTA.md` once it exists). It is the product UI/companion: project/package management, device profiling/qualification/benchmarking, running exported packages through an on-device runtime, update diagnostics. It does not train 7B models on-phone. |
| 0003 | GitHub Actions is the only build system (no EAS). |

## 2. System shape

```
                     +---------------------------+
                     |  Android app (Kotlin)     |   product UI / companion
                     |  - project & package mgr  |
                     |  - device profiler/bench  |<--- Device Profile, Benchmark Result (JSON)
                     |  - on-device runtime      |
                     |  - update diagnostics     |
                     +-----^---------------+-----+
        Export Package (dir/zip, manifests) |  ^ Deployment Recommendation (JSON)
                     |                       |  |
 +-------------------+---------------------- v--+---------------------------------+
 |  Python factory core  ("factory")  - CLI + library + (optional) local HTTP API |
 |                                                                               |
 |  workspace -> ingest -> rights -> corpus -> dataset -> train -> eval          |
 |       \             \                                  \         |             |
 |        \             +--> reference/retrieval builder   +--> model registry     |
 |         \                                                     |   quantize    |
 |          +--> license gate (base-model registry) <------------+   qualify     |
 |                                                                   export      |
 +-------------------------------------------------------------------------------+
        ^ runs on: developer workstation, rented/owned GPU box, GitHub Actions (CPU tests)
```

The two halves communicate only through versioned, file-based artifact contracts (`schemas/`, owned by another agent) plus, optionally, a local HTTP/JSON API served by the factory for workstation-attached use. There is no shared in-process code between Python and Kotlin (ADR 0002). That is the property that keeps models "not trapped inside LLM Trainer": a consuming app needs only an Export Package and a runtime, never the factory.

Compute reality: LoRA/QLoRA on 7B-8B and dataset generation need an NVIDIA/AMD GPU (or Apple Silicon via a different path); neither GitHub-hosted CI runners nor phones are suitable for that. CI therefore tests the factory on CPU with tiny fixtures; real training runs on workstation/cloud (owner-authorized spend only, per `CLAUDE.md`).

## 3. Module boundaries

Each module is a Python package inside the factory with a narrow public interface (typed with pydantic v2, ADR 0014), a CLI subcommand group, and no import of any other module's internals. Interfaces exchange artifacts by path + content hash, not by in-memory objects.

| # | Module | Responsibility | Inputs -> outputs (contracts) | Language / key components | Why |
|---|---|---|---|---|---|
| 1 | Workspace / Project Manager | Create/open a workspace; project config; immutable-run layout; lockfile of tool versions; garbage-collect unreferenced blobs | -> Specialist Project | Python; pydantic, filesystem layout, content-addressed blob store (sha256) | Pure data/IO; shared by CLI and any API |
| 2 | Source Ingestion | Import files/URLs/dirs; hash; extract text/structure/tables/page maps; keep originals read-only and separate | raw file -> Source record + extracted document (JSONL/Parquet with page/section/chunk offsets) | Python; Docling (MIT) as primary PDF/office parser, pypdf/trafilatura fallbacks; PyMuPDF only behind an explicit license flag (AGPL, ADR 0013) | Python has the only mature document-parsing ecosystem |
| 3 | Provenance / Rights Registry | Source rights status, license text, who attested, scope (train? redistribute? quote?), removal; lineage graph source -> chunk -> example -> dataset -> model | Source Manifest; lineage edges | Python; SQLite (WAL) for the index + append-only JSONL event log as the source of truth | Rights are the legal spine; must be queryable and auditable offline; SQLite is also readable from Kotlin if ever needed |
| 4 | Corpus Processing | Clean, de-noise (headers/footers), dedupe (exact + MinHash), classify, segment, table treatment, OCR-quality flags, contradiction/outdated-revision flags | extracted docs -> processed corpus + quality report | Python; datasketch (MIT), regex/heuristics, optional small classifier | Same ecosystem as 2 |
| 5 | Dataset Generation | Turn chunks into SFT examples (instruction/QA/diagnostic dialogs/procedures) with per-example provenance, confidence filters; split train/val/test **by source document/section** before generation to prevent leakage | processed corpus -> Dataset Manifest + train/val/test shards | Python; generator is pluggable (local model via llama.cpp/vLLM, or an explicitly authorized external API; external use is off by default) | Generation backends are Python/server-side |
| 6 | Training Orchestrator | Resolve base model via registry + license gate; build training config; run LoRA/QLoRA SFT; record seeds, env, versions, GPU; checkpoint/resume; produce adapter + metrics | Dataset Manifest + base-model entry -> Training Run + adapter | Python; transformers + PEFT + TRL + bitsandbytes (ADR 0004/0005); optional Unsloth accelerator plugin | The training ecosystem is Python/PyTorch |
| 7 | Evaluation Harness | Domain evals on unseen split + general-regression evals; base vs specialist vs quantized; unsupported-claim and citation checks; latency/memory where measurable | models + eval sets -> Evaluation Run | Python; thin in-house harness over generation backends; lm-evaluation-harness (MIT) for general regression (ADR 0007) | Reproducible, scriptable, CI-testable |
| 8 | Retrieval / Reference Builder | Build the companion exact-reference package: chunk store, FTS5 index, optional vectors, citation anchors, table store | processed corpus -> Reference Package | Python builds; consumer reads with SQLite from any language (ADR 0010) | Build in Python, query in Kotlin/anywhere via SQLite |
| 9 | Model Registry | Two registries: (a) base-model registry (`registry/base-models/*.json`, license gate); (b) specialist registry (every trained/exported specialist: manifest, hashes, lineage, never two different artifacts with one identity) | JSON entries -> Model Manifest | Python; JSON files + SQLite index | Plain files are inspectable and diffable in Git |
| 10 | Quantization / Optimization | Merge adapter (when permitted), convert to GGUF, quantize (imatrix), calibrate, optionally distill to a student; emit one artifact per deployment grade | adapter + base -> quantized models | Python orchestrating llama.cpp tools (convert_hf_to_gguf, convert_lora_to_gguf, llama-quantize, llama-imatrix) (ADR 0008) | llama.cpp tooling is invoked as pinned subprocesses; Python glue only |
| 11 | Device Qualification | Estimate and (where possible) consume measured device data to pick the best safe profile; define Performance / Balanced / Max-Quality-Within-Envelope | Device Profile + model catalog -> Deployment Recommendation | **Algorithm in Python (reference + tests); same algorithm re-implemented in Kotlin as the on-device authority**, both validated against a shared set of golden JSON test vectors (ADR 0011) | Phone must decide offline with live readings; the Python implementation exists for CI, workstation planning and as the executable spec |
| 12 | Export / Package | Assemble a self-describing Export Package: models, tokenizer/config, reference package, eval report, provenance manifest, license/attribution bundle, runtime manifest, hashes; refuse export when the license gate fails | all above -> Export Package | Python; no runtime deps on the consumer side | Enables "not trapped inside LLM Trainer" |
| 13 | UI | Project workflow Create -> Sources -> Review -> Base model -> Dataset -> Train -> Evaluate -> Optimize -> Benchmark -> Export; warnings (license, hardware, safety, regression) | reads/writes the same artifacts; talks to a workstation factory over the optional local API | **Kotlin (Android)** now; a desktop/web UI is deliberately out of scope for v1 | Fixed ADR 0002; on-phone UI is a manager/companion, not the trainer |
| 14 | CLI / Automation | `llmt` command (Typer) over every module; deterministic, scriptable, JSON output mode, exit codes; same entry points used by CI | - | Python; Typer (MIT) | CLI-first requirement; UI must never be the only way to operate |

### 3.1 Python vs Kotlin/other, summarized

- **Python (factory):** modules 1-12 and 14. Justification: the ML/data stack (transformers, PEFT, TRL, bitsandbytes, datasets, Docling, llama.cpp tooling) is Python; reproducibility and CI testing are easiest in one language; the work targets GPU hosts.
- **Kotlin (Android):** module 13 plus the on-device half of module 11 (profiler, benchmark runner, safety-envelope evaluator), the package importer/validator, the runtime bridge, and the reference-package reader. Justification: ADR 0002; Android system APIs (ActivityManager, PowerManager thermal, ApplicationExitInfo) are Kotlin/Java-first.
- **C/C++ (vendored, not authored):** inference runtime (llama.cpp via JNI, ADR 0009). We write a thin JNI/C++ bridge only.
- **SQL/SQLite:** the on-disk reference format and the provenance index (ADR 0010, 0006).
- **YAML/JSON:** project configs and all contracts. No custom DSL.

## 4. Data flow and contracts

All stages are pure functions of (input artifact hashes, config, tool versions, seed) -> output artifact, written to a content-addressed, immutable run directory. Nothing is silently overwritten; re-running with identical inputs must produce an identical manifest hash (model weights excepted where GPU nondeterminism applies; those runs record the nondeterminism flag).

```
workspace/
  sources/<sha256>/original.*            read-only originals (never modified)
  sources/<sha256>/source.json           Source Manifest entry (rights, provenance)
  extracted/<run-id>/...                 extraction output + page/offset maps
  corpus/<corpus-hash>/...               processed corpus + quality report
  datasets/<dataset-hash>/{train,val,test}.jsonl + dataset.json
  runs/train/<run-id>/ adapter/ config.json env.json metrics.jsonl
  runs/eval/<run-id>/ ...
  models/<model-id>/ manifest.json + artifacts
  refpkg/<refpkg-hash>/ reference.sqlite + manifest.json
  exports/<export-id>/ package + manifest.json
  lineage.sqlite  events.jsonl            provenance graph and append-only log
```

Contracts (schemas are owned by the schema agent; this document only fixes what each must carry): Specialist Project, Source Manifest, Dataset Manifest, Training Run, Evaluation Run, Model Manifest, Device Profile, Deployment Recommendation, Export Package; all start honestly at v1 with `schema_version`, canonical-JSON sha256 identity, and references by hash (ADR 0014).

### 4.1 Source removal and rebuild

Removing a source is a first-class operation: mark source `revoked` in the rights registry; the lineage graph yields every chunk, example, dataset, training run, model and export derived from it; the factory marks those artifacts `tainted` and refuses to export them; `llmt rebuild --without <source>` re-derives dataset (and optionally retrains). Weights already trained on a removed source cannot be "edited"; the system must say so rather than pretend (open owner decision on policy, see section 8).

### 4.2 License gate

Before any download/train/merge/export, the base-model registry entry for the requested model is checked: each required permission must be `yes` (or `conditional` with all conditions acknowledged and generated into the license/attribution bundle). `unverified` blocks export by default and prints what is missing. The gate also checks that the registry entry's recorded license hash matches the LICENSE file actually downloaded (not yet implementable until `hf_hub` access is available; HF was blocked in this research round).

## 5. Training and evaluation split policy

Split is assigned at the *source-section* level before example generation; generated examples inherit their parent's split; near-duplicate (MinHash) clusters are assigned atomically; test split is frozen by hash and a leakage check (n-gram overlap and embedding proximity between test items and train corpus) is a blocking gate. Evaluations compare base vs specialist vs quantized specialist on the identical frozen set (ADR 0007).

## 6. Device qualification (summary; detail in ADR 0011)

Three distinct verdicts, never conflated: storage fit, RAM fit, safe sustained operating fit. A candidate profile = model + quant + context + runtime + retrieval budget + settings. Memory estimate = weights + runtime overhead + KV cache (architecture-aware; hybrid/linear-attention models such as Qwen3.5 need a family-specific formula, see BASE_MODELS.md) + retrieval/index + host-app + safety reserve; RAM budget derives from *measured* `availMem`/`threshold` and reserves, not total RAM. A profile is only "safe" if a sustained benchmark shows no thermal-status escalation beyond configured limit, no throttling cliff, no low-memory kill/ANR, and UI-frame budget held. Missing measurements trigger conservative fallbacks (smaller context, lower quant tier, "unmeasured" label, never "recommended").

## 7. Build, CI, release

- GitHub Actions only (ADR 0003): factory job (lint, type-check, pytest on CPU fixtures, schema validation, registry JSON validation, license-gate tests, package-validation tests); Android job (Gradle build, unit tests, golden-vector tests of the Kotlin qualifier against the Python reference); OTA bundle job (lead). No release publication on green build; releases need owner authorization (`CLAUDE.md`).
- Heavy jobs (real training, GPU evals) are manual `workflow_dispatch`/self-hosted runner jobs, never default.
- Python pinned with a lock file; llama.cpp pinned by commit (tag format `bNNNN`), recorded in each manifest.

## 8. Open owner decisions

1. Policy when a source is revoked after training (retrain-required vs warn-and-label).
2. Which external services (if any) may be used for dataset generation; default is local-only.
3. Whether Llama-family bases (custom license with naming/AUP obligations) are enabled by default or opt-in.
4. Whether a desktop/web UI is ever wanted for workstation-side use (v1 assumes CLI + Android).
5. Per-device-class safety-reserve defaults (proposed numbers are in ADR 0011 and need hardware validation).

## 9. What is not decided here

Exact schema field names (schema agent), OTA bundle mechanics (lead), concrete Android UI screens (lead), and any claim of model quality (needs evaluation evidence).
