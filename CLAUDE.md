# LLM Trainer --- CLAUDE.md

## Project Identity

**Product:** LLM Trainer\
**Studio:** Hot Attic Games\
**Purpose:** Build a reusable specialist-model factory that converts
curated, authorized domain material into evaluated, portable specialist
LLM packages for use inside LLM Trainer and other applications.

This repository currently begins as a planning/placeholder project. Do
not treat an earlier provisional stack suggestion as an irreversible
architecture decision.

------------------------------------------------------------------------

## Product Vision

LLM Trainer should let a user create a specialist project such as:

-   Motorcycle Mechanic
-   HVAC Technician
-   Urban / Town Design
-   Automotive Diagnostics
-   Electrical Troubleshooting
-   or another knowledge-intensive specialty

The user supplies curated source material such as manuals, training
books, technical papers, standards, documentation, and other material
they are authorized to use.

The application then provides an understandable workflow to:

1.  ingest and catalog sources;
2.  preserve source provenance and usage-rights metadata;
3.  clean, deduplicate, classify, and segment the corpus;
4.  generate and quality-filter training material;
5.  separate training, validation, and genuinely unseen evaluation
    material;
6.  fine-tune an appropriate open-weight base model;
7.  evaluate the specialist against the original base model;
8.  detect regressions, unsupported claims, and overfitting;
9.  maintain a retrievable reference package for exact facts;
10. quantize and optimize models for target hardware;
11. benchmark target devices;
12. export reusable, versioned specialist packages for other
    applications.

**LLM Trainer is the factory. Models produced by it must not be trapped
inside LLM Trainer.**

------------------------------------------------------------------------

## Core Product Principles

### 1. Specialist quality first, deployment optimization second

Do not design the training system around the smallest model that might
run on a phone.

Support multiple model sizes and families. A 7B/8B model is a normal and
important target, not an edge case. Larger and smaller models should
also be possible when licensing, hardware, and training infrastructure
permit.

Training should seek the best specialist quality appropriate to the
project. Quantization, distillation, context sizing, and target-device
packaging are separate optimization stages.

### 2. Open-weight and license-aware

The system requires access to model weights for fine-tuning and portable
deployment.

Do not assume that "downloadable," "open," or "open source"
automatically grants all required rights.

Maintain machine-readable metadata for each supported base model,
including where determinable:

-   model family and exact version;
-   license;
-   commercial-use status;
-   fine-tuning permission;
-   adapter/derivative permission;
-   redistribution permission;
-   attribution requirements;
-   relevant restrictions;
-   supported runtimes/export formats.

Never silently package or redistribute a model when its license is
incompatible with the requested use.

### 3. Source rights and provenance are first-class

Every source and generated training example must remain traceable.

Track, at minimum:

-   source identity;
-   source version/date when available;
-   source hash;
-   ingestion date;
-   provenance;
-   usage-rights/license status;
-   page/section/chunk origin where practical;
-   transformations performed;
-   generated examples derived from that source.

Do not assume that possession of a PDF/book/manual grants commercial
training or redistribution rights.

A source must be removable later, and the system should be capable of
rebuilding affected datasets/artifacts.

### 4. Fine-tuning is not a substitute for exact-reference retrieval

Fine-tuning should develop domain language, reasoning patterns,
diagnostic behavior, procedures, concepts, and specialist competence.

Do not rely on model weights alone for exact facts such as:

-   torque specifications;
-   electrical values;
-   clearances;
-   capacities;
-   model/year-specific procedures;
-   tabular technical specifications;
-   exact standards language.

Create an optional/companion retrievable specialist knowledge package so
consuming applications can ground answers in exact source material and
provide provenance/citations where appropriate.

### 5. Evaluation must use unseen material

Do not grade the model primarily on questions it saw during training.

Maintain real train/validation/test separation and generate or curate
unseen domain evaluations.

Compare at least:

-   original base model;
-   trained specialist;
-   quantized/exported specialist where applicable.

Useful metrics may include:

-   domain correctness;
-   task completion;
-   unsupported/hallucinated claims;
-   retrieval grounding;
-   citation/source accuracy;
-   regression on general capabilities relevant to the use case;
-   latency;
-   memory use;
-   model size;
-   tokens/second.

Metrics must be appropriate to the domain. Do not reduce quality to one
meaningless aggregate score.

### 6. Portable outputs

A completed training project should be capable of producing reusable
artifacts, as licensing and runtime support permit:

-   original adapter (for example LoRA/QLoRA);
-   merged/master model where appropriate;
-   quantized deployment models;
-   reference/retrieval knowledge package;
-   tokenizer/configuration;
-   evaluation report;
-   provenance manifest;
-   license/attribution bundle;
-   compatibility/runtime manifest;
-   hashes and version identity.

Other Hot Attic Games applications must be able to consume exported
specialists without embedding LLM Trainer itself.

### 7. Device-aware deployment means NO DEGRADED DEVICE OPERATION

This is a non-negotiable product requirement.

Do **not** define "fits" as "the model can allocate enough RAM to
launch."

For a target device, recommend the **highest-quality specialist
configuration that can operate without materially degrading normal
device operation**.

Account for:

-   physical RAM;
-   currently available RAM;
-   OS/system reserve;
-   host application memory;
-   normal background-process reserve;
-   inference runtime overhead;
-   model weights;
-   KV cache/context size;
-   retrieval/index memory;
-   concurrency;
-   free storage and required storage reserve;
-   sustained thermal behavior;
-   battery/power behavior where measurable;
-   UI responsiveness;
-   memory pressure;
-   background-process kills;
-   sustained inference throttling.

A model that technically loads but causes OS pressure, app stutter,
excessive heating, background-app termination, instability, or severe
sustained throttling does **not** fit by this project's definition.

Use a configurable safety reserve rather than consuming all available
RAM/storage.

Where possible, benchmark the actual target device and measure:

-   time to first token;
-   tokens/second;
-   peak RAM;
-   sustained RAM;
-   memory pressure;
-   thermal behavior;
-   sustained throttling;
-   responsiveness;
-   crashes/ANRs;
-   background-process impact.

The deployment recommendation should select the best complete profile,
not merely a parameter count:

**model + quantization + context window + inference runtime + retrieval
budget + relevant runtime settings.**

Consider user-facing profiles such as:

-   Performance
-   Balanced / Recommended
-   Maximum Quality Within Safe Envelope

Advanced overrides may exist, but unsafe configurations must be clearly
identified.

### 8. Multiple deployment grades

A specialist may have multiple exports from one project, for example:

-   master/training artifact;
-   desktop/workstation;
-   high-end mobile;
-   balanced mobile;
-   lightweight mobile;
-   server.

Do not artificially force a flagship device to use a tiny model if a
stronger model runs inside the no-degradation envelope.

Likewise, do not force a midrange device to run a 7B model merely
because it can technically start.

### 9. Distillation is an optimization option, not the default assumption

It should be possible to train a strong specialist first and later
distill that capability into smaller models.

Evaluate the resulting students against the stronger specialist rather
than assuming a smaller model is adequate.

### 10. Usable application, not merely scripts

Python/PyTorch/Hugging Face and command-line/config-driven pipelines may
be appropriate implementation components, but they are not the product
definition.

The finished product should provide a usable application/workflow for
creating projects, adding sources, inspecting ingestion, selecting
models, configuring training, monitoring jobs, reviewing evaluations,
benchmarking devices, and exporting artifacts.

A CLI may coexist with the UI and is desirable for
automation/reproducibility.

------------------------------------------------------------------------

## Architecture Expectations

Before committing to the final architecture:

1.  inspect the repository and existing state;
2.  research current, maintained training/inference/export tooling;
3.  verify current licenses and compatibility from authoritative
    sources;
4.  identify the intended host platforms and available compute;
5.  design clear module boundaries;
6.  document major architecture decisions.

Prefer reproducible, inspectable pipelines.

Keep the training/data/evaluation core separable from the presentation
layer.

Likely conceptual modules include:

-   Project/Workspace Manager
-   Source Ingestion
-   Provenance/Rights Registry
-   Corpus Processing
-   Dataset Builder
-   Training Orchestrator
-   Evaluation Harness
-   Retrieval/Reference Builder
-   Model Registry
-   Quantization/Optimization
-   Device Profiler/Benchmark
-   Export/Packaging
-   UI
-   CLI/Automation API

These are conceptual boundaries, not mandated filenames or frameworks.

------------------------------------------------------------------------

## Training Safety and Data Quality

Never blindly convert all ingested text into training data.

Detect and handle where practical:

-   duplicate passages;
-   contradictory sources;
-   outdated revisions;
-   corrupted extraction;
-   headers/footers/navigation noise;
-   tables requiring structured treatment;
-   OCR errors;
-   low-confidence generated examples;
-   leakage between train and evaluation sets.

Preserve original source material separately from derived datasets.

Do not silently overwrite training artifacts. Training runs must be
versioned and reproducible.

Use deterministic seeds where practical and record all material training
configuration.

------------------------------------------------------------------------

## Model Registry and Versioning

Every trained/exported specialist must have an unambiguous identity.

A model manifest should be able to describe:

-   specialist name/version;
-   base model exact identity/version;
-   base model license;
-   adapter/training method;
-   dataset version/hash;
-   source manifest version/hash;
-   training configuration;
-   tokenizer;
-   context assumptions;
-   quantization;
-   runtime requirements;
-   evaluation results;
-   export date;
-   artifact hashes;
-   target profile/device class;
-   known limitations.

Never label two materially different model artifacts with the same
identity.

------------------------------------------------------------------------

## Research Use

Academic papers and technical research may inform training strategy,
evaluation, retrieval, or a specialist domain.

Do not merely dump papers into a model and assume expertise emerged.

Extract defensible, testable domain knowledge and preserve provenance.

When research is being translated into deterministic software rules
rather than model knowledge, document that distinction.

------------------------------------------------------------------------

## UI/UX Direction

The application should make a complex process understandable without
pretending training is trivial.

A future project workflow should naturally support:

**Create Project → Add Sources → Review Corpus → Choose Base Model →
Build Dataset → Train → Evaluate → Optimize → Benchmark Target →
Export**

Show important warnings clearly, especially:

-   incompatible licenses;
-   insufficient hardware;
-   unsafe device deployment;
-   evaluation regressions;
-   questionable source quality;
-   unsupported export targets.

Do not bury expert controls. Use sensible defaults with an advanced
path.

------------------------------------------------------------------------

## Hot Attic Games Branding

This is a Hot Attic Games product.

Use the repository-provided canonical Hot Attic Games studio artwork for
the studio splash. Do not redraw, approximate, recolor, crop, stretch,
or substitute the canonical asset.

Target launch order:

**native/system startup → Hot Attic Games studio splash → LLM Trainer
product experience**

The Hot Attic Games splash must be the first branded image where
technically practical.

Preserve aspect ratio and transparency. Avoid duplicate splash
presentation and fake dead time.

The user may also provide a product icon and/or LLM Trainer-specific
splash artwork. Use repository assets rather than inventing replacements
when supplied.

------------------------------------------------------------------------

## Testing and Quality Gates

This is infrastructure that may create models used by other
applications. Treat correctness and reproducibility seriously.

Build automated coverage for:

-   project/model manifests;
-   source provenance;
-   dataset splitting/leakage protection;
-   source removal/rebuild behavior;
-   model compatibility rules;
-   licensing gates;
-   training configuration;
-   evaluation;
-   artifact hashing;
-   export/package validation;
-   device-fit calculations;
-   safety-reserve calculations;
-   target profile selection;
-   import of exported packages by a reference consumer where practical.

Use integration tests around actual pipeline boundaries.

Do not claim a model is "better" without evaluation evidence.

Do not claim a target device is compatible merely because a model file
fits in storage.

------------------------------------------------------------------------

## Security and Privacy

Assume some source libraries may be private/proprietary.

Do not upload source documents, datasets, model artifacts, secrets, or
proprietary material to external services without explicit
authorization.

Do not commit credentials, tokens, private signing keys, paid API
secrets, or private source corpora to public Git history.

Prefer local/offline processing where practical and clearly identify any
workflow that transmits data externally.

------------------------------------------------------------------------

## Git / Engineering Discipline

-   Preserve working baselines before risky changes.
-   Keep commits coherent and descriptive.
-   Do not rewrite shared history casually.
-   Do not publish releases merely because a build succeeds.
-   Separate engineering completion from release authorization.
-   Use CI for reproducible validation.
-   Fix engineering failures that can reasonably be fixed autonomously
    rather than making the owner routine QA.
-   Ask the owner only for genuine product/rights/creative decisions or
    physical validation that automation cannot replace.

------------------------------------------------------------------------

## Scope Discipline

Do not invent unrelated features during implementation.

When the project reaches feature freeze, respect it strictly.

For now, foundational work should prioritize:

1.  correct architecture;
2.  reproducible training;
3.  provenance and licensing;
4.  real evaluation;
5.  reusable export;
6.  safe device qualification;
7.  usable workflow.

Do not optimize for flashy demos at the expense of those foundations.

------------------------------------------------------------------------

## Definition of Success

LLM Trainer succeeds when a user can take an authorized, curated
specialist corpus, create a demonstrably improved specialist model,
prove the improvement on unseen evaluation material, retain
exact-reference grounding, export the specialist for reuse, and deploy
the strongest configuration a target device can sustain **without
degrading normal device operation**.

That is the product.

------------------------------------------------------------------------

## GitHub Actions Budget (STANDING OWNER DIRECTIVE)

GitHub-hosted Actions minutes are shared across the owner's projects and the monthly budget is deliberately small. Treat them
as a scarce resource, not as the default way to validate a change. Before starting any hosted workflow ask:

**"Does this need GitHub Actions, or can I prove it locally?"**

Use Actions only when hosted execution gives meaningful, necessary evidence.

**Prove locally first:** JVM suites (`./gradlew --offline :ota-core:test :qualify:test :extract:test :studio-api:test
:studio-core:test :engine-adapter:test`), `runtime/host-test/run.sh`, `cd factory && python -m pytest`, `scripts/catalog/tests`,
`python -m pytest native/engine/tests` (x86 engine), YAML parse checks, shell `bash -n`. Do not use CI as a debugger.

**Actions ARE appropriate for:** final validation of a candidate approaching release/OTA; checks that cannot be reproduced
locally (Android emulator/instrumented tests, 16 KB page size, NDK cross-build, anything needing huggingface.co, which the
authoring sandbox cannot reach); an APK/AAB the owner actually needs for physical testing or release; OTA publication with its
safety, compatibility and signing checks; release builds and release verification.

**Actions are NOT appropriate for:** building every platform after every push; Windows/desktop artifacts nobody asked for;
Android artifacts for an OTA-only change; re-running an expensive workflow just to see if a flaky test passes; rebuilding the
same SHA when a verified result can be reused; full release validation for docs/research/comments/bookkeeping.

**How the workflows are wired (keep it this way):**

| Workflow | Runs on | Cost |
| --- | --- | --- |
| `ci.yml` job `core-tests` | pull-request updates (not docs-only) | cheap: JVM + Python |
| `ci.yml` jobs `android`, `native-engine` | `workflow_dispatch`, or a push to a `ci/**` branch, ON PURPOSE | expensive: APK build, 3 emulators |
| `engine.yml` | `workflow_dispatch`, or a push to an `engine/**` branch, ON PURPOSE | expensive: real-model grid |
| `catalog-refresh.yml` | push to `catalog/refresh`, dispatch | cheap |
| `release-apk.yml` | push to `release/v*`, dispatch | release only; keep every gate |
| `publish-ota.yml` | push to `ota/v*`, dispatch | OTA only; keep every gate |

Start an expensive run deliberately, once per candidate, e.g. `git push origin HEAD:refs/heads/ci/android`. Never add a plain
`push: branches: ['**']` trigger. Documentation-only changes use zero minutes. Superseded runs are cancelled by `concurrency`.
Batch fixes locally and push once instead of pushing each small step.

**Release safety is NOT negotiable to save minutes:** never bypass signing verification, runtime/OTA compatibility checks,
rollback protections or any release gate. Do not publish a release/OTA/APK as a side effect of an audit.
