# LLM Trainer --- Initial Claude Code Execution Prompt

You are beginning the first real engineering/design round for **LLM
Trainer**, a Hot Attic Games internal/product tool.

## FIRST ACTION

Read the repository's `CLAUDE.md` completely before making architecture
decisions or writing implementation code.

Treat `CLAUDE.md` as the permanent product/engineering constitution for
this repository.

Also inventory all repository files and assets, including any Hot Attic
Games studio logo, LLM Trainer icon, product splash, README, prior
notes, or placeholder files.

Do not assume the old placeholder README or any earlier provisional
suggestion such as "PyTorch + Hugging Face + CLI/config files" is an
approved final architecture. Those technologies may still be
appropriate, but they must earn their place through the architecture
review.

## PRODUCT IN ONE SENTENCE

Build a reusable **specialist-model factory** that ingests authorized
domain material such as manuals, training books, technical papers and
documentation; creates and evaluates specialist LLMs; builds
exact-reference knowledge packages; optimizes them for target hardware;
and exports versioned models/packages that can be used both inside LLM
Trainer and in other applications.

Examples include:

-   Motorcycle Mechanic
-   HVAC Technician
-   Urban / Town Design
-   Automotive Diagnostics

This is NOT merely a PDF chat/RAG application, and it is NOT merely a
collection of training scripts.

## IMPORTANT PRODUCT REQUIREMENTS

The architecture must support:

-   open-weight, license-compatible base models;
-   multiple model families and sizes;
-   7B/8B-class specialists as normal supported targets;
-   smaller and larger models where appropriate;
-   LoRA/QLoRA or other appropriate fine-tuning methods;
-   real train/validation/unseen-test separation;
-   base-vs-specialist evaluation;
-   provenance from source → derived training example → model version;
-   source-rights/license tracking;
-   exact-reference retrieval packages for facts that should not be
    trusted to weights alone;
-   quantization and target-specific exports;
-   optional distillation workflows;
-   reusable model packages consumable by other apps;
-   device profiling and benchmarking;
-   safe deployment recommendations.

### NON-NEGOTIABLE DEVICE RULE

Do not define "fits" as "the model loads."

The deployment system must recommend the **highest-quality
model/configuration that can run without materially degrading normal
device operation**.

Preserve RAM/storage/OS/app/background-process headroom and account for
runtime overhead, context/KV cache, retrieval, thermals, responsiveness,
memory pressure and sustained throttling.

If a 7B model technically starts but causes harmful memory pressure, UI
degradation, background-app killing, instability, excessive heat or
severe sustained throttling, it does NOT fit.

If a 7B model runs comfortably inside the safe envelope, do not
arbitrarily downgrade the user to a smaller model.

## THIS FIRST ROUND

Do NOT immediately build the entire product.

Perform a serious foundation/architecture round first, then implement
only the amount of scaffolding/core infrastructure that is justified by
that analysis.

### Phase 1 --- Repository Archaeology

Inspect:

-   current repository state;
-   git history/branches/tags if present;
-   README and prior planning;
-   supplied branding/icon/splash assets;
-   existing code, if any.

Report what actually exists versus what is merely planned.

### Phase 2 --- Current Technical Research

Research the current ecosystem using authoritative/current sources where
available.

Evaluate practical options for:

-   training/fine-tuning;
-   LoRA/QLoRA;
-   dataset handling;
-   evaluation;
-   model registries/manifests;
-   quantization;
-   GGUF and/or other appropriate deployment formats;
-   mobile inference;
-   desktop inference;
-   GPU training;
-   retrieval/index packaging;
-   device profiling/benchmarking;
-   UI/application framework;
-   CLI/automation interface.

Also research representative open-weight base models suitable for
specialist fine-tuning across several size classes.

Do NOT permanently select models based on memory or popularity alone.

For each candidate family, verify current licensing and intended
commercial/fine-tuning/redistribution implications from authoritative
sources.

Do not make unsupported legal claims. Record uncertainties explicitly.

### Phase 3 --- Architecture Proposal

Produce a concrete architecture that separates at least:

-   workspace/project management;
-   source ingestion;
-   provenance/rights registry;
-   corpus processing;
-   dataset generation;
-   training orchestration;
-   evaluation;
-   retrieval/reference package generation;
-   model registry;
-   quantization/optimization;
-   device qualification;
-   export/package generation;
-   user interface;
-   CLI/automation.

Specify which parts should be Python or another language/framework and
why.

The backend must be independently testable and automatable.

The UI must not become the only way to operate the system.

### Phase 4 --- Artifact Contracts

Design versioned schemas/contracts for at least:

1.  Specialist Project
2.  Source Manifest
3.  Dataset Manifest
4.  Training Run
5.  Evaluation Run
6.  Model Manifest
7.  Device Profile
8.  Deployment Recommendation
9.  Export Package

Include hashes/version identities so outputs are reproducible and
traceable.

Do not create fake migration history. Start schema versions honestly at
v1.

### Phase 5 --- Device Qualification Design

Design the no-degradation device selection algorithm.

It must distinguish:

-   storage fit;
-   RAM fit;
-   safe sustained operating fit.

Define how the system will estimate and, where possible, measure:

-   model memory;
-   runtime overhead;
-   KV/context cost;
-   retrieval/index cost;
-   host app cost;
-   system reserve;
-   background-process reserve;
-   free-storage reserve;
-   TTFT;
-   tokens/sec;
-   peak/sustained RAM;
-   thermal behavior;
-   throttling;
-   UI responsiveness;
-   crashes/ANRs;
-   background-process impact.

Define a conservative fallback when measurements are incomplete.

Design user-facing profiles such as:

-   Performance
-   Balanced / Recommended
-   Maximum Quality Within Safe Envelope

### Phase 6 --- Training and Evaluation Design

Define the first reproducible end-to-end training experiment.

It should be small enough to execute economically but representative
enough to prove the architecture.

Do NOT use copyrighted/proprietary training material from the repository
unless its use is clearly authorized.

If no suitable training corpus exists yet, create a synthetic/test
fixture or use an appropriately licensed small public dataset solely to
validate the pipeline.

The experiment must demonstrate:

source → processed corpus → dataset → training/fine-tuning → unseen
evaluation → versioned artifact → export validation

Do not claim domain improvement without measured evidence.

### Phase 7 --- Initial Implementation

After the architecture is coherent:

Implement the minimum durable foundation needed to prove the
architecture.

Prioritize:

-   project/schema definitions;
-   manifests/hashing;
-   provenance;
-   configuration;
-   reproducibility;
-   model/license registry structure;
-   dataset split/leakage protections;
-   evaluation interfaces;
-   export manifest/package contract;
-   automated tests.

Avoid building disposable demo code that will immediately need
replacement.

If actual model training would consume significant compute, do not
silently launch an expensive job. Prepare the reproducible
command/config and clearly identify the expected resource requirement.

### Phase 8 --- Branding Foundation

Inventory the supplied Hot Attic Games and product branding assets.

Use exact supplied assets.

Establish the intended launch order for the eventual application:

native/system startup → Hot Attic Games studio splash → LLM Trainer
product experience

Do not redraw or approximate supplied artwork.

Do not spend this first architecture round polishing visual presentation
ahead of the training foundation.

## AUTONOMY

Work independently on engineering questions that do not require owner
judgment.

Do not stop after every small decision to ask permission.

You may create coherent commits and push engineering work to the working
branch if that is consistent with repository practice.

Do NOT publish a public release, upload private corpora, incur paid
external compute charges, or expose proprietary material without
explicit authorization.

## REQUIRED FIRST-ROUND HANDOFF

When the foundation round reaches a coherent stopping point, report:

1.  Repository/branch/SHA.
2.  What existed before this round.
3.  Architecture selected and why.
4.  Technologies evaluated and rejected.
5.  Initial supported model/runtime strategy.
6.  License/provenance strategy.
7.  Data and artifact schemas created.
8.  Training pipeline design.
9.  Evaluation methodology.
10. Exact-reference/RAG package design.
11. Device no-degradation qualification design.
12. Export/package design.
13. What was implemented.
14. Tests and results.
15. Any CI created and status.
16. Branding assets discovered and their hashes.
17. Remaining risks/open decisions.
18. The next recommended engineering round.

Do not publish a release.

The objective of this first round is to establish a production-quality
foundation for a reusable specialist-model factory, not to rush out a
toy fine-tuning demo.
