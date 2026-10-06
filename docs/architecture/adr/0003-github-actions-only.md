# ADR 0003: GitHub Actions is the only build system

- Status: accepted (fixed by lead; recorded, not re-litigated)
- Date: 2026-10-05

## Decision
All builds and automated validation run in GitHub Actions. No EAS or other hosted build service.

## Implications for architecture
- Hosted runners are CPU-only: CI covers unit/integration tests on tiny fixtures, schema/registry validation, Gradle build, Kotlin unit tests, and golden-vector parity between the Python and Kotlin qualifiers. It cannot run GPU training or on-device benchmarks.
- Heavy work (real training, GPU evals) is manual `workflow_dispatch` on an owner-authorized runner or run locally; the reproducible command is committed (see `FIRST_EXPERIMENT.md`).
- Pinned toolchains (Python lock file, Gradle wrapper, llama.cpp commit) so CI is reproducible. Build success does not authorize releases (`CLAUDE.md`).

## Alternatives rejected
EAS, other hosted CI/build services: excluded by the decision. Self-hosted GPU runners are not excluded as *runners* of GitHub Actions jobs, but cost/authorization is an owner decision.
