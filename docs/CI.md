# CI map and Actions budget

Policy: see `CLAUDE.md`, "GitHub Actions Budget". GitHub-hosted minutes are scarce; prove things locally first.

| Want | Do | Approx. cost |
| --- | --- | --- |
| Fast feedback on a PR | nothing: `core-tests` runs on PR updates (skipped for docs-only) | ~5 min |
| Android APK build + 16 KB gate + emulators with real models | `git push origin HEAD:refs/heads/ci/android` (or workflow_dispatch) | ~60+ min |
| Real-model specialization grid, engine tests | `git push origin HEAD:refs/heads/engine/<name>` (or workflow_dispatch) | ~30-60 min |
| Refresh model catalog from Hugging Face | `git push -f origin HEAD:refs/heads/catalog/refresh` | ~2 min |
| Native release APK | `release/vN` branch (see `docs/RELEASE.md`) | release |
| OTA publication | `ota/vN` branch (see `docs/OTA.md`) | release |

Run locally before any of the above: the JVM suites, `runtime/host-test/run.sh`, `factory` pytest, `scripts/catalog/tests`,
and the x86 engine tests. Re-pushing to the same `ci/**`/`engine/**` ref cancels the superseded run.
