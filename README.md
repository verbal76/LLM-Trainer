# LLM Trainer — Hot Attic Games

A reusable **specialist-model factory**: ingest authorized domain material, build and evaluate specialist LLMs,
package exact-reference knowledge, optimize for target hardware, and export portable versioned packages.
Authoritative requirements: [`CLAUDE.md`](CLAUDE.md) and [`docs/START_PROMPT.md`](docs/START_PROMPT.md).

| Path | What |
|---|---|
| `factory/` | Python factory core (schemas v1, provenance, license gate, splits, evaluation, export, device qualification), CLI-first |
| `ota-core/` | Pure-JVM OTA/update core: signed bundles, compatibility rules, rollback state machine, CI tool |
| `host-api/` | Stable Kotlin contract between the native host and OTA bundles |
| `bundle/` | OTA-updatable LLM Trainer product experience (dexed + signed to `.hagb`) |
| `app/` | Native Android host: Hot Attic Games splash, loader, updater, safe mode |
| `branding/` | Canonical studio artwork (used byte-for-byte) |
| `docs/` | `OTA.md`, architecture + ADRs, research |
| `registry/` | Machine-readable base-model license registry |
| `.github/workflows/` | CI (always), manual `Release APK` and `Publish OTA bundle` |

Builds run **only on GitHub Actions** (no EAS, no paid build services). Releases are manual and publish
`LLM-Trainer-vN.apk` as a GitHub Release asset. No app-store publishing.

Local: `./gradlew :ota-core:test` (JVM only; Android modules need an SDK) and `cd factory && python -m pytest`.
