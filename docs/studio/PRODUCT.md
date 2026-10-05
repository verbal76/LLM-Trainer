# LLM Trainer Studio — owner-facing product design (OTA milestone)

Authoritative: `CLAUDE.md`, `docs/OTA.md`. This file pins the product flow, the honest on-device/desktop split, and the
hard boundaries of the installed v1 host that the OTA bundle must live inside.

## Owner workflow (all screens live in the OTA bundle; no ADB/terminal/JSON editing)
Dashboard (projects) → Create specialist (name, domain, purpose) → Device profile → Model recommendations (catalog + device
qualifier + license state) → Model detail (license evidence, storage, feasibility) → Select base model → [Acquire/import model:
plan + explicit confirmation] → Add sources (document picker) → Ingestion report → Dataset build + REVIEW (include/exclude,
origin, split) → Method (honest options) → Training job package (export to desktop) / RAG reference package → Evaluation
(held-out set export; import desktop results; base vs specialist) → Specialist package export → Import elsewhere.

## Honest capability split (never blur these in UI text)
* **Retrieval/RAG** = exact-reference lookup over source chunks. It is NOT training. Label it "Reference package".
* **Prompt/system specialization** = instructions only. NOT training.
* **Adapter training (LoRA/QLoRA)** = parameter adaptation. v1 host has NO inference/training runtime (nativeAbi 1,
  runtime "none-v1"), so adaptation runs on desktop/GPU: the phone prepares a **training job package**
  (dataset JSONL + config + manifest + license evidence) that `llmtrainer import-job` consumes. The phone never claims to
  have trained anything.
* **Evaluation** needs a model to run. Until an on-device inference runtime ships (next APK, new native ABI), evaluation
  runs on desktop (`llmtrainer evaluate`) and the phone IMPORTS and displays the report. The phone prepares the held-out set.
* **Export** = specialist package (adapter/model refs + hashes, base model identity/version, license state, dataset/eval
  metadata, compatibility) consumable without LLM Trainer.
* Private source material never leaves the device except via an owner-initiated export/share of a package.

## License safety (fail closed — unchanged)
States: VERIFIED / UNVERIFIED / DISALLOWED (see factory/llmtrainer/licenses.py). Registry baseline is UNVERIFIED for models whose
authoritative license text has not been read. The phone has normal network access, so the product provides an owner-facing
**evidence flow**: fetch the license text from the model's authoritative URL (or import a license file), show it, record
sha256 + source URL + fetched_at, and let the OWNER attest which uses it permits (commercial / fine-tune / adapter /
redistribution / attribution). That attestation is stored per model in the workspace and exported with job/specialist packages.
Only then does the entry become VERIFIED (scope text: "owner-reviewed license text from <url>, hash <sha>; permissions as
attested by owner"). The app does not interpret legal text and says so. Secondary-source claims never become VERIFIED.

## Downloads
Never automatic. Show model, approx size, storage impact, device compatibility, license state, intended use; require explicit
confirm; HTTPS only; Range-resumable `.part` files; storage-headroom check; works only while the app process is alive
(no foreground service in the v1 host) and resumes on next launch. Blocked unless license is VERIFIED for the intended use.

## Persistence
All state under `filesDir/studio/` (per-project directories; atomic JSON writes; per-file ingestion status; resumable
operations) so process death never loses a project. No global mutable model state.

## OTA boundary of the installed v1 host (HARD constraints)
* Bundle = pure Kotlin `classes*.dex` + assets, built with Kotlin stdlib (host-provided), platform `android.*` and platform
  `org.json`. No native code, no new manifest entries (no new activities/services/permissions/intent filters).
* Host API level 1 only: `HostServices` {hostVersionName, hostApiLevel, nativeAbi, nativeRuntimeId, diagnosticsJson(),
  deviceSnapshotJson(), checkForUpdates(cb), restartApp()}. The Activity passed to `createContentView(context)` IS the host
  Activity, so the bundle may use platform APIs through it (files, ContentResolver, startActivity, network).
* `INTERNET` permission exists in the host manifest. There is NO storage permission; use the Storage Access Framework.
* Activity results are NOT forwarded to the bundle. File pick/create results must be received via a bundle-owned
  `android.app.Fragment` added to the host Activity's FragmentManager (`startActivityForResult` from the fragment).
* `HostServices.diagnosticsJson()` reports runningSource "none"/v0 when called during `createContentView` (host bug,
  fixed for the next APK); bundle code must read diagnostics lazily AFTER the view is attached.
* Anything that needs the native boundary (inference runtime, foreground service, new permissions, splash/host fixes) is
  recorded for the next APK and is NOT faked in the bundle.
