# Model manager — artifact registry, CI refresh, downloader/store

Phone-first: the owner picks a device-appropriate base LLM, the phone downloads and verifies it, runs it and (where feasible)
specializes it locally. **Nothing downloads automatically**; every download shows size, storage impact, compatibility,
license state and purpose and needs explicit confirmation.

## Artifact registry (`registry/artifacts/<id>.json`)
One file per exact downloadable GGUF: source repo, file, **revision (commit sha)**, immutable `resolve/<revision>/<file>` URL,
size, sha256 (LFS oid), quantization/precision, exact parameter count, architecture (layers, KV heads, head dim, embd, ffn,
trained context, vocab), chat-template presence, runtime compatibility (llama.cpp / hag-engine architecture support), training
class + reasons, license identity + evidence. Curated input: `registry/artifacts/sources.json` (repos, file-name globs,
license-text URLs). Python loader/validator: `factory/llmtrainer/artifacts.py`.

Starter set: SmolLM2-135M/360M-Instruct, Qwen3-0.6B/1.7B/4B/8B (quantizations as published; F16/F32/BF16 siblings where
published, for local training of the small ones).

### Unrefreshed until CI has run
huggingface.co is unreachable from the authoring sandbox, so the committed files are `refresh_state: "unrefreshed"` placeholders:
no hash, size or revision (the loader and tests reject guessed values). The app treats them as **NOT downloadable**
(`ARTIFACT_UNREFRESHED`), shows an estimated size labelled as such, and still offers manual import.

### Refresh (CI)
`.github/workflows/catalog-refresh.yml` (push to branch `catalog/refresh`, or manual dispatch) runs
`scripts/catalog/refresh_catalog.py` (stdlib only): HF API (`?blobs=true`, tree API fallback) -> revision, exact file names (a glob
must match exactly one file; nothing is guessed), sizes, LFS sha256; GGUF header via Range requests (never a full download);
license text + model-card license -> `license_text_sha256`. Result: workflow artifact `catalog-artifacts` AND branch
`catalog-data` (`git fetch origin catalog-data && git checkout catalog-data -- registry/artifacts`). A failed entry never overwrites
a previously refreshed one. Offline tests with a fake HF: `python -m pytest scripts/catalog/tests`.

### License verification rule (fail closed)
An artifact license is `VERIFIED` only if (1) the license text fetched by CI is byte-identical (sha256) to a canonical reference
text in `registry/licenses/` (Apache-2.0 from apache.org; MIT from SPDX with its single copyright line normalised) AND (2) the
model-card license id agrees (the repo and its base repo). Everything else (Llama, Gemma, Qwen research licenses, unknown, any
byte difference, missing text) is `UNVERIFIED` and uses the existing owner-attestation flow. In the app, CI evidence unlocks
downloading for that model; an owner attestation or a registry DISALLOWED still wins. Job packages for desktop training still
carry the OWNER attestation (the importer requires `attested_by == owner`).

## Downloader / store (studio-core)
* Resumable Range downloads to `workspace/models/<variant>/<file>.part`; the final file appears only after size AND sha256 match
  (streaming hash, fsync, atomic move) — works for multi-GB files with no in-memory buffering (tested with 2.3 GB).
* Storage headroom: checked at plan, at start/resume, and every 64 MiB mid-transfer against a floor; disk-full is a STORAGE error
  (partial kept, resumable), not a network error.
* Recovery: a partial longer than the file is deleted; resumed data that fails verification is deleted and the download restarts
  once from zero; a verified final file whose manifest was lost in a crash is adopted without network.
* Cancel/resume across process death: operations are persisted; interrupted downloads reload PAUSED and resume only on request.
* Partials are never installed: `installedFile()` needs the manifest AND a final file of the recorded size.
* Install manifest (`acquired.json`, schema 2): path, size, sha256, source (`download` | `import` | `user_import`), url, revision,
  artifact id, license state/evidence at install time, provenance (`verified_download`, `download_unverified_checksum`,
  `verified_import`, `unverified_provenance`), GGUF architecture info for imports.
* Uninstall (refused while a transfer runs), free-space accounting (`free`, `reserve`, `installed`, `partial`, `headroom`).
* Manual import of a user-supplied GGUF: streamed copy with sha256, GGUF magic required, duplicate detection, recorded with
  `unverified_provenance` and license state UNVERIFIED.

API: `studio-api/.../ModelsApi.kt` (additive; `ModelsFactory.of(studio)` in studio-core). The existing `Studio` plan/download/
import/operation calls keep working and use the hardened service.
