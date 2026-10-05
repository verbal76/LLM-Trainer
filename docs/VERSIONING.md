# Versioning — one vocabulary, never ambiguous (Hot Attic Games rule)

| Term | Meaning | Where it lives | Example |
|---|---|---|---|
| **Native version** | The APK generation. Changes only when a new APK is built (native code, manifest, host). | `versionName` of the APK; GitHub Release `vN` with `LLM-Trainer-vN.apk` | `2` |
| **Application version** | The OTA-updatable product layer running on a native generation: `<native>.<minor>`. `<native>.0` is the layer built into the APK. | `bundleVersionName` in the signed bundle manifest | `2.3` |
| **OTA sequence** | Internal, monotonically increasing integer of published bundles. Engineering only. | `bundleVersion` | `#7` |
| **Runtime / ABI** | Exact native-runtime compatibility gate. | `nativeAbi` (+ `nativeRuntimeId`) | `2` / `hag-engine 1; llama.cpp 0c1e570` |
| **Source** | Git commit the artifact was built from. | `BuildConfig.GIT_SHA`, release notes | `a1b2c3d` |

Rules
1. **An OTA never creates a new native version.** "LLM Trainer v2" exists if and only if a GitHub Release `v2` carries `LLM-Trainer-v2.apk`.
2. Application updates on native N are `N.1`, `N.2`, … ; a release is always announced as either **NATIVE APK RELEASE** or **APPLICATION-LAYER OTA UPDATE**.
3. An application layer declares the native runtime it needs (`requires.nativeAbi` exact match, plus capabilities). An old native host never receives an OTA that needs a newer runtime.
4. Owner-facing screens (About / Diagnostics) show all five identities in one block.
5. `ota/native-map.json` maps `nativeAbi → native version`; the publish workflow rejects an application version whose major is not the native version of its `nativeAbi`.

## Where each identity lives in code
| Identity | Build input | Runtime surface |
|---|---|---|
| Native version | `-PhostVersionName` / `-PhostVersionCode` (default 2) -> `BuildConfig.VERSION_NAME` | `diagnostics.identity.nativeVersion`, `HostServices.hostVersionName` |
| Application version | bundle `-PbundleVersionName` (default `2.0`, convention `<native>.<minor>`) -> manifest `bundleVersionName` | `diagnostics.identity.appVersion` (from the *running* bundle) |
| OTA sequence | bundle `-PbundleVersion` (default 3) -> manifest `bundleVersion` | `diagnostics.identity.otaSequence` |
| Runtime / ABI | `NATIVE_ABI` (2) + `-PnativeRuntimeId` -> `BuildConfig.NATIVE_ABI/NATIVE_RUNTIME_ID` | `diagnostics.identity.nativeAbi / nativeRuntimeId` |
| Source | `-PgitSha=<short sha>` (workflows; `dev` locally) -> `BuildConfig.GIT_SHA` | `diagnostics.identity.sourceSha` |

`HostRuntime.diagnosticsJson()` carries `identity` (structured) and `identityBlock` (the formatted owner-facing text):

```
Native version: 2
Application version: 2.0
OTA sequence: #3 (builtin)
Runtime: ABI 2, hag-engine 1; llama.cpp 0c1e570
Source: a1b2c3d
```

The running identity is published *before* the bundle builds its view (source `starting:<builtin|ota>` while loading),
so "running none / v0" can only mean genuine host safe mode, where the block says `none (safe mode)`.

`ota/native-map.json` (`nativeAbi -> native version`) is the single source for rule 5; `tools/ci/check-native-map.py`
enforces it in `publish-ota.yml`, and `release-apk.yml` refuses a native version that is not in the map.

## Reconciled history (native v1)
| Native | Application | OTA sequence | Notes |
|---|---|---|---|
| v1 (APK `v1`, source `2e146d4`) | v1.0 | #1 | the bundle built into the APK (manifest name `1.0.1`) |
| v1 | **v1.1** | #2 | first (and only) OTA on native v1; source `28de300`; manifest name `1.0.2` (legacy label — the owner-facing name is **v1.1**). Earlier messages called this "OTA v2": that was wrong wording; no native v2 existed. |
| v2 (APK `v2`, this round) | v2.0 | #3 | built-in layer of the native v2 APK; `nativeAbi` 2 |
