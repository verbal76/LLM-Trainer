# Releasing — NATIVE APK vs APPLICATION-LAYER OTA

Vocabulary is defined in [VERSIONING.md](VERSIONING.md). Every release announcement (GitHub Release notes, chat,
changelog) starts with exactly one of:

* **NATIVE APK RELEASE** — a new installable APK generation `vN` (this document, section 1).
* **APPLICATION-LAYER OTA UPDATE** — a signed bundle `N.M` published to the OTA channel; never a native version (section 2).

An OTA is **never** called "v2", "version 2", or any native version. "LLM Trainer v2" exists if and only if the GitHub
Release `v2` carries `LLM-Trainer-v2.apk`.

## 1. NATIVE APK RELEASE (`release-apk.yml`)

**Trigger (owner-authorized only):** push branch `release/v<N>`, or `workflow_dispatch` with `version=<N>`
(optional `prerelease`, default **false**; optional `extra_limits` line for the notes).

**Naming**

| Thing | Value for native N |
|---|---|
| Git tag / Release | `v<N>`, title `LLM Trainer v<N>` (not a pre-release unless the input says so) |
| Assets (exactly two) | `LLM-Trainer-v<N>.apk`, `LLM-Trainer-v<N>.apk.sha256` |
| APK `versionCode` / `versionName` | `<N>` / `<N>` |
| Package / signer | `com.hotatticgames.llmtrainer`, the permanent identity (cert SHA-256 `C9:02:BC:41:93:82:0D:F5:92:E5:43:8C:5A:F5:FA:93:66:9F:34:A9:FC:2B:55:BE:F0:81:C9:F2:F5:74:11:C5`), same as v1 so it upgrades in place |
| Built-in application layer | `<N>.0`, OTA sequence = `bundleVersion` default in `bundle/build.gradle.kts` |

**Before pushing the branch** (one commit that is the release SHA): `ota/native-map.json` has the entry for the new
ABI (`"<abi>": <N>`), `app/build.gradle.kts` defaults match (`NATIVE_ABI`, `hostVersionCode/Name`, builtin
`bundleVersion` strictly above every published sequence), `bundle/build.gradle.kts` defaults match
(`bundleVersionName = "<N>.0"`, `nativeAbi`). The workflow refuses to run if these disagree.

**Gates (all must pass; nothing is published otherwise)**

1. Build from the exact triggering SHA (`-PgitSha=<short sha>` is baked into `BuildConfig.GIT_SHA`).
2. Unit tests of all JVM modules (`:ota-core :qualify :extract :studio-api :studio-core`), Python factory tests,
   native engine tests (`tools/ci/native-engine-tests.sh`, must exist).
3. Emulator suites on **API 34 and API 36**: OTA qualification (including exact-ABI cases), HAG splash, restore guard, studio flow.
4. `tools/ci/verify-apk.sh`: package, `versionCode == versionName == N`, minSdk 26, targetSdk 36, single signer equal to the
   permanent identity, canonical studio-logo sha256 (`e3d9bb56...48c4e`), built-in bundle verifies against
   `ota/keys/prod.pub`, built-in application version is `<N>.0` with `nativeAbi` = the APK's, `lib/arm64-v8a/*.so` present.
5. `tools/ci/check-16kb.sh <apk>` (16 KB page-size alignment; must exist).
6. **Install-over test** (`tools/ci/install-over-test.sh`): the *published* v1 APK is installed and run, then the
   candidate is installed with `adb install -r`; asserts same signer, higher versionCode, app data survives, the built-in
   layer reaches a healthy first frame, and no previous-ABI OTA slot survives.
7. Publish: re-verify the SHA-256, tag must still not exist, create the Release with notes stating **NATIVE APK RELEASE**
   plus the five identities (native / application / OTA sequence / runtime-ABI / source), signer, APK SHA-256 and known
   limits, attach exactly the two assets (checked after upload).

Secrets (repository secrets, never in git): `ANDROID_KEYSTORE_B64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`,
`ANDROID_KEY_PASSWORD`, `OTA_SIGNING_KEY`. Qualification tests use an **ephemeral** CI key; only the `build` job sees the
production keys.

**Engineering completion is not release authorization.** A green CI on a feature branch never publishes; only the owner
pushing `release/v<N>` (or dispatching the workflow) does.

## 2. APPLICATION-LAYER OTA UPDATE (`publish-ota.yml`)

Trigger: dispatch with `bundle_version` (OTA sequence, strictly above every published one), `version_name`
(`<native>.<minor>`, e.g. `2.1`), `native_abi` (default 2); or push branch `ota/v<N>` with the name in `ota/app-version.txt`.

Rules enforced by the workflow:

* `version_name` must be `<native>.<minor>` with `<native> == ota/native-map.json[native_abi]` and `minor >= 1`
  (`<native>.0` is the layer inside the APK). `tools/ci/check-native-map.py` rejects anything else, e.g. `1.1` for ABI 2.
* The native release `v<native>` must already exist: an OTA cannot create a native version.
* Verified against the production key for several Android levels, published to `ota-stable`, re-verified from the public
  URL, then a **canary** installs the *released native APK* on an emulator and checks it stages and persists the update;
  on failure the previous `channel.json` is restored automatically.
* The channel is shared by all native generations. A host only ever stages bundles with its **exact** `nativeAbi`; bundles
  for a newer ABI are reported as "needs a new APK", bundles for an older ABI are ignored (see OTA.md).

## 3. What changes where

| Change | Pipeline | Native version | Application version |
|---|---|---|---|
| Kotlin UI / logic / studio screens | OTA | unchanged | `N.M` -> `N.(M+1)` |
| JNI surface, shipped `.so`, manifest, host classes, native engine | NATIVE APK RELEASE | `N` -> `N+1` | resets to `(N+1).0` |
| Host API (append-only addition) | NATIVE APK RELEASE (new host), then OTA may use it | `N+1` | `(N+1).0` |
