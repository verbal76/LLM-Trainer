# OTA / update architecture

## Goal
Ship ordinary Kotlin, UI, business-logic and model-management changes **without a new APK**, and make it
structurally impossible to push a native/runtime change through OTA.

## Two layers
| Layer | What it is | How it changes |
|---|---|---|
| **Host (APK)** | Studio splash, bundle loader, update client, rollback, safe mode, host API implementation, `kotlin-stdlib`, and (later) the native inference runtime (llama.cpp/JNI) | New APK only |
| **Bundle (`.hagb`)** | Pure Kotlin `classes*.dex` + optional `assets/`; the LLM Trainer product experience | OTA |

The splash, loader, verifier and rollback logic live in the host on purpose: an OTA bundle can never break
the thing that rescues the app from a bad OTA bundle.

## Compatibility rules (enforced twice: before download and again on-device against the signed manifest)
A bundle declares `requires` in its signed manifest. It is installable only if **all** hold:

1. **Host API range** – `hostApi.min <= host.HOST_API_LEVEL <= hostApi.max`. The host API
   (`host-api` module) is append-only inside an API *major* (`level / 100`). Adding a method bumps the level by one
   (old bundles keep working; new bundles declare `min`). A breaking change jumps to the next hundred, so every
   older bundle is incompatible *by declaration* and is rejected instead of crashing. Bundle builds default
   `max` to the top of their major.
2. **Native ABI exact match** – `nativeAbi == host.NATIVE_ABI`. Anything that changes JNI signatures or the shipped
   `.so` set bumps it. Bundles never contain native code: `.so`, `lib/**`, `.jar`, `.apk` are refused by the
   format itself even if signed.
3. **Capabilities** – every capability the bundle lists must be advertised by the host (e.g. `core.v1`,
   `device.snapshot.v1`, later `inference.gguf.v1`). This lets bundles depend on a *feature*, not an exact APK,
   so one APK serves many bundle versions and vice versa.
4. **minSdk** of the device.
5. **Strictly newer** than everything already held and than the bundle embedded in the APK. A bundle version that
   was quarantined on a device is never retried; fixes ship as a higher version.
6. **Identity** – `bundleId` and `channel` match.

`kotlin-stdlib` is provided by the host (bundles are compiled without it); the host never minifies. The stdlib
version is therefore part of the host API contract: upgrading it incompatibly is a host-API-major change.

If the only newer bundles need a newer host, the app reports **"needs new APK"** (diagnostics + UI) and changes
nothing. It never installs an APK by itself.

## Trust & integrity
* ECDSA P-256/SHA-256 signature over the raw bytes of `manifest.json`; verification keys are embedded in the APK
  (`keyId`, rotation = ship a host trusting both). Private key lives only in the `OTA_SIGNING_KEY` repo secret.
* Manifest lists every payload file with sha256+size; unlisted/missing/modified files, path traversal, duplicate
  entries, zip bombs (entry/size caps) are rejected. Nothing is written before full verification succeeds.
* The channel index is **not** trusted: it only locates bundles. A lying index cannot smuggle an incompatible or
  unsigned bundle (covered by tests). Residual risk: an attacker controlling the index can withhold updates
  (freeze), not install code.
* Staged dex files are set read-only (Android 14+ requirement) and re-hashed on every boot.

## Boot / rollback protocol (`UpdateStore`)
```
download -> verify -> stage as PENDING (trial)   (takes effect next launch)
launch: pending(trial) -> active -> lastKnownGood -> built-in bundle (in APK) -> host safe mode
```
* The boot attempt is persisted **before** loading, so a hard crash counts. Trial budget 2 unhealthy boots,
  active/LKG budget 3; then the slot is quarantined and the next candidate runs.
* A slot becomes good (`active` + `lastKnownGood`) only after `selfTest()` passed **and** the bundle drew its first
  frame (host-driven).
* In-process failures (load error, constructor throws, selfTest fails, view creation throws) quarantine at once
  and fall back within the same launch. An uncaught exception whose stack contains bundle code within 2 minutes of
  start quarantines that slot.
* Installing a newer APK drops any OTA slot not newer than the APK's built-in bundle.
* Corrupt `state.json` resets to built-in rather than bricking.
* Known limit: a bundle that hangs the main thread without crashing is only caught once the OS/user kills the app
  (the boot counter then rolls it back after 2–3 launches).

## Diagnostics
In-app "Diagnostics" (and safe mode) show APK version, host API level, native ABI/runtime id, capabilities,
running source/slot/version, active/LKG/pending slots, quarantine reasons, last 20 update events. Logcat tag
`HagOta`.

## Qualification (automated, GitHub Actions)
* `:ota-core:test` – 49 JVM tests: compatibility, verification attack cases, rollback state machine, coordinator.
* Emulator job (`OtaQualificationTest`): real `DexClassLoader`; full stage→restart→promote cycle; rollback from a
  failing selfTest and a throwing entry; crash-loop quarantine → built-in; incompatible-ABI bundle refused without
  download; tampered bundle with a matching index hash rejected; logo asset hash. Fixtures are signed in CI with an
  ephemeral key that the test APK trusts.
* Release workflow re-verifies the real built-in bundle against `ota/keys/prod.pub` and the logo hash inside the
  APK before publishing.

## Distribution
* Channel: GitHub release `ota-stable` (pre-release, rolling): `channel.json` + `*.hagb` assets
  (`https://github.com/verbal76/LLM-Trainer/releases/download/ota-stable/channel.json`). Published by the
  `Publish OTA bundle` workflow (manual dispatch; checks monotonic version and compatibility first).
* APKs: GitHub Release `vN` with `LLM-Trainer-vN.apk` attached, via the manual `Release APK` workflow.
* No Google Play / other store publishing.

## What requires a new APK (and nothing else)
Native libraries or JNI changes (e.g. adding llama.cpp), host API breaking changes, kotlin-stdlib incompatibility,
manifest permissions/components, the splash/loader/updater itself, signing-key rotation of the verification keys.

## Required one-time owner setup (secrets)
`ANDROID_KEYSTORE_B64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD` (stable APK
signing identity so sideloaded updates install over each other) and `OTA_SIGNING_KEY` (the exact text of the `.key` file: base64 PKCS#8, matching
`ota/keys/prod.pub`). Release workflows fail loudly if any are missing; CI never needs them.
