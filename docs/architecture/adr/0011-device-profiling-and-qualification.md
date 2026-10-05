# ADR 0011: Device profiling = in-app measurement with Android system APIs; qualification algorithm duplicated (Kotlin authority + Python spec) with golden vectors

- Status: proposed (numeric defaults need hardware validation)
- Date: 2026-10-05

## Decision
The Kotlin app profiles the actual device, runs a staged benchmark of candidate profiles (model + quant + context + runtime + retrieval budget + settings), and decides locally. The Python factory holds the reference implementation (planning, CI, workstation "what would fit class X" estimates). Both implement the same pure function `qualify(device_profile, candidate_catalog, policy) -> deployment_recommendation`, checked against shared golden JSON vectors in CI.

## Signals (Android APIs, references fetched 2026-10-05 from developer.android.com)
| Need | API | Notes |
|---|---|---|
| Physical/available RAM, low-memory state | `ActivityManager.MemoryInfo`: `totalMem`, `availMem`, `threshold`, `lowMemory` | `availMem` is a point-in-time value; sample repeatedly during benchmark |
| Memory pressure events | `onTrimMemory`/`ComponentCallbacks2` | standard Android API; not fetched this round |
| Thermal | `PowerManager.getCurrentThermalStatus()`, `addThermalStatusListener()` (API 29), `getThermalHeadroom()`; statuses NONE..SHUTDOWN | Headroom forecasting API availability varies by device; treat as optional signal |
| Kills/ANR/crash history | `ActivityManager.getHistoricalProcessExitReasons()` -> `ApplicationExitInfo` (`REASON_LOW_MEMORY`, `REASON_ANR`, `REASON_CRASH`, `REASON_NATIVE_CRASH`...) | Check after each benchmark stage and across runs |
| Per-process memory | `Debug.MemoryInfo`/`/proc/self/smaps_rollup` (PSS) | not fetched this round; confirm at implementation |
| Storage | `StatFs` free bytes on target volume | |
| UI responsiveness | frame timing (`FrameMetrics`/Jank stats) during inference with the app UI active | not fetched this round |
| Battery/power | `BatteryManager` current/charge counters where exposed | device-dependent |

## Three separate verdicts
1. **Storage fit:** model + reference package + temp space + configurable free-storage reserve <= free storage.
2. **RAM fit (static estimate):** `weights (mmap-resident) + runtime_overhead + KV(context) + index + host_app + system_reserve + background_reserve + safety_margin <= availMem_budget`, where budget is derived from measured `availMem` and `threshold`, *not* `totalMem`. KV(context) is architecture-aware: standard GQA = `2 * layers * kv_heads * head_dim * bytes * ctx`; hybrid models (Qwen3.5) apply it only to full-attention layers plus fixed recurrent state; sliding-window layers (Gemma 4: window 512) cap per-layer context. The base-model registry stores the needed fields; unknown fields force the conservative fallback.
3. **Safe sustained operating fit (measured):** a profile passes only if during a sustained run (default 10 min, configurable) there is no thermal status above the policy ceiling (default: stay at/under MODERATE), tokens/sec stays above `min_ratio` of the first-minute rate (no throttling cliff), no `REASON_LOW_MEMORY`/ANR/crash exit for the app or recorded background processes, memory pressure signals stay below policy, and UI frame-drop rate stays under a threshold while the foreground app is active. 

## Conservative fallback when measurements are incomplete
Never label a profile "Recommended" without a sustained measurement. Missing signal => reduce context, move down one quant tier, add margin (default multiply overhead by 1.25), mark `confidence: estimated`, and show it as "Unverified on this device".

## Profiles
- **Performance:** lowest latency/heat; smaller model or quant, short context.
- **Balanced / Recommended:** highest-quality profile passing all three verdicts with default reserves.
- **Maximum Quality Within Safe Envelope:** highest-quality profile passing all three verdicts with reduced (not zero) reserves, and the user told the trade-off; never above the unsafe line.
- Advanced overrides allowed; any configuration failing a verdict is labeled UNSAFE.
Reserve defaults (e.g. system+background reserve, storage reserve %, margin) are **proposals requiring device data**; they live in a `policy` file, not in code.

## Alternatives rejected
- Static RAM tables per device model: stale and ignores current load.
- "Does it load" probe: forbidden by `CLAUDE.md`.
- Perfetto/Macrobenchmark as the only measurement: valuable for lab validation and CI-on-device (adb) but not available inside a user's installed app; used as the lab ground truth for calibrating the in-app estimator (not evaluated further here).
- Python-only qualifier with the phone as a dumb client: breaks offline operation and live readings.

## Consequences
Golden vectors must cover boundary cases (low RAM, high thermal, missing signals, hybrid-KV model). Benchmark results are saved as Device Profile / Benchmark Result artifacts and can be shared with the factory.
