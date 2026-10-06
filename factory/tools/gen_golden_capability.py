#!/usr/bin/env python3
"""Generate the shared golden vectors for the device-profiler v2 (per-artifact capabilities).

The Python implementation (``llmtrainer.device``, section "Device profiler v2") is the executable spec; the Kotlin port
(``qualify/.../Capability.kt``) must reproduce every expected value in the generated file.

    python factory/tools/gen_golden_capability.py           # (re)write factory/tests/golden/device_capability.v1.json
    python factory/tools/gen_golden_capability.py --check   # exit 1 if the committed file drifted

Also holds the SPEC of ``state_from_snapshot`` (host ``deviceSnapshotJson`` -> transient DeviceState).

The artifact specs below are ILLUSTRATIVE fixtures (geometry from model cards as remembered by the author, sizes derived
from parameter count x bits per weight). They are NOT catalog data: real specs come from registry/artifacts after the
catalog-refresh CI job has read the GGUF headers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from llmtrainer import device as dv  # noqa: E402
from llmtrainer.device import ArtifactSpec, CapabilityPolicy, DeviceState, MeasurementRecord  # noqa: E402
from llmtrainer.schemas import SafetyPolicy  # noqa: E402

_spec = importlib.util.spec_from_file_location("gen_golden_device", HERE / "gen_golden_device.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

GOLDEN_PATH = HERE.parent / "tests" / "golden" / "device_capability.v1.json"
FORMAT = "device_capability.v1"

BPW = {"Q8_0": 8.5, "Q4_K_M": 4.85, "F16": 16.0, "F32": 32.0}
MODELS = {
    "smollm2-135m": dict(p=134.5e6, nom=0.135, L=30, kv=3, hd=64, embd=576, ffn=1536, vocab=49152, ctx=8192),
    "smollm2-360m": dict(p=362e6, nom=0.36, L=32, kv=5, hd=64, embd=960, ffn=2560, vocab=49152, ctx=8192),
    "qwen3-0.6b": dict(p=596e6, nom=0.6, L=28, kv=8, hd=128, embd=1024, ffn=3072, vocab=151936, ctx=40960),
    "qwen3-1.7b": dict(p=1.72e9, nom=1.7, L=28, kv=8, hd=128, embd=2048, ffn=6144, vocab=151936, ctx=40960),
    "qwen3-4b": dict(p=4.02e9, nom=4.0, L=36, kv=8, hd=128, embd=2560, ffn=9728, vocab=151936, ctx=40960),
    "qwen3-8b": dict(p=8.19e9, nom=8.0, L=36, kv=8, hd=128, embd=4096, ffn=12288, vocab=151936, ctx=40960),
}


def mk(m, q, prec, cls, src=None, dl=True, lic="VERIFIED", **over) -> dict:
    d = MODELS[m]
    spec = dict(artifact_id=f"{m}-{q.lower()}", model_id=m, quantization=q, precision=prec, sha256="sha256:" + "ab" * 32,
                size_bytes=int(d["p"] * BPW[q] / 8), params=d["p"], nominal_params_b=d["nom"], layers=d["L"], kv_heads=d["kv"],
                head_dim=d["hd"], embd=d["embd"], ffn=d["ffn"], vocab=d["vocab"], ctx_train=d["ctx"], training_class=cls,
                training_source_artifact_id=src, downloadable=dl, license_state=lic, runtime_supported="unverified")
    spec.update(over)
    return spec


def illustrative_specs() -> list[dict]:
    return [
        mk("smollm2-135m", "Q8_0", "quantized", "inference_only", "smollm2-135m-f16"),
        mk("smollm2-135m", "F16", "f16", "local_full"),
        mk("smollm2-360m", "Q8_0", "quantized", "inference_only", "smollm2-360m-f16"),
        mk("smollm2-360m", "F16", "f16", "local_full"),
        mk("qwen3-0.6b", "Q8_0", "quantized", "inference_only", "qwen3-0.6b-f16"),
        mk("qwen3-0.6b", "F16", "f16", "local_partial"),
        mk("qwen3-1.7b", "Q8_0", "quantized", "inference_only"),
        mk("qwen3-1.7b", "F16", "f16", "external_only"),
        mk("qwen3-4b", "Q4_K_M", "quantized", "inference_only"),
        mk("qwen3-4b", "Q8_0", "quantized", "inference_only"),
        mk("qwen3-8b", "Q4_K_M", "quantized", "inference_only"),
        mk("qwen3-8b", "Q8_0", "quantized", "inference_only"),
    ]


def unrefreshed_specs() -> list[dict]:
    """What the app sees BEFORE the first CI refresh: nominal size only, nothing downloadable."""
    out = []
    for m, q, prec, cls, src in [("smollm2-360m", "Q8_0", "quantized", "inference_only", "smollm2-360m-f16"), ("smollm2-360m", "F16", "f16", "local_full", None),
                                 ("qwen3-4b", "Q4_K_M", "quantized", "inference_only", None), ("qwen3-8b", "Q4_K_M", "quantized", "inference_only", None)]:
        out.append(mk(m, q, prec, cls, src, dl=False, sha256=None, size_bytes=None, params=None, layers=None, kv_heads=None, head_dim=None,
                      embd=None, ffn=None, vocab=None, ctx_train=None))
    return out


SPEC_SETS = {"illustrative": illustrative_specs(), "unrefreshed": unrefreshed_specs()}


def state_from_snapshot(s: dict) -> DeviceState:
    """SPEC (Kotlin DeviceSnapshot.stateFromSnapshot mirrors this)."""
    def num(k):
        v = s.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    charging = s.get("charging", s.get("isCharging"))
    return DeviceState(thermal_status=base._thermal_index(s.get("thermalStatus")), battery_pct=num("batteryPct"),
                       charging=charging if isinstance(charging, bool) else None,
                       power_save=s.get("powerSaveMode") if isinstance(s.get("powerSaveMode"), bool) else None)


def apply_patch(specs: list[dict], patch: dict | None) -> list[dict]:
    out = []
    for s in specs:
        s2 = dict(s)
        s2.update((patch or {}).get(s["artifact_id"], {}))
        out.append(s2)
    return out


# ----------------------------- serialization of expectations ---------------- #


def opt_row(o):
    return None if o is None else o.model_dump(mode="json")


def cap_row(c) -> dict:
    d = c.model_dump(mode="json")
    return d


def choice_row(p) -> dict:
    return p.model_dump(mode="json")


def rec(rid, artifact, kind, **kw) -> dict:
    d = dict(record_id=rid, artifact_id=artifact, device_id="Google-Pixel 10 Pro XL", recorded_at="2026-10-06T00:00:00Z", kind=kind)
    d.update(kw)
    return d


def cap_case(name, device, set_name="illustrative", *, state=None, patch=None, records=None, safety=None, cap=None) -> dict:
    specs = [ArtifactSpec(**s) for s in apply_patch(SPEC_SETS[set_name], patch)]
    recs = [MeasurementRecord(**r) for r in (records or [])]
    sp = SafetyPolicy(**(safety or {}))
    cp = CapabilityPolicy(**(cap or {}))
    st = DeviceState(**state) if state is not None else None
    rep = dv.choose_models(device, specs, sp, cp, st, recs)
    return dict(
        name=name, kind="capability", spec_set=set_name, spec_patch=patch or {}, device=device.model_dump(mode="json", exclude={"content_hash", "schema_version", "kind"}),
        state=state, safety=safety or {}, cap_policy=cap or {}, records=records or [],
        expected=dict(capabilities=[cap_row(c) for c in rep.capabilities], choices=[choice_row(c) for c in rep.choices]),
    )


def snap_case(name, snap_: dict, set_name="illustrative", **kw) -> dict:
    prof, notes, withhold = base.profile_from_snapshot(snap_)
    state = state_from_snapshot(snap_)
    case = cap_case(name, prof, set_name, state=state.model_dump(mode="json"), **kw)
    case["kind"] = "capability_snapshot"
    case["snapshot"] = snap_
    del case["device"]
    case["expected"]["notes"] = notes
    return case


def pixel(avail=2.1, **kw):
    gb = 1024**3
    d = dict(manufacturer="Google", model="Pixel 10 Pro XL", sdkInt=37, abis="arm64-v8a", totalRamBytes=int(15.2 * gb), availRamBytes=int(avail * gb),
             lowMemoryThresholdBytes=400 * 1024 * 1024, lowMemory=False, isLowRamDevice=False, memoryClassMb=512, freeStorageBytes=int(54.5 * gb),
             totalStorageBytes=int(228.4 * gb), cpuCores=8, thermalStatus=0)
    d.update(kw)
    return {k: v for k, v in d.items() if v is not None}


def merge_case(name, existing, new) -> dict:
    merged = dv.merge_records([MeasurementRecord(**r) for r in existing], [MeasurementRecord(**r) for r in new])
    return dict(name=name, kind="merge", existing=existing, new=new, expected=dict(record_ids=[r.record_id for r in merged]))


def build() -> dict:
    pid = "Google-Pixel 10 Pro XL"
    full_inf = dict(context_tokens=1024, ttft_ms=2500, tokens_per_s=7.0, peak_ram_mb=5000, sustained_ram_mb=4900, thermal_throttle_ratio=0.85,
                    ui_jank_pct=1.0, crashes=0, anrs=0, background_kills=0, sustained_minutes=15)
    cases = []
    q = cases.append
    q(snap_case("pixel10/regression-15gb-2.1gb-available", pixel()))
    q(snap_case("pixel10/charging-cool-ready", pixel(charging=True, batteryPct=80, powerSaveMode=False)))
    q(snap_case("pixel10/on-battery-low", pixel(charging=False, batteryPct=20)))
    q(snap_case("pixel10/on-battery-high-not-charging", pixel(charging=False, batteryPct=90)))
    q(snap_case("pixel10/thermal-moderate-power-save", pixel(thermalStatus=2, powerSaveMode=True, charging=True)))
    q(snap_case("pixel10/low-free-storage", pixel(freeStorageBytes=int(3.0 * 1024**3))))
    q(snap_case("pixel10/low-memory-state", pixel(avail=0.2, lowMemory=True)))
    q(snap_case("pixel10/unrefreshed-catalog", pixel(), "unrefreshed"))
    q(snap_case("phone8/snapshot", base.snap(7.4, 3.5, 40)))
    q(snap_case("phone6/snapshot", base.snap(5.5, 2.5, 50)))
    q(snap_case("phone4/snapshot", base.snap(3.7, 1.4, 12)))
    q(cap_case("flagship12/profile", base.flagship12()))
    q(cap_case("phone8/profile-no-state", base.phone8()))
    q(cap_case("phone4/profile", base.phone4()))
    q(cap_case("flagship12/disallowed-license-never-offered", base.flagship12(), patch={"qwen3-4b-q8_0": {"license_state": "DISALLOWED"}, "qwen3-4b-q4_k_m": {"license_state": "DISALLOWED"}}))
    q(cap_case("flagship12/runtime-unsupported", base.flagship12(), patch={"qwen3-8b-q4_k_m": {"runtime_supported": "no"}}))
    q(cap_case("flagship12/loose-min-tps", base.flagship12(), safety=dict(min_tps=1.0)))
    q(cap_case("flagship12/stricter-training-utilization", base.flagship12(), cap=dict(train_max_utilization=0.5)))
    q(cap_case("flagship12/long-context-policy", base.flagship12(), cap=dict(infer_contexts=[512, 2048], eval_min_context=2048)))
    # measured behavior
    q(cap_case("measured/8b-inference-passes", base.flagship12(device_id=pid), records=[rec("r1", "qwen3-8b-q4_k_m", "inference", **full_inf)]))
    q(cap_case("measured/8b-inference-slow", base.flagship12(device_id=pid), records=[rec("r1", "qwen3-8b-q4_k_m", "inference", **{**full_inf, "tokens_per_s": 1.0})]))
    q(cap_case("measured/4b-crashes", base.flagship12(device_id=pid), records=[rec("r1", "qwen3-4b-q4_k_m", "inference", **{**full_inf, "context_tokens": 2048, "crashes": 1})]))
    q(cap_case("measured/partial-inference-medium", base.flagship12(device_id=pid),
               records=[rec("r1", "qwen3-4b-q4_k_m", "inference", context_tokens=2048, tokens_per_s=8.0, peak_ram_mb=3500)]))
    q(cap_case("measured/load-record-overrides-load-ram", base.flagship12(device_id=pid), records=[rec("r1", "qwen3-8b-q8_0", "load", load_time_ms=9000, load_peak_ram_mb=9000)]))
    q(cap_case("measured/other-device-ignored", base.flagship12(device_id=pid),
               records=[{**rec("r1", "qwen3-8b-q4_k_m", "inference", **full_inf), "device_id": "someone-else"}]))
    q(cap_case("measured/hash-mismatch-ignored", base.flagship12(device_id=pid),
               records=[{**rec("r1", "qwen3-8b-q4_k_m", "inference", **full_inf), "artifact_sha256": "sha256:" + "cd" * 32}]))
    q(cap_case("measured/training-success-beyond-estimate", base.flagship12(device_id=pid),
               records=[rec("t1", "qwen3-0.6b-f16", "training", trainable_last_layers=20, train_peak_ram_mb=4800, train_tokens_per_s=40.0,
                            checkpoint_size_mb=1100, train_minutes=30, train_completed=True, train_oom=False, thermal_max_status=1, battery_drop_pct=5)]))
    q(cap_case("measured/training-oom-caps-layers", base.flagship12(device_id=pid),
               records=[rec("t1", "qwen3-0.6b-f16", "training", trainable_last_layers=5, train_peak_ram_mb=9999, train_oom=True, train_completed=False)]))
    q(cap_case("measured/training-full-oom-360m", base.flagship12(device_id=pid),
               records=[rec("t1", "smollm2-135m-f16", "training", train_peak_ram_mb=9999, train_oom=True, train_completed=False)]))
    q(cap_case("measured/training-partial-record-medium", base.flagship12(device_id=pid),
               records=[rec("t1", "smollm2-135m-f16", "training", trainable_last_layers=8, train_peak_ram_mb=1500)]))
    # merge logic
    a = rec("a1", "qwen3-8b-q4_k_m", "inference", context_tokens=2048, recorded_at="2026-10-01T00:00:00Z", tokens_per_s=3.0)
    b = rec("a2", "qwen3-8b-q4_k_m", "inference", context_tokens=2048, recorded_at="2026-10-02T00:00:00Z", tokens_per_s=6.0)
    c = rec("a3", "qwen3-8b-q4_k_m", "inference", context_tokens=4096, recorded_at="2026-09-01T00:00:00Z", tokens_per_s=5.0)
    t = rec("t1", "qwen3-0.6b-f16", "training", trainable_last_layers=8, train_peak_ram_mb=1)
    t2 = rec("t2", "qwen3-0.6b-f16", "training", train_peak_ram_mb=2)
    q(merge_case("merge/latest-wins-per-key", [a, c], [b]))
    q(merge_case("merge/older-new-record-loses", [b], [a]))
    q(merge_case("merge/training-keys-distinct-full-vs-partial", [t], [t2]))
    q(merge_case("merge/same-timestamp-later-argument-wins", [a], [{**a, "record_id": "a9"}]))
    return dict(format=FORMAT, algorithm_version=dv.CAPABILITY_ALGORITHM_VERSION, generated_by="factory/tools/gen_golden_capability.py",
                spec_sets=SPEC_SETS, cases=cases)


def render(doc: dict) -> str:
    head = {k: v for k, v in doc.items() if k != "cases"}
    lines = [json.dumps(head, separators=(",", ":"))[:-1] + ',"cases":[']
    n = len(doc["cases"])
    lines += [json.dumps(c, separators=(",", ":")) + ("," if i < n - 1 else "") for i, c in enumerate(doc["cases"])]
    lines.append("]}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = render(build())
    if args.check:
        if not GOLDEN_PATH.exists() or GOLDEN_PATH.read_text() != text:
            print(f"golden vectors drifted: regenerate with python {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    GOLDEN_PATH.write_text(text)
    print(f"wrote {GOLDEN_PATH} ({len(text)} bytes, {len(json.loads(text)['cases'])} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
