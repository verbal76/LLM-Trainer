#!/usr/bin/env python3
"""Generate the shared golden vectors for the device-qualification algorithm.

The Python implementation (``llmtrainer.device``) is the executable spec; the Kotlin port in ``qualify/``
(the on-device authority, ADR 0011) must reproduce every expected value in the generated file.

    python factory/tools/gen_golden_device.py           # (re)write factory/tests/golden/device_qualification.v1.json
    python factory/tools/gen_golden_device.py --check   # exit 1 if the committed file drifted

This file also holds the SPEC of the pieces that are not in ``llmtrainer.device``:
  * ``catalog_candidates``     - model-config catalog -> candidate configurations;
  * ``profile_from_snapshot``  - host ``deviceSnapshotJson`` -> DeviceProfile (+ notes, + withhold decision).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from llmtrainer import device as dv  # noqa: E402
from llmtrainer.schemas import (  # noqa: E402
    CandidateConfig, DeviceProfile, Measurement, RetrievalBudget, SafetyPolicy,
)

GOLDEN_PATH = HERE.parent / "tests" / "golden" / "device_qualification.v1.json"
FORMAT = "device_qualification.v1"
MIB = dv.MIB
NOW = "2026-10-05"

# --------------------------------------------------------------------------- #
# model-config catalog (JSON shape shared with the Kotlin bundle)
# --------------------------------------------------------------------------- #


def reference_catalog() -> dict:
    """The generic size-class catalog embedded in the bundle. Weights size per quantization is explicit."""
    models = []
    for m in dv.REFERENCE_MODELS:
        d = m.model_dump(mode="json", exclude={"quality_score"})
        d["weights_mb"] = {q.name: round(m.params_b * 1e9 * q.bits_per_weight / 8 / MIB, 1) for q in dv.REFERENCE_QUANTS}
        models.append(d)
    return {
        "catalog_version": 1,
        "description": "ILLUSTRATIVE generic size classes, not specific vendor models.",
        "models": models,
        "quants": [q.model_dump(mode="json") for q in dv.REFERENCE_QUANTS],
        "runtimes": [r.model_dump(mode="json") for r in dv.REFERENCE_RUNTIMES],
        "contexts": list(dv.REFERENCE_CONTEXTS),
        "retrieval": [{"index_ram_mb": 128.0, "index_storage_mb": 256.0, "top_k": 4}],
    }


def catalog_candidates(cat: dict, retrievals: list[dict] | None = None) -> list[CandidateConfig]:
    from llmtrainer.schemas import ModelSpec, QuantSpec, RuntimeSpec

    models = [ModelSpec(**{k: v for k, v in m.items() if k != "weights_mb"}) for m in cat["models"]]
    exact = {m["model_id"]: m.get("weights_mb", {}) for m in cat["models"]}
    rets = [RetrievalBudget(**r) for r in (retrievals if retrievals is not None else cat["retrieval"])]
    cands = dv.generate_candidates(
        models, [QuantSpec(**q) for q in cat["quants"]], cat["contexts"], [RuntimeSpec(**r) for r in cat["runtimes"]], rets
    )
    out = []
    for c in cands:
        w = exact.get(c.model.model_id, {}).get(c.quant.name)
        out.append(c.model_copy(update={"weights_file_mb": w}) if w is not None else c)
    return out


# --------------------------------------------------------------------------- #
# host snapshot -> DeviceProfile (SPEC; Kotlin DeviceSnapshot mirrors this)
# --------------------------------------------------------------------------- #

# (max total RAM MB, os reserve, background reserve, host app)
RESERVE_TABLE = [(4096, 1800, 800, 400), (6144, 2200, 1000, 450), (8192, 2600, 1200, 500), (12288, 3000, 1500, 600), (None, 3500, 1800, 700)]
THERMAL_NAMES = ["NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN"]
WITHHOLD_TEXT = {
    "missing_core_fields": "Qualification withheld: the device snapshot lacks total RAM or storage figures.",
    "thermal_throttled": "Qualification withheld: device is thermally throttled (thermal status SEVERE or worse). Retry when the device has cooled.",
}


def _num(d: dict, k: str):
    v = d.get(k)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _thermal_index(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        i = int(v)
        return i if 0 <= i <= 6 else None
    if isinstance(v, str) and v.upper() in THERMAL_NAMES:
        return THERMAL_NAMES.index(v.upper())
    return None


def profile_from_snapshot(s: dict) -> tuple[DeviceProfile | None, list[str], list[str]]:
    """Return (profile or None, notes, withhold_codes). Notes are stable codes."""
    notes: list[str] = []
    withhold: list[str] = []
    total_b, free_b, tot_st_b = _num(s, "totalRamBytes"), _num(s, "freeStorageBytes"), _num(s, "totalStorageBytes")
    if total_b is None or free_b is None or tot_st_b is None or total_b <= 0:
        withhold.append("missing_core_fields")
        return None, notes, withhold
    total_mb = int(total_b) // MIB
    for bound, os_r, bg_r, host in RESERVE_TABLE:
        if bound is None or total_mb <= bound:
            break
    if s.get("isLowRamDevice") is True:
        os_r, bg_r = os_r * 6 // 5, bg_r * 6 // 5
        notes.append("low_ram_device")
    mc = _num(s, "memoryClassMb")
    if mc is not None and mc > host:
        host = int(mc)
    avail_b = _num(s, "availRamBytes")
    if s.get("lowMemory") is True:
        typical = 0
        notes.append("low_memory")
    elif avail_b is None:
        typical = total_mb // 2
        notes.append("avail_ram_unknown")
    else:
        thr = int(_num(s, "lowMemoryThresholdBytes") or 0) // MIB
        typical = max(0, int(avail_b) // MIB - thr)
    bw = _num(s, "memBandwidthGBpsEstimate")
    bw = float(bw) if bw is not None and bw > 0 else None
    t = _thermal_index(s.get("thermalStatus"))
    factor = 1.0
    if t is None:
        notes.append("thermal_unknown")
    elif t >= 3:
        withhold.append("thermal_throttled")
    elif t == 2:
        factor *= 0.8
        notes.append("thermal_moderate")
    if s.get("powerSaveMode") is True:
        factor *= 0.8
        notes.append("power_save")
    if factor < 1.0:
        bw = (bw if bw is not None else 25.0) * factor
    pct = _num(s, "batteryPct")
    if pct is not None and pct <= 15:
        notes.append("battery_low")
    maker, model = str(s.get("manufacturer", "")), str(s.get("model", ""))
    prof = DeviceProfile(
        device_id=f"{maker}-{model}".strip("-") or "unknown", name=f"{maker} {model}".strip() or "unknown", device_class="probed",
        soc=s.get("socModel") if isinstance(s.get("socModel"), str) else None, os="android", total_ram_mb=total_mb,
        typical_available_ram_mb=typical, os_reserve_mb=os_r, background_reserve_mb=bg_r, host_app_mb=host,
        storage_total_mb=int(tot_st_b) // MIB, storage_free_mb=int(free_b) // MIB, mem_bandwidth_gbps=bw, source="probed",
    )
    return prof, notes, withhold


def recommend_for_snapshot(s: dict, cands: list[CandidateConfig]):
    """Returns (profile|None, notes, withhold, pick rows). Withheld => every pick is tier none."""
    prof, notes, withhold = profile_from_snapshot(s)
    if withhold:
        text = WITHHOLD_TEXT[withhold[0]]
        picks = [
            dict(profile=k, label=dv.PROFILE_LABELS[k], config_id=None, tier="none", recommended=False, confidence=None,
                 reason=text, warnings=[], verified_fallback_config_id=None)
            for k in ("performance", "balanced", "max_quality")
        ]
        return prof, notes, withhold, picks
    rec = dv.recommend(prof, cands, created_on=NOW)
    return prof, notes, withhold, [pick_row(p) for p in rec.picks]


# --------------------------------------------------------------------------- #
# serialization of expectations
# --------------------------------------------------------------------------- #


def pick_row(p) -> dict:
    return dict(
        profile=p.profile, label=p.label, config_id=p.config.config_id if p.config else None, tier=p.tier, recommended=p.recommended,
        confidence=p.confidence, reason=p.reason, warnings=list(p.warnings), verified_fallback_config_id=p.verified_fallback_config_id,
    )


def assessed_row(a) -> dict:
    return dict(
        config_id=a.config.config_id, storage_verdict=a.storage_verdict, ram_verdict=a.ram_verdict, sustained_verdict=a.sustained_verdict,
        confidence=a.confidence, quality_score=a.quality_score, safe_to_deploy=a.safe_to_deploy, eligible=a.eligible,
        blocking_reasons=list(a.blocking_reasons), storage_headroom_mb=a.storage.headroom_mb, ram_weights_mb=a.ram.weights_mb,
        ram_runtime_mb=a.ram.runtime_mb, ram_kv_cache_mb=a.ram.kv_cache_mb, ram_required_mb=a.ram.required_model_side_mb,
        ram_budget_mb=a.ram.ram_budget_mb, ram_headroom_mb=a.ram.headroom_mb, ram_utilization=a.ram.utilization,
        ram_measured_override=a.ram.measured_override, est_tps=a.sustained.est_tokens_per_s, est_ttft_ms=a.sustained.est_ttft_ms,
        missing_measurements=list(a.sustained.missing_measurements), sustained_reasons=list(a.sustained.reasons),
    )


def profile_row(p: DeviceProfile) -> dict:
    return dict(
        total_ram_mb=p.total_ram_mb, typical_available_ram_mb=p.typical_available_ram_mb, os_reserve_mb=p.os_reserve_mb,
        background_reserve_mb=p.background_reserve_mb, host_app_mb=p.host_app_mb, storage_total_mb=p.storage_total_mb,
        storage_free_mb=p.storage_free_mb, mem_bandwidth_gbps=p.mem_bandwidth_gbps,
    )


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def dev(**kw) -> DeviceProfile:
    base = dict(device_id="d", name="d", device_class="c", os="android", typical_available_ram_mb=None)
    base.update(kw)
    return DeviceProfile(**base)


def _with(base: dict, kw: dict) -> DeviceProfile:
    return dev(**{**base, **kw})


def flagship12(**kw):
    return _with(dict(device_id="flagship-12gb", name="12GB flagship", device_class="flagship", total_ram_mb=12288, os_reserve_mb=3000,
               background_reserve_mb=1500, host_app_mb=600, storage_total_mb=256000, storage_free_mb=120000, mem_bandwidth_gbps=68), kw)


def phone8(**kw):
    return _with(dict(device_id="phone-8gb", name="8GB phone", device_class="upper-mid", total_ram_mb=8192, os_reserve_mb=2600,
               background_reserve_mb=1200, host_app_mb=500, storage_total_mb=128000, storage_free_mb=40000, mem_bandwidth_gbps=40), kw)


def phone6(**kw):
    return _with(dict(device_id="phone-6gb", name="6GB phone", device_class="mid", total_ram_mb=6144, os_reserve_mb=2200,
               background_reserve_mb=1000, host_app_mb=450, storage_total_mb=128000, storage_free_mb=50000, mem_bandwidth_gbps=30), kw)


def phone4(**kw):
    return _with(dict(device_id="phone-4gb", name="4GB phone", device_class="entry", total_ram_mb=4096, os_reserve_mb=1800,
               background_reserve_mb=800, host_app_mb=400, storage_total_mb=64000, storage_free_mb=12000, mem_bandwidth_gbps=15), kw)


def meas(cid, **over) -> Measurement:
    d = dict(config_id=cid, ttft_ms=2500, tokens_per_s=7.0, peak_ram_mb=5200, sustained_ram_mb=5100, thermal_throttle_ratio=0.85,
             ui_jank_pct=1.0, crashes=0, anrs=0, background_kills=0, sustained_minutes=15)
    d.update(over)
    return Measurement(**d)


def find_id(cands, model, quant, ctx) -> str:
    return next(c.config_id for c in cands if c.model.model_id == model and c.quant.name == quant and c.context_tokens == ctx)


def qualify_case(name, device: DeviceProfile, cands, *, catalog_ref=None, policy=None, require_retrieval=False, full=True) -> dict:
    pol = policy or SafetyPolicy()
    rec = dv.recommend(device, cands, pol, created_on=NOW, require_retrieval=require_retrieval)
    case = dict(
        name=name, kind="qualify",
        device=device.model_dump(mode="json", exclude={"content_hash", "schema_version", "kind"}),
        policy=(policy.model_dump(mode="json") if policy else {}), require_retrieval=require_retrieval,
    )
    if catalog_ref:
        case["catalog"] = catalog_ref
    else:
        case["candidates"] = [c.model_dump(mode="json") for c in cands]
    exp = dict(picks=[pick_row(p) for p in rec.picks], eligible_ids=[a.config.config_id for a in rec.assessed if a.eligible])
    if full:
        exp["assessed"] = [assessed_row(a) for a in rec.assessed]
    case["expected"] = exp
    return case


def snapshot_case(name, snap_: dict, cands) -> dict:
    prof, notes, withhold, picks = recommend_for_snapshot(snap_, cands)
    exp = dict(withheld=withhold, notes=notes, profile=profile_row(prof) if prof else None, picks=picks)
    return dict(name=name, kind="snapshot", snapshot=snap_, expected=exp)


def snap(total_gb, avail_gb, free_gb, **kw) -> dict:
    g = 1024**3
    d = dict(manufacturer="Acme", model="Phone", sdkInt=34, abis="arm64-v8a,armeabi-v7a", totalRamBytes=int(total_gb * g),
             availRamBytes=int(avail_gb * g), lowMemoryThresholdBytes=int(0.25 * g), lowMemory=False, isLowRamDevice=False,
             memoryClassMb=256, freeStorageBytes=int(free_gb * g), totalStorageBytes=int(max(free_gb * 2, 64) * g), cpuCores=8,
             socModel="X1", thermalStatus=0)
    d.update(kw)
    return {k: v for k, v in d.items() if v is not None}


def build() -> dict:
    cat = reference_catalog()
    ref = catalog_candidates(cat)
    ref_ref = {"name": "reference"}
    no_ret = [dict(index_ram_mb=0.0, index_storage_mb=0.0, top_k=0)]
    small_ret = [dict(index_ram_mb=8.0, index_storage_mb=16.0, top_k=2)]
    cases = []
    q = cases.append

    q(qualify_case("flagship12/estimates-only", flagship12(), ref, catalog_ref=ref_ref))
    q(qualify_case("phone8/estimates-only", phone8(), ref, catalog_ref=ref_ref))
    q(qualify_case("phone6/estimates-only", phone6(), ref, catalog_ref=ref_ref))
    q(qualify_case("phone4/with-retrieval-nothing-fits", phone4(), ref, catalog_ref=ref_ref))
    q(qualify_case("phone4/no-retrieval", phone4(), catalog_candidates(cat, no_ret), catalog_ref={"name": "reference", "retrieval": no_ret}))
    q(qualify_case("phone4/small-retrieval", phone4(), catalog_candidates(cat, small_ret), catalog_ref={"name": "reference", "retrieval": small_ret}))
    q(qualify_case("flagship12/typical-available-5000", flagship12(typical_available_ram_mb=5000), ref, catalog_ref=ref_ref, full=False))
    q(qualify_case("flagship12/low-memory-state", flagship12(typical_available_ram_mb=0), ref, catalog_ref=ref_ref, full=False))
    q(qualify_case("flagship12/tiny-free-storage", flagship12(storage_free_mb=3000), ref, catalog_ref=ref_ref, full=False))
    q(qualify_case("flagship12/storage-reserve-edge", flagship12(storage_free_mb=30000), ref, catalog_ref=ref_ref, full=False))
    q(qualify_case("flagship12/slow-bandwidth", flagship12(mem_bandwidth_gbps=5), ref, catalog_ref=ref_ref, full=False))
    q(qualify_case("flagship12/unknown-bandwidth", flagship12(mem_bandwidth_gbps=None), ref, catalog_ref=ref_ref, full=False))
    q(qualify_case("flagship12/thermal-derated-bandwidth", flagship12(mem_bandwidth_gbps=68 * 0.8), ref, catalog_ref=ref_ref, full=False))
    q(qualify_case("flagship12/tight-policy", flagship12(), ref, catalog_ref=ref_ref,
                   policy=SafetyPolicy(safety_reserve_frac=0.2, balanced_max_utilization=0.8), full=False))
    q(qualify_case("flagship12/loose-policy", flagship12(), ref, catalog_ref=ref_ref,
                   policy=SafetyPolicy(safety_reserve_frac=0.0, safety_reserve_min_mb=0, max_quant_penalty=5.0), full=False))
    mixed = ref + dv.generate_candidates(dv.REFERENCE_MODELS, dv.REFERENCE_QUANTS, [2048], dv.REFERENCE_RUNTIMES)
    q(qualify_case("flagship12/require-retrieval", flagship12(), mixed, require_retrieval=True, full=False))
    q(qualify_case("negative-budget-2gb", dev(total_ram_mb=2048, os_reserve_mb=1500, background_reserve_mb=800, host_app_mb=300,
                                               storage_total_mb=64000, storage_free_mb=30000), ref, catalog_ref=ref_ref, full=False))

    # measured behavior (explicit candidate lists keep these cases self-describing)
    small = [c for c in ref if c.model.model_id in ("ref-3b", "ref-7b") and c.quant.name == "Q4_K_M"]
    id3, id7 = find_id(ref, "ref-3b", "Q4_K_M", 4096), find_id(ref, "ref-7b", "Q4_K_M", 4096)
    q(qualify_case("measured/7b-passes-recommended", flagship12(measurements=[meas(id7, peak_ram_mb=5600, sustained_ram_mb=5500)]), small))
    q(qualify_case("measured/only-3b-verified-fallback", flagship12(measurements=[meas(id3, peak_ram_mb=3000, sustained_ram_mb=2900, tokens_per_s=12)]),
                   ref, catalog_ref=ref_ref, full=False))
    for label, over in [("low-tps", dict(tokens_per_s=1.0)), ("slow-ttft", dict(ttft_ms=30000)), ("throttle", dict(thermal_throttle_ratio=0.4)),
                        ("jank", dict(ui_jank_pct=20)), ("crash", dict(crashes=1)), ("anr", dict(anrs=2)), ("bg-kill", dict(background_kills=1)),
                        ("ram-over", dict(peak_ram_mb=99999)), ("short-run", dict(sustained_minutes=2))]:
        q(qualify_case(f"measured/7b-fails-{label}", flagship12(measurements=[meas(id7, **over)]), small))
    q(qualify_case("measured/partial", flagship12(measurements=[Measurement(config_id=id7, tokens_per_s=8.0, peak_ram_mb=5000)]), small))
    q(qualify_case("measured/partial-with-failure", flagship12(measurements=[Measurement(config_id=id7, crashes=3)]), small))
    q(qualify_case("measured/for-other-config-ignored", flagship12(measurements=[meas("some-other-config")]), small))
    q(qualify_case("measured/pessimistic-estimate-overridden",
                   phone8(measurements=[meas(id7, peak_ram_mb=3200, sustained_ram_mb=3100, tokens_per_s=6.0)]), small))

    # host snapshots
    q(snapshot_case("snapshot/12gb-flagship", snap(11.3, 6.0, 120, memBandwidthGBpsEstimate=68), ref))
    q(snapshot_case("snapshot/12gb-roomy-7b-class", snap(11.3, 9.0, 120, memBandwidthGBpsEstimate=68), ref))
    q(snapshot_case("snapshot/12gb-no-bandwidth", snap(11.3, 6.0, 120), ref))
    q(snapshot_case("snapshot/8gb", snap(7.4, 3.5, 40), ref))
    q(snapshot_case("snapshot/6gb", snap(5.5, 2.5, 50), ref))
    q(snapshot_case("snapshot/4gb", snap(3.7, 1.4, 12), ref))
    q(snapshot_case("snapshot/low-memory", snap(11.3, 0.2, 120, lowMemory=True), ref))
    q(snapshot_case("snapshot/avail-below-threshold", snap(7.4, 0.2, 40), ref))
    q(snapshot_case("snapshot/thermal-moderate", snap(11.3, 6.0, 120, thermalStatus=2, memBandwidthGBpsEstimate=68), ref))
    q(snapshot_case("snapshot/thermal-severe-withheld", snap(11.3, 6.0, 120, thermalStatus=3), ref))
    q(snapshot_case("snapshot/thermal-name-critical-withheld", snap(11.3, 6.0, 120, thermalStatus="CRITICAL"), ref))
    q(snapshot_case("snapshot/thermal-missing", snap(11.3, 6.0, 120, thermalStatus=None), ref))
    q(snapshot_case("snapshot/tiny-free-storage", snap(11.3, 6.0, 2.5), ref))
    q(snapshot_case("snapshot/low-ram-device", snap(3.7, 1.4, 12, isLowRamDevice=True), ref))
    q(snapshot_case("snapshot/optional-extras", snap(11.3, 6.0, 120, largeMemoryClassMb=512, vulkanLevel=1, vulkanVersion=4198400,
                                                     hasNeon=True, cpuFeatures="neon,fp16", batteryPct=10, powerSaveMode=True,
                                                     memBandwidthGBpsEstimate=51.2), ref))
    q(snapshot_case("snapshot/missing-avail", snap(7.4, 3.5, 40, availRamBytes=None, lowMemoryThresholdBytes=None), ref))
    q(snapshot_case("snapshot/big-memory-class", snap(7.4, 3.5, 40, memoryClassMb=768), ref))
    q(snapshot_case("snapshot/missing-total-ram-withheld", snap(7.4, 3.5, 40, totalRamBytes=None), ref))
    q(snapshot_case("snapshot/missing-storage-withheld", snap(7.4, 3.5, 40, freeStorageBytes=None), ref))
    return dict(format=FORMAT, algorithm_version=dv.ALGORITHM_VERSION, generated_by="factory/tools/gen_golden_device.py",
                reference_catalog=cat, cases=cases)


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
