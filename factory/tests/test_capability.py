"""Device profiler v2: per-artifact capabilities, training memory model, measurement merge, golden-vector drift."""

import importlib.util
import json
from pathlib import Path

import pytest

from llmtrainer import device as dv
from llmtrainer.device import ArtifactSpec, CapabilityPolicy, DeviceState, MeasurementRecord, classify_artifact, choose_models, merge_records
from llmtrainer.schemas import SafetyPolicy

TOOLS = Path(__file__).resolve().parent.parent / "tools"
spec = importlib.util.spec_from_file_location("gen_golden_capability", TOOLS / "gen_golden_capability.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
base = gen.base
MIB = dv.MIB


def S(artifact_id, **over) -> ArtifactSpec:
    rows = {s["artifact_id"]: s for s in gen.illustrative_specs()}
    d = dict(rows[artifact_id])
    d.update(over)
    return ArtifactSpec(**d)


def pixel_profile():
    prof, _, _ = base.profile_from_snapshot(gen.pixel())
    return prof


def caps(device, artifact_id, **kw):
    return classify_artifact(device, S(artifact_id, **kw.pop("spec", {})), **kw)


# ----------------------------- training memory model ------------------------ #


def test_training_memory_is_f32_weights_plus_12_bytes_per_trainable_param_plus_activations():
    s = S("smollm2-135m-f16")
    cap, safety = CapabilityPolicy(), SafetyPolicy()
    full, trainable = dv.training_ram_mb(s, None, cap, safety)
    assert trainable == s.params
    weights = s.params * 4 / MIB
    state = s.params * 12 / MIB
    act = (s.layers * 4 * cap.train_ctx_tokens * (16 * s.embd + 2 * s.ffn) + 2 * 4 * cap.train_ctx_tokens * s.vocab) / MIB
    expect = (weights + state) * safety.weights_inflation + (act + cap.train_overhead_mb) * safety.estimate_inflation
    assert full == pytest.approx(expect)
    # tuning only the last k layers costs far less state memory
    part, tr = dv.training_ram_mb(s, 4, cap, safety)
    assert tr == pytest.approx(s.params / s.layers * 4) and part < full
    assert part > weights  # F32 weights are always resident


def test_training_memory_is_strictly_larger_than_inference_for_the_same_model():
    dev = pixel_profile()
    c = caps(dev, "smollm2-135m-f16")
    assert c.full is not None and c.infer_ram_mb is not None and c.full.ram_mb > 2.5 * c.infer_ram_mb


def test_activation_fallback_when_hidden_size_is_unknown():
    s = S("smollm2-135m-f16", embd=None, ffn=None, vocab=None)
    ram, _ = dv.training_ram_mb(s, None, CapabilityPolicy(), SafetyPolicy())
    w = s.params * 4 / MIB
    assert ram == pytest.approx((w + s.params * 12 / MIB) * 1.05 + (0.25 * w + 300.0) * 1.25)


# ----------------------------- inference != training ------------------------ #


def test_inference_feasibility_never_implies_training_feasibility():
    dev = pixel_profile()
    for aid in ("qwen3-4b-q4_k_m", "qwen3-1.7b-q8_0", "smollm2-360m-q8_0"):
        c = caps(dev, aid)
        assert c.can_infer and not c.can_specialize_full and not c.can_specialize_partial and c.external_compute_required
        assert "training_needs_full_precision_artifact" in c.reasons
    c = caps(dev, "qwen3-1.7b-f16")  # 1.7B can run inference, but its class is external_only
    assert c.can_infer and not c.can_specialize_partial and c.external_compute_required and "training_class_external_only" in c.reasons


def test_quantized_artifact_points_at_its_full_precision_source():
    c = caps(pixel_profile(), "smollm2-135m-q8_0")
    assert c.specialize_via_artifact_id == "smollm2-135m-f16" and c.full is None and c.partial is None


def test_full_vs_partial_depends_on_class_and_ram():
    dev = pixel_profile()
    assert caps(dev, "smollm2-135m-f16").can_specialize_full
    c = caps(dev, "qwen3-0.6b-f16")  # local_partial class: never offers full tuning
    assert not c.can_specialize_full and c.can_specialize_partial and 2 <= c.trainable_last_layers < 28 and c.full is None
    c = caps(dev, "smollm2-360m-f16")  # local_full class but the RAM envelope only allows partial
    assert not c.can_specialize_full and c.can_specialize_partial and c.full is not None and not c.full.fits


def test_small_phone_cannot_specialize_and_needs_external_compute():
    prof, _, _ = base.profile_from_snapshot(base.snap(3.7, 1.4, 12))
    c = classify_artifact(prof, S("smollm2-135m-f16"))
    assert not c.can_load and not c.can_specialize_full and not c.can_specialize_partial and c.external_compute_required
    assert "training_ram_insufficient" in c.reasons


def test_partial_layer_count_is_monotone_in_ram():
    ks = []
    for gb in (6, 8, 12, 16):
        prof, _, _ = base.profile_from_snapshot(base.snap(gb, gb / 2, 100))
        c = classify_artifact(prof, S("qwen3-0.6b-f16"))
        ks.append(c.trainable_last_layers or 0)
    assert ks == sorted(ks) and ks[-1] > ks[0]


# ----------------------------- download / storage --------------------------- #


def test_download_needs_storage_after_reserve_and_a_downloadable_artifact():
    prof, _, _ = base.profile_from_snapshot(gen.pixel(freeStorageBytes=int(3.0 * 1024**3)))
    c = classify_artifact(prof, S("qwen3-4b-q8_0"))
    assert not c.can_download and "storage_insufficient" in c.reasons
    c = classify_artifact(pixel_profile(), S("qwen3-4b-q8_0", downloadable=False))
    assert not c.can_download and "not_downloadable" in c.reasons


def test_unrefreshed_artifacts_are_previews_never_recommended_and_low_confidence():
    rep = choose_models(pixel_profile(), [ArtifactSpec(**s) for s in gen.SPEC_SETS["unrefreshed"]])
    assert all(c.can_download is False for c in rep.capabilities)
    for ch in rep.choices:
        assert ch.tier in ("preview", "none") and not ch.downloadable
    for c in rep.capabilities:
        assert c.confidence_infer == "low" and c.confidence_train == "low"
        assert {"geometry_estimated", "size_estimated", "params_nominal"} <= set(c.reasons)


def test_downloadable_artifacts_beat_unrefreshed_ones_in_choices():
    specs = [S("qwen3-4b-q4_k_m"), S("qwen3-8b-q4_k_m", downloadable=False, size_bytes=None)]
    rep = choose_models(pixel_profile(), specs, SafetyPolicy(min_tps=1.0))
    best = next(c for c in rep.choices if c.choice == "best_quality")
    assert best.artifact_id == "qwen3-4b-q4_k_m" and best.tier == "provisional" and best.downloadable


# ----------------------------- choices -------------------------------------- #


def test_three_choices_plus_specialize_are_distinct_questions():
    rep = choose_models(pixel_profile(), [ArtifactSpec(**s) for s in gen.illustrative_specs()])
    by = {c.choice: c for c in rep.choices}
    assert set(by) == {"fastest", "balanced", "best_quality", "best_specialize"}
    caps_by = {c.artifact_id: c for c in rep.capabilities}
    assert caps_by[by["fastest"].artifact_id].est_tokens_per_s >= caps_by[by["balanced"].artifact_id].est_tokens_per_s
    assert caps_by[by["best_quality"].artifact_id].quality_score >= caps_by[by["balanced"].artifact_id].quality_score
    assert by["best_specialize"].artifact_id in ("smollm2-135m-f16", "smollm2-360m-f16", "qwen3-0.6b-f16")
    assert caps_by[by["best_specialize"].artifact_id].quality_score < caps_by[by["best_quality"].artifact_id].quality_score  # specialize != best model
    assert all(c.tier != "recommended" for c in rep.choices)  # estimates are never "recommended"


def test_disallowed_license_is_never_offered():
    specs = [S("qwen3-4b-q4_k_m", license_state="DISALLOWED"), S("smollm2-135m-q8_0")]
    rep = choose_models(pixel_profile(), specs)
    assert all(c.artifact_id != "qwen3-4b-q4_k_m" for c in rep.choices)


def test_runtime_unsupported_blocks_inference():
    c = caps(pixel_profile(), "qwen3-4b-q4_k_m", spec=dict(runtime_supported="no"))
    assert not c.can_infer and "runtime_unsupported" in c.reasons


def test_nothing_fits_on_a_tiny_device():
    prof, _, _ = base.profile_from_snapshot(base.snap(3.7, 1.4, 12))
    rep = choose_models(prof, [ArtifactSpec(**s) for s in gen.illustrative_specs()])
    assert all(c.artifact_id is None and c.tier == "none" for c in rep.choices)


# ----------------------------- Pixel 10 Pro XL regression ------------------------------ #


def test_pixel_10_pro_xl_15gb_with_2_1gb_available_stays_sensible():
    prof = pixel_profile()
    assert prof.total_ram_mb == 15564 and prof.typical_available_ram_mb is None  # transient availMem must not cap a capable phone
    rep = choose_models(prof, [ArtifactSpec(**s) for s in gen.illustrative_specs()])
    by = {c.choice: c for c in rep.choices}
    assert all(c.artifact_id for c in by.values())
    c4 = next(c for c in rep.capabilities if c.artifact_id == "qwen3-4b-q4_k_m")
    assert c4.can_infer and c4.infer_utilization < 0.6  # a 4B model is comfortable on a 15 GB phone
    assert by["best_quality"].artifact_id.startswith("qwen3-4b")
    assert by["best_specialize"].artifact_id in ("smollm2-135m-f16", "smollm2-360m-f16", "qwen3-0.6b-f16")
    low = base.profile_from_snapshot(gen.pixel(avail=0.2, lowMemory=True))[0]
    assert all(not classify_artifact(low, S("qwen3-4b-q4_k_m")).can_load for _ in [0])  # real low-memory state is still respected


# ----------------------------- training conditions -------------------------- #


def conds(state):
    c = classify_artifact(pixel_profile(), S("smollm2-135m-f16"), state=state)
    return {x.name: x.status for x in c.training_conditions}, c.ready_to_train_now


def test_conditions_unknown_unmet_met_and_readiness():
    got, ready = conds(None)
    assert got["thermal"] == "unknown" and got["charging"] == "unknown" and not ready  # missing facts never count as met
    got, ready = conds(DeviceState(thermal_status=0, battery_pct=80, charging=True, power_save=False))
    assert set(got.values()) == {"met"} and ready
    got, ready = conds(DeviceState(thermal_status=3, battery_pct=80, charging=True, power_save=False))
    assert got["thermal"] == "unmet" and not ready
    got, ready = conds(DeviceState(thermal_status=0, battery_pct=20, charging=False, power_save=False))
    assert got["battery"] == "unmet" and got["charging"] == "unmet" and not ready
    got, ready = conds(DeviceState(thermal_status=0, battery_pct=90, charging=False, power_save=False))
    assert got["charging"] == "unmet" and not ready  # training requires charging even with a full battery
    got, ready = conds(DeviceState(thermal_status=0, charging=True, power_save=True))
    assert got["not_power_save"] == "unmet" and not ready


def test_snapshot_state_extraction():
    st = gen.state_from_snapshot(gen.pixel(charging=True, batteryPct=55, powerSaveMode=False, thermalStatus="LIGHT"))
    assert st == DeviceState(thermal_status=1, battery_pct=55.0, charging=True, power_save=False)
    assert gen.state_from_snapshot({}) == DeviceState()


# ----------------------------- measurement records -------------------------- #


PID = "Google-Pixel 10 Pro XL"


def R(rid, artifact, kind, **kw) -> MeasurementRecord:
    return MeasurementRecord(**gen.rec(rid, artifact, kind, **kw))


FULL = dict(context_tokens=1024, ttft_ms=2500, tokens_per_s=7.0, peak_ram_mb=5000, sustained_ram_mb=4900, thermal_throttle_ratio=0.85,
            ui_jank_pct=1.0, crashes=0, anrs=0, background_kills=0, sustained_minutes=15)


def test_complete_measurement_replaces_estimate_and_raises_confidence_to_high():
    dev = pixel_profile()
    est = classify_artifact(dev, S("qwen3-8b-q4_k_m"), safety=SafetyPolicy(min_tps=1.0))
    assert est.confidence_infer == "low"
    rec = R("r", "qwen3-8b-q4_k_m", "inference", **{**FULL, "context_tokens": est.infer_context})
    got = classify_artifact(dev, S("qwen3-8b-q4_k_m"), safety=SafetyPolicy(min_tps=1.0), records=[rec])
    assert got.confidence_infer == "high" and got.est_tokens_per_s == 7.0 and got.can_infer


def test_partial_measurement_is_only_medium():
    got = classify_artifact(pixel_profile(), S("qwen3-4b-q4_k_m"), records=[R("r", "qwen3-4b-q4_k_m", "inference", context_tokens=2048, tokens_per_s=8.0, peak_ram_mb=3500)])
    assert got.confidence_infer == "medium"


def test_measurement_for_another_context_does_not_make_the_choice_high():
    got = classify_artifact(pixel_profile(), S("qwen3-4b-q4_k_m"), records=[R("r", "qwen3-4b-q4_k_m", "inference", **{**FULL, "context_tokens": 512})])
    assert got.confidence_infer == "medium"


def test_failed_measurement_overrides_a_rosy_estimate():
    dev = pixel_profile()
    assert classify_artifact(dev, S("qwen3-4b-q4_k_m")).can_infer
    ctxs = (512, 1024, 2048, 4096, 8192)
    recs = [R(f"r{c}", "qwen3-4b-q4_k_m", "inference", **{**FULL, "context_tokens": c, "crashes": 1}) for c in ctxs]
    got = classify_artifact(dev, S("qwen3-4b-q4_k_m"), records=recs)
    assert not got.can_infer
    slow = classify_artifact(dev, S("qwen3-4b-q4_k_m"), records=[R(f"s{c}", "qwen3-4b-q4_k_m", "inference", **{**FULL, "context_tokens": c, "tokens_per_s": 0.5}) for c in ctxs])
    assert not slow.can_infer


def test_records_for_other_devices_or_other_files_are_ignored_with_a_reason():
    dev = pixel_profile()
    s = S("qwen3-8b-q4_k_m")
    other = MeasurementRecord(**{**gen.rec("o", s.artifact_id, "inference", **FULL), "device_id": "elsewhere"})
    wrong = MeasurementRecord(**{**gen.rec("w", s.artifact_id, "inference", **FULL), "artifact_sha256": "sha256:" + "cd" * 32})
    got = classify_artifact(dev, s, records=[other, wrong])
    assert got.confidence_infer == "low" and sorted(got.ignored_measurements) == ["o:other_device", "w:artifact_hash_mismatch"]


def test_training_oom_caps_the_layer_count_and_success_extends_it():
    dev = pixel_profile()
    base_k = classify_artifact(dev, S("qwen3-0.6b-f16")).trainable_last_layers
    assert base_k == 13
    oom = classify_artifact(dev, S("qwen3-0.6b-f16"), records=[R("t", "qwen3-0.6b-f16", "training", trainable_last_layers=8, train_peak_ram_mb=9999, train_oom=True, train_completed=False)])
    assert oom.trainable_last_layers == 7 and oom.confidence_train == "medium"
    ok = classify_artifact(dev, S("qwen3-0.6b-f16"), records=[R("t", "qwen3-0.6b-f16", "training", trainable_last_layers=20, train_peak_ram_mb=4800,
                                                               train_tokens_per_s=40.0, checkpoint_size_mb=1100, train_completed=True, train_oom=False)])
    assert ok.trainable_last_layers == 20 and ok.confidence_train == "high" and ok.partial.measured and ok.partial.est_tokens_per_s == 40.0
    full_oom = classify_artifact(dev, S("smollm2-135m-f16"), records=[R("t", "smollm2-135m-f16", "training", train_oom=True, train_completed=False)])
    assert not full_oom.can_specialize_full and full_oom.can_specialize_partial and full_oom.trainable_last_layers == 29


def test_load_record_overrides_the_load_estimate():
    dev = pixel_profile()
    assert classify_artifact(dev, S("qwen3-8b-q4_k_m")).can_load
    got = classify_artifact(dev, S("qwen3-8b-q4_k_m"), records=[R("l", "qwen3-8b-q4_k_m", "load", load_time_ms=9000, load_peak_ram_mb=99999)])
    assert not got.can_load and got.confidence_infer == "medium"


def test_merge_latest_wins_per_key_and_keys_are_independent():
    a1 = R("a1", "x", "inference", context_tokens=2048, recorded_at="2026-10-01T00:00:00Z")
    a2 = R("a2", "x", "inference", context_tokens=2048, recorded_at="2026-10-02T00:00:00Z")
    a3 = R("a3", "x", "inference", context_tokens=4096, recorded_at="2026-09-01T00:00:00Z")
    assert [r.record_id for r in merge_records([a1, a3], [a2])] == ["a2", "a3"]
    assert [r.record_id for r in merge_records([a2], [a1])] == ["a2"]  # an older record never replaces a newer one
    t1 = R("t1", "x", "training", trainable_last_layers=8)
    t2 = R("t2", "x", "training")
    assert {r.record_id for r in merge_records([t1], [t2])} == {"t1", "t2"}  # full and partial runs are different keys


def test_measurement_record_schema_roundtrip():
    r = R("r", "x", "training", trainable_last_layers=4, train_peak_ram_mb=1.5, train_completed=True)
    assert MeasurementRecord.model_validate_json(r.model_dump_json()) == r
    with pytest.raises(Exception):
        MeasurementRecord(**{**gen.rec("r", "x", "inference"), "bogus": 1})


# ----------------------------- golden vectors ------------------------------- #


def test_golden_capability_vectors_have_not_drifted():
    assert gen.GOLDEN_PATH.read_text() == gen.render(gen.build()), (
        "device_capability.v1.json is out of date vs llmtrainer.device: run `python factory/tools/gen_golden_capability.py` "
        "and port any behavior change to qualify/ (Kotlin Capability.kt)")


def test_golden_capability_vectors_cover_the_required_situations():
    doc = json.loads(gen.GOLDEN_PATH.read_text())
    names = {c["name"] for c in doc["cases"]}
    for needed in ("pixel10/regression-15gb-2.1gb-available", "pixel10/unrefreshed-catalog", "measured/training-oom-caps-layers",
                   "measured/8b-inference-passes", "merge/latest-wins-per-key", "pixel10/charging-cool-ready", "phone4/snapshot"):
        assert needed in names
    assert len(doc["cases"]) >= 30
    for c in doc["cases"]:
        if c["kind"] == "merge":
            continue
        for ch in c["expected"]["choices"]:
            if not c["name"].startswith("measured/"):
                assert ch["tier"] != "recommended", c["name"]
