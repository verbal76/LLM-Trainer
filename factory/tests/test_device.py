import math

import pytest
from conftest import flagship12, make_device, phone4, phone8

from llmtrainer import device as dv
from llmtrainer.schemas import (
    CandidateConfig, Measurement, ModelSpec, QuantSpec, RetrievalBudget, RuntimeSpec, SafetyPolicy,
)

NOW = "2026-10-05"
M7 = ModelSpec(model_id="m7", params_b=7.0, layers=32, kv_heads=8, head_dim=128)
Q4 = QuantSpec(name="Q4", bits_per_weight=4.5, quality_penalty=1.8)
RT = RuntimeSpec(name="rt", overhead_mb=300, scratch_mb_per_1k_ctx=16)


def cand(model=M7, quant=Q4, ctx=4096, rt=RT, ret=None, **kw) -> CandidateConfig:
    ret = ret or RetrievalBudget(index_ram_mb=128, index_storage_mb=256, top_k=4)
    return CandidateConfig(config_id=dv.make_config_id(model, quant, ctx, rt, ret), model=model, quant=quant, context_tokens=ctx, runtime=rt, retrieval=ret, **kw)


def full_measurement(cid, **over):
    d = dict(config_id=cid, ttft_ms=2500, tokens_per_s=7.0, peak_ram_mb=5200, sustained_ram_mb=5100, thermal_throttle_ratio=0.85,
             ui_jank_pct=1.0, crashes=0, anrs=0, background_kills=0, sustained_minutes=15)
    d.update(over)
    return Measurement(**d)


# ---------------------------- component math ---------------------------------

def test_kv_cache_formula():
    # 2 (K,V) * layers * kv_heads * head_dim * ctx * 2 bytes
    assert dv.kv_cache_mb(32, 8, 128, 4096, "f16") == pytest.approx(2 * 32 * 8 * 128 * 4096 * 2 / 2**20)
    assert dv.kv_cache_mb(32, 8, 128, 4096, "f16") == pytest.approx(512)
    assert dv.kv_cache_mb(32, 8, 128, 4096, "q8_0") < 512 * 0.6
    assert dv.kv_cache_mb(32, 8, 128, 8192) == pytest.approx(2 * dv.kv_cache_mb(32, 8, 128, 4096))


def test_weights_from_params_or_exact_file():
    c = cand()
    assert dv.weights_mb(c) == pytest.approx(7e9 * 4.5 / 8 / 2**20)
    assert dv.weights_mb(cand(weights_file_mb=4000.0)) == 4000.0


def test_runtime_scratch_scales_with_context():
    assert dv.runtime_mb(cand(ctx=2048)) < dv.runtime_mb(cand(ctx=8192))


def test_ram_budget_components():
    d = flagship12()
    pol = SafetyPolicy()
    budget, reserve = dv.ram_budget_mb(d, pol)
    assert reserve == pytest.approx(0.08 * 12288)
    assert budget == pytest.approx(12288 - 3000 - 1500 - 600 - reserve)


def test_typical_available_ram_caps_budget():
    d = flagship12(); d2 = flagship12(typical_available_ram_mb=5000)
    b1, _ = dv.ram_budget_mb(d, SafetyPolicy())
    b2, _ = dv.ram_budget_mb(d2, SafetyPolicy())
    assert b2 < b1 and b2 == pytest.approx(5000 - 600 - dv.safety_reserve_mb(d2, SafetyPolicy()))


def test_safety_reserve_has_floor():
    tiny = make_device(total_ram_mb=1000, os_reserve_mb=0, background_reserve_mb=0, host_app_mb=0, storage_total_mb=1, storage_free_mb=1)
    assert dv.safety_reserve_mb(tiny, SafetyPolicy()) == 256


def test_safety_reserve_is_configurable():
    d = flagship12()
    loose, tight = SafetyPolicy(safety_reserve_frac=0.0, safety_reserve_min_mb=0), SafetyPolicy(safety_reserve_frac=0.2)
    assert dv.ram_budget_mb(d, loose)[0] > dv.ram_budget_mb(d, tight)[0]


# ---------------------------- three distinct verdicts ------------------------

def test_storage_fit_ram_fit_and_sustained_are_independent():
    # fits storage, fails RAM: big model on a small-RAM, big-storage device
    d = make_device(total_ram_mb=4096, os_reserve_mb=1800, background_reserve_mb=800, host_app_mb=400, storage_total_mb=512000,
                    storage_free_mb=400000, mem_bandwidth_gbps=60)
    a = dv.assess(d, cand())
    assert a.storage_verdict == "fit" and a.ram_verdict == "no_fit" and a.sustained_verdict == "no_fit"
    assert not a.eligible and not a.safe_to_deploy
    # fits RAM, fails storage
    d2 = flagship12().model_copy(update={"storage_free_mb": 3000})
    a2 = dv.assess(d2, cand())
    assert a2.ram_verdict == "fit" and a2.storage_verdict == "no_fit" and not a2.eligible
    assert any(r.startswith("storage") for r in a2.blocking_reasons)


def test_model_that_merely_loads_is_not_a_fit():
    """Weights alone fit in physical RAM, but not once OS/background/host/safety reserves are honoured."""
    d = phone8()
    c = cand(ret=RetrievalBudget())
    assert dv.weights_mb(c) < d.total_ram_mb  # it would 'load'
    a = dv.assess(d, c)
    assert a.ram_verdict == "no_fit" and not a.eligible


def test_storage_reserve_is_enforced():
    d = flagship12().model_copy(update={"storage_free_mb": 4000 + 3000})  # reserve = max(2048, 10% of 256000)=25600 -> no fit
    assert dv.assess(d, cand()).storage_verdict == "no_fit"
    d2 = flagship12().model_copy(update={"storage_free_mb": 60000})
    assert dv.assess(d2, cand()).storage_verdict == "fit"


def test_estimates_are_inflated_and_marked_estimated():
    d = flagship12()
    a = dv.assess(d, cand())
    raw = a.ram.weights_mb + a.ram.runtime_mb + a.ram.kv_cache_mb + a.ram.retrieval_mb
    assert a.ram.required_model_side_mb > raw
    assert not a.ram.measured_override and a.confidence == "estimated"
    assert a.sustained_verdict == "unverified" and not a.safe_to_deploy and a.eligible


def test_never_recommended_on_estimates_alone():
    rec = dv.recommend(flagship12(), dv.reference_candidates(), created_on=NOW)
    for p in rec.picks:
        assert p.tier == "provisional" and p.recommended is False
        assert any("estimates" in w.lower() for w in p.warnings)
        assert p.confidence == "estimated"


# ---------------------------- measurements -----------------------------------

def test_full_passing_measurements_yield_fit_and_recommended():
    c = cand()
    d = flagship12(measurements=[full_measurement(c.config_id)])
    a = dv.assess(d, c)
    assert a.sustained_verdict == "fit" and a.confidence == "measured" and a.safe_to_deploy
    assert a.ram.measured_override and a.ram.required_model_side_mb == 5200
    rec = dv.recommend(d, [c], created_on=NOW)
    assert all(p.tier == "recommended" and p.recommended for p in rec.picks if p.config)


@pytest.mark.parametrize(
    "override,text",
    [
        (dict(tokens_per_s=1.0), "tok/s"),
        (dict(ttft_ms=30000), "TTFT"),
        (dict(thermal_throttle_ratio=0.4), "throttle"),
        (dict(ui_jank_pct=20), "jank"),
        (dict(crashes=1), "crashes"),
        (dict(anrs=2), "ANRs"),
        (dict(background_kills=1), "background-process kills"),
        (dict(peak_ram_mb=99999), "RAM"),
    ],
)
def test_any_failed_measurement_blocks(override, text):
    c = cand()
    d = flagship12(measurements=[full_measurement(c.config_id, **override)])
    a = dv.assess(d, c)
    assert not a.eligible and not a.safe_to_deploy
    assert any(text in r for r in a.blocking_reasons), a.blocking_reasons


def test_measured_overrides_a_pessimistic_estimate():
    c = cand(ctx=8192, model=ModelSpec(model_id="m8", params_b=8.0, layers=32, kv_heads=8, head_dim=128))
    d = flagship12()
    est = dv.assess(d, c)
    assert est.ram_verdict == "no_fit"
    # real measurement shows it actually runs within the envelope
    d2 = flagship12(measurements=[full_measurement(c.config_id, peak_ram_mb=5600, sustained_ram_mb=5500)])
    assert dv.assess(d2, c).ram_verdict == "fit"


def test_partial_measurements_are_not_enough():
    c = cand()
    d = flagship12(measurements=[Measurement(config_id=c.config_id, tokens_per_s=8.0, peak_ram_mb=5000)])
    a = dv.assess(d, c)
    assert a.confidence == "partial" and a.sustained_verdict == "unverified" and not a.safe_to_deploy
    assert {"thermal_throttle_ratio", "ui_jank_pct", "crashes"} <= set(a.sustained.missing_measurements)


def test_partial_measurement_with_a_failure_still_blocks():
    c = cand()
    d = flagship12(measurements=[Measurement(config_id=c.config_id, crashes=3)])
    assert dv.assess(d, c).sustained_verdict == "no_fit"


def test_short_benchmark_is_inconclusive_not_a_pass():
    c = cand()
    d = flagship12(measurements=[full_measurement(c.config_id, sustained_minutes=2)])
    a = dv.assess(d, c)
    assert a.sustained_verdict == "unverified" and not a.safe_to_deploy


def test_measurement_for_another_config_is_ignored():
    c = cand()
    d = flagship12(measurements=[full_measurement("some-other-config")])
    assert dv.assess(d, c).confidence == "estimated"


def test_slow_estimate_blocks_even_without_measurement():
    d = flagship12().model_copy(update={"mem_bandwidth_gbps": 5})
    a = dv.assess(d, cand())
    assert a.sustained_verdict == "no_fit" and any("estimated" in r for r in a.blocking_reasons)


def test_unknown_bandwidth_uses_conservative_fallback():
    d = flagship12().model_copy(update={"mem_bandwidth_gbps": None})
    assert dv.estimate_decode_tps(d, cand(), SafetyPolicy()) < dv.estimate_decode_tps(flagship12(), cand(), SafetyPolicy())


# ---------------------------- recommendations on reference devices -----------

def picks(dev, **kw):
    rec = dv.recommend(dev, dv.reference_candidates(**kw.pop("ret", {})), created_on=NOW, **kw)
    return {p.profile: p for p in rec.picks}


def test_12gb_flagship_runs_7b_class_not_a_tiny_model():
    p = picks(flagship12())
    bal = p["balanced"].config
    assert bal.model.params_b >= 7 and bal.quant.name == "Q4_K_M"
    assert p["max_quality"].config.model.params_b >= 8
    assert p["performance"].config.model.params_b < bal.model.params_b
    for pick in p.values():
        assert pick.config.retrieval.index_ram_mb == 128  # complete config includes retrieval budget


def test_no_needless_downgrade_when_7b_fits():
    d = flagship12()
    rec = dv.recommend(d, dv.reference_candidates(), created_on=NOW)
    best_fit_params = max(a.config.model.params_b for a in rec.assessed if a.eligible and a.ram.utilization <= 0.92)
    bal = next(x for x in rec.picks if x.profile == "balanced")
    assert bal.config.model.params_b == best_fit_params


def test_8gb_phone_gets_mid_size_not_7b():
    p = picks(phone8())
    assert 2 <= p["balanced"].config.model.params_b <= 4
    assert all(x.config.model.params_b < 7 for x in p.values())


def test_4gb_phone_never_gets_large_model_and_may_get_none():
    p = picks(phone4())
    assert all(x.config is None and x.tier == "none" for x in p.values())  # 128 MB retrieval index leaves no safe config
    assert all("fits" in x.reason for x in p.values())
    p2 = picks(phone4(), ret={"retrieval_ram_mb": 0, "retrieval_storage_mb": 0})
    assert p2["max_quality"].config.model.params_b <= 0.5
    assert p2["balanced"].config is None  # nothing keeps balanced headroom: honest "no", not a risky pick
    p3 = picks(phone4(), ret={"retrieval_ram_mb": 8, "retrieval_storage_mb": 16})
    assert p3["max_quality"].config is not None and p3["max_quality"].config.model.params_b <= 1.2


def test_every_assessed_candidate_for_4gb_excludes_7b():
    rec = dv.recommend(phone4(), dv.reference_candidates(), created_on=NOW)
    assert not any(a.eligible and a.config.model.params_b >= 3 for a in rec.assessed)


def test_profiles_are_ordered_by_headroom_and_quality():
    p = picks(flagship12())
    q = {k: dv.quality_score(v.config) for k, v in p.items()}
    assert q["max_quality"] >= q["balanced"] >= q["performance"]
    assert p["performance"].config.model.params_b <= p["balanced"].config.model.params_b


def test_picks_respect_profile_utilization_limits():
    rec = dv.recommend(flagship12(), dv.reference_candidates(), created_on=NOW)
    by = {a.config.config_id: a for a in rec.assessed}
    pol = rec.policy
    for pick in rec.picks:
        a = by[pick.config.config_id]
        limit = {"performance": pol.performance_max_utilization, "balanced": pol.balanced_max_utilization, "max_quality": pol.max_quality_max_utilization}[pick.profile]
        assert a.ram.utilization <= limit


def test_quality_destroying_quant_is_excluded():
    rec = dv.recommend(flagship12(), dv.reference_candidates(), created_on=NOW)
    for pick in rec.picks:
        assert pick.config.quant.name != "Q3_K_M"
    assert any(not a.eligible and a.config.quant.name == "Q3_K_M" for a in rec.assessed)


def test_provisional_pick_points_to_verified_fallback():
    d0 = flagship12()
    rec0 = dv.recommend(d0, dv.reference_candidates(), created_on=NOW)
    small = next(a.config for a in rec0.assessed if a.config.model.model_id == "ref-3b" and a.config.quant.name == "Q4_K_M" and a.config.context_tokens == 4096)
    d = flagship12(measurements=[full_measurement(small.config_id, peak_ram_mb=3000, sustained_ram_mb=2900, tokens_per_s=12)])
    rec = dv.recommend(d, dv.reference_candidates(), created_on=NOW)
    bal = next(p for p in rec.picks if p.profile == "balanced")
    assert bal.tier == "provisional" and bal.verified_fallback_config_id == small.config_id


def test_require_retrieval_filters_candidates():
    cands = dv.reference_candidates() + dv.generate_candidates(dv.REFERENCE_MODELS, dv.REFERENCE_QUANTS, [2048], dv.REFERENCE_RUNTIMES)
    rec = dv.recommend(flagship12(), cands, created_on=NOW, require_retrieval=True)
    assert all(p.config.retrieval.index_ram_mb > 0 for p in rec.picks)


def test_recommendation_is_sealed_deterministic_and_serializable():
    r1 = dv.recommend(flagship12().seal(), dv.reference_candidates(), created_on=NOW)
    r2 = dv.recommend(flagship12().seal(), dv.reference_candidates(), created_on=NOW)
    assert r1.verify() and r1.content_hash == r2.content_hash
    assert r1.device_profile_hash == flagship12().seal().content_hash
    r1.to_json()  # no inf/nan


def test_generate_candidates_respects_model_max_context():
    m = ModelSpec(model_id="short", params_b=1, layers=8, kv_heads=2, head_dim=64, max_context=2048)
    cs = dv.generate_candidates([m], [Q4], [2048, 4096], [RT])
    assert [c.context_tokens for c in cs] == [2048]


def test_negative_budget_device_is_no_fit_not_crash():
    d = make_device(total_ram_mb=2048, os_reserve_mb=1500, background_reserve_mb=800, host_app_mb=300, storage_total_mb=64000, storage_free_mb=30000)
    a = dv.assess(d, cand(model=ModelSpec(model_id="t", params_b=0.1, layers=4, kv_heads=1, head_dim=32), ret=RetrievalBudget()))
    assert a.ram_verdict == "no_fit" and math.isfinite(a.ram.utilization)
