"""Device qualification: pure functions, no I/O.

Three distinct verdicts per candidate configuration:

* storage fit   - files fit on flash after the free-storage reserve;
* RAM fit       - model-side memory fits the RAM envelope left after OS,
                  background, host app and safety reserves;
* sustained fit - the configuration keeps running without degrading the
                  device (throughput, thermal, UI jank, crashes/ANRs, kills).

"Loads" is never "fits". Measurements for a candidate override estimates.
When measurements are missing, estimates are inflated and the result can be
at best ``provisional``; only a fully measured, passing candidate earns the
``recommended`` tier.

Quality ranking (``quality_score``) is a monotone heuristic used only for
ordering candidates; replace it with measured evaluation deltas when available.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import product

from .schemas import (
    CandidateAssessment,
    CandidateConfig,
    DeploymentRecommendation,
    DeviceProfile,
    Measurement,
    ModelSpec,
    ProfilePick,
    QuantSpec,
    RamBreakdown,
    RetrievalBudget,
    RuntimeSpec,
    SafetyPolicy,
    StorageBreakdown,
    SustainedAssessment,
)

ALGORITHM_VERSION = "1"
MIB = 1024 * 1024
KV_DTYPE_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
PROFILE_LABELS = {
    "performance": "Performance",
    "balanced": "Balanced (Recommended)",
    "max_quality": "Maximum Quality Within Safe Envelope",
}
REQUIRED_MEASUREMENTS = (
    "ttft_ms",
    "tokens_per_s",
    "peak_ram_mb",
    "sustained_ram_mb",
    "thermal_throttle_ratio",
    "ui_jank_pct",
    "crashes",
    "anrs",
    "background_kills",
    "sustained_minutes",
)


# ----------------------------- component math ------------------------------ #


def kv_cache_mb(layers: int, kv_heads: int, head_dim: int, context: int, kv_dtype: str = "f16") -> float:
    """K and V tensors for every layer over the full context window."""
    return 2 * layers * kv_heads * head_dim * context * KV_DTYPE_BYTES[kv_dtype] / MIB


def weights_mb(c: CandidateConfig) -> float:
    if c.weights_file_mb is not None:
        return c.weights_file_mb
    return c.model.params_b * 1e9 * c.quant.bits_per_weight / 8 / MIB


def runtime_mb(c: CandidateConfig) -> float:
    return c.runtime.overhead_mb + c.runtime.scratch_mb_per_1k_ctx * c.context_tokens / 1024


def quality_score(c: CandidateConfig) -> float:
    base = c.model.quality_score if c.model.quality_score is not None else 10 * math.log2(1 + c.model.params_b)
    return round(base - c.quant.quality_penalty, 4)


def make_config_id(model: ModelSpec, quant: QuantSpec, ctx: int, runtime: RuntimeSpec, ret: RetrievalBudget) -> str:
    return f"{model.model_id}.{quant.name}.c{ctx}.{runtime.name}.kv-{runtime.kv_dtype}.r{int(ret.index_ram_mb)}"


def generate_candidates(
    models: Iterable[ModelSpec],
    quants: Iterable[QuantSpec],
    contexts: Iterable[int],
    runtimes: Iterable[RuntimeSpec],
    retrievals: Iterable[RetrievalBudget] = (RetrievalBudget(),),
) -> list[CandidateConfig]:
    out = []
    retrievals = list(retrievals)
    contexts = list(contexts)
    runtimes = list(runtimes)
    quants = list(quants)
    for m, q, ctx, rt, ret in product(models, quants, contexts, runtimes, retrievals):
        if ctx > m.max_context:
            continue
        out.append(
            CandidateConfig(config_id=make_config_id(m, q, ctx, rt, ret), model=m, quant=q, context_tokens=ctx, runtime=rt, retrieval=ret)
        )
    return out


# ----------------------------- envelopes ----------------------------------- #


def safety_reserve_mb(device: DeviceProfile, policy: SafetyPolicy) -> float:
    return max(policy.safety_reserve_min_mb, policy.safety_reserve_frac * device.total_ram_mb)


def ram_budget_mb(device: DeviceProfile, policy: SafetyPolicy) -> tuple[float, float]:
    """Return (model-side RAM budget, safety reserve). Budget may be negative on tiny devices."""
    ceiling = device.total_ram_mb - device.os_reserve_mb - device.background_reserve_mb
    if device.typical_available_ram_mb is not None:
        ceiling = min(ceiling, device.typical_available_ram_mb)  # never assume more than is typically free
    reserve = safety_reserve_mb(device, policy)
    return ceiling - device.host_app_mb - reserve, reserve


def storage_reserve_mb(device: DeviceProfile, policy: SafetyPolicy) -> float:
    return max(policy.storage_reserve_min_mb, policy.storage_reserve_frac * device.storage_total_mb)


def find_measurement(device: DeviceProfile, config_id: str) -> Measurement | None:
    found = [m for m in device.measurements if m.config_id == config_id]
    return found[-1] if found else None


# ----------------------------- verdicts ------------------------------------ #


def assess_storage(device: DeviceProfile, c: CandidateConfig, policy: SafetyPolicy) -> tuple[str, StorageBreakdown]:
    model_file = weights_mb(c)
    required = model_file + c.retrieval.index_storage_mb
    reserve = storage_reserve_mb(device, policy)
    avail = device.storage_free_mb - reserve
    sb = StorageBreakdown(
        model_file_mb=round(model_file, 1),
        retrieval_mb=c.retrieval.index_storage_mb,
        required_mb=round(required, 1),
        free_mb=device.storage_free_mb,
        reserve_mb=round(reserve, 1),
        available_after_reserve_mb=round(avail, 1),
        headroom_mb=round(avail - required, 1),
    )
    return ("fit" if required <= avail else "no_fit"), sb


def assess_ram(
    device: DeviceProfile, c: CandidateConfig, policy: SafetyPolicy, m: Measurement | None
) -> tuple[str, RamBreakdown]:
    w, rt = weights_mb(c), runtime_mb(c)
    kv = kv_cache_mb(c.model.layers, c.model.kv_heads, c.model.head_dim, c.context_tokens, c.runtime.kv_dtype)
    ret = c.retrieval.index_ram_mb
    budget, reserve = ram_budget_mb(device, policy)
    estimated = w * policy.weights_inflation + (rt + kv + ret) * policy.estimate_inflation
    measured_vals = [v for v in (m.peak_ram_mb, m.sustained_ram_mb) if v is not None] if m else []
    overridden = bool(measured_vals)
    required = max(measured_vals) if overridden else estimated
    rb = RamBreakdown(
        weights_mb=round(w, 1),
        runtime_mb=round(rt, 1),
        kv_cache_mb=round(kv, 1),
        retrieval_mb=ret,
        estimate_inflation=1.0 if overridden else policy.estimate_inflation,
        weights_inflation=1.0 if overridden else policy.weights_inflation,
        required_model_side_mb=round(required, 1),
        measured_override=overridden,
        total_ram_mb=device.total_ram_mb,
        os_reserve_mb=device.os_reserve_mb,
        background_reserve_mb=device.background_reserve_mb,
        host_app_mb=device.host_app_mb,
        safety_reserve_mb=round(reserve, 1),
        ram_budget_mb=round(budget, 1),
        headroom_mb=round(budget - required, 1),
        utilization=round(required / budget, 4) if budget > 0 else (999.0 if required > 0 else 0.0),
    )
    return ("fit" if budget > 0 and required <= budget else "no_fit"), rb


def estimate_decode_tps(device: DeviceProfile, c: CandidateConfig, policy: SafetyPolicy) -> float:
    """Memory-bandwidth-bound decode estimate, derated for sustained thermal behavior."""
    bw = device.mem_bandwidth_gbps if device.mem_bandwidth_gbps else policy.fallback_bandwidth_gbps
    bytes_per_token = weights_mb(c) * MIB
    return bw * 1e9 * policy.bandwidth_efficiency * policy.estimate_thermal_derate / bytes_per_token


def assess_sustained(
    device: DeviceProfile, c: CandidateConfig, policy: SafetyPolicy, m: Measurement | None, ram_ok: bool
) -> SustainedAssessment:
    est_tps = estimate_decode_tps(device, c, policy)
    est_ttft = 512 / (est_tps * 6) * 1000  # reference prompt, assumed prefill ~6x decode speed; informational only
    missing = [f for f in REQUIRED_MEASUREMENTS if m is None or getattr(m, f) is None]
    reasons: list[str] = []
    failed: list[str] = []
    if m is not None:
        if m.tokens_per_s is not None and m.tokens_per_s < policy.min_tps:
            failed.append(f"measured {m.tokens_per_s:.1f} tok/s < minimum {policy.min_tps}")
        if m.ttft_ms is not None and m.ttft_ms > policy.max_ttft_ms:
            failed.append(f"TTFT {m.ttft_ms:.0f} ms > {policy.max_ttft_ms:.0f} ms")
        if m.thermal_throttle_ratio is not None and m.thermal_throttle_ratio < policy.min_thermal_ratio:
            failed.append(f"sustained throttle ratio {m.thermal_throttle_ratio:.2f} < {policy.min_thermal_ratio}")
        if m.ui_jank_pct is not None and m.ui_jank_pct > policy.max_ui_jank_pct:
            failed.append(f"UI jank {m.ui_jank_pct:.1f}% > {policy.max_ui_jank_pct}%")
        for name, val in (("crashes", m.crashes), ("ANRs", m.anrs), ("background-process kills", m.background_kills)):
            if val:
                failed.append(f"{val} {name} observed")
        if m.sustained_minutes is not None and m.sustained_minutes < policy.min_sustained_minutes:
            reasons.append(f"run lasted {m.sustained_minutes:.1f} min < required {policy.min_sustained_minutes} min; inconclusive")
            missing = sorted(set(missing) | {"sustained_minutes"})
    if not ram_ok:
        failed.append("RAM envelope exceeded")
    if m is None or all(getattr(m, f) is None for f in REQUIRED_MEASUREMENTS):
        basis = "estimated"
    elif not missing:
        basis = "measured"
    else:
        basis = "partial"
    # estimates stand in for missing throughput data
    eff_tps = m.tokens_per_s if (m and m.tokens_per_s is not None) else est_tps
    if m is None or m.tokens_per_s is None:
        if est_tps < policy.min_tps:
            failed.append(f"estimated sustained {est_tps:.1f} tok/s < minimum {policy.min_tps}")
    if failed:
        verdict = "no_fit"
        reasons = failed + reasons
    elif basis == "measured":
        verdict = "fit"
    else:
        verdict = "unverified"
        reasons.append("sustained behavior not fully measured on this device; estimates only")
    return SustainedAssessment(
        verdict=verdict,
        basis=basis,
        est_tokens_per_s=round(est_tps, 2),
        est_ttft_ms=round(est_ttft),
        tokens_per_s=m.tokens_per_s if m else None,
        ttft_ms=m.ttft_ms if m else None,
        reasons=reasons,
        missing_measurements=missing if basis != "measured" else [],
    )


def assess(device: DeviceProfile, c: CandidateConfig, policy: SafetyPolicy | None = None) -> CandidateAssessment:
    policy = policy or SafetyPolicy()
    m = find_measurement(device, c.config_id)
    sv, sb = assess_storage(device, c, policy)
    rv, rb = assess_ram(device, c, policy, m)
    su = assess_sustained(device, c, policy, m, ram_ok=rv == "fit")
    blocking = []
    if sv == "no_fit":
        blocking.append(f"storage: needs {sb.required_mb:.0f} MB, {sb.available_after_reserve_mb:.0f} MB available after reserve")
    if rv == "no_fit":
        blocking.append(f"RAM: needs {rb.required_model_side_mb:.0f} MB, budget {rb.ram_budget_mb:.0f} MB")
    if su.verdict == "no_fit":
        blocking += su.reasons
    if c.quant.quality_penalty > policy.max_quant_penalty:
        blocking.append(f"quantization {c.quant.name} exceeds the allowed quality penalty ({policy.max_quant_penalty})")
    verdicts = (sv, rv, su.verdict)
    return CandidateAssessment(
        config=c,
        storage_verdict=sv,  # type: ignore[arg-type]
        ram_verdict=rv,  # type: ignore[arg-type]
        sustained_verdict=su.verdict,
        storage=sb,
        ram=rb,
        sustained=su,
        confidence=su.basis,
        quality_score=quality_score(c),
        safe_to_deploy=all(v == "fit" for v in verdicts) and not blocking,
        eligible=not blocking,
        blocking_reasons=blocking,
    )


# ----------------------------- recommendation ------------------------------ #


def _tps(a: CandidateAssessment) -> float:
    return a.sustained.tokens_per_s if a.sustained.tokens_per_s is not None else (a.sustained.est_tokens_per_s or 0.0)


def _quality_key(a: CandidateAssessment):
    c = a.config
    return (a.quality_score, c.retrieval.index_ram_mb, c.context_tokens, -a.ram.utilization)


def _pick(profile: str, pool: list[CandidateAssessment], pick_fn, policy: SafetyPolicy, reason_ok: str, why_none: str) -> ProfilePick:
    label = PROFILE_LABELS[profile]
    best = pick_fn(pool)
    if best is None:
        return ProfilePick(profile=profile, label=label, config=None, tier="none", recommended=False, reason=why_none)  # type: ignore[arg-type]
    verified = [a for a in pool if a.safe_to_deploy]
    verified_best = pick_fn(verified) if verified else None
    measured = best.safe_to_deploy
    warnings = []
    if not measured:
        warnings.append(
            "Based on conservative estimates only. Benchmark this exact configuration on the device "
            "before treating it as recommended."
        )
        if best.sustained.missing_measurements:
            warnings.append("missing measurements: " + ", ".join(best.sustained.missing_measurements))
    return ProfilePick(
        profile=profile,  # type: ignore[arg-type]
        label=label,
        config=best.config,
        tier="recommended" if measured else "provisional",
        recommended=measured,
        confidence=best.confidence,
        reason=reason_ok + f" Quality score {best.quality_score}, RAM utilization {best.ram.utilization:.0%}, ~{_tps(best):.1f} tok/s.",
        warnings=warnings,
        verified_fallback_config_id=None if measured or verified_best is None else verified_best.config.config_id,
    )


def recommend(
    device: DeviceProfile,
    candidates: list[CandidateConfig],
    policy: SafetyPolicy | None = None,
    *,
    created_on: str,
    require_retrieval: bool = False,
) -> DeploymentRecommendation:
    """Pick the best COMPLETE configuration for each user-facing profile.

    No model is excluded for size alone: a larger model is chosen whenever it
    stays inside the safe envelope. Only configurations whose verdicts are not
    ``no_fit`` are eligible.
    """
    policy = policy or SafetyPolicy()
    if require_retrieval:
        candidates = [c for c in candidates if c.retrieval.index_ram_mb > 0]
    assessed = [assess(device, c, policy) for c in candidates]
    eligible = [a for a in assessed if a.eligible]

    def max_quality(pool):
        pool = [a for a in pool if a.ram.utilization <= policy.max_quality_max_utilization]
        return max(pool, key=_quality_key, default=None)

    def balanced(pool):
        pool = [
            a
            for a in pool
            if a.ram.utilization <= policy.balanced_max_utilization
            and _tps(a) >= policy.balanced_min_tps
            and a.config.context_tokens >= policy.balanced_min_context
        ]
        return max(pool, key=_quality_key, default=None)

    def performance(pool):
        fast = [
            a
            for a in pool
            if a.ram.utilization <= policy.performance_max_utilization and _tps(a) >= policy.performance_min_tps
        ]
        if fast:
            return max(fast, key=_quality_key)
        return max(pool, key=lambda a: (_tps(a), a.quality_score), default=None)

    none_msg = "No configuration fits the safe envelope for this profile on this device; shrink the retrieval budget or context, free storage, or choose a smaller model class."
    picks = [
        _pick("performance", eligible, performance, policy, "Fast and light: leaves generous headroom.", none_msg),
        _pick("balanced", eligible, balanced, policy, "Best quality that keeps comfortable headroom and usable speed.", none_msg),
        _pick(
            "max_quality",
            eligible,
            max_quality,
            policy,
            "Highest quality that still fits the safe envelope.",
            none_msg,
        ),
    ]
    return DeploymentRecommendation(
        algorithm_version=ALGORITHM_VERSION,
        device_profile_hash=device.content_hash or device.seal().content_hash,
        device_id=device.device_id,
        policy=policy,
        picks=picks,
        assessed=assessed,
        created_on=created_on,
    ).seal()


# ----------------------------- reference catalog --------------------------- #
# ILLUSTRATIVE geometry for generic model size classes. These are NOT entries
# for any specific vendor model; real candidates must come from the model
# registry with their true architecture and file sizes.

REFERENCE_MODELS = [
    ModelSpec(model_id="ref-0.5b", params_b=0.5, layers=24, kv_heads=2, head_dim=64, max_context=8192),
    ModelSpec(model_id="ref-1b", params_b=1.2, layers=16, kv_heads=8, head_dim=64, max_context=8192),
    ModelSpec(model_id="ref-3b", params_b=3.2, layers=28, kv_heads=8, head_dim=128, max_context=8192),
    ModelSpec(model_id="ref-7b", params_b=7.2, layers=32, kv_heads=8, head_dim=128, max_context=8192),
    ModelSpec(model_id="ref-8b", params_b=8.0, layers=32, kv_heads=8, head_dim=128, max_context=8192),
    ModelSpec(model_id="ref-13b", params_b=13.0, layers=40, kv_heads=8, head_dim=128, max_context=8192),
]
REFERENCE_QUANTS = [
    QuantSpec(name="Q8_0", bits_per_weight=8.5, quality_penalty=0.3),
    QuantSpec(name="Q6_K", bits_per_weight=6.6, quality_penalty=0.6),
    QuantSpec(name="Q5_K_M", bits_per_weight=5.7, quality_penalty=1.0),
    QuantSpec(name="Q4_K_M", bits_per_weight=4.85, quality_penalty=1.8),
    QuantSpec(name="Q3_K_M", bits_per_weight=3.9, quality_penalty=4.0),
]
REFERENCE_RUNTIMES = [RuntimeSpec(name="llamacpp-android", overhead_mb=300, scratch_mb_per_1k_ctx=16, kv_dtype="f16")]
REFERENCE_CONTEXTS = [2048, 4096, 8192]


def reference_candidates(retrieval_ram_mb: float = 128, retrieval_storage_mb: float = 256) -> list[CandidateConfig]:
    ret = RetrievalBudget(index_ram_mb=retrieval_ram_mb, index_storage_mb=retrieval_storage_mb, top_k=4)
    return generate_candidates(REFERENCE_MODELS, REFERENCE_QUANTS, REFERENCE_CONTEXTS, REFERENCE_RUNTIMES, [ret])
