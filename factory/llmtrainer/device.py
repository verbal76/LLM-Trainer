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
from typing import Literal

from pydantic import Field

from .schemas import (
    Base,
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


# =========================================================================== #
# Device profiler v2: per-ARTIFACT capabilities (ALGORITHM v2 of the capability layer)
#
# The v1 functions above answer "which complete deployment configuration is safe?". v2 answers, for EACH candidate
# downloadable artifact, a set of explicit and SEPARATE capabilities:
#
#   CAN_DOWNLOAD / CAN_LOAD / CAN_INFER / CAN_EVALUATE /
#   CAN_SPECIALIZE_LOCALLY_FULL / CAN_SPECIALIZE_LOCALLY_PARTIAL (+ trainable_last_layers) / EXTERNAL_COMPUTE_REQUIRED
#
# Inference feasibility is NEVER training feasibility. Training memory (F32 trainer, see docs/studio/DEVICE_PROFILER.md):
#       F32 weights + (4 + 8) bytes x trainable params + activation estimate (+ runtime overhead)
# Everything here is an ESTIMATE until a MeasurementRecord from the device replaces it; confidence is "low" until then.
# Reasons are stable CODES (no formatted numbers) so the Kotlin port can be compared string-for-string.
# Operation order is part of the contract: qualify/Capability.kt mirrors it so floating-point results match bit for bit.
# =========================================================================== #

CAPABILITY_ALGORITHM_VERSION = "2"
QUANT_TABLE = {  # name -> (bits per weight, quality penalty); unknown names use the conservative fallback below
    "F32": (32.0, 0.0), "F16": (16.0, 0.0), "BF16": (16.0, 0.0), "Q8_0": (8.5, 0.3), "Q6_K": (6.6, 0.6),
    "Q5_K_M": (5.7, 1.0), "Q4_K_M": (4.85, 1.8), "Q3_K_M": (3.9, 4.0),
}
UNKNOWN_QUANT = (8.5, 1.0)
CHOICE_LABELS = {
    "fastest": "Fastest / lowest resource use",
    "balanced": "Balanced (Recommended)",
    "best_quality": "Best quality this device can reasonably handle",
    "best_specialize": "Best model you can specialize on this phone",
}


class CapabilityPolicy(Base):
    """Thresholds of the capability layer (the v1 SafetyPolicy still supplies reserves, inflation and min tok/s)."""

    infer_contexts: list[int] = Field(default_factory=lambda: [512, 1024, 2048, 4096, 8192])
    load_context: int = 512
    eval_min_context: int = 1024
    eval_min_tps: float = 5.0
    train_ctx_tokens: int = 256
    train_overhead_mb: float = 300.0
    train_max_utilization: float = 0.85  # of the RAM budget: training is a heavier, longer load than inference
    min_partial_layers: int = 2
    train_dataset_storage_mb: float = 256.0
    train_ref_tokens: int = 200000
    max_practical_train_hours: float = 48.0
    train_min_battery_pct: float = 30.0
    train_max_thermal_status: int = 1  # 0 NONE, 1 LIGHT


class DeviceState(Base):
    """Transient device condition (not part of the sealed DeviceProfile). Unknown fields stay None -> condition unknown."""

    thermal_status: int | None = None  # android thermal status 0..6
    battery_pct: float | None = None
    charging: bool | None = None
    power_save: bool | None = None


class ArtifactSpec(Base):
    """One candidate downloadable artifact as the profiler sees it (flat, derived from registry/artifacts/<id>.json)."""

    artifact_id: str
    model_id: str
    quantization: str
    precision: Literal["quantized", "f32", "f16", "bf16"]
    sha256: str | None = None
    size_bytes: int | None = None
    params: float | None = None  # exact parameter count (GGUF header); None until the catalog refresh read it
    nominal_params_b: float
    layers: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    embd: int | None = None
    ffn: int | None = None
    vocab: int | None = None
    ctx_train: int | None = None
    training_class: Literal["local_full", "local_partial", "inference_only", "external_only"]
    training_source_artifact_id: str | None = None
    downloadable: bool = False
    license_state: Literal["VERIFIED", "UNVERIFIED", "DISALLOWED"] = "UNVERIFIED"
    runtime_supported: Literal["yes", "unverified", "no"] = "unverified"


class MeasurementRecord(Base):
    """One REAL on-device measurement. Replaces the matching estimate and raises confidence (see merge/apply below)."""

    schema_id: Literal["device_measurement.v1"] = "device_measurement.v1"
    record_id: str
    artifact_id: str
    artifact_sha256: str | None = None  # must equal the artifact's hash, else the record is ignored
    device_id: str
    runtime_id: str | None = None
    app_version: str | None = None
    recorded_at: str
    kind: Literal["load", "inference", "training"]
    # load
    load_time_ms: float | None = None
    load_peak_ram_mb: float | None = None
    # inference (same fields as the v1 Measurement)
    context_tokens: int | None = None
    ttft_ms: float | None = None
    tokens_per_s: float | None = None
    peak_ram_mb: float | None = None
    sustained_ram_mb: float | None = None
    thermal_throttle_ratio: float | None = None
    ui_jank_pct: float | None = None
    crashes: int | None = None
    anrs: int | None = None
    background_kills: int | None = None
    sustained_minutes: float | None = None
    # training (trainable_last_layers None with kind=training means FULL tuning)
    trainable_last_layers: int | None = None
    train_peak_ram_mb: float | None = None
    train_tokens_per_s: float | None = None
    checkpoint_size_mb: float | None = None
    train_minutes: float | None = None
    train_completed: bool | None = None
    train_oom: bool | None = None
    thermal_max_status: int | None = None
    battery_drop_pct: float | None = None


class Condition(Base):
    name: str
    status: Literal["met", "unmet", "unknown"]


class TrainingOption(Base):
    mode: Literal["full", "partial"]
    trainable_last_layers: int
    trainable_params_m: float
    ram_mb: float
    utilization: float
    est_tokens_per_s: float
    est_minutes_for_reference_tokens: float
    checkpoint_mb: float
    storage_required_mb: float
    fits: bool
    measured: bool


class ArtifactCapabilities(Base):
    artifact_id: str
    can_download: bool
    can_load: bool
    can_infer: bool
    can_evaluate: bool
    can_specialize_full: bool
    can_specialize_partial: bool
    external_compute_required: bool
    infer_context: int | None
    est_tokens_per_s: float | None
    infer_ram_mb: float | None
    infer_utilization: float | None
    load_ram_mb: float | None
    quality_score: float
    trainable_last_layers: int | None
    full: TrainingOption | None
    partial: TrainingOption | None
    specialize_via_artifact_id: str | None
    training_conditions: list[Condition]
    ready_to_train_now: bool
    confidence_infer: Literal["low", "medium", "high"]
    confidence_train: Literal["low", "medium", "high"]
    reasons: list[str]
    ignored_measurements: list[str] = Field(default_factory=list)


class ChoicePick(Base):
    choice: str
    label: str
    artifact_id: str | None
    tier: Literal["recommended", "provisional", "preview", "none"]
    confidence: Literal["low", "medium", "high"] | None = None
    context_tokens: int | None = None
    trainable_last_layers: int | None = None
    license_state: str | None = None
    downloadable: bool = False
    reason: str


class CapabilityReport(Base):
    algorithm_version: str = CAPABILITY_ALGORITHM_VERSION
    device_id: str
    capabilities: list[ArtifactCapabilities]
    choices: list[ChoicePick]


# ----------------------------- artifact -> v1 types ------------------------- #


def spec_params(spec: ArtifactSpec) -> float:
    return spec.params if spec.params is not None else spec.nominal_params_b * 1e9


def _quant_row(spec: ArtifactSpec) -> tuple[float, float]:
    return QUANT_TABLE.get(spec.quantization.upper(), UNKNOWN_QUANT)


def spec_file_mb(spec: ArtifactSpec) -> float:
    if spec.size_bytes is not None:
        return spec.size_bytes / MIB
    return spec_params(spec) * _quant_row(spec)[0] / 8 / MIB


def spec_geometry(spec: ArtifactSpec) -> tuple[int, int, int, bool]:
    """(layers, kv_heads, head_dim, estimated?). Unknown geometry falls back to the first reference class that is not smaller."""
    if spec.layers is not None and spec.kv_heads is not None and spec.head_dim is not None:
        return spec.layers, spec.kv_heads, spec.head_dim, False
    p = spec.nominal_params_b
    ref = next((m for m in REFERENCE_MODELS if m.params_b >= p), REFERENCE_MODELS[-1])
    return ref.layers, ref.kv_heads, ref.head_dim, True


def spec_candidate(spec: ArtifactSpec, ctx: int) -> CandidateConfig:
    layers, kvh, hd, _ = spec_geometry(spec)
    bits, pen = _quant_row(spec)
    max_ctx = min(spec.ctx_train if spec.ctx_train is not None else 8192, 8192)
    model = ModelSpec(model_id=spec.artifact_id, params_b=spec_params(spec) / 1e9, layers=layers, kv_heads=kvh, head_dim=hd, max_context=max_ctx)
    quant = QuantSpec(name=spec.quantization, bits_per_weight=bits, quality_penalty=pen)
    rt = REFERENCE_RUNTIMES[0]
    ret = RetrievalBudget()
    return CandidateConfig(config_id=make_config_id(model, quant, ctx, rt, ret), model=model, quant=quant, context_tokens=ctx, runtime=rt,
                           retrieval=ret, weights_file_mb=spec_file_mb(spec))


# ----------------------------- training model ------------------------------- #


def train_gflops_prior(device: DeviceProfile) -> float:
    """Sustained F32 GFLOPS prior by RAM class (heuristic, LOW confidence)."""
    t = device.total_ram_mb
    if t >= 12000:
        return 40.0
    if t >= 8000:
        return 25.0
    if t >= 6000:
        return 15.0
    return 8.0


def training_ram_mb(spec: ArtifactSpec, k: int | None, cap: CapabilityPolicy, safety: SafetyPolicy) -> tuple[float, float]:
    """(model-side training RAM in MB, trainable params). ``k`` = number of trailing layers tuned; None = full tuning.

    RAM = (F32 weights + 12 bytes x trainable) x weights_inflation + (activations + runtime overhead) x estimate_inflation.
    Per-layer params are params/layers (includes the embedding share: conservative for partial tuning).
    """
    layers = spec_geometry(spec)[0]
    p = spec_params(spec)
    per_layer = p / layers
    trainable = p if k is None else per_layer * k
    t_layers = layers if k is None else k
    weights = p * 4 / MIB
    state = trainable * 12 / MIB
    if spec.embd is not None:
        ffn = spec.ffn if spec.ffn is not None else 4 * spec.embd
        vocab = spec.vocab if spec.vocab is not None else 0
        act = (t_layers * 4 * cap.train_ctx_tokens * (16 * spec.embd + 2 * ffn) + 2 * 4 * cap.train_ctx_tokens * vocab) / MIB
    else:
        act = 0.25 * weights
    total = (weights + state) * safety.weights_inflation + (act + cap.train_overhead_mb) * safety.estimate_inflation
    return total, trainable


def train_tps(device: DeviceProfile, spec: ArtifactSpec, k: int | None, safety: SafetyPolicy) -> float:
    layers = spec_geometry(spec)[0]
    p = spec_params(spec)
    per_layer = p / layers
    flops = 6 * p if k is None else 2 * p + 4 * k * per_layer
    return train_gflops_prior(device) * 1e9 * safety.estimate_thermal_derate / flops


# ----------------------------- measurement records -------------------------- #


def valid_records(spec: ArtifactSpec, device: DeviceProfile, records: list[MeasurementRecord]) -> tuple[list[MeasurementRecord], list[str]]:
    """Keep records that describe THIS artifact file on THIS device. Returns (usable, ignored reason codes)."""
    ok, ignored = [], []
    for r in records:
        if r.artifact_id != spec.artifact_id:
            continue
        if r.device_id != device.device_id:
            ignored.append(f"{r.record_id}:other_device")
        elif r.artifact_sha256 is not None and spec.sha256 is not None and r.artifact_sha256 != spec.sha256:
            ignored.append(f"{r.record_id}:artifact_hash_mismatch")
        else:
            ok.append(r)
    return ok, ignored


def record_key(r: MeasurementRecord) -> tuple:
    ctx = r.context_tokens if r.kind == "inference" else None
    k = (-1 if r.trainable_last_layers is None else r.trainable_last_layers) if r.kind == "training" else None
    return (r.artifact_id, r.device_id, r.kind, -1 if ctx is None else ctx, -2 if k is None else k)


def merge_records(existing: list[MeasurementRecord], new: list[MeasurementRecord]) -> list[MeasurementRecord]:
    """Union keyed by (artifact, device, kind, context, trainable layers); the latest ``recorded_at`` wins (ties: the later argument)."""
    best: dict[tuple, MeasurementRecord] = {}
    for r in list(existing) + list(new):
        key = record_key(r)
        cur = best.get(key)
        if cur is None or r.recorded_at >= cur.recorded_at:
            best[key] = r
    return [best[k] for k in sorted(best)]


def _inference_measurement(r: MeasurementRecord, config_id: str) -> Measurement:
    return Measurement(config_id=config_id, ttft_ms=r.ttft_ms, tokens_per_s=r.tokens_per_s, peak_ram_mb=r.peak_ram_mb, sustained_ram_mb=r.sustained_ram_mb,
                       thermal_throttle_ratio=r.thermal_throttle_ratio, ui_jank_pct=r.ui_jank_pct, crashes=r.crashes, anrs=r.anrs,
                       background_kills=r.background_kills, sustained_minutes=r.sustained_minutes)


# ----------------------------- classification ------------------------------- #


def training_conditions(state: DeviceState | None, cap: CapabilityPolicy, storage_ok: bool) -> list[Condition]:
    s = state or DeviceState()

    def tri(v: bool | None) -> str:
        return "unknown" if v is None else ("met" if v else "unmet")

    thermal = None if s.thermal_status is None else s.thermal_status <= cap.train_max_thermal_status
    if s.charging is True:
        battery: bool | None = True
    elif s.battery_pct is not None:
        battery = s.battery_pct >= cap.train_min_battery_pct
    else:
        battery = None
    return [
        Condition(name="thermal", status=tri(thermal)),  # type: ignore[arg-type]
        Condition(name="charging", status=tri(s.charging)),  # type: ignore[arg-type]
        Condition(name="battery", status=tri(battery)),  # type: ignore[arg-type]
        Condition(name="not_power_save", status=tri(None if s.power_save is None else (not s.power_save))),  # type: ignore[arg-type]
        Condition(name="storage_for_checkpoints", status="met" if storage_ok else "unmet"),
    ]


def classify_artifact(
    device: DeviceProfile,
    spec: ArtifactSpec,
    safety: SafetyPolicy | None = None,
    cap: CapabilityPolicy | None = None,
    state: DeviceState | None = None,
    records: list[MeasurementRecord] | None = None,
) -> ArtifactCapabilities:
    safety = safety or SafetyPolicy()
    cap = cap or CapabilityPolicy()
    reasons: list[str] = []
    usable, ignored = valid_records(spec, device, records or [])
    layers, _kvh, _hd, geo_estimated = spec_geometry(spec)
    if geo_estimated:
        reasons.append("geometry_estimated")
    if spec.size_bytes is None:
        reasons.append("size_estimated")
    if spec.params is None:
        reasons.append("params_nominal")
    if spec.runtime_supported == "unverified":
        reasons.append("runtime_support_unverified")
    if spec.runtime_supported == "no":
        reasons.append("runtime_unsupported")

    # ---- download: storage after the reserve, and an artifact the catalog can actually verify ----
    reserve = storage_reserve_mb(device, safety)
    file_mb = spec_file_mb(spec)
    avail = device.storage_free_mb - reserve
    storage_ok = file_mb <= avail
    can_download = spec.downloadable and storage_ok
    if not spec.downloadable:
        reasons.append("not_downloadable")
    if not storage_ok:
        reasons.append("storage_insufficient")

    # ---- load / infer (reuses the v1 assessment; measured records override the estimates) ----
    max_ctx = min(spec.ctx_train if spec.ctx_train is not None else 8192, 8192)
    inf_meas = [_inference_measurement(r, spec_candidate(spec, r.context_tokens).config_id)
                for r in usable if r.kind == "inference" and r.context_tokens is not None]
    dev = device.model_copy(update={"measurements": list(device.measurements) + inf_meas}) if inf_meas else device
    load_c = spec_candidate(spec, min(cap.load_context, max_ctx))
    load_a = assess(dev, load_c, safety)
    load_ram = load_a.ram.required_model_side_mb
    can_load = load_a.ram_verdict == "fit"
    load_recs = [r for r in usable if r.kind == "load"]
    peaks = [r.load_peak_ram_mb for r in load_recs if r.load_peak_ram_mb is not None]
    if peaks:
        load_ram = max(peaks)
        can_load = load_a.ram.ram_budget_mb > 0 and load_ram <= load_a.ram.ram_budget_mb
    if not can_load:
        reasons.append("ram_insufficient_to_load")

    best_ok = None
    best_comfortable = None
    for ctx in sorted(cap.infer_contexts):
        if ctx > max_ctx:
            continue
        a = assess(dev, spec_candidate(spec, ctx), safety)
        ok = (a.ram_verdict == "fit" and a.sustained_verdict != "no_fit"
              and a.config.quant.quality_penalty <= safety.max_quant_penalty and spec.runtime_supported != "no")
        if ok:
            best_ok = a
            if a.ram.utilization <= safety.balanced_max_utilization:
                best_comfortable = a
    chosen = best_comfortable if best_comfortable is not None else best_ok
    can_infer = chosen is not None
    if not can_infer and can_load and spec.runtime_supported != "no":
        reasons.append("infer_too_slow_or_unsafe")
    est_tps = infer_util = infer_ram = None
    ctx_out = None
    if chosen is not None:
        ctx_out = chosen.config.context_tokens
        est_tps = chosen.sustained.tokens_per_s if chosen.sustained.tokens_per_s is not None else chosen.sustained.est_tokens_per_s
        infer_util = chosen.ram.utilization
        infer_ram = chosen.ram.required_model_side_mb
    can_eval = can_infer and ctx_out is not None and ctx_out >= cap.eval_min_context and est_tps is not None and est_tps >= cap.eval_min_tps
    q = quality_score(spec_candidate(spec, min(2048, max_ctx)))

    # ---- specialization (training): a SEPARATE question from inference ----
    budget, _reserve = ram_budget_mb(device, safety)
    train_recs = [r for r in usable if r.kind == "training"]
    failed_layers = [layers if r.trainable_last_layers is None else r.trainable_last_layers
                     for r in train_recs if r.train_oom or r.train_completed is False]
    min_failed = min(failed_layers) if failed_layers else None

    def option(k: int | None) -> TrainingOption:
        est_ram, trainable = training_ram_mb(spec, k, cap, safety)
        found = [r for r in train_recs if r.trainable_last_layers == k]
        rec = found[-1] if found else None
        measured = rec is not None and rec.train_peak_ram_mb is not None
        ram = rec.train_peak_ram_mb if (rec is not None and rec.train_peak_ram_mb is not None) else est_ram
        tps = rec.train_tokens_per_s if (rec is not None and rec.train_tokens_per_s is not None) else train_tps(device, spec, k, safety)
        ckpt = rec.checkpoint_size_mb if (rec is not None and rec.checkpoint_size_mb is not None) else trainable * 4 / MIB
        storage_req = 2 * ckpt + cap.train_dataset_storage_mb
        eff_k = layers if k is None else k
        failed = rec is not None and (bool(rec.train_oom) or rec.train_completed is False)
        fits = (budget > 0 and ram <= budget * cap.train_max_utilization and storage_req <= avail
                and not (min_failed is not None and eff_k >= min_failed) and not failed)
        return TrainingOption(
            mode="full" if k is None else "partial", trainable_last_layers=eff_k, trainable_params_m=round(trainable / 1e6, 2), ram_mb=round(ram, 1),
            utilization=round(ram / budget, 4) if budget > 0 else 999.0, est_tokens_per_s=round(tps, 3),
            est_minutes_for_reference_tokens=round(cap.train_ref_tokens / tps / 60, 1), checkpoint_mb=round(ckpt, 1),
            storage_required_mb=round(storage_req, 1), fits=fits, measured=measured)

    full_opt: TrainingOption | None = None
    partial_opt: TrainingOption | None = None
    best_k: int | None = None
    cls = spec.training_class
    full_precision_artifact = spec.precision != "quantized"
    if cls == "inference_only" or not full_precision_artifact:
        reasons.append("training_needs_full_precision_artifact")
    elif cls == "external_only":
        reasons.append("training_class_external_only")
    else:
        if cls == "local_full":
            full_opt = option(None)
        k = layers if cls == "local_full" else layers - 1
        while k >= cap.min_partial_layers:
            o = option(k)
            if o.fits:
                partial_opt, best_k = o, k
                break
            k -= 1
        if partial_opt is None and layers >= cap.min_partial_layers:
            partial_opt = option(cap.min_partial_layers)  # diagnostic only: fits is False
    can_full = full_opt is not None and full_opt.fits
    can_partial = partial_opt is not None and partial_opt.fits
    if (cls in ("local_full", "local_partial") and full_precision_artifact) and not can_full and not can_partial:
        reasons.append("training_ram_insufficient")
    if can_partial and partial_opt is not None and partial_opt.est_minutes_for_reference_tokens / 60 > cap.max_practical_train_hours:
        reasons.append("training_very_slow")
    external = not can_full and not can_partial
    via = spec.training_source_artifact_id if (cls == "inference_only" or not full_precision_artifact) else None
    options = [o for o in (full_opt, partial_opt) if o is not None]
    conds: list[Condition] = []
    if options:
        conds = training_conditions(state, cap, any(o.storage_required_mb <= avail for o in options))
    ready = (can_full or can_partial) and all(c.status == "met" for c in conds)

    conf_infer = "low"
    if inf_meas:
        # "high" needs a COMPLETE measurement of the very configuration that was chosen; other contexts only give "medium"
        at_ctx = [m for m in inf_meas if chosen is not None and m.config_id == chosen.config.config_id]
        full_meas = [m for m in at_ctx if all(getattr(m, f) is not None for f in REQUIRED_MEASUREMENTS)]
        conf_infer = "high" if full_meas else "medium"
    elif load_recs:
        conf_infer = "medium"
    done = [r for r in train_recs if r.train_completed and not r.train_oom and r.train_peak_ram_mb is not None and r.train_tokens_per_s is not None]
    conf_train = "high" if done else ("medium" if train_recs else "low")
    return ArtifactCapabilities(
        artifact_id=spec.artifact_id, can_download=can_download, can_load=can_load, can_infer=can_infer, can_evaluate=can_eval,
        can_specialize_full=can_full, can_specialize_partial=can_partial, external_compute_required=external,
        infer_context=ctx_out, est_tokens_per_s=None if est_tps is None else round(est_tps, 2),
        infer_ram_mb=None if infer_ram is None else round(infer_ram, 1), infer_utilization=infer_util, load_ram_mb=round(load_ram, 1),
        quality_score=q, trainable_last_layers=best_k, full=full_opt, partial=partial_opt, specialize_via_artifact_id=via,
        training_conditions=conds, ready_to_train_now=ready, confidence_infer=conf_infer, confidence_train=conf_train,  # type: ignore[arg-type]
        reasons=reasons, ignored_measurements=ignored)


# ----------------------------- owner-facing choices ------------------------- #


def _tier(downloadable: bool, conf: str) -> str:
    if not downloadable:
        return "preview"
    return "recommended" if conf == "high" else "provisional"


def _first_max(pool: list, key):
    best = None
    for item in pool:
        if best is None or key(item) > key(best):
            best = item
    return best


def choose_models(
    device: DeviceProfile,
    specs: list[ArtifactSpec],
    safety: SafetyPolicy | None = None,
    cap: CapabilityPolicy | None = None,
    state: DeviceState | None = None,
    records: list[MeasurementRecord] | None = None,
) -> CapabilityReport:
    """Three owner-facing choices (fastest, balanced, best quality) + separately the best model to specialize here.

    Artifacts that are not downloadable yet (unrefreshed catalog) only produce ``preview`` picks, and only when nothing
    downloadable qualifies. DISALLOWED licenses are never offered. Estimate-only picks are never ``recommended``.
    """
    safety = safety or SafetyPolicy()
    cap = cap or CapabilityPolicy()
    caps = [classify_artifact(device, s, safety, cap, state, records) for s in specs]
    pairs = [(s, c) for s, c in zip(specs, caps)
             if s.license_state != "DISALLOWED" and c.can_infer and c.can_load and "storage_insufficient" not in c.reasons]

    def pools(pred):
        sel = [(s, c) for s, c in pairs if pred(s, c)]
        dl = [(s, c) for s, c in sel if s.downloadable]
        return (dl, True) if dl else (sel, False)

    def util(c: ArtifactCapabilities) -> float:
        return c.infer_utilization if c.infer_utilization is not None else 999.0

    def tps(c: ArtifactCapabilities) -> float:
        return c.est_tokens_per_s if c.est_tokens_per_s is not None else 0.0

    def pick(choice: str, pool_info, key, why_ok: str, why_none: str) -> ChoicePick:
        pool, dl = pool_info
        best = _first_max(pool, key)
        if best is None:
            return ChoicePick(choice=choice, label=CHOICE_LABELS[choice], artifact_id=None, tier="none", reason=why_none)
        s, c = best
        return ChoicePick(choice=choice, label=CHOICE_LABELS[choice], artifact_id=s.artifact_id, tier=_tier(dl, c.confidence_infer),  # type: ignore[arg-type]
                          confidence=c.confidence_infer, context_tokens=c.infer_context, license_state=s.license_state, downloadable=s.downloadable, reason=why_ok)

    fastest = pick("fastest", pools(lambda s, c: True), lambda it: (tps(it[1]), -spec_file_mb(it[0])),
                   "fastest_decode_within_envelope", "nothing_fits")
    balanced = pick("balanced",
                    pools(lambda s, c: util(c) <= safety.balanced_max_utilization and tps(c) >= safety.balanced_min_tps
                          and (c.infer_context or 0) >= safety.balanced_min_context),
                    lambda it: (it[1].quality_score, it[1].infer_context or 0, -util(it[1])),
                    "best_quality_with_comfortable_headroom", "nothing_comfortable")
    best_q = pick("best_quality", pools(lambda s, c: util(c) <= safety.max_quality_max_utilization),
                  lambda it: (it[1].quality_score, it[1].infer_context or 0, -util(it[1])),
                  "highest_quality_within_envelope", "nothing_fits")

    def spec_minutes(c: ArtifactCapabilities) -> float:
        o = c.full if c.can_specialize_full else c.partial
        return o.est_minutes_for_reference_tokens if o is not None else 0.0

    sp_pool, sp_dl = pools(lambda s, c: c.can_specialize_full or c.can_specialize_partial)
    sbest = _first_max(sp_pool, lambda it: (it[1].quality_score, 1 if it[1].can_specialize_full else 0, -spec_minutes(it[1])))
    if sbest is None:
        sp = ChoicePick(choice="best_specialize", label=CHOICE_LABELS["best_specialize"], artifact_id=None, tier="none",
                        reason="no_local_specialization_possible")
    else:
        s, c = sbest
        layers_out = c.full.trainable_last_layers if (c.can_specialize_full and c.full is not None) else c.trainable_last_layers
        sp = ChoicePick(choice="best_specialize", label=CHOICE_LABELS["best_specialize"], artifact_id=s.artifact_id,
                        tier=_tier(sp_dl, c.confidence_train), confidence=c.confidence_train, context_tokens=c.infer_context,  # type: ignore[arg-type]
                        trainable_last_layers=layers_out, license_state=s.license_state, downloadable=s.downloadable,
                        reason="full_tuning_fits" if c.can_specialize_full else "partial_tuning_fits")
    return CapabilityReport(device_id=device.device_id, capabilities=caps, choices=[fastest, balanced, best_q, sp])
