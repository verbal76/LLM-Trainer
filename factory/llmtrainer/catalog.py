"""Model catalog and acquisition planner.

The catalog is a read-only *view* over the base-model registry
(``registry/base-models/*.json``): identity, source, architecture, variants,
license verification state, Android-inference feasibility and desktop
training feasibility. Unknown facts are ``None`` - nothing here is guessed.

The acquisition planner never downloads anything and performs no network I/O.
It only reports whether a download would be permitted. A download is
authorized only with an explicit confirmation flag AND a VERIFIED license that
grants the intended use; multi-GB (or unknown-size) downloads must be
confirmed with the size printed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .device import kv_cache_mb
from .licenses import LicenseRegistry, RequestedUse, evaluate_gate
from .schemas import BaseModelLicenseEntry, ModelVariant
from .training import ExperimentConfig, estimate_resources

# Downloads at or above this size (or of unknown size) need --confirm-download and a printed size.
LARGE_DOWNLOAD_BYTES = 1 * 1024**3

ACQUISITION_CODES = {
    "NO_SOURCE_URL": "no authoritative source URL is recorded for this variant",
    "LICENSE_BLOCKED": "license gate does not permit the intended use (see gate.reason_codes)",
    "CONFIRMATION_REQUIRED": "explicit --confirm-download is required",
}


class AcquisitionRefused(RuntimeError):
    def __init__(self, plan: "AcquisitionPlan", why: list[str]):
        self.plan = plan
        self.why = why
        super().__init__(f"download of {plan.model_id}/{plan.variant_id} refused: " + "; ".join(why))


def default_registry_dir() -> Path:
    """``$LLMTRAINER_REGISTRY`` or the repository's ``registry/base-models`` next to the factory."""
    env = os.environ.get("LLMTRAINER_REGISTRY")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "registry" / "base-models"


def load_registry(path: str | Path | None = None) -> LicenseRegistry:
    p = Path(path) if path else default_registry_dir()
    if not p.exists():
        raise FileNotFoundError(f"registry not found: {p} (pass --registry or set LLMTRAINER_REGISTRY)")
    return LicenseRegistry.load(p)


def resolve_model(reg: LicenseRegistry, query: str) -> BaseModelLicenseEntry:
    """Find an entry by exact id, exact HF repo id, or a unique case-insensitive substring of the id."""
    if query in reg.entries:
        return reg.entries[query]
    q = query.lower()
    by_repo = [e for e in reg.entries.values() if (e.hf_repo_or_source or "").lower() == q]
    if len(by_repo) == 1:
        return by_repo[0]
    hits = [e for i, e in reg.entries.items() if q in i.lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise KeyError(f"no catalog model matches {query!r}")
    raise KeyError(f"{query!r} is ambiguous; matches: " + ", ".join(sorted(e.entry_id for e in hits)))


def parse_params_b(parameter_count: str | None) -> tuple[float | None, str]:
    """Leading 'N.NB' from the recorded string. Returns (billions or None, note).

    'Effective' parameter counts (e.g. Gemma E-series) understate real weight
    memory, so they are not used for training estimates.
    """
    if not parameter_count:
        return None, "parameter count not recorded"
    if "effective" in parameter_count.lower():
        return None, "recorded count is 'effective' parameters; real weight size is larger, so no estimate is made"
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)\s*B\b", parameter_count)
    if not m:
        return None, "parameter count not parseable"
    return float(m.group(1)), "nominal parameter count taken from the first figure in the recorded string"


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "unknown (not recorded; will not be guessed)"
    return f"{n / 1024**3:.2f} GiB" if n >= 1024**3 else f"{n / 1024**2:.1f} MiB"


def training_feasibility(entry: BaseModelLicenseEntry, context: int = 2048, train_tokens: int = 1_000_000) -> dict[str, Any]:
    """LoRA/QLoRA/full memory estimates from the existing order-of-magnitude estimator (not measured)."""
    params_b, note = parse_params_b(entry.parameter_count)
    out: dict[str, Any] = {"basis": "training.estimate_resources (order-of-magnitude, not measured)", "params_b": params_b,
                           "params_note": note, "context_tokens": context, "methods": {}}
    if params_b is None:
        return out
    for method in ("qlora", "lora", "full"):
        cfg = ExperimentConfig(base_model_id=entry.entry_id, method=method, base_params_b=params_b,
                               context_tokens=context, micro_batch_size=1)
        est = estimate_resources(cfg, train_tokens)
        out["methods"][method] = {"gpu_memory_gib": est.gpu_memory_gib, "trainable_params_m": est.trainable_params_m,
                                  "gpu_hours_at_default_throughput": est.gpu_hours, "train_tokens": train_tokens}
    return out


def android_feasibility(entry: BaseModelLicenseEntry, context: int = 4096) -> dict[str, Any]:
    a = entry.android_inference
    out: dict[str, Any] = {
        "state": a.state, "runtime_candidates": list(a.runtime_candidates), "evidence": list(a.evidence), "notes": a.notes,
        "claim": "feasible" if a.state == "evidenced" else ("infeasible" if a.state == "infeasible" else "unverified"),
        "kv_cache_mb_f16": None,
    }
    arch = entry.architecture
    if arch and None not in (arch.layers, arch.kv_heads, arch.head_dim):
        out["kv_cache_mb_f16"] = round(kv_cache_mb(arch.layers, arch.kv_heads, arch.head_dim, context), 1)
        out["kv_cache_context"] = context
    return out


def catalog_entry(entry: BaseModelLicenseEntry) -> dict[str, Any]:
    """Full catalog record for one model (all variants)."""
    v = entry.verification
    return {
        "id": entry.entry_id,
        "identity": {"family": entry.model_family, "version": entry.exact_version},
        "source": entry.hf_repo_or_source,
        "parameter_count": entry.parameter_count,
        "context_length": entry.context_length,
        "architecture": entry.architecture.model_dump() if entry.architecture else None,
        "variants": [x.model_dump() for x in entry.variants],
        "license": {
            "license_id": entry.license_id, "license_url": entry.license_url, "state": v.state,
            "evidence_level": v.evidence_level, "license_text_sha256": v.license_text_sha256,
            "verified_on": v.verified_on, "claims_are_binding": v.state == "VERIFIED",
            "recorded_claims": {
                "commercial_use": entry.commercial_use, "fine_tuning_permitted": entry.fine_tuning_permitted,
                "derivative_adapter_permitted": entry.derivative_adapter_permitted,
                "redistribution_permitted": entry.redistribution_permitted, "attribution_required": entry.attribution_required,
            },
            "uncertainties": list(v.uncertainties),
        },
        "supported_formats": list(entry.supported_formats),
        "android_inference": android_feasibility(entry),
        "training_feasibility": training_feasibility(entry),
        "provenance": {"license_source_urls": list(v.source_urls), "registry_schema_version": entry.schema_version},
    }


def catalog_rows(reg: LicenseRegistry) -> list[dict[str, Any]]:
    rows = []
    for eid in sorted(reg.entries):
        e = reg.entries[eid]
        rows.append({
            "id": eid, "parameter_count": e.parameter_count, "context_length": e.context_length, "license_id": e.license_id,
            "license_state": e.verification.state, "evidence_level": e.verification.evidence_level,
            "android": e.android_inference.state, "variants": [x.variant_id for x in e.variants],
        })
    return rows


@dataclass
class AcquisitionPlan:
    model_id: str
    variant_id: str | None
    format: str | None
    quantization: str | None
    source_url: str | None
    expected_size_bytes: int | None
    size_human: str
    large_download: bool
    license_state: str
    intended_use: dict[str, Any]
    gate: dict[str, Any]
    download_permitted: bool
    confirmation_required: bool
    blocked_codes: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    performs_download: bool = False  # this tool never transfers anything

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _pick_variant(entry: BaseModelLicenseEntry, variant_id: str | None) -> ModelVariant | None:
    if variant_id is None:
        if len(entry.variants) == 1:
            return entry.variants[0]
        if not entry.variants:
            return None
        raise KeyError("model has several variants; choose one of: " + ", ".join(v.variant_id for v in entry.variants))
    for v in entry.variants:
        if v.variant_id == variant_id:
            return v
    known = ", ".join(v.variant_id for v in entry.variants) or "(none recorded)"
    raise KeyError(f"unknown variant {variant_id!r} for {entry.entry_id}; known: {known}")


def plan_acquire(entry: BaseModelLicenseEntry, variant_id: str | None = None, use: RequestedUse | None = None) -> AcquisitionPlan:
    """Describe a download. Pure: no network, no files. Licence gate is evaluated for the intended use."""
    use = use or RequestedUse(fine_tune=True, produce_adapter=True)
    variant = _pick_variant(entry, variant_id)
    gate = evaluate_gate(entry, use)
    size = variant.size_bytes if variant else None
    large = size is None or size >= LARGE_DOWNLOAD_BYTES
    codes: list[str] = []
    msgs: list[str] = []
    if variant is None or not variant.source_url:
        codes.append("NO_SOURCE_URL")
        msgs.append(ACQUISITION_CODES["NO_SOURCE_URL"])
    if not gate.allowed:
        codes.append("LICENSE_BLOCKED")
        msgs.extend(gate.remediation or gate.blocking)
    if size is None:
        msgs.append("expected download size is unknown; treat as multi-GB")
    permitted = not codes
    if permitted and large:
        msgs.append(f"download is {_human_bytes(size)}: re-run with --confirm-download to authorize")
    return AcquisitionPlan(
        model_id=entry.entry_id,
        variant_id=variant.variant_id if variant else None,
        format=variant.format if variant else None,
        quantization=variant.quantization if variant else None,
        source_url=variant.source_url if variant else None,
        expected_size_bytes=size,
        size_human=_human_bytes(size),
        large_download=large,
        license_state=entry.verification.state,
        intended_use=use.as_dict(),
        gate={"allowed": gate.allowed, "reason_codes": gate.reason_codes, "remediation": gate.remediation},
        download_permitted=permitted,
        confirmation_required=True,  # every download needs explicit confirmation; large ones also print the size
        blocked_codes=codes,
        messages=msgs,
    )


def authorize_download(plan: AcquisitionPlan, confirm_download: bool) -> AcquisitionPlan:
    """Return the plan if a download may proceed, else raise AcquisitionRefused. Still transfers nothing."""
    why = list(plan.blocked_codes)
    if not confirm_download:
        why.append("CONFIRMATION_REQUIRED")
    if why:
        raise AcquisitionRefused(plan, why)
    return plan
