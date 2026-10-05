"""License gating for real specialization and for packaging.

Rules (deliberately stricter than "the file downloads"):

* Training requires the base-model entry to permit fine-tuning and derivative
  adapters (and commercial use when the project intends it) with recorded evidence
  -> state ``VERIFIED``.
* ``UNVERIFIED`` / ``CONDITIONAL`` can be run only as a *local experiment* with the explicit
  ``allow_unverified_license_for_local_experiment`` opt-in. That fact is recorded in the
  TrainingRun and permanently blocks packaging/export of anything derived from the run.
* An explicit ``no`` (``RESTRICTED``) is never overridable.
* Packaging additionally requires redistribution to be ``yes`` (state recomputed with
  ``redistribute=True``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..licenses import LicenseGateError, RequestedUse, evaluate_gate
from ..schemas import BaseModelLicenseEntry, GateResult, LicenseState


class PackagingRefused(RuntimeError):
    """Raised when an export must not be produced (license state or local-experiment run)."""


def classify_state(entry: BaseModelLicenseEntry, gate: GateResult) -> LicenseState:
    evidence = bool(entry.verification.source_urls and entry.verification.verified_on)
    if gate.allowed and evidence:
        return "VERIFIED"
    failing = [c.value for c in gate.checks if not c.passed and c.field != "supported_formats"]
    if "no" in failing:
        return "RESTRICTED"
    if "conditional" in failing:
        return "CONDITIONAL"
    return "UNVERIFIED"


@dataclass(frozen=True)
class TrainingLicenseDecision:
    gate: GateResult
    license_state: LicenseState
    local_experiment: bool
    warnings: list[str] = field(default_factory=list)


def decide_training_license(
    entry: BaseModelLicenseEntry, *, commercial: bool, allow_unverified_local: bool
) -> TrainingLicenseDecision:
    gate = evaluate_gate(entry, RequestedUse(fine_tune=True, produce_adapter=True, commercial=commercial))
    state = classify_state(entry, gate)
    if state == "VERIFIED":
        return TrainingLicenseDecision(gate, state, False)
    if state == "RESTRICTED" or not allow_unverified_local:
        raise LicenseGateError(gate)
    return TrainingLicenseDecision(
        gate,
        state,
        True,
        [
            f"LOCAL EXPERIMENT ONLY: base-model license state is {state}; this run can never be packaged or exported.",
        ],
    )


def redistribution_gate(entry: BaseModelLicenseEntry, *, commercial: bool = False) -> tuple[GateResult, LicenseState]:
    """Gate for packaging: fine-tune + adapter + redistribution (+ commercial if intended)."""
    gate = evaluate_gate(entry, RequestedUse(fine_tune=True, produce_adapter=True, commercial=commercial, redistribute=True))
    return gate, classify_state(entry, gate)
