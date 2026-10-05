"""Base-model license registry and the license gate.

Policy (fail closed): an entry whose ``verification.state`` is not VERIFIED
(UNVERIFIED or DISALLOWED) blocks every use and its recorded permission values
are ignored - they are claims, not facts. For a VERIFIED entry a requested use
is allowed only if every required permission is exactly ``"yes"``;
``"conditional"``, ``"no"`` and ``"unverified"`` all block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import hash_obj
from .schemas import BaseModelLicenseEntry, GateCheck, GateResult

# Machine-readable reason codes (stable API; also stored in GateResult.reason_codes).
LICENSE_UNVERIFIED = "LICENSE_UNVERIFIED"
LICENSE_DISALLOWED = "LICENSE_DISALLOWED"
LICENSE_EVIDENCE_INVALID = "LICENSE_EVIDENCE_INVALID"
PERMISSION_NOT_GRANTED = "PERMISSION_NOT_GRANTED"
PERMISSION_CONDITIONAL = "PERMISSION_CONDITIONAL"
PERMISSION_UNVERIFIED = "PERMISSION_UNVERIFIED"
FORMAT_NOT_SUPPORTED = "FORMAT_NOT_SUPPORTED"

EVIDENCE_REMEDIATION = (
    "Read the LICENSE (and model card) from the model's official repository, record the sha256 of the exact "
    "text inspected in verification.license_text_sha256, set verification.evidence_level to a primary_* level, "
    "fill source_urls/verified_on/verified_scope, then set verification.state to VERIFIED."
)


class LicenseGateError(RuntimeError):
    def __init__(self, result: GateResult):
        self.result = result
        super().__init__(
            f"license gate refused {result.entry_id}: " + "; ".join(result.blocking)
        )


@dataclass(frozen=True)
class RequestedUse:
    fine_tune: bool = True
    produce_adapter: bool = True
    commercial: bool = False
    redistribute: bool = False
    export_format: str | None = None

    def as_dict(self) -> dict:
        return {
            "fine_tune": self.fine_tune,
            "produce_adapter": self.produce_adapter,
            "commercial": self.commercial,
            "redistribute": self.redistribute,
            "export_format": self.export_format,
        }


def entry_hash(entry: BaseModelLicenseEntry) -> str:
    return hash_obj(entry.model_dump(mode="json"))


def _state_check(entry: BaseModelLicenseEntry) -> GateCheck | None:
    """Fail-closed check on verification state. Returns None only for a consistent VERIFIED entry."""
    v = entry.verification
    if v.state == "DISALLOWED":
        return GateCheck(
            requirement="license verification", field="verification.state", value=v.state, passed=False,
            code=LICENSE_DISALLOWED,
            reason=f"DISALLOWED: known incompatible with the requested use ({v.disallowed_reason or 'no reason recorded'}); "
            "recorded permission claims are ignored",
        )
    if v.state != "VERIFIED":
        return GateCheck(
            requirement="license verification", field="verification.state", value=v.state, passed=False,
            code=LICENSE_UNVERIFIED,
            reason=f"UNVERIFIED (evidence_level={v.evidence_level}): recorded permission claims are ignored. "
            f"To unblock: {EVIDENCE_REMEDIATION}",
        )
    problems = v.evidence_problems()  # defence in depth against entries built without validation
    if problems:
        return GateCheck(
            requirement="license verification", field="verification", value="VERIFIED", passed=False,
            code=LICENSE_EVIDENCE_INVALID,
            reason="state is VERIFIED but evidence is inconsistent: " + "; ".join(problems),
        )
    return None


def _permission_checks(entry: BaseModelLicenseEntry, use: RequestedUse) -> list[GateCheck]:
    checks: list[GateCheck] = []

    def need(requirement: str, field_name: str) -> None:
        value = getattr(entry, field_name)
        if value == "yes":
            code, reason = "OK", "permitted"
        elif value == "unverified":
            code, reason = PERMISSION_UNVERIFIED, "unverified is treated as blocked"
        elif value == "conditional":
            code, reason = PERMISSION_CONDITIONAL, "conditional permission: conditions must be resolved and the entry updated to 'yes'"
        else:
            code, reason = PERMISSION_NOT_GRANTED, "not permitted"
        checks.append(GateCheck(requirement=requirement, field=field_name, value=value, passed=value == "yes", reason=reason, code=code))

    if use.fine_tune:
        need("fine-tune the model", "fine_tuning_permitted")
    if use.produce_adapter:
        need("create derivative adapter/weights", "derivative_adapter_permitted")
    if use.commercial:
        need("commercial use", "commercial_use")
    if use.redistribute:
        need("redistribute model or derivative", "redistribution_permitted")
    if use.export_format is not None:
        ok = use.export_format in entry.supported_formats
        checks.append(
            GateCheck(
                requirement=f"export as {use.export_format}",
                field="supported_formats",
                value=use.export_format,
                passed=ok,
                reason="listed" if ok else "format not listed in supported_formats",
                code="OK" if ok else FORMAT_NOT_SUPPORTED,
            )
        )
    return checks


def evaluate_gate(entry: BaseModelLicenseEntry, use: RequestedUse) -> GateResult:
    """Fail-closed license gate. Permission claims count only when verification.state == VERIFIED."""
    state_check = _state_check(entry)
    checks = [state_check] if state_check is not None else _permission_checks(entry, use)
    blocking = [f"[{c.code}] {c.requirement}: {c.field}={c.value} ({c.reason})" for c in checks if not c.passed]
    codes = sorted({c.code for c in checks if not c.passed})
    remediation: list[str] = []
    if LICENSE_UNVERIFIED in codes or LICENSE_EVIDENCE_INVALID in codes:
        remediation.append(EVIDENCE_REMEDIATION)
    if LICENSE_DISALLOWED in codes:
        remediation.append("Choose a different base model or a different intended use; this entry is recorded as incompatible.")
    if PERMISSION_CONDITIONAL in codes:
        remediation.append("Resolve the license conditions with a human reviewer, then record the outcome as 'yes' with evidence.")
    if PERMISSION_NOT_GRANTED in codes or PERMISSION_UNVERIFIED in codes:
        remediation.append("The verified license does not grant the requested use; drop that use or pick another model.")
    if FORMAT_NOT_SUPPORTED in codes:
        remediation.append("Export in a listed format or update supported_formats with evidence.")
    return GateResult(
        entry_id=entry.entry_id,
        entry_hash=entry_hash(entry),
        requested_use=use.as_dict(),
        allowed=not blocking,
        checks=checks,
        blocking=blocking,
        attribution_required=entry.attribution_required if state_check is None else "unverified",
        restrictions=list(entry.restrictions),
        verification_state=entry.verification.state,
        reason_codes=codes,
        remediation=remediation,
    )


def enforce_gate(entry: BaseModelLicenseEntry, use: RequestedUse) -> GateResult:
    result = evaluate_gate(entry, use)
    if not result.allowed:
        raise LicenseGateError(result)
    return result


@dataclass
class LicenseRegistry:
    entries: dict[str, BaseModelLicenseEntry] = field(default_factory=dict)

    def add(self, entry: BaseModelLicenseEntry) -> None:
        if entry.entry_id in self.entries and self.entries[entry.entry_id] != entry:
            raise ValueError(f"conflicting registry entry for {entry.entry_id}")
        self.entries[entry.entry_id] = entry

    def get(self, entry_id: str) -> BaseModelLicenseEntry:
        try:
            return self.entries[entry_id]
        except KeyError:
            raise KeyError(f"base model {entry_id!r} is not in the license registry") from None

    @classmethod
    def load(cls, path: str | Path) -> "LicenseRegistry":
        """Load one JSON file or every ``*.json`` file in a directory (file may hold an object or a list).

        Entries must be schema_version 2; v1 entries are rejected, never silently upgraded.
        """
        p = Path(path)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        reg = cls()
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else [data]:
                try:
                    reg.add(BaseModelLicenseEntry.model_validate(item))
                except ValueError as e:  # includes pydantic ValidationError; v1 entries land here
                    raise ValueError(f"{f.name}: invalid base-model entry: {e}") from e
        return reg


def gate_sources_for_training(manifest) -> tuple[list[str], list[str]]:
    """Return (allowed_source_ids, blocked reasons). Only permitted_training == 'yes' is allowed."""
    allowed, blocked = [], []
    for s in manifest.active_sources():
        if s.rights.permitted_training == "yes":
            allowed.append(s.source_id)
        else:
            blocked.append(
                f"{s.source_id}: permitted_training={s.rights.permitted_training} (status={s.rights.status})"
            )
    return allowed, blocked
