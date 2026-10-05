"""Base-model license registry and the license gate.

Policy: a requested use is allowed only if every required permission is
exactly ``"yes"``. ``"conditional"``, ``"no"`` and ``"unverified"`` all block;
unverified is treated as blocked, never as permissive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import hash_obj
from .schemas import BaseModelLicenseEntry, GateCheck, GateResult


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


def evaluate_gate(entry: BaseModelLicenseEntry, use: RequestedUse) -> GateResult:
    checks: list[GateCheck] = []

    def need(requirement: str, field_name: str) -> None:
        value = getattr(entry, field_name)
        passed = value == "yes"
        if passed:
            reason = "permitted"
        elif value == "unverified":
            reason = "unverified is treated as blocked"
        elif value == "conditional":
            reason = "conditional permission: conditions must be resolved and the entry updated to 'yes'"
        else:
            reason = "not permitted"
        checks.append(GateCheck(requirement=requirement, field=field_name, value=value, passed=passed, reason=reason))

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
            )
        )
    blocking = [f"{c.requirement}: {c.field}={c.value} ({c.reason})" for c in checks if not c.passed]
    return GateResult(
        entry_id=entry.entry_id,
        entry_hash=entry_hash(entry),
        requested_use=use.as_dict(),
        allowed=not blocking,
        checks=checks,
        blocking=blocking,
        attribution_required=entry.attribution_required,
        restrictions=list(entry.restrictions),
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
        """Load one JSON file or every ``*.json`` file in a directory (file may hold an object or a list)."""
        p = Path(path)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        reg = cls()
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else [data]:
                reg.add(BaseModelLicenseEntry.model_validate(item))
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
