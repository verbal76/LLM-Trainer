"""Owner license attestation -> workspace registry entry.

Fail closed: an entry becomes VERIFIED only from a *complete* attestation (hash, url, fetched_at, all five attested
permissions, attested_by == owner, evidence level primary_license_text_read, scope note) whose model id matches the base
model. Everything else leaves the model UNVERIFIED. A DISALLOWED catalog entry is never overridden, and an entry the
catalog already VERIFIED is kept as is. The override is marked ``OWNER-ATTESTED`` inside the entry itself so the entry hash
(and therefore every later manifest/package that records ``license_entry_hash``) covers it.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..catalog import default_registry_dir
from ..licenses import LicenseRegistry, entry_hash
from ..schemas import BaseModelLicenseEntry, LicenseVerification
from .schema import JobBaseModel, LicenseEvidence

OWNER_ATTESTED_PREFIX = "OWNER-ATTESTED:"
PERMISSION_FIELDS = ("commercial_use", "fine_tuning_permitted", "derivative_adapter_permitted", "redistribution_permitted",
                     "attribution_required")
_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass
class LicenseOutcome:
    state: str  # VERIFIED | UNVERIFIED | DISALLOWED (verification.state of the workspace entry)
    owner_attested: bool
    entry: BaseModelLicenseEntry
    entry_hash: str
    reasons: list[str] = field(default_factory=list)  # why an attestation was not applied (empty when applied / absent)
    source: str = "catalog"  # catalog | owner_attestation | placeholder

    def as_dict(self) -> dict:
        return {"state": self.state, "owner_attested": self.owner_attested, "entry_id": self.entry.entry_id,
                "entry_hash": self.entry_hash, "source": self.source, "reasons": self.reasons}


def is_owner_attested(entry: BaseModelLicenseEntry) -> bool:
    v = entry.verification
    return v.state == "VERIFIED" and (v.verified_scope or "").startswith(OWNER_ATTESTED_PREFIX)


def attestation_problems(ev: LicenseEvidence | None, base: JobBaseModel, license_text: bytes | None = None) -> list[str]:
    """Why ``ev`` cannot make the entry VERIFIED. Empty list == complete and consistent."""
    if ev is None:
        return ["no license_evidence in the job: model stays UNVERIFIED"]
    p: list[str] = []
    if not ev.model_id:
        p.append("license_evidence.model_id missing")
    elif ev.model_id != base.registry_id:
        p.append(f"license_evidence.model_id {ev.model_id!r} != base_model.registry_id {base.registry_id!r}")
    if not ev.license_url or not ev.license_url.startswith("https://"):
        p.append("license_evidence.license_url missing or not https")
    try:
        _dt.datetime.fromisoformat(ev.fetched_at or "")
    except ValueError:
        p.append("license_evidence.fetched_at missing or not ISO-8601")
    if not ev.text_sha256 or not _SHA.match(ev.text_sha256):
        p.append("license_evidence.text_sha256 missing or not sha256:<hex>")
    if ev.evidence_level != "primary_license_text_read":
        p.append("license_evidence.evidence_level must be primary_license_text_read")
    if ev.attested_by != "owner":
        p.append("license_evidence.attested_by must be 'owner'")
    try:
        _dt.date.fromisoformat((ev.attested_on or "")[:10])
    except ValueError:
        p.append("license_evidence.attested_on missing or not an ISO date")
    perms = ev.permissions
    if not isinstance(perms, dict):
        p.append("license_evidence.permissions missing")
    else:
        for f in PERMISSION_FIELDS:
            if perms.get(f) not in ("yes", "no", "conditional"):
                p.append(f"license_evidence.permissions.{f} must be yes|no|conditional (got {perms.get(f)!r})")
        extra = set(perms) - set(PERMISSION_FIELDS)
        if extra:
            p.append(f"license_evidence.permissions has unknown keys {sorted(extra)}")
    if not (ev.scope_note or "").strip():
        p.append("license_evidence.scope_note missing")
    if license_text is not None and ev.text_sha256:
        from ..hashing import sha256_bytes

        if sha256_bytes(license_text) != ev.text_sha256:
            p.append("license/license_text.txt does not hash to license_evidence.text_sha256")
    return p


def _catalog_entry(registry_id: str, catalog_dir: Path | None) -> BaseModelLicenseEntry | None:
    d = catalog_dir or default_registry_dir()
    try:
        if not Path(d).exists():
            return None
        return LicenseRegistry.load(d).entries.get(registry_id)
    except (ValueError, OSError):
        return None


def _placeholder(base: JobBaseModel) -> BaseModelLicenseEntry:
    return BaseModelLicenseEntry(
        schema_version=2, model_family=base.family, exact_version=base.exact_version, license_id="unknown",
        license_url=base.source_url or "unknown", commercial_use="unverified", fine_tuning_permitted="unverified",
        derivative_adapter_permitted="unverified", redistribution_permitted="unverified", attribution_required="unverified",
        verification=LicenseVerification(state="UNVERIFIED", evidence_level="none",
                                        uncertainties=["Model is not in the license catalog; no license evidence was read."]),
        hf_repo_or_source=base.source_url,
    )


def resolve_license(base: JobBaseModel, ev: LicenseEvidence | None, license_text: bytes | None = None,
                    catalog_dir: Path | None = None) -> LicenseOutcome:
    cat = _catalog_entry(base.registry_id, catalog_dir)
    problems = attestation_problems(ev, base, license_text)
    if cat is not None and cat.verification.state == "DISALLOWED":
        return LicenseOutcome("DISALLOWED", False, cat, entry_hash(cat),
                              ["catalog entry is DISALLOWED; an owner attestation can never override it"], "catalog")
    if cat is not None and cat.verification.state == "VERIFIED":
        reasons = ["catalog entry is already VERIFIED; owner attestation not needed and not applied"] if ev is not None else []
        return LicenseOutcome("VERIFIED", False, cat, entry_hash(cat), reasons, "catalog")
    if problems:
        entry = cat if cat is not None else _placeholder(base)
        return LicenseOutcome(entry.verification.state, False, entry, entry_hash(entry), problems,
                              "catalog" if cat is not None else "placeholder")
    assert ev is not None and ev.permissions is not None
    base_entry = cat if cat is not None else _placeholder(base)
    scope = (f"{OWNER_ATTESTED_PREFIX} owner-reviewed license text from {ev.license_url}, hash {ev.text_sha256}; "
             f"permissions as attested by owner. {ev.scope_note}")
    notes = list(base_entry.verification.uncertainties)
    notes.append("Permissions are owner-attested, not independently interpreted by LLM Trainer; no legal review was performed.")
    if cat is not None and cat.verification.license_text_sha256 and cat.verification.license_text_sha256 != ev.text_sha256:
        notes.append("Attested license text hash differs from the hash recorded in the catalog entry.")
    verification = LicenseVerification(
        state="VERIFIED", evidence_level="primary_license_text_read",
        source_urls=[ev.license_url], verified_on=ev.attested_on[:10], license_text_sha256=ev.text_sha256,
        license_text_url=ev.license_url, verified_scope=scope, uncertainties=notes,
    )
    entry = BaseModelLicenseEntry.model_validate({
        **base_entry.model_dump(mode="json"),
        **{f: ev.permissions[f] for f in PERMISSION_FIELDS},
        "verification": verification.model_dump(mode="json"),
    })
    return LicenseOutcome("VERIFIED", True, entry, entry_hash(entry), [], "owner_attestation")
