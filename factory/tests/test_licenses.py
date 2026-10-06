import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmtrainer.licenses import LicenseGateError, LicenseRegistry, RequestedUse, enforce_gate, evaluate_gate, gate_sources_for_training
from llmtrainer.schemas import BaseModelLicenseEntry, RightsInfo, SourceManifest


VERIFIED_V = {
    "state": "VERIFIED", "evidence_level": "primary_license_text_read",
    "source_urls": ["https://example.invalid/license"], "verified_on": "2026-10-05",
    "license_text_sha256": "sha256:" + "ab" * 32, "verified_scope": "test text", "uncertainties": [],
}
UNVERIFIED_V = {"state": "UNVERIFIED", "evidence_level": "secondary_source_only", "source_urls": ["https://example.invalid/x"], "verified_on": "2026-10-05"}


def entry(**over) -> BaseModelLicenseEntry:
    d = dict(
        schema_version=2, model_family="fam", exact_version="1.0", license_id="TEST-1", license_url="https://example.invalid/license",
        commercial_use="yes", fine_tuning_permitted="yes", derivative_adapter_permitted="yes",
        redistribution_permitted="yes", attribution_required="yes", restrictions=["r"], supported_formats=["gguf"],
        verification=dict(VERIFIED_V),
    )
    d.update(over)
    return BaseModelLicenseEntry.model_validate(d)


FULL = RequestedUse(fine_tune=True, produce_adapter=True, commercial=True, redistribute=True, export_format="gguf")


def test_all_yes_allows():
    r = evaluate_gate(entry(), FULL)
    assert r.allowed and not r.blocking and r.entry_id == "fam@1.0"


@pytest.mark.parametrize("value", ["no", "conditional", "unverified"])
@pytest.mark.parametrize(
    "field,use",
    [
        ("fine_tuning_permitted", RequestedUse(fine_tune=True, produce_adapter=False)),
        ("derivative_adapter_permitted", RequestedUse(fine_tune=False, produce_adapter=True)),
        ("commercial_use", RequestedUse(fine_tune=False, produce_adapter=False, commercial=True)),
        ("redistribution_permitted", RequestedUse(fine_tune=False, produce_adapter=False, redistribute=True)),
    ],
)
def test_non_yes_blocks_requested_use(field, use, value):
    e = entry(**{field: value})
    r = evaluate_gate(e, use)
    assert not r.allowed
    with pytest.raises(LicenseGateError) as ei:
        enforce_gate(e, use)
    assert field in str(ei.value)


def test_unverified_is_blocked_with_clear_reason():
    r = evaluate_gate(entry(commercial_use="unverified"), RequestedUse(commercial=True))
    assert any("unverified is treated as blocked" in c.reason for c in r.checks)


def test_unrequested_permissions_do_not_block():
    e = entry(commercial_use="no", redistribution_permitted="no")
    assert evaluate_gate(e, RequestedUse(fine_tune=True, produce_adapter=True)).allowed


def test_unsupported_format_blocks():
    assert not evaluate_gate(entry(), RequestedUse(export_format="mlc")).allowed


def test_fully_unverified_entry_blocks_everything():
    e = entry(
        commercial_use="unverified", fine_tuning_permitted="unverified", derivative_adapter_permitted="unverified",
        redistribution_permitted="unverified", attribution_required="unverified",
        verification={"state": "UNVERIFIED", "evidence_level": "none"},
    )
    assert not evaluate_gate(e, RequestedUse()).allowed
    assert evaluate_gate(e, RequestedUse()).attribution_required == "unverified"


def test_permission_claims_need_source_urls():
    with pytest.raises(ValidationError):
        entry(verification={"state": "UNVERIFIED", "evidence_level": "none"})


def test_invalid_permission_value_rejected():
    with pytest.raises(ValidationError):
        entry(commercial_use="maybe")


def test_gate_result_hashes_entry():
    a, b = entry(), entry(restrictions=["different"])
    assert evaluate_gate(a, FULL).entry_hash != evaluate_gate(b, FULL).entry_hash


def test_registry_load_and_lookup(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(entry().model_dump(mode="json")))
    (tmp_path / "b.json").write_text(json.dumps([entry(exact_version="2.0").model_dump(mode="json")]))
    reg = LicenseRegistry.load(tmp_path)
    assert set(reg.entries) == {"fam@1.0", "fam@2.0"}
    with pytest.raises(KeyError):
        reg.get("missing@1")


def test_registry_conflicting_entry_rejected():
    reg = LicenseRegistry()
    reg.add(entry())
    reg.add(entry())  # identical is fine
    with pytest.raises(ValueError):
        reg.add(entry(commercial_use="no"))


def test_bundled_stub_entry_is_valid_and_blocks_commercial():
    p = Path(__file__).resolve().parents[1] / "llmtrainer" / "data" / "registry"
    reg = LicenseRegistry.load(p)
    stub = reg.get("pipeline-validation-stub@1")
    assert not evaluate_gate(stub, RequestedUse(commercial=True)).allowed
    assert evaluate_gate(stub, RequestedUse(redistribute=True, export_format="stub-json")).allowed


def test_source_rights_gate_only_yes_passes():
    from llmtrainer.ingest import ingest_bytes

    def src(text, perm):
        rec, _ = ingest_bytes(text.encode(), filename="f.md", title="t", origin="o", rights=RightsInfo(permitted_training=perm), ingested_on="2026-10-05")
        return rec

    m = SourceManifest(project_id="p", sources=[src("# A\n\nfirst source body text here", "yes"), src("# B\n\nsecond source body text", "unverified"), src("# C\n\nthird source body text", "conditional")])
    allowed, blocked = gate_sources_for_training(m)
    assert len(allowed) == 1 and len(blocked) == 2
