"""Fail-closed verification semantics (entry schema v2) and the shipped registry."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmtrainer import cli
from llmtrainer.hashing import sha256_bytes
from llmtrainer.licenses import (
    LICENSE_DISALLOWED,
    LICENSE_EVIDENCE_INVALID,
    LICENSE_UNVERIFIED,
    LicenseGateError,
    LicenseRegistry,
    RequestedUse,
    enforce_gate,
    evaluate_gate,
)
from llmtrainer.schemas import BaseModelLicenseEntry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "registry" / "base-models"
STUB_DIR = Path(cli.__file__).parent / "data" / "registry"

GOOD_V = {
    "state": "VERIFIED", "evidence_level": "primary_license_text_read",
    "source_urls": ["https://example.invalid/LICENSE"], "verified_on": "2026-10-05",
    "license_text_sha256": "sha256:" + "cd" * 32, "verified_scope": "unit test",
}


def mk(verification=None, **over) -> dict:
    d = dict(
        schema_version=2, model_family="m", exact_version="1", license_id="X", license_url="u",
        commercial_use="yes", fine_tuning_permitted="yes", derivative_adapter_permitted="yes",
        redistribution_permitted="yes", attribution_required="yes", supported_formats=["gguf"],
        verification=verification if verification is not None else dict(GOOD_V),
    )
    d.update(over)
    return d


def ent(verification=None, **over) -> BaseModelLicenseEntry:
    return BaseModelLicenseEntry.model_validate(mk(verification, **over))


ANY_USE = RequestedUse(fine_tune=True, produce_adapter=True, commercial=True, redistribute=True, export_format="gguf")


# ---- validator ------------------------------------------------------------

@pytest.mark.parametrize("missing", ["license_text_sha256", "verified_on", "source_urls", "verified_scope"])
def test_verified_requires_each_evidence_item(missing):
    v = dict(GOOD_V)
    v[missing] = [] if missing == "source_urls" else None
    with pytest.raises(ValidationError, match="VERIFIED requires"):
        ent(v)


@pytest.mark.parametrize("level", ["secondary_source_only", "none"])
def test_verified_rejects_non_primary_evidence(level):
    with pytest.raises(ValidationError, match="primary evidence"):
        ent({**GOOD_V, "evidence_level": level})


def test_primary_model_card_read_is_accepted():
    assert ent({**GOOD_V, "evidence_level": "primary_model_card_read"}).verification.state == "VERIFIED"


@pytest.mark.parametrize("level", ["secondary_source_only", "none"])
def test_disallowed_needs_primary_evidence_and_reason(level):
    with pytest.raises(ValidationError):
        ent({"state": "DISALLOWED", "evidence_level": level, "disallowed_reason": "r", "source_urls": ["u"]})
    with pytest.raises(ValidationError, match="disallowed_reason"):
        ent({"state": "DISALLOWED", "evidence_level": "primary_license_text_read", "source_urls": ["u"]})


def test_bad_hash_format_rejected():
    with pytest.raises(ValidationError):
        ent({**GOOD_V, "license_text_sha256": "abc"})


# ---- gate -----------------------------------------------------------------

def test_verified_entry_passes_when_claims_yes():
    r = evaluate_gate(ent(), ANY_USE)
    assert r.allowed and r.verification_state == "VERIFIED" and r.reason_codes == []


@pytest.mark.parametrize("level", ["secondary_source_only", "none"])
def test_unverified_blocks_even_with_all_yes_claims(level):
    e = ent({"state": "UNVERIFIED", "evidence_level": level, "source_urls": ["https://x"]})
    r = evaluate_gate(e, ANY_USE)
    assert not r.allowed
    assert r.reason_codes == [LICENSE_UNVERIFIED]
    assert r.attribution_required == "unverified"
    assert any("sha256" in m and "LICENSE" in m for m in r.remediation)
    with pytest.raises(LicenseGateError, match="LICENSE_UNVERIFIED"):
        enforce_gate(e, RequestedUse())  # even a minimal use


def test_unverified_blocks_a_no_use_request_too():
    e = ent({"state": "UNVERIFIED", "evidence_level": "none"}, commercial_use="unverified", fine_tuning_permitted="unverified",
            derivative_adapter_permitted="unverified", redistribution_permitted="unverified", attribution_required="unverified")
    assert not evaluate_gate(e, RequestedUse(fine_tune=False, produce_adapter=False)).allowed


def test_disallowed_blocks_with_its_own_code_and_reason():
    e = ent({"state": "DISALLOWED", "evidence_level": "primary_license_text_read", "source_urls": ["https://x"],
             "disallowed_reason": "non-commercial licence"})
    r = evaluate_gate(e, ANY_USE)
    assert not r.allowed and r.reason_codes == [LICENSE_DISALLOWED]
    assert "non-commercial licence" in r.blocking[0]


def test_forged_verified_without_evidence_is_caught_by_gate():
    e = ent().model_copy(deep=True)
    e.verification.license_text_sha256 = None  # bypasses validators (assignment is unvalidated)
    r = evaluate_gate(e, ANY_USE)
    assert not r.allowed and r.reason_codes == [LICENSE_EVIDENCE_INVALID]


def test_gate_result_is_machine_readable():
    r = evaluate_gate(ent({"state": "UNVERIFIED", "evidence_level": "none"}, commercial_use="unverified",
                          fine_tuning_permitted="unverified", derivative_adapter_permitted="unverified",
                          redistribution_permitted="unverified", attribution_required="unverified"), ANY_USE)
    d = r.model_dump(mode="json")
    assert d["reason_codes"] == ["LICENSE_UNVERIFIED"] and d["checks"][0]["code"] == "LICENSE_UNVERIFIED"


# ---- loader / schema version ---------------------------------------------

def test_v1_entry_rejected_not_upgraded(tmp_path):
    d = mk()
    d["schema_version"] = 1
    with pytest.raises(ValidationError, match="not supported"):
        BaseModelLicenseEntry.model_validate(d)
    d.pop("schema_version")  # unversioned == v1
    with pytest.raises(ValidationError, match="not supported"):
        BaseModelLicenseEntry.model_validate(d)
    (tmp_path / "old.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="old.json"):
        LicenseRegistry.load(tmp_path)


def test_entry_without_verification_state_rejected():
    d = mk()
    d["verification"] = {"source_urls": ["u"], "verified_on": "2026-10-05"}
    with pytest.raises(ValidationError):
        BaseModelLicenseEntry.model_validate(d)


# ---- shipped registry -----------------------------------------------------

def test_shipped_registry_loads_and_has_no_unearned_verified():
    reg = LicenseRegistry.load(REGISTRY)
    assert len(reg.entries) == 22
    for eid, e in reg.entries.items():
        v = e.verification
        if v.state == "VERIFIED":
            assert v.license_text_sha256 and v.evidence_level.startswith("primary_"), eid
        else:
            assert v.state == "UNVERIFIED" and v.evidence_level in ("secondary_source_only", "none"), eid


def test_shipped_apache_claims_do_not_unblock_anything():
    reg = LicenseRegistry.load(REGISTRY)
    q = next(e for e in reg.entries.values() if e.model_family == "Qwen3" and e.exact_version.startswith("Qwen3-8B"))
    assert q.commercial_use == "yes"  # the claim is retained...
    r = evaluate_gate(q, RequestedUse(redistribute=True))
    assert not r.allowed and r.reason_codes == [LICENSE_UNVERIFIED]  # ...but ignored


def test_shipped_llama_is_verified_but_conditional_still_blocks():
    reg = LicenseRegistry.load(REGISTRY)
    llamas = [e for e in reg.entries.values() if e.model_family.startswith("Llama")]
    assert len(llamas) == 2
    for e in llamas:
        assert e.verification.state == "VERIFIED"
        r = evaluate_gate(e, RequestedUse(commercial=True))
        assert not r.allowed and "PERMISSION_CONDITIONAL" in r.reason_codes


def test_stub_license_hash_matches_file_on_disk():
    reg = LicenseRegistry.load(STUB_DIR)
    stub = reg.get("pipeline-validation-stub@1")
    text = (STUB_DIR / "pipeline-validation-stub.LICENSE.txt").read_bytes()
    assert stub.verification.license_text_sha256 == sha256_bytes(text)


def test_cli_check_license_fails_closed_with_codes(capsys):
    eid = next(i for i in LicenseRegistry.load(REGISTRY).entries if i.startswith("Qwen3@Qwen3-8B"))
    assert cli.main(["check-license", str(REGISTRY), eid]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["allowed"] is False and out["reason_codes"] == ["LICENSE_UNVERIFIED"]
