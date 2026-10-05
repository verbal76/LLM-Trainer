import json

import pytest
from pydantic import ValidationError

from llmtrainer import schema_export
from llmtrainer.hashing import canonical_json, hash_file, hash_obj, hash_text
from llmtrainer.schemas import (
    ARTIFACT_TYPES,
    BaseModelLicenseEntry,
    EvaluationRun,
    SourceManifest,
    SpecialistProject,
    SplitConfig,
    load_artifact,
)


def test_canonical_json_sorts_keys_and_is_compact():
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == b'{"a":[1,{"c":3,"d":2}],"b":1}'


def test_hash_is_key_order_independent_and_prefixed():
    a, b = {"x": 1, "y": [1, 2]}, {"y": [1, 2], "x": 1}
    assert hash_obj(a) == hash_obj(b)
    assert hash_obj(a).startswith("sha256:") and len(hash_obj(a)) == 71
    assert hash_obj({"x": 1, "y": [2, 1]}) != hash_obj(a)


def test_hash_known_vector():
    assert hash_text("") == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_nan_rejected():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_hash_file(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert hash_file(p) == hash_text("abc")


def test_unicode_stable():
    assert canonical_json({"k": "é"}) == '{"k":"é"}'.encode()


def _project():
    return SpecialistProject(project_id="p", name="P", domain="d", created_on="2026-10-05").seal()


def test_seal_verify_and_tamper_detection():
    p = _project()
    assert p.schema_version == 1 and p.verify()
    tampered = p.model_copy(update={"name": "Other"})
    assert not tampered.verify()
    assert tampered.seal().content_hash != p.content_hash


def test_identity_requires_seal():
    unsealed = SpecialistProject(project_id="p", name="P", domain="d", created_on="2026-10-05")
    with pytest.raises(ValueError):
        unsealed.identity
    assert not unsealed.verify()


def test_hash_deterministic_across_roundtrip():
    p = _project()
    again = SpecialistProject.model_validate_json(p.to_json())
    assert again.verify() and again.content_hash == p.content_hash


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        SpecialistProject(project_id="p", name="P", domain="d", created_on="x", bogus=1)


def test_schema_version_is_one_only():
    with pytest.raises(ValidationError):
        SpecialistProject(schema_version=2, project_id="p", name="P", domain="d", created_on="x")


def test_split_ratios_validated():
    with pytest.raises(ValidationError):
        SplitConfig(ratios={"train": 0.5, "validation": 0.2, "test": 0.2})
    with pytest.raises(ValidationError):
        SplitConfig(ratios={"train": 1.0, "validation": 0.0, "test": 0.0})


def test_load_artifact_dispatch_and_unknown_kind():
    assert isinstance(load_artifact(json.loads(_project().to_json())), SpecialistProject)
    with pytest.raises(ValueError):
        load_artifact({"kind": "nope"})


def test_required_artifacts_present():
    expected = {
        "specialist_project", "source_manifest", "dataset_manifest", "training_run", "evaluation_run",
        "model_manifest", "device_profile", "deployment_recommendation", "export_package",
    }
    assert expected <= set(ARTIFACT_TYPES)


def test_every_artifact_schema_has_hash_and_version_fields():
    for kind, schema in schema_export.generate().items():
        props = schema["properties"]
        if kind == "base_model_license_entry":
            continue
        assert {"schema_version", "kind", "content_hash"} <= set(props), kind


def test_checked_in_schemas_match_generated():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "schemas" / "v1"
    if not root.exists():
        pytest.skip("schemas/ directory not present")
    for kind, schema in schema_export.generate().items():
        f = root / f"{kind}.schema.json"
        assert f.exists(), f"missing {f}; run `python -m llmtrainer export-schemas ../schemas`"
        assert f.read_text() == schema_export.render(schema), f"{kind} schema is stale"


def test_evaluation_requires_base_and_specialist():
    from llmtrainer.schemas import EvalSubject, EvaluatorInfo

    h = "sha256:" + "0" * 64
    common = dict(
        eval_id="e", project_id="p", dataset_hash=h, n_eval_examples=1, train_eval_group_overlap=0,
        evaluator=EvaluatorInfo(name="x", version="1", is_stub=False), metrics=[], comparisons=[],
        improvement_claim_allowed=False, evaluated_on="2026-10-05",
    )
    with pytest.raises(ValidationError):
        EvaluationRun(subjects=[EvalSubject(role="specialist", model_ref="s")], **common)
    ok = EvaluationRun(subjects=[EvalSubject(role="base", model_ref="b"), EvalSubject(role="specialist", model_ref="s")], **common)
    assert ok.seal().verify()


def test_stub_evaluator_cannot_allow_claim():
    from llmtrainer.schemas import EvalSubject, EvaluatorInfo

    h = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        EvaluationRun(
            eval_id="e", project_id="p", dataset_hash=h, n_eval_examples=1, train_eval_group_overlap=0,
            evaluator=EvaluatorInfo(name="x", version="1", is_stub=True), metrics=[], comparisons=[],
            improvement_claim_allowed=True, evaluated_on="d",
            subjects=[EvalSubject(role="base", model_ref="b"), EvalSubject(role="specialist", model_ref="s")],
        )


def test_source_manifest_helpers():
    m = SourceManifest(project_id="p")
    assert m.active_sources() == []
    with pytest.raises(KeyError):
        m.get("nope")
