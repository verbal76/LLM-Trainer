"""Artifact registry: committed placeholders, invariants, the CI-script <-> loader contract, license gate, profiler specs."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from llmtrainer import device as dv
from llmtrainer.artifacts import DownloadableArtifact, REPO_ROOT, license_gate, load_artifacts, load_canonical

SCRIPTS = REPO_ROOT / "scripts" / "catalog"
sys.path.insert(0, str(SCRIPTS))
import refresh_catalog as rc  # noqa: E402

_spec = importlib.util.spec_from_file_location("refresh_catalog_tests", SCRIPTS / "tests" / "test_refresh_catalog.py")
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def committed():
    return load_artifacts()


def test_committed_registry_loads_and_is_entirely_unrefreshed_until_ci_runs():
    arts = committed()
    sources = json.loads((REPO_ROOT / "registry/artifacts/sources.json").read_text())
    assert {a.artifact_id for a in arts} == {f["id"] for r in sources["repos"] for f in r["files"]}
    for a in arts:
        if a.refresh_state == "unrefreshed":
            assert not a.downloadable and a.sha256 is None and a.size_bytes is None and a.source.revision is None
            assert a.license.state == "UNVERIFIED"
    wanted = {"smollm2-135m-instruct-q8_0", "smollm2-360m-instruct-q8_0", "qwen3-0.6b-q8_0", "qwen3-1.7b-q8_0", "qwen3-4b-q4_k_m", "qwen3-8b-q4_k_m"}
    assert wanted <= {a.artifact_id for a in arts}


def test_starter_set_training_classes():
    by = {a.artifact_id: a for a in committed()}
    for aid, a in by.items():
        if a.precision == "quantized":
            assert a.training.klass == "inference_only", aid
    assert by["smollm2-135m-instruct-f16"].training.klass == "local_full"
    assert by["smollm2-360m-instruct-f16"].training.klass == "local_full"
    assert by["qwen3-0.6b-f16"].training.klass == "local_partial"
    assert by["smollm2-360m-instruct-q8_0"].training.training_source_artifact_id == "smollm2-360m-instruct-f16"
    assert by["qwen3-0.6b-q8_0"].training.training_source_artifact_id == "qwen3-0.6b-f16"
    assert by["qwen3-8b-q4_k_m"].training.model_class_if_full_precision == "external_only"


def test_every_committed_artifact_converts_to_a_profiler_spec_and_is_classified_without_downloads():
    gd = importlib.util.spec_from_file_location("g", Path(__file__).resolve().parent.parent / "tools" / "gen_golden_device.py")
    g = importlib.util.module_from_spec(gd)
    gd.loader.exec_module(g)
    device = g.flagship12()
    specs = [a.to_spec() for a in committed()]
    rep = dv.choose_models(device, specs)
    arts = committed()
    for a, c in zip(arts, rep.capabilities):
        assert c.confidence_infer == "low"             # estimates are never "recommended" without a measurement
        if a.refresh_state != "refreshed":             # only a CI-refreshed (pinned, hashed) file may be offered for download
            assert not c.can_download and "not_downloadable" in c.reasons, a.artifact_id
    if all(a.refresh_state != "refreshed" for a in arts):
        assert all(ch.tier in ("preview", "none") for ch in rep.choices)


def test_class_rule_is_identical_in_the_ci_script_and_the_profiler():
    for p in (0.01, 0.135, 0.36, 0.449, 0.45, 0.451, 0.6, 1.0, 1.0001, 1.7, 8.0):
        assert rc.model_ceiling(p) == dv.full_precision_class_for(p), p


# ---- the CI script's output is accepted by the loader (cross-component contract) -----------------------------------


@pytest.fixture
def refreshed_dir(tmp_path):
    canon = rc.load_canon()
    hf = fx.world()
    rc.refresh(fx.mini_sources(), tmp_path, hf, canon, fx.NOW)
    return tmp_path


def test_script_output_loads_validates_and_is_downloadable(refreshed_dir):
    arts = {a.artifact_id: a for a in load_artifacts(refreshed_dir)}
    q8 = arts["tiny-q8_0"]
    assert q8.downloadable and q8.license.state == "VERIFIED" and q8.training.klass == "inference_only"
    assert q8.source.download_url == f"https://huggingface.co/Example/Tiny-GGUF/resolve/{fx.REV}/Tiny-Q8_0.gguf"
    assert not arts["tiny-f32"].downloadable and arts["tiny-f32"].published is False
    spec = q8.to_spec()
    assert spec.downloadable and spec.sha256 == q8.sha256 and spec.layers == 28 and spec.params == q8.parameter_count
    assert license_gate(q8) == (True, [])


def test_license_gate_blocks_unverified_and_unknown(refreshed_dir):
    arts = load_artifacts()
    ok, why = license_gate(arts[0])
    assert not ok and why[0].startswith("LICENSE_UNVERIFIED")


# ---- invariants: tampered files are rejected --------------------------------------------------------------------------


def doc(refreshed_dir, name="tiny-q8_0"):
    return json.loads((refreshed_dir / f"{name}.json").read_text())


def bad(d):
    with pytest.raises(Exception):
        DownloadableArtifact.model_validate(d)


def test_tampering_is_rejected(refreshed_dir):
    d = doc(refreshed_dir)
    DownloadableArtifact.model_validate(d)
    x = copy.deepcopy(d); x["license"]["evidence"]["match"] = "mismatch"; bad(x)                     # VERIFIED without a canonical match
    x = copy.deepcopy(d); x["license"]["evidence"]["model_card_agrees"] = False; bad(x)
    x = copy.deepcopy(d); x["license"]["evidence"]["license_text_sha256"] = "sha256:" + "cd" * 32; bad(x)   # exact match but different hash
    x = copy.deepcopy(d); x["license"]["spdx_id"] = "MIT"; bad(x)
    x = copy.deepcopy(d); x["sha256"] = "abc"; bad(x)
    x = copy.deepcopy(d); x["source"]["revision"] = "main"; bad(x)
    x = copy.deepcopy(d); x["source"]["download_url"] = d["source"]["download_url"].replace(d["source"]["revision"], "main"); bad(x)   # mutable URL
    x = copy.deepcopy(d); x["source"]["download_url"] = d["source"]["download_url"].replace("https://", "http://"); bad(x)
    x = copy.deepcopy(d); x["training"]["class"] = "local_full"; bad(x)                              # quantized weights are never trainable
    x = copy.deepcopy(d); x["unknown_field"] = 1; bad(x)


def test_unrefreshed_artifacts_may_not_carry_guessed_values(tmp_path):
    # the shipped files may already be CI-refreshed: generate the UNREFRESHED placeholder form with the script's own scaffold
    import subprocess, sys
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/catalog/refresh_catalog.py"), "--scaffold", "--out", str(tmp_path)],
                   check=True, capture_output=True, cwd=REPO_ROOT)
    d = json.loads((tmp_path / "qwen3-4b-q4_k_m.json").read_text())
    DownloadableArtifact.model_validate(d)
    for path, val in ((("sha256",), "sha256:" + "ab" * 32), (("size_bytes",), 2_500_000_000), (("source", "revision"), "0" * 40),
                      (("source", "download_url"), "https://huggingface.co/x/resolve/main/f.gguf"), (("parameter_count",), 4_000_000_000)):
        x = copy.deepcopy(d)
        t = x
        for k in path[:-1]:
            t = t[k]
        t[path[-1]] = val
        bad(x)
    x = copy.deepcopy(d); x["license"]["state"] = "VERIFIED"; bad(x)


def test_loader_cross_checks_verified_licenses_against_the_canonical_texts(refreshed_dir):
    p = refreshed_dir / "tiny-q8_0.json"
    d = json.loads(p.read_text())
    d["license"]["evidence"]["canonical_sha256"] = "sha256:" + "ee" * 32
    d["license"]["evidence"]["license_text_sha256"] = d["license"]["evidence"]["canonical_sha256"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="canonical text this repository does not hold"):
        load_artifacts(refreshed_dir)


def test_canonical_texts_are_pinned():
    c = load_canonical()
    assert c["Apache-2.0"]["sha256"] == "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    assert c["MIT"]["match_mode"] == "mit_copyright_line"
    assert all(v["source_url"].startswith("https://") for v in c.values())
