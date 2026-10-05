"""The checked-in studio fixtures must stay valid and reproducible."""

import json
from pathlib import Path

import pytest

from llmtrainer.jobs.importer import import_job
from llmtrainer.jobs.results import validate_results_zip
from llmtrainer.jobs.sample import build_job_dict, job_files, sample_license_evidence
from llmtrainer.jobs.zipio import JobRejected, read_zip, write_zip

FIX = Path(__file__).parent / "fixtures" / "studio"


def test_job_fixtures_are_byte_reproducible(tmp_path):
    write_zip(tmp_path / "a.zip", job_files(build_job_dict(license_evidence=sample_license_evidence())))
    assert (tmp_path / "a.zip").read_bytes() == (FIX / "sample_job.zip").read_bytes()
    write_zip(tmp_path / "b.zip", job_files(build_job_dict(license_evidence=None, job_id="job-sample-unattested")))
    assert (tmp_path / "b.zip").read_bytes() == (FIX / "sample_job_unattested.zip").read_bytes()


def test_sample_job_imports_attested_and_unattested_stays_unverified(tmp_path):
    a = import_job(FIX / "sample_job.zip", tmp_path / "a")
    assert a.license["state"] == "VERIFIED" and a.license["owner_attested"] and a.examples == {"train": 72, "validation": 18, "test": 18}
    b = import_job(FIX / "sample_job_unattested.zip", tmp_path / "b")
    assert b.license["state"] == "UNVERIFIED" and not b.license["owner_attested"]


def test_bad_checksum_fixture_is_rejected(tmp_path):
    with pytest.raises(JobRejected, match="checksum mismatch for 'chunks.jsonl'"):
        import_job(FIX / "sample_job_bad_checksum.zip", tmp_path / "w")


@pytest.mark.parametrize("name,status,kind", [
    ("sample_results_planned", "planned_only", "none"),
    ("sample_results_stub", "stub", "none"),
    ("sample_results_adapter_local_experiment", "stub", "adapter"),
])
def test_results_fixtures_validate_and_never_claim_improvement(name, status, kind):
    p = FIX / f"{name}.zip"
    assert validate_results_zip(p) == []
    files = read_zip(p)
    m, r = json.loads(files["manifest.json"]), json.loads(files["evaluation_report.json"])
    assert (m["status"], m["specialist"]["kind"]) == (status, kind)
    assert r["improvement_claim_allowed"] is False and r["held_out_only"] is True and m["job_id"].startswith("job-sample")
