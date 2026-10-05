"""Kotlin producer -> Python consumer: the training-job zip emitted by studio-core must be accepted by `import-job`.

The zip is produced by the JVM test `SampleEmitTest` (`./gradlew :studio-core:test`), which writes
studio-core/build/sample/job.zip. Point LLMT_KOTLIN_JOB_ZIP at it to run this test; it is skipped otherwise so the Python
suite stays independent of a JVM build.

    ./gradlew --no-daemon :studio-core:test
    LLMT_KOTLIN_JOB_ZIP=studio-core/build/sample/job.zip pytest factory/tests/test_kotlin_job_interop.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmtrainer import pipeline as pl
from llmtrainer.jobs.importer import import_job
from llmtrainer.jobs.verify import verify_workspace_dataset

ENV = "LLMT_KOTLIN_JOB_ZIP"
pytestmark = pytest.mark.skipif(not os.environ.get(ENV) or not Path(os.environ.get(ENV, "")).is_file(),
                                reason=f"set {ENV} to a studio-core job zip (studio-core/build/sample/job.zip)")


def test_python_accepts_kotlin_job(tmp_path):
    z = Path(os.environ[ENV])
    rep = import_job(z, tmp_path / "ws")
    ws = pl.Workspace(tmp_path / "ws")
    ds = pl.latest_dataset(ws)
    assert ds.verify() and verify_workspace_dataset(ws, ds) == []      # Python re-ran leakage + split protection itself
    assert all(rep.examples.values()) and rep.heldout_items > 0
    assert rep.license["state"] == "VERIFIED" and rep.license["owner_attested"] is True
    assert rep.license["source"] == "owner_attestation"
    # the phone never claims more than the data supports: examples are template-built and labelled so
    for s in ds.splits.values():
        assert s.n_examples > 0
    assert ds.builder.type == "template"
    assert {e.origin for e in ds.examples} <= {"source_derived", "synthetic"}
    assert not any(e.origin == "synthetic" and e.split == "test" for e in ds.examples)


def test_kotlin_job_import_is_deterministic(tmp_path):
    z = Path(os.environ[ENV])
    a = import_job(z, tmp_path / "a")
    b = import_job(z, tmp_path / "b")
    assert a.dataset_hash == b.dataset_hash and a.job_content_hash == b.job_content_hash
