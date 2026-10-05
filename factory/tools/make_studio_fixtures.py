"""Regenerate the studio package fixtures under factory/tests/fixtures/studio/ (see README.md there).

    cd factory && python tools/make_studio_fixtures.py

Job zips are byte-reproducible. Results zips are produced by the real run-job code path (dry-run / stub / fake trainer) and are
validated structurally by tests; they embed run environment info, so they are not byte-reproducible across machines.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["LLMTRAINER_FIXED_NOW"] = "2026-10-05T12:00:00+00:00"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmtrainer.jobs.run import run_job  # noqa: E402
from llmtrainer.jobs.sample import build_job_dict, job_files, sample_license_evidence  # noqa: E402
from llmtrainer.jobs.zipio import checksums_bytes, write_zip  # noqa: E402
from llmtrainer.training import TrainerResult  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "studio"


class FakeTrainer:
    """Writes tiny placeholder bytes: fixtures only, NOT a trained model."""
    name, version, is_stub = "fixture-fake-trainer", "0", False

    def train(self, cfg, train_rows, val_rows, out_dir):
        (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
        (out_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
        a, c, t = out_dir / "adapter" / "adapter_model.safetensors", out_dir / "adapter" / "adapter_config.json", out_dir / "tokenizer" / "tokenizer.json"
        a.write_bytes(b"FIXTURE-PLACEHOLDER-NOT-A-REAL-ADAPTER")
        c.write_text('{"r": 16, "note": "fixture"}')
        t.write_text('{"note": "fixture"}')
        return TrainerResult(metrics={"train_loss": 0.0}, artifacts={"adapter/adapter_model.safetensors": a, "adapter/adapter_config.json": c,
                                                                     "tokenizer/tokenizer.json": t})


def sample_job_zip(path: Path, **kw) -> Path:
    kw.setdefault("license_evidence", sample_license_evidence())
    write_zip(path, job_files(build_job_dict(**kw)))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.zip"):
        old.unlink()
    job = sample_job_zip(OUT / "sample_job.zip")
    unattested = sample_job_zip(OUT / "sample_job_unattested.zip", license_evidence=None, job_id="job-sample-unattested")
    # negative fixture: a valid job whose chunks.jsonl was altered after checksums were computed
    files = job_files(build_job_dict(license_evidence=sample_license_evidence(), job_id="job-sample-badsum"))
    members = {**files, "checksums.json": checksums_bytes(files)}
    members["chunks.jsonl"] = members["chunks.jsonl"].replace(b"Tighten", b"Loosen", 1)
    import zipfile
    with zipfile.ZipFile(OUT / "sample_job_bad_checksum.zip", "w", zipfile.ZIP_STORED) as z:
        for n in sorted(members):
            zi = zipfile.ZipInfo(n, date_time=(2026, 10, 5, 0, 0, 0))
            zi.external_attr = 0o100644 << 16
            z.writestr(zi, members[n])
    tmp = Path(tempfile.mkdtemp())
    try:
        r = run_job(job, tmp / "planned")
        shutil.copy(r["results"]["results"], OUT / "sample_results_planned.zip")
        r = run_job(job, tmp / "stub", eval_backend="stub")
        shutil.copy(r["results"]["results"], OUT / "sample_results_stub.zip")
        r = run_job(unattested, tmp / "local", execute=True, allow_unverified_local=True, trainer=FakeTrainer(), eval_backend="stub")
        shutil.copy(r["results"]["results"], OUT / "sample_results_adapter_local_experiment.zip")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for f in sorted(OUT.glob("*.zip")):
        print(f.relative_to(OUT.parents[2]), f.stat().st_size)


if __name__ == "__main__":
    main()
