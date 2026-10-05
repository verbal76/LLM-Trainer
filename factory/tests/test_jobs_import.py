"""import-job: format validation, hostile inputs, leakage re-verification, license attestation."""

import copy
import json
import zipfile
from pathlib import Path

import pytest
from jobs_helpers import (OWNED_RIGHTS, chunk_text_of, first_test_ref, make_job, other_model, write_job, write_members)

from llmtrainer import cli
from llmtrainer import pipeline as pl
from llmtrainer.hashing import hash_text
from llmtrainer.jobs.importer import import_job
from llmtrainer.jobs.license_override import OWNER_ATTESTED_PREFIX, is_owner_attested
from llmtrainer.jobs.sample import job_files, sample_license_evidence
from llmtrainer.jobs.verify import verify_workspace_dataset
from llmtrainer.jobs.zipio import JobRejected, ZipLimits, checksums_bytes, write_zip
from llmtrainer.licenses import LicenseGateError, RequestedUse, enforce_gate, evaluate_gate
from llmtrainer.schemas import BaseModelLicenseEntry


def rejected(zip_path, tmp_path, **kw):
    ws = tmp_path / "ws"
    with pytest.raises(JobRejected) as ei:
        import_job(zip_path, ws, **kw)
    assert not ws.exists(), "nothing may be written on rejection"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".ws")], "no temp leftovers"
    return "\n".join(ei.value.problems)


# ------------------------------------------------------------------ happy path

def test_import_builds_workspace_and_reverifies(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    rep = import_job(z, tmp_path / "ws")
    ws = pl.Workspace(tmp_path / "ws")
    project = ws.project()
    assert project.project_id == "proj-pump-service" and project.verify()
    sm = ws.sources()
    assert sm.verify() and len(sm.active_sources()) == 6
    ds = pl.latest_dataset(ws)
    assert ds.verify() and ds.dataset_id == rep.dataset_id
    assert {s: ds.splits[s].n_examples for s in ds.splits} == rep.examples and all(rep.examples.values())
    assert verify_workspace_dataset(ws, ds) == []
    assert rep.heldout_items > 0 and (tmp_path / "ws" / "evals" / "heldout.jsonl").is_file()
    assert (tmp_path / "ws" / "job" / "job_state.json").is_file() and (tmp_path / "ws" / "job" / "import_report.json").is_file()
    # corpus rows are readable by the existing corpus tooling
    from llmtrainer import corpus
    assert corpus.inspect_corpus(ws)["totals"]["train"] == rep.chunks["train"]
    # training data loader of the existing pipeline accepts it
    from llmtrainer.specialize.data import load_training_data
    tr, va = load_training_data(ws.root / "datasets", ds)
    assert len(tr) == rep.examples["train"] and len(va) == rep.examples["validation"]


def test_import_is_deterministic(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    a = import_job(z, tmp_path / "a")
    b = import_job(z, tmp_path / "b")
    assert (a.dataset_hash, a.job_content_hash, a.license["entry_hash"]) == (b.dataset_hash, b.job_content_hash, b.license["entry_hash"])


def test_refuses_nonempty_workspace_and_reuses_same_job(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    import_job(z, tmp_path / "ws")
    with pytest.raises(JobRejected, match="not empty"):
        import_job(z, tmp_path / "ws")
    again = import_job(z, tmp_path / "ws", reuse_existing=True)
    assert again.reused
    other = write_job(tmp_path / "k.zip", make_job(job_id="job-other", n_docs=7))
    with pytest.raises(JobRejected, match="not empty"):
        import_job(other, tmp_path / "ws", reuse_existing=True)


def test_empty_existing_directory_is_accepted(tmp_path):
    (tmp_path / "ws").mkdir()
    import_job(write_job(tmp_path / "j.zip", make_job()), tmp_path / "ws")
    assert (tmp_path / "ws" / "project.json").is_file()


# ------------------------------------------------------------------ zip hygiene / hostile packages

def test_bad_checksum_rejected(tmp_path):
    files = job_files(make_job())
    members = dict(files)
    members["checksums.json"] = checksums_bytes(files)
    members["chunks.jsonl"] = members["chunks.jsonl"].replace(b"Tighten", b"Loosen", 1)
    msg = rejected(write_members(tmp_path / "j.zip", members, checksums="as-given"), tmp_path)
    assert "checksum mismatch for 'chunks.jsonl'" in msg


def test_missing_and_extra_checksum_entries_rejected(tmp_path):
    files = job_files(make_job())
    assert "missing checksums.json" in rejected(write_members(tmp_path / "a.zip", files, checksums="none"), tmp_path)
    members = {**files, "extra.txt": b"x"}
    members["checksums.json"] = checksums_bytes(files)
    assert "not listed" in rejected(write_members(tmp_path / "b.zip", members, checksums="as-given"), tmp_path)
    ghost = checksums_bytes({**files, "ghost.txt": b"y"})
    members = {**files, "checksums.json": ghost}
    assert "not in the zip" in rejected(write_members(tmp_path / "c.zip", members, checksums="as-given"), tmp_path)


def test_unlisted_member_with_valid_checksum_is_still_rejected(tmp_path):
    files = {**job_files(make_job()), "extra.txt": b"x"}
    assert "unexpected members" in rejected(write_members(tmp_path / "a.zip", files), tmp_path)


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt", "dataset\\train.jsonl", "C:/evil.txt", "a//b.txt", "./x.txt"])
def test_path_traversal_names_rejected(tmp_path, name):
    members = {**job_files(make_job()), name: b"pwn"}
    msg = rejected(write_members(tmp_path / "t.zip", members), tmp_path)
    assert "unsafe member name" in msg
    assert not (tmp_path.parent / "evil.txt").exists()


def test_symlink_member_rejected(tmp_path):
    p = tmp_path / "s.zip"
    files = job_files(make_job())
    with zipfile.ZipFile(p, "w") as z:
        for n, b in {**files, "checksums.json": checksums_bytes(files)}.items():
            z.writestr(n, b)
        zi = zipfile.ZipInfo("license/link")
        zi.external_attr = 0o120777 << 16
        z.writestr(zi, "/etc/passwd")
    assert "not a regular file" in rejected(p, tmp_path)


def test_duplicate_member_rejected(tmp_path):
    p = tmp_path / "d.zip"
    files = job_files(make_job())
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(p, "w") as z:
            for n, b in {**files, "checksums.json": checksums_bytes(files)}.items():
                z.writestr(n, b)
            z.writestr("manifest.json", b"{}")
    assert "duplicate member" in rejected(p, tmp_path)


def test_oversized_rejected(tmp_path):
    z = write_job(tmp_path / "j.zip", make_job())
    assert "over the" in rejected(z, tmp_path, limits=ZipLimits(max_member_bytes=1000))
    assert "over the" in rejected(z, tmp_path, limits=ZipLimits(max_total_bytes=1000))
    assert "over the" in rejected(z, tmp_path, limits=ZipLimits(max_zip_bytes=1000))
    assert "members, over" in rejected(z, tmp_path, limits=ZipLimits(max_members=3))


def test_compression_bomb_rejected(tmp_path):
    files = job_files(make_job())
    files["license/license_text.txt"] = b"A" * (3 << 20)  # 3 MiB of zeros-like data compresses > 200:1
    p = tmp_path / "bomb.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in {**files, "checksums.json": checksums_bytes(files)}.items():
            z.writestr(n, b)
    assert "compression ratio" in rejected(p, tmp_path)


def test_not_a_zip_and_missing_file(tmp_path):
    (tmp_path / "x.zip").write_bytes(b"not a zip")
    assert "not a valid zip" in rejected(tmp_path / "x.zip", tmp_path)
    assert "not a file" in rejected(tmp_path / "nope.zip", tmp_path)


# ------------------------------------------------------------------ schema

def mutate_manifest(tmp_path, fn, name="m.zip"):
    job = make_job()
    fn(job["manifest"])
    return write_job(tmp_path / name, job)


@pytest.mark.parametrize("fn,needle", [
    (lambda m: m.update(version=2), "unsupported format/version"),
    (lambda m: m.update(format="something-else"), "unsupported format/version"),
    (lambda m: m.update(bogus=1), "bogus"),
    (lambda m: m["project"].update(id="../x"), "project.id"),
    (lambda m: m["base_model"].update(registry_id="other@thing"), "registry_id"),
    (lambda m: m["sources"][0]["rights"].update(permitted_training="maybe"), "permitted_training"),
    (lambda m: m["sources"][0].update(sha256="abc"), "sha256"),
    (lambda m: m["sources"].append(copy.deepcopy(m["sources"][0])), "duplicate source ids"),
    (lambda m: m.update(created_at="yesterday"), "ISO-8601"),
])
def test_manifest_schema_violations(tmp_path, fn, needle):
    assert needle in rejected(mutate_manifest(tmp_path, fn), tmp_path)


def test_chunk_hash_mismatch_and_unknown_source(tmp_path):
    job = make_job()
    job["chunks"][0]["text"] += " tampered"
    assert "sha256 does not match text" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)
    job = make_job()
    job["chunks"][0]["source_id"] = "src-unknown"
    assert "unknown source_id" in rejected(write_job(tmp_path / "b.zip", job), tmp_path)


def test_dataset_manifest_inconsistencies(tmp_path):
    job = make_job()
    job["dataset_manifest"]["counts"]["train"] += 1
    assert "counts" in rejected(write_job(tmp_path / "a.zip", job, fix_counts=False), tmp_path)
    job = make_job()
    job["dataset_manifest"]["split_config"]["ratios"] = {"train": 1.0, "validation": 0.0, "test": 0.0}
    assert "split_config" in rejected(write_job(tmp_path / "b.zip", job), tmp_path)


# ------------------------------------------------------------------ leakage injected (Python is authoritative)

def test_example_in_wrong_split_file_rejected(tmp_path):
    job = make_job()
    job["examples"]["test"].append(job["examples"]["train"].pop())
    assert "assigned to train" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)


def test_group_assigned_to_two_splits_via_relabelled_example(tmp_path):
    job = make_job()
    # a train-group example relabelled with the test group id and placed in test.jsonl: assignment now consistent,
    # but its chunk belongs to a train document, so the chunk-level checks must still catch it
    ex = copy.deepcopy(job["examples"]["train"].pop())
    ex["group_id"] = job["examples"]["test"][0]["group_id"]
    job["examples"]["test"].append(ex)
    msg = rejected(write_job(tmp_path / "a.zip", job), tmp_path)
    assert "does not match its chunk's group" in msg


def test_test_chunk_text_copied_into_train_example_rejected(tmp_path):
    job = make_job()
    sid, cid = first_test_ref(job)
    text = chunk_text_of(job, sid, cid)
    tr = job["examples"]["train"][0]
    tr["prompt"], tr["response"] = text[: len(text) // 2], text[len(text) // 2:]
    msg = rejected(write_job(tmp_path / "a.zip", job), tmp_path)
    assert "contained in test chunk" in msg or "near-duplicate" in msg


def test_train_example_deriving_from_test_chunk_rejected(tmp_path):
    job = make_job()
    sid, cid = first_test_ref(job)
    job["examples"]["train"][0]["derived_from"] = [{"source_id": sid, "chunk_id": cid}]
    msg = rejected(write_job(tmp_path / "a.zip", job), tmp_path)
    assert "does not match its chunk's group" in msg


def test_near_duplicate_test_example_in_train_rejected(tmp_path):
    job = make_job()
    t = job["examples"]["test"][0]
    dup = copy.deepcopy(job["examples"]["train"][0])
    dup["example_id"] = "ex-injected"
    dup["prompt"], dup["response"] = t["prompt"], t["response"]
    job["examples"]["train"].append(dup)
    assert "near-duplicate across splits" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)


def test_synthetic_example_in_test_rejected(tmp_path):
    job = make_job()
    job["examples"]["test"][0]["origin"] = "synthetic"
    assert "synthetic example in the test split" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)


def test_synthetic_chunk_in_test_rejected(tmp_path):
    job = make_job()
    sid, cid = first_test_ref(job)
    for c in job["chunks"]:
        if (c["source_id"], c["id"]) == (sid, cid):
            c["origin"] = "synthetic"
    assert "synthetic chunk" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)


def test_reference_or_excluded_chunk_cannot_back_examples(tmp_path):
    job = make_job()
    sid, cid = job["examples"]["train"][0]["derived_from"][0].values()
    for c in job["chunks"]:
        if (c["source_id"], c["id"]) == (sid, cid):
            c["role"] = "reference"
    assert "only train chunks may feed examples" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)


def test_heldout_item_on_train_chunk_rejected(tmp_path):
    job = make_job()
    tr = job["examples"]["train"][0]["derived_from"][0]
    job["heldout"][0]["gold_refs"] = [f"{tr['source_id']}/{tr['chunk_id']}"]
    msg = rejected(write_job(tmp_path / "a.zip", job), tmp_path)
    assert "not a test-split chunk" in msg


def test_heldout_item_malformed_rejected(tmp_path):
    job = make_job()
    job["heldout"][0]["gold_refs"] = ["src-sample-00/c9999"]
    assert "unknown chunk" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)
    job = make_job()
    job["heldout"][0].update(kind="fact", expected=None)
    assert "held-out items invalid" in rejected(write_job(tmp_path / "b.zip", job), tmp_path)
    job = make_job()
    job["heldout"][0]["origin"] = "synthetic"
    assert "held-out items invalid" in rejected(write_job(tmp_path / "c.zip", job), tmp_path)


def test_heldout_content_overlapping_training_text_rejected(tmp_path):
    job = make_job()
    # make a TRAIN example that is verbatim the held-out item's gold chunk text (in a train group, so only the item guard sees it)
    ref = job["heldout"][0]["gold_refs"][0]
    sid, cid = ref.split("/")
    text = chunk_text_of(job, sid, cid)
    job["examples"]["train"][0]["prompt"] = "Q"
    job["examples"]["train"][0]["response"] = text
    msg = rejected(write_job(tmp_path / "a.zip", job), tmp_path)
    assert "contained" in msg or "overlaps" in msg or "near-duplicate" in msg


def test_untrainable_source_feeding_training_rejected(tmp_path):
    job = make_job()
    sid = job["examples"]["train"][0]["group_id"]
    next(s for s in job["manifest"]["sources"] if s["id"] == sid)["rights"]["permitted_training"] = "no"
    assert "not cleared for training" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)


def test_too_few_groups_cannot_form_splits(tmp_path):
    job = make_job()
    job["examples"]["validation"].clear()
    job["dataset_manifest"]["counts"]["validation"] = 0
    assert "validation split is empty" in rejected(write_job(tmp_path / "a.zip", job), tmp_path)


# ------------------------------------------------------------------ license attestation

def entry_of(ws):
    return next(e for e in ws.registry().entries.values() if e.model_family == "Qwen3")


def test_complete_attestation_becomes_owner_attested_verified(tmp_path):
    rep = import_job(write_job(tmp_path / "j.zip", make_job()), tmp_path / "ws")
    ws = pl.Workspace(tmp_path / "ws")
    e = entry_of(ws)
    assert e.verification.state == "VERIFIED" and e.verification.evidence_level == "primary_license_text_read"
    assert e.verification.verified_scope.startswith(OWNER_ATTESTED_PREFIX) and "hash sha256:" in e.verification.verified_scope
    assert e.verification.license_text_sha256 == sample_license_evidence()["text_sha256"]
    assert any("owner-attested" in u for u in e.verification.uncertainties)
    assert is_owner_attested(e) and rep.license["owner_attested"] and rep.license["state"] == "VERIFIED"
    # catalog facts survive the override
    assert e.hf_repo_or_source == "Qwen/Qwen3-1.7B" and e.variants
    # the existing gate now allows the attested uses
    assert evaluate_gate(e, RequestedUse(fine_tune=True, produce_adapter=True, redistribute=True)).allowed
    att = json.loads((tmp_path / "ws" / "job" / "license_attestation.json").read_text())
    assert att["outcome"]["owner_attested"] is True and att["evidence_as_received"]["attested_by"] == "owner"


def test_attested_no_permission_still_blocks_gate(tmp_path):
    ev = sample_license_evidence()
    ev["permissions"]["redistribution_permitted"] = "no"
    import_job(write_job(tmp_path / "j.zip", make_job(license_evidence=ev)), tmp_path / "ws")
    e = entry_of(pl.Workspace(tmp_path / "ws"))
    assert e.verification.state == "VERIFIED"
    assert evaluate_gate(e, RequestedUse(fine_tune=True, produce_adapter=True)).allowed
    assert not evaluate_gate(e, RequestedUse(fine_tune=True, produce_adapter=True, redistribute=True)).allowed


def test_no_attestation_stays_unverified_and_gate_blocks(tmp_path):
    job = make_job()
    job["manifest"].pop("license_evidence")
    rep = import_job(write_job(tmp_path / "j.zip", job), tmp_path / "ws")
    e = entry_of(pl.Workspace(tmp_path / "ws"))
    assert rep.license["state"] == "UNVERIFIED" and not rep.license["owner_attested"] and e.verification.state == "UNVERIFIED"
    assert not is_owner_attested(e)
    assert not evaluate_gate(e, RequestedUse(fine_tune=True, produce_adapter=True)).allowed
    assert any("UNVERIFIED" in w for w in rep.warnings)


def _drop(k):
    return lambda ev: ev.pop(k)


def _set(k, v):
    return lambda ev: ev.__setitem__(k, v)


@pytest.mark.parametrize("name,fn,needle", [
    ("no_hash", _drop("text_sha256"), "text_sha256"),
    ("bad_hash", _set("text_sha256", "sha256:xyz"), "text_sha256"),
    ("no_url", _drop("license_url"), "license_url"),
    ("http_url", _set("license_url", "http://example.invalid/l"), "license_url"),
    ("no_fetched", _drop("fetched_at"), "fetched_at"),
    ("no_permissions", _drop("permissions"), "permissions missing"),
    ("partial_permissions", lambda ev: ev["permissions"].pop("commercial_use"), "commercial_use"),
    ("unverified_permission", lambda ev: ev["permissions"].__setitem__("fine_tuning_permitted", "unverified"), "fine_tuning_permitted"),
    ("not_owner", _set("attested_by", "phone"), "attested_by"),
    ("no_attester", _drop("attested_by"), "attested_by"),
    ("secondary_evidence", _set("evidence_level", "secondary_source_only"), "evidence_level"),
    ("no_scope", _drop("scope_note"), "scope_note"),
    ("blank_scope", _set("scope_note", "  "), "scope_note"),
    ("no_date", _drop("attested_on"), "attested_on"),
    ("wrong_model", _set("model_id", "Other@1"), "model_id"),
    ("no_model", _drop("model_id"), "model_id"),
])
def test_incomplete_attestation_stays_unverified(tmp_path, name, fn, needle):
    ev = sample_license_evidence()
    fn(ev)
    rep = import_job(write_job(tmp_path / "j.zip", make_job(license_evidence=ev)), tmp_path / "ws")
    e = entry_of(pl.Workspace(tmp_path / "ws"))
    assert e.verification.state == "UNVERIFIED" and not rep.license["owner_attested"]
    assert needle in " ".join(rep.license["reasons"])


def test_license_text_file_must_match_attested_hash(tmp_path):
    job = make_job()
    job["license_text"] = b"a different license text\n"
    rep = import_job(write_job(tmp_path / "j.zip", job), tmp_path / "ws")
    assert rep.license["state"] == "UNVERIFIED" and "license_text.txt" in " ".join(rep.license["reasons"])


def test_disallowed_catalog_entry_is_never_overridden(tmp_path):
    cat = tmp_path / "cat"
    cat.mkdir()
    e = json.loads((Path(__file__).resolve().parents[2] / "registry" / "base-models" / "qwen3-1.7b.json").read_text())
    e["verification"] = {"state": "DISALLOWED", "evidence_level": "primary_license_text_read", "disallowed_reason": "test", "source_urls": ["https://example.invalid/x"],
                         "verified_on": "2026-10-05", "uncertainties": []}
    (cat / "q.json").write_text(json.dumps(e))
    rep = import_job(write_job(tmp_path / "j.zip", make_job()), tmp_path / "ws", catalog_dir=cat)
    assert rep.license["state"] == "DISALLOWED" and not rep.license["owner_attested"]
    assert "can never override" in " ".join(rep.license["reasons"])
    with pytest.raises(LicenseGateError):
        enforce_gate(entry_of(pl.Workspace(tmp_path / "ws")), RequestedUse())


def test_unknown_model_attested_builds_entry_unknown_unattested_gets_placeholder(tmp_path):
    job = other_model(make_job())
    rep = import_job(write_job(tmp_path / "a.zip", job), tmp_path / "a")
    e = pl.Workspace(tmp_path / "a").registry().get("Nonexistent@1")
    assert rep.license["state"] == "VERIFIED" and rep.license["owner_attested"] and e.license_id == "unknown"
    job = other_model(make_job())
    job["manifest"].pop("license_evidence")
    rep = import_job(write_job(tmp_path / "b.zip", job), tmp_path / "b")
    e = pl.Workspace(tmp_path / "b").registry().get("Nonexistent@1")
    assert rep.license["state"] == "UNVERIFIED" and e.fine_tuning_permitted == "unverified"


def test_owner_attestation_recorded_in_downstream_entry_hash(tmp_path):
    a = import_job(write_job(tmp_path / "a.zip", make_job()), tmp_path / "a")
    ev = sample_license_evidence()
    ev["scope_note"] = "different note"
    b = import_job(write_job(tmp_path / "b.zip", make_job(license_evidence=ev)), tmp_path / "b")
    assert a.license["entry_hash"] != b.license["entry_hash"]  # the attestation text is inside the hashed entry


# ------------------------------------------------------------------ CLI

def test_cli_import_job(tmp_path, capsys):
    z = write_job(tmp_path / "j.zip", make_job())
    assert cli.main(["import-job", str(z), "--workspace", str(tmp_path / "ws")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["license"]["owner_attested"] is True and out["examples"]["test"] > 0
    bad = write_members(tmp_path / "bad.zip", {**job_files(make_job()), "../x": b"1"})
    assert cli.main(["import-job", str(bad), "--workspace", str(tmp_path / "ws2")]) == 1
    assert "REJECTED" in capsys.readouterr().err and not (tmp_path / "ws2").exists()
