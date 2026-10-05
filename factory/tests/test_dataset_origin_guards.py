import json

import pytest
from conftest import SYNTH, fixture_files
from pydantic import ValidationError

from llmtrainer import corpus
from llmtrainer import pipeline as pl
from llmtrainer import splits as sp
from llmtrainer.datasets import build_dataset, load_split, verify_no_leakage
from llmtrainer.schemas import ExampleRecord, GeneratorInfo


def test_synthetic_forbidden_in_test_by_schema():
    base = dict(example_id="e", group_id="g", task="t", derived_from=[], generator=GeneratorInfo(name="g", version="1", type="llm"),
                text_sha256="sha256:" + "0" * 64, quality_score=1.0)
    ExampleRecord(split="train", origin="synthetic", **base)
    with pytest.raises(ValidationError):
        ExampleRecord(split="test", origin="synthetic", **base)


def _synthetic_for(ws, d, split):
    """A synthetic example citing a chunk whose group landed in `split`."""
    src = ws.sources()
    group_split = {e.group_id: e.split for e in d.examples}
    for s in src.active_sources():
        for ch in s.chunks:
            if group_split.get(s.source_id) == split:
                return {"source_id": s.source_id, "chunk_id": ch.chunk_id, "prompt": "Q about " + ch.chunk_id,
                        "response": "A synthetic answer unrelated to the text", "quality": 0.9}
    raise AssertionError


def test_synthetic_origin_labelled_train_only_and_dataset_id_changes(built):
    ws, d = built
    project = ws.project()
    allowed = [s.source_id for s in ws.sources().active_sources()]

    def build(syn):
        return build_dataset(ws.sources(), ws.root / "corpus", ws.root / "datasets", project_id=project.project_id, seed=project.seed,
                             config=project.split_config, allowed_source_ids=allowed, synthetic_examples=syn)

    syn_train = _synthetic_for(ws, d, "train")
    syn_test = _synthetic_for(ws, d, "test")
    d2 = build([syn_train, syn_test])
    assert d2.dataset_id != d.dataset_id  # inputs changed -> identity changed
    syn = [e for e in d2.examples if e.origin == "synthetic"]
    assert len(syn) == 1 and syn[0].split == "train"
    assert all(e.origin == "source_derived" for e in d2.examples if e.origin != "synthetic")
    assert any("synthetic" in n and "held-out" in n for n in d2.leakage_report.notes)
    assert verify_no_leakage(d2, ws.root / "datasets") == []
    assert all(r["origin"] == "source_derived" for r in load_split(ws.root / "datasets", d2, "test"))
    with pytest.raises(ValueError):
        build([{**syn_train, "split": "test"}])
    with pytest.raises(ValueError):
        build([{**syn_train, "chunk_id": "c9999"}])


def test_config_change_changes_dataset_id(built):
    ws, d = built
    p = ws.project()
    cfg = p.split_config.model_copy(update={"near_duplicate_threshold": 0.7})
    allowed = [s.source_id for s in ws.sources().active_sources()]
    d2 = build_dataset(ws.sources(), ws.root / "corpus", ws.root / "datasets", project_id=p.project_id, seed=p.seed, config=cfg, allowed_source_ids=allowed)
    assert d2.dataset_id != d.dataset_id


def test_containment_catches_short_copy_of_long_chunk():
    long = " ".join(f"w{i}" for i in range(60))
    short = " ".join(f"w{i}" for i in range(10, 25))
    assert sp.find_near_duplicates({"a": long, "b": short}) == []  # Jaccard misses it
    assert sp.containment_leaks({"b": short}, {"a": long}, 0.8, 5)[0][:2] == ("b", "a")


PROSE = [
    "The compressor stage raises air pressure before the combustion chamber receives the airflow from upstream.",
    "Lubrication oil circulates through galleries drilled in the bearing housing to carry away friction heat.",
    "Ignition timing is advanced under light load so that the flame front finishes burning near top centre.",
    "Exhaust gas temperature sensors protect the turbine wheel by triggering a fuel cutback when limits rise.",
]


def test_tables_go_to_reference_set_not_training(tmp_path):
    pl.init_project(tmp_path / "p", "Mixed", "d")
    ws = pl.Workspace(tmp_path / "p")
    srcdir = tmp_path / "in"
    srcdir.mkdir()
    for i, topic in enumerate(["alpha", "bravo", "charlie", "delta"]):
        (srcdir / f"{topic}.txt").write_text(PROSE[i])
    (srcdir / "torque.csv").write_text("bolt,torque\nA,25 Nm\nB,40 Nm\n")
    rep = corpus.ingest_paths(ws, [srcdir], origin="t", rights=SYNTH)
    assert len(rep.ingested) == 5
    d = pl.build_dataset_for_project(ws)
    ref = (ws.root / "datasets" / d.dataset_id / "reference_chunks.jsonl").read_text().splitlines()
    assert len(ref) == 1 and json.loads(ref[0])["table"]["rows"][1] == ["B", "40 Nm"]
    assert not any("40 Nm" in json.dumps(load_split(ws.root / "datasets", d, s)) for s in ("train", "validation", "test"))
