import pytest
from conftest import SYNTH, fixture_files

from llmtrainer import pipeline as pl
from llmtrainer import provenance as pv
from llmtrainer.training import ExperimentConfig


@pytest.fixture
def full(built):
    ws, d = built
    out = pl.run_experiment(ws, ExperimentConfig(base_model_id=pl.STUB_MODEL_ID))
    return ws, d, out


def test_graph_lineage_source_to_model(full):
    ws, d, out = full
    g = pv.build_graph(ws.sources(), [d], [(out["model"], d)], [out["package"]])
    lin = g.lineage(pv.package_node(out["package"].content_hash))
    assert set(lin) == {"model", "dataset", "example", "chunk", "source"}
    assert len(lin["source"]) == 4
    ex = d.examples[0]
    desc = g.descendants(pv.source_node(ex.derived_from[0].source_id))
    assert pv.example_node(d.content_hash, ex.example_id) in desc
    assert pv.model_node(out["model"].content_hash) in desc


def test_removal_plan_marks_models_packages_and_datasets(full):
    ws, d, out = full
    train_ex = next(e for e in d.examples if e.split == "train")
    sid = train_ex.derived_from[0].source_id
    plan = pl.removal_plan(ws, [sid])
    assert d.content_hash in plan.datasets_to_rebuild
    assert out["model"].content_hash in plan.models_to_retrain
    assert out["package"].content_hash in plan.packages_to_withdraw
    assert plan.reference_index_rebuild
    assert any("retrain" in s for s in plan.steps)
    affected = set(plan.affected_examples[d.content_hash])
    assert affected == {e.example_id for e in d.examples if any(r.source_id == sid for r in e.derived_from)}


def test_removal_affecting_only_test_split_means_reevaluate(full):
    ws, d, out = full
    by_source = {}
    for e in d.examples:
        by_source.setdefault(e.derived_from[0].source_id, set()).add(e.split)
    only_test = [s for s, sp_ in by_source.items() if sp_ == {"test"}]
    assert only_test, "document-level grouping should give a test-only document"
    plan = pl.removal_plan(ws, only_test)
    assert out["model"].content_hash in plan.models_to_reevaluate
    assert out["model"].content_hash not in plan.models_to_retrain


def test_unknown_source_rejected(full):
    ws, *_ = full
    with pytest.raises(KeyError):
        pl.removal_plan(ws, ["src-nope"])


def test_remove_source_deletes_files_and_tombstones(full):
    ws, d, out = full
    sid = d.examples[0].derived_from[0].source_id
    plan = pl.remove_source(ws, [sid], reason="rights withdrawn")
    assert plan.removed_sources == [sid]
    assert not (ws.root / "sources" / "raw" / sid).exists()
    assert not (ws.root / "corpus" / f"{sid}.jsonl").exists()
    m = ws.sources()
    rec = m.get(sid)
    assert rec.status == "removed" and rec.chunks == [] and rec.removal.reason == "rights withdrawn"
    assert rec.removal.removed_chunk_ids and rec.sha256  # audit trail retained, content gone
    assert m.manifest_version == 6 and m.verify()
    # old dataset no longer matches the manifest; a rebuild excludes the removed source
    d2 = pl.build_dataset_for_project(ws)
    assert all(r.source_id != sid for e in d2.examples for r in e.derived_from)
    # removed material cannot be silently re-added
    with pytest.raises(pl.WorkspaceError):
        pl.add_source(ws, fixture_files()[0], title="x", origin="o", rights=SYNTH)


def test_old_package_fails_validation_semantics_after_removal(full):
    """Source manifest in the old package still lists the source as active (it is a snapshot); the plan says withdraw it."""
    ws, d, out = full
    sid = d.examples[0].derived_from[0].source_id
    plan = pl.removal_plan(ws, [sid])
    assert out["package"].content_hash in plan.packages_to_withdraw
