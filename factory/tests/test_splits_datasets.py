import json

import pytest

from llmtrainer import pipeline as pl
from llmtrainer import splits as sp
from llmtrainer.datasets import leak_text, load_split, verify_no_leakage
from llmtrainer.schemas import SplitConfig


def test_near_duplicate_detection_threshold():
    a = "the quick brown fox jumps over the lazy dog near the river bank today"
    b = "the quick brown fox jumps over the lazy dog near the river bank tonight"
    c = "completely unrelated sentence about turbine commissioning checklists and vibration"
    pairs = sp.find_near_duplicates({"a": a, "b": b, "c": c}, threshold=0.6)
    assert [(x, y) for x, y, _ in pairs] == [("a", "b")]
    assert sp.find_near_duplicates({"a": a, "b": b}, threshold=0.99) == []


def test_exact_duplicates_have_jaccard_one():
    t = "one two three four five six seven eight nine ten"
    assert sp.find_near_duplicates({"x": t, "y": t})[0][2] == 1.0


def test_assign_groups_deterministic_and_order_independent():
    w = {f"g{i}": 3 + i % 4 for i in range(12)}
    r = {"train": 0.7, "validation": 0.15, "test": 0.15}
    a = sp.assign_groups(w, r, seed=7)
    b = sp.assign_groups(dict(reversed(list(w.items()))), r, seed=7)
    assert a == b
    assert set(a.values()) == {"train", "validation", "test"}
    assert sp.assign_groups(w, r, seed=8) != a  # seed matters


def test_assign_groups_needs_three_groups():
    with pytest.raises(ValueError):
        sp.assign_groups({"a": 1, "b": 1}, {"train": 0.7, "validation": 0.15, "test": 0.15}, 1)


def test_resolve_leakage_drops_train_side_first():
    t = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    texts = {"tr": t, "te": t, "va": "unrelated words entirely different content here for validation set only"}
    split_of = {"tr": "train", "te": "test", "va": "validation"}
    remaining, dropped, found = sp.resolve_leakage(texts, split_of, 0.8, 5)
    assert dropped == ["tr"] and found == 1 and "te" in remaining
    # validation vs test duplicate: validation is dropped, test kept
    r2, d2, _ = sp.resolve_leakage({"va": t, "te": t}, {"va": "validation", "te": "test"}, 0.8, 5)
    assert d2 == ["va"] and "te" in r2


def test_same_split_duplicates_are_not_leakage():
    t = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    _, dropped, found = sp.resolve_leakage({"a": t, "b": t}, {"a": "train", "b": "train"}, 0.8, 5)
    assert dropped == [] and found == 0


def test_leak_text_strips_template():
    assert leak_text('Complete the statement from section "X": hello', "world") == "hello world"


def test_build_dataset_splits_clean_and_deterministic(built):
    ws, d = built
    assert d.verify() and d.schema_version == 1
    assert {k: v.n_examples > 0 for k, v in d.splits.items()} == {"train": True, "validation": True, "test": True}
    assert d.leakage_report.group_overlap_pairs == 0
    assert d.leakage_report.residual_cross_split_near_duplicates == 0
    assert d.leakage_report.effective_group_by == "document"
    assert verify_no_leakage(d, ws.root / "datasets") == []
    # every example links back to source chunks that exist
    src = ws.sources()
    for ex in d.examples:
        for ref in ex.derived_from:
            assert any(c.chunk_id == ref.chunk_id for c in src.get(ref.source_id).chunks)
    # rebuilding gives an identical manifest
    again = pl.build_dataset_for_project(ws)
    assert again.content_hash == d.content_hash


def test_groups_never_straddle_splits(built):
    _, d = built
    seen = {}
    for ex in d.examples:
        assert seen.setdefault(ex.group_id, ex.split) == ex.split


def test_seed_changes_assignment(ws):
    import dataclasses

    p = ws.project()
    hashes = set()
    for seed in range(6):
        ws.project_file.write_text(p.model_copy(update={"seed": seed}).seal().to_json())
        hashes.add(pl.build_dataset_for_project(ws).content_hash)
    assert len(hashes) > 1


def test_cross_document_near_duplicate_is_removed_from_train(ws):
    """The fixture copies one sentence between two documents; for some seeds they land in different splits."""
    p = ws.project()
    found_case = False
    for seed in range(40):
        ws.project_file.write_text(p.model_copy(update={"seed": seed}).seal().to_json())
        d = pl.build_dataset_for_project(ws)
        if d.leakage_report.near_duplicate_pairs_found:
            found_case = True
            assert d.leakage_report.dropped_example_ids
            assert d.leakage_report.residual_cross_split_near_duplicates == 0
            assert verify_no_leakage(d, ws.root / "datasets") == []
            kept = {e.example_id for e in d.examples}
            assert not set(d.leakage_report.dropped_example_ids) & kept
            break
    assert found_case, "no seed placed the duplicated sentence across splits"


def test_verify_no_leakage_detects_tampering(built):
    ws, d = built
    root = ws.root / "datasets"
    train = load_split(root, d, "train")
    test_rows = load_split(root, d, "test")
    # inject a test example into train => file hash mismatch is caught by load_split
    path = root / d.splits["train"].file
    with path.open("a") as fh:
        fh.write(json.dumps(test_rows[0]) + "\n")
    with pytest.raises(ValueError):
        verify_no_leakage(d, root)


def test_rights_gate_blocks_unverified_sources(ws):
    from llmtrainer.schemas import RightsInfo
    from llmtrainer.pipeline import WorkspaceError
    from conftest import fixture_files

    f = ws.root / "extra.md"
    f.write_text("# Unknown\n\nThis source has no recorded usage rights at all in the manifest.\n")
    pl.add_source(ws, f, title="x", origin="somewhere", rights=RightsInfo())
    with pytest.raises(WorkspaceError, match="rights gate"):
        pl.build_dataset_for_project(ws)
    d = pl.build_dataset_for_project(ws, allow_blocked_sources=True)  # explicit override is recorded
    assert d.excluded_sources


def test_few_documents_fall_back_to_section_groups(tmp_path):
    from conftest import SYNTH, fixture_files

    pl.init_project(tmp_path / "p", "Two", "d")
    w = pl.Workspace(tmp_path / "p")
    for f in fixture_files()[:2]:
        pl.add_source(w, f, title=f.stem, origin="o", rights=SYNTH)
    d = pl.build_dataset_for_project(w)
    assert d.leakage_report.effective_group_by == "section"
    assert any("fell back" in n for n in d.leakage_report.notes)


def test_split_config_default_ratios():
    assert SplitConfig().ratios["test"] == 0.15
