import re

import pytest

from llmtrainer import pipeline as pl
from llmtrainer.datasets import load_split
from llmtrainer.evalsuite import CallableSubject, LeakageError, LexicalRetriever, build_items, check_leakage, load_test_chunks, run_suite, to_evaluation_run
from llmtrainer.evalsuite import metrics as m
from llmtrainer.schemas import RightsInfo

REAL = RightsInfo(status="owned", permitted_training="yes", permitted_commercial="yes", permitted_redistribution="yes", evidence="test")


def doc(i):
    parts = [f"# Doc {i}\n"]
    for j in range(6):
        u = lambda n: f"z{i}{j}q{n}"  # unique tokens so no 5-gram is shared across documents
        parts.append(
            f"## Procedure {i}-{j}\n\nTighten {u(1)} the flange{i}{j} {u(2)} bolt {u(3)} to {10 + i + j} N·m {u(4)} using {u(5)} the calibrated {u(6)} wrench "
            f"before {u(7)} commissioning the {u(8)} assembly. Maintain {u(9)} a clearance {u(10)} of 0.{i}{j + 1} mm {u(11)} between the {u(12)} rotor{i}{j} "
            f"{u(13)} housing and {u(14)} the guard {u(15)} during inspection. The {u(16)} technician must {u(17)} record the {u(18)} actuator{i}{j} "
            f"{u(19)} reading in {u(20)} the logbook {u(21)} after every {u(22)} shift change.\n"
        )
    return "\n".join(parts)


@pytest.fixture
def evalset(tmp_path):
    pl.init_project(tmp_path / "p", "Eval Fixture", "pump service")
    ws = pl.Workspace(tmp_path / "p")
    for i in range(5):
        f = tmp_path / f"d{i}.md"
        f.write_text(doc(i))
        pl.add_source(ws, f, title=f"d{i}", origin="test", rights=REAL)
    ds = pl.build_dataset_for_project(ws)
    chunks = load_test_chunks(ds, ws.root / "corpus", ws.sources())
    items = build_items(chunks, domain="pump service")
    return ws, ds, chunks, items


def test_metric_numeric_normalization():
    exp = m.extract_quantities("torque 12 N·m")[0]
    assert m.fact_correct("Use 12 Nm.", exp) and m.fact_correct("about 8.85 lb-ft", exp)  # cross-unit conversion within 2%
    assert m.fact_correct("It is 12.0 N-m", exp)
    assert not m.fact_correct("Use 21 N·m", exp)
    assert not m.fact_correct("12 N·m or maybe 20 N·m", exp)  # hedging with conflicting values is wrong
    assert not m.fact_correct("12 mm", exp)  # wrong dimension
    c = m.extract_quantities("clearance 0.15 mm")[0]
    assert m.fact_correct("0.15mm", c) and m.fact_correct("0,15 mm", c) and m.fact_correct("0.0059 in", c)


def test_unsupported_claims_heuristic():
    src = "Tighten the drain plug to 12 N·m using the calibrated wrench before commissioning."
    assert m.unsupported_claims("Tighten the drain plug using the calibrated wrench.", src) == (0, 1)
    assert m.unsupported_claims("Tighten the drain plug to 15 N·m using the wrench.", src) == (1, 1)  # number not in source
    assert m.unsupported_claims("Always replace the turbine blades with titanium alloy parts.", src) == (1, 1)
    assert m.unsupported_claims("I do not know.", src) == (0, 0)
    assert m.parse_citations("see [a/b, c/d] and [e/f]") == ["a/b", "c/d", "e/f"]


def test_items_come_only_from_test_chunks_and_pass_guard(evalset):
    ws, ds, chunks, items = evalset
    test_refs = {f"{r.source_id}/{r.chunk_id}" for e in ds.examples if e.split == "test" for r in e.derived_from}
    assert items and {r for i in items for r in i.gold_refs} <= test_refs
    assert any(i.kind == "fact" for i in items) and any(i.kind == "concept" for i in items)
    train = load_split(ws.root / "datasets", ds, "train") + load_split(ws.root / "datasets", ds, "validation")
    assert check_leakage(items, ds, train)["items_checked"] == len(items)


def test_leakage_guard_catches_train_chunk_and_text_overlap(evalset):
    ws, ds, chunks, items = evalset
    train = load_split(ws.root / "datasets", ds, "train")
    train_ref = next(f"{r.source_id}/{r.chunk_id}" for e in ds.examples if e.split == "train" for r in e.derived_from)
    leaked = items[0].__class__(**{**items[0].__dict__, "gold_refs": (train_ref,)})
    with pytest.raises(LeakageError, match="also used by train"):
        check_leakage([leaked], ds, train)
    # eval content copied verbatim from a training example
    copied = items[0].__class__(**{**items[0].__dict__, "content_text": train[0]["prompt"] + " " + train[0]["response"]})
    with pytest.raises(LeakageError, match="overlaps training text"):
        check_leakage([copied], ds, train)


def make_generators(items, quality):
    by_q = {i.question: i for i in items}

    def gen(prompt):
        if "Sources:" in prompt:  # RAG prompt
            q = prompt.split("Question: ")[1]
            it = by_q[q]
            if quality == "good":
                return f"{it.content_text.split('.')[0]}. [{it.gold_refs[0]}]"
            return "The turbine uses a proprietary titanium rotor rated at 999 bar. [made/up]"
        if prompt in by_q:
            it = by_q[prompt]
            if it.kind == "concept":
                return " ".join(it.required_terms) if quality == "good" else "It is a section about things."
            return "I am not sure."
        if quality == "worse":
            return "unknown"
        if any(q == prompt for q, _ in m.GENERAL_PROBES):
            return _probe_answer(prompt)
        return ""
    return gen


def _probe_answer(q):
    return {"12 plus 15": "27", "9 times 8": "72", "France": "Paris", "water": "H2O", "Red Planet": "Mars", "opposite": "cold", "mouse": "mice",
            "week": "7", "sun rises": "east", "100 divided": "25", "17 prime": "yes", "banana": "BANANA"}[next(k for k in
            ["12 plus 15", "9 times 8", "France", "water", "Red Planet", "opposite", "mouse", "week", "sun rises", "100 divided", "17 prime", "banana"] if k in q)]


def subjects(items, spec_quality, base_quality="base"):
    return [CallableSubject("base", "fake-base", make_generators(items, base_quality)),
            CallableSubject("specialist", "fake-spec", make_generators(items, spec_quality))]


def test_better_specialist_measured_per_metric(evalset):
    ws, ds, chunks, items = evalset
    r = LexicalRetriever([{"ref": c["ref"], "text": c["text"]} for c in chunks])
    s = run_suite(subjects(items, "good"), items, retrieve=r)
    cmp = {c.metric: c for c in s.comparisons}
    assert set(cmp) == {"terminology_coverage", "closed_book_exact_fact_accuracy", "general_probe_pass_rate", "rag_exact_fact_accuracy",
                        "rag_unsupported_claim_rate", "citation_accuracy"}
    assert cmp["terminology_coverage"].specialist_minus_base > 0.5 and cmp["terminology_coverage"].base < 0.1
    assert cmp["rag_unsupported_claim_rate"].specialist < cmp["rag_unsupported_claim_rate"].base
    assert cmp["citation_accuracy"].specialist_minus_base > 0.5
    assert not any(c.regression for c in s.comparisons)
    run = to_evaluation_run(s, project_id="p", dataset=ds, train_eval_group_overlap=0, evaluated_on="2026-10-05", subjects=subjects(items, "good"))
    assert run.verify() and len(run.comparisons) == 6 and run.evaluator.is_stub is False
    assert any("95%CI" in n for n in run.notes) and any("performance[base]" in n for n in run.notes)
    # real (non-synthetic) data but only a handful of items => no improvement claim (needs >= 50)
    assert len(items) < 50 and not run.improvement_claim_allowed


def test_worse_specialist_flags_regressions_and_no_claim(evalset):
    ws, ds, chunks, items = evalset
    r = LexicalRetriever([{"ref": c["ref"], "text": c["text"]} for c in chunks])
    subs = subjects(items, "worse", "good")
    s = run_suite(subs, items, retrieve=r)
    cmp = {c.metric: c for c in s.comparisons}
    assert cmp["terminology_coverage"].regression and cmp["general_probe_pass_rate"].regression and cmp["citation_accuracy"].regression
    run = to_evaluation_run(s, project_id="p", dataset=ds, train_eval_group_overlap=0, evaluated_on="2026-10-05", subjects=subs)
    assert not run.improvement_claim_allowed


def test_claim_possible_only_with_enough_real_items_and_forced_off_for_synthetic(evalset):
    ws, ds, chunks, items = evalset
    big = [items[i % len(items)].__class__(**{**items[i % len(items)].__dict__, "item_id": f"x{i}"}) for i in range(60)]
    r = LexicalRetriever([{"ref": c["ref"], "text": c["text"]} for c in chunks])
    subs = subjects(items, "good")
    s = run_suite(subs, big, retrieve=r)
    ok = to_evaluation_run(s, project_id="p", dataset=ds, train_eval_group_overlap=0, evaluated_on="2026-10-05", subjects=subs)
    assert ok.improvement_claim_allowed
    syn = to_evaluation_run(s, project_id="p", dataset=ds, train_eval_group_overlap=0, evaluated_on="2026-10-05", subjects=subs, synthetic=True)
    assert not syn.improvement_claim_allowed and any("synthetic" in n for n in syn.notes)


def test_no_retrieval_skips_grounding_metrics_and_timing_hooks(evalset):
    ws, ds, chunks, items = evalset
    ticks = iter(range(0, 10000))
    s = run_suite(subjects(items, "good"), items, clock=lambda: next(ticks) * 0.01, count_tokens=lambda t: 10)
    assert {c.metric for c in s.comparisons} == {"terminology_coverage", "closed_book_exact_fact_accuracy", "general_probe_pass_rate"}
    assert any("not measured" in c for c in s.caveats)
    perf = s.reports["base"].performance()
    assert perf["latency_ms_mean"] == pytest.approx(10.0) and perf["tokens_per_s"] == pytest.approx(1000.0)


def test_quantized_subject_is_compared_to_specialist(evalset):
    ws, ds, chunks, items = evalset
    subs = subjects(items, "good") + [CallableSubject("quantized", "fake-q", make_generators(items, "worse"))]
    s = run_suite(subs, items)
    assert any(c.quantization_regression for c in s.comparisons)
