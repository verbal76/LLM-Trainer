import pytest

from llmtrainer import evaluation as ev
from llmtrainer import training as tr
from llmtrainer.schemas import EvaluatorInfo, MetricResult


def result(metric, role, value, higher=True, cat="domain"):
    return MetricResult(metric=metric, subject_role=role, value=value, n=10, higher_is_better=higher, category=cat)


SPECS = (ev.MetricSpec("acc", "domain"), ev.MetricSpec("gen", "general_regression"), ev.MetricSpec("halluc", "safety", higher_is_better=False))


def test_compare_is_per_metric_no_aggregate():
    res = [
        result("acc", "base", 0.5), result("acc", "specialist", 0.7), result("acc", "quantized", 0.69),
        result("gen", "base", 0.8), result("gen", "specialist", 0.7), result("gen", "quantized", 0.7),
        result("halluc", "base", 0.2, False, "safety"), result("halluc", "specialist", 0.1, False, "safety"), result("halluc", "quantized", 0.2, False, "safety"),
    ]
    cmp = {c.metric: c for c in ev.compare(res, SPECS)}
    assert len(cmp) == 3
    assert cmp["acc"].specialist_minus_base == pytest.approx(0.2) and not cmp["acc"].regression and not cmp["acc"].quantization_regression
    assert cmp["gen"].regression  # general capability regressed beyond tolerance
    assert cmp["halluc"].quantization_regression  # lower is better and quantized got worse (0.1 -> 0.2)
    assert not cmp["halluc"].regression


def test_compare_without_quantized():
    res = [result("acc", "base", 0.5), result("acc", "specialist", 0.6)]
    c = ev.compare(res, (ev.MetricSpec("acc", "domain"),))[0]
    assert c.quantized is None and c.quantized_minus_specialist is None and not c.quantization_regression


def comps(domain_delta, regress=False):
    res = [result("acc", "base", 0.5), result("acc", "specialist", 0.5 + domain_delta)]
    out = ev.compare(res, (ev.MetricSpec("acc", "domain"),))
    return out


REAL = EvaluatorInfo(name="real", version="1", is_stub=False)
STUB = EvaluatorInfo(name="stub", version="1", is_stub=True)


def test_claim_rules():
    assert ev.improvement_claim_allowed(REAL, comps(0.1), 100)
    assert not ev.improvement_claim_allowed(STUB, comps(0.1), 100)  # stub never
    assert not ev.improvement_claim_allowed(REAL, comps(0.1), 10)  # too few unseen examples
    assert not ev.improvement_claim_allowed(REAL, comps(0.0), 100)  # no gain
    assert not ev.improvement_claim_allowed(REAL, comps(-0.1), 100)


def test_claim_blocked_by_quantization_regression():
    res = [result("acc", "base", 0.5), result("acc", "specialist", 0.7), result("acc", "quantized", 0.4)]
    assert not ev.improvement_claim_allowed(REAL, ev.compare(res, (ev.MetricSpec("acc", "domain"),)), 100)


def test_protocols_runtime_checkable():
    assert isinstance(ev.StubEvaluator(), ev.Evaluator)
    assert isinstance(ev.base_subject(), ev.Subject)
    assert isinstance(tr.StubTrainer(), tr.Trainer)


def test_run_evaluation_requires_all_metrics():
    class Bad:
        info = STUB
        metrics = (ev.MetricSpec("a", "domain"), ev.MetricSpec("b", "domain"))

        def evaluate(self, subject, rows):
            return {"a": 1.0}

    with pytest.raises(ValueError):
        ev.run_evaluation(Bad(), [ev.base_subject()], [])


def test_stub_trainer_deterministic_and_labeled(tmp_path):
    rows = [{"prompt": "p one two", "response": "alpha beta gamma"}] * 3
    cfg = tr.ExperimentConfig(base_model_id="x")
    a = tr.StubTrainer().train(cfg, rows, rows, tmp_path / "a")
    b = tr.StubTrainer().train(cfg, rows, rows, tmp_path / "b")
    assert tr.hash_artifacts(a.artifacts) == tr.hash_artifacts(b.artifacts)
    assert tr.StubTrainer.is_stub and ev.StubEvaluator.info.is_stub
    assert "alpha" in a.subject.vocab and "alpha" not in ev.base_subject().vocab


def test_stub_specialist_covers_more_than_base_but_quantization_prunes(tmp_path):
    rows = [{"prompt": "x", "response": "zork zork blip"}, {"prompt": "x", "response": "zork zork blip"}]
    s = tr.StubTrainer().train(tr.ExperimentConfig(base_model_id="x"), rows, rows, tmp_path).subject
    q = ev.quantize_subject(s)
    e = ev.StubEvaluator()
    assert e.evaluate(s, rows)["token_coverage"] > e.evaluate(ev.base_subject(), rows)["token_coverage"]
    assert set(q.vocab) <= set(s.vocab)
    rare = tr.StubTrainer().train(tr.ExperimentConfig(base_model_id="x"), [{"prompt": "x", "response": "singleton"}], rows, tmp_path / "r").subject
    assert "singleton" in rare.vocab and "singleton" not in ev.quantize_subject(rare).vocab


def test_run_id_changes_with_config_and_dataset():
    c1 = tr.ExperimentConfig(base_model_id="x", seed=1)
    c2 = tr.ExperimentConfig(base_model_id="x", seed=2)
    h = "sha256:" + "1" * 64
    assert tr.run_id_for(c1, h) == tr.run_id_for(c1, h)
    assert tr.run_id_for(c1, h) != tr.run_id_for(c2, h)
    assert tr.run_id_for(c1, h) != tr.run_id_for(c1, "sha256:" + "2" * 64)


def test_resource_estimates_scale_sensibly():
    q7 = tr.estimate_resources(tr.ExperimentConfig(base_model_id="x", method="qlora", base_params_b=7), 5_000_000)
    l7 = tr.estimate_resources(tr.ExperimentConfig(base_model_id="x", method="lora", base_params_b=7), 5_000_000)
    f7 = tr.estimate_resources(tr.ExperimentConfig(base_model_id="x", method="full", base_params_b=7), 5_000_000)
    q1 = tr.estimate_resources(tr.ExperimentConfig(base_model_id="x", method="qlora", base_params_b=1), 5_000_000)
    assert q7.gpu_memory_gib < l7.gpu_memory_gib < f7.gpu_memory_gib
    assert q1.gpu_memory_gib < q7.gpu_memory_gib and q1.gpu_hours < q7.gpu_hours
    assert 8 < q7.gpu_memory_gib < 24  # a 7B QLoRA run is a single-consumer-GPU job
    assert tr.estimate_resources(tr.ExperimentConfig(base_model_id="x"), 100).gpu_hours == 0.0  # stub
