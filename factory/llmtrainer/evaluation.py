"""Evaluation interfaces, comparison logic and the pipeline-validation stub evaluator.

The stub evaluator measures unigram vocabulary coverage of a toy bag-of-words
"model". It exists ONLY to exercise the pipeline plumbing. Numbers it
produces say nothing about real model quality, and ``improvement_claim_allowed``
is forced to False whenever a stub evaluator is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from . import splits as sp
from .schemas import EvaluatorInfo, MetricComparison, MetricResult

Role = str  # "base" | "specialist" | "quantized"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    category: str
    higher_is_better: bool = True
    regression_tolerance: float = 0.02


@runtime_checkable
class Subject(Protocol):
    """Anything that can be evaluated: a handle on a model/adapter/quantized artifact."""

    role: Role
    ref: str


@runtime_checkable
class Evaluator(Protocol):
    info: EvaluatorInfo
    metrics: tuple[MetricSpec, ...]

    def evaluate(self, subject: Subject, rows: list[dict]) -> dict[str, float]:
        """Return metric name -> value for the given held-out rows."""
        ...


def run_evaluation(evaluator: Evaluator, subjects: list[Subject], rows: list[dict]) -> list[MetricResult]:
    specs = {m.name: m for m in evaluator.metrics}
    results = []
    for subj in subjects:
        values = evaluator.evaluate(subj, rows)
        missing = set(specs) - set(values)
        if missing:
            raise ValueError(f"evaluator did not report metrics {sorted(missing)} for {subj.role}")
        for name, spec in specs.items():
            results.append(
                MetricResult(
                    metric=name,
                    subject_role=subj.role,  # type: ignore[arg-type]
                    value=float(values[name]),
                    n=len(rows),
                    higher_is_better=spec.higher_is_better,
                    category=spec.category,  # type: ignore[arg-type]
                )
            )
    return results


def compare(results: list[MetricResult], specs: tuple[MetricSpec, ...]) -> list[MetricComparison]:
    """Per-metric base vs specialist vs quantized. Deliberately no aggregate score."""
    by = {(r.metric, r.subject_role): r.value for r in results}
    out = []
    for spec in specs:
        base, spec_v = by[(spec.name, "base")], by[(spec.name, "specialist")]
        quant = by.get((spec.name, "quantized"))
        sign = 1 if spec.higher_is_better else -1
        d_bs = spec_v - base
        d_qs = None if quant is None else quant - spec_v
        out.append(
            MetricComparison(
                metric=spec.name,
                category=spec.category,
                higher_is_better=spec.higher_is_better,
                base=base,
                specialist=spec_v,
                quantized=quant,
                specialist_minus_base=round(d_bs, 10),
                quantized_minus_specialist=None if d_qs is None else round(d_qs, 10),
                regression=sign * d_bs < -spec.regression_tolerance,
                regression_tolerance=spec.regression_tolerance,
                quantization_regression=d_qs is not None and sign * d_qs < -spec.regression_tolerance,
            )
        )
    return out


def improvement_claim_allowed(
    evaluator: EvaluatorInfo, comparisons: list[MetricComparison], n_examples: int, min_examples: int = 50
) -> bool:
    """A real-improvement claim needs a non-stub evaluator, enough unseen examples, a gain on every domain metric
    and no regression anywhere."""
    if evaluator.is_stub or n_examples < min_examples:
        return False
    domain = [c for c in comparisons if c.category == "domain"]
    if not domain:
        return False
    gains = all((c.specialist_minus_base > 0) == c.higher_is_better and c.specialist_minus_base != 0 for c in domain)
    return gains and not any(c.regression or c.quantization_regression for c in comparisons)


# --------------------------------------------------------------------------- #
# Pipeline-validation stub
# --------------------------------------------------------------------------- #

BASE_VOCAB = (
    "the a an of to in and or is are was be it for on with as by at from that this which not can may "
    "should must will when where how what why do does if then than also each all any more most other "
    "use used using before after during between into over under about one two three first second"
).split()
GENERAL_PROBES = [
    "the first step is to use the tool when it is not in use",
    "what is the most common reason that this can be used with the other one",
]


@dataclass(frozen=True)
class StubSubject:
    role: Role
    ref: str
    vocab: dict[str, int]


def base_subject() -> StubSubject:
    return StubSubject("base", "stub-base", {w: 1 for w in BASE_VOCAB})


def quantize_subject(subject: StubSubject, role: Role = "quantized") -> StubSubject:
    """Toy 'quantization': drop rare learned words (count < 2) that are not base vocabulary."""
    kept = {w: c for w, c in subject.vocab.items() if c >= 2 or w in BASE_VOCAB}
    return StubSubject(role, subject.ref + "-q", kept)


class StubEvaluator:
    info = EvaluatorInfo(name="unigram_coverage_stub", version="1", is_stub=True)
    metrics = (
        MetricSpec("token_coverage", "domain"),
        MetricSpec("general_probe_coverage", "general_regression"),
    )

    @staticmethod
    def _coverage(vocab: dict[str, int], toks: list[str]) -> float:
        return sum(t in vocab for t in toks) / len(toks) if toks else 0.0

    def evaluate(self, subject: StubSubject, rows: list[dict]) -> dict[str, float]:  # type: ignore[override]
        dom = [self._coverage(subject.vocab, sp.words(r["response"])) for r in rows]
        gen = [self._coverage(subject.vocab, sp.words(p)) for p in GENERAL_PROBES]
        return {
            "token_coverage": round(sum(dom) / len(dom), 6) if dom else 0.0,
            "general_probe_coverage": round(sum(gen) / len(gen), 6),
        }
