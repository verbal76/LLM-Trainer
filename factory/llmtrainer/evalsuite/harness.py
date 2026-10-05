"""Model-agnostic evaluation harness: base vs specialist (vs quantized) on the same held-out items.

A subject is just ``generate(prompt) -> str``. Retrieval is an optional ``retrieve(query, k) -> [{"ref","text"}]``.
Metrics are reported one by one; there is no aggregate score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .. import evaluation as ev
from ..hashing import hash_obj, short
from ..schemas import DatasetManifest, EvalSubject, EvaluationRun, EvaluatorInfo, MetricResult
from . import metrics as m
from .items import EvalItem

EVALUATOR = EvaluatorInfo(name="evalsuite_heuristic", version="1", is_stub=False)
RAG_PROMPT = (
    "Answer using only the sources below. Cite the sources you use in square brackets, e.g. [source/chunk].\n\n"
    "Sources:\n{sources}\n\nQuestion: {question}"
)
MIN_CLAIM_ITEMS = 50
SMALL_N = 30


@dataclass
class CallableSubject:
    role: str  # base | specialist | quantized
    ref: str
    generate: Callable[[str], str]
    artifact_hash: str | None = None


Pairs = list[tuple[float, float]]  # per item (numerator, denominator)


@dataclass
class SubjectReport:
    pairs: dict[str, Pairs] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)
    tokens: int = 0
    seconds: float = 0.0

    def performance(self) -> dict[str, float]:
        lat = sorted(self.latencies_ms)
        if not lat:
            return {}
        return {
            "latency_ms_mean": round(sum(lat) / len(lat), 3), "latency_ms_p50": round(lat[len(lat) // 2], 3),
            "latency_ms_p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
            "tokens_per_s": round(self.tokens / self.seconds, 3) if self.seconds > 0 else 0.0,
        }


# metric name -> (category, higher_is_better)
METRIC_SPECS = {
    "terminology_coverage": ("domain", True),
    "closed_book_exact_fact_accuracy": ("grounding", True),
    "general_probe_pass_rate": ("general_regression", True),
    "rag_exact_fact_accuracy": ("domain", True),
    "rag_unsupported_claim_rate": ("safety", False),
    "citation_accuracy": ("grounding", True),
}


@dataclass
class SuiteResult:
    items: list[EvalItem]
    reports: dict[str, SubjectReport]
    metrics: list[MetricResult]
    specs: tuple[ev.MetricSpec, ...]
    comparisons: list
    delta_ci: dict[str, tuple[float, float]]
    intervals: dict[tuple[str, str], tuple[float, float]]
    caveats: list[str]
    used_retrieval: bool

    def deltas_significant(self) -> bool:
        """Every domain metric improved with a paired-bootstrap 95% CI that excludes 0."""
        dom = [c for c in self.comparisons if c.category == "domain"]
        return bool(dom) and all((self.delta_ci[c.metric][0] > 0) if c.higher_is_better else (self.delta_ci[c.metric][1] < 0) for c in dom)


def _timed(subject: CallableSubject, prompt: str, rep: SubjectReport, clock, count_tokens) -> str:
    t0 = clock()
    out = subject.generate(prompt)
    dt = clock() - t0
    rep.latencies_ms.append(dt * 1000)
    rep.seconds += dt
    rep.tokens += count_tokens(out)
    return out


def run_suite(
    subjects: list[CallableSubject], items: list[EvalItem], *, retrieve=None, k: int = 3, seed: int = 0,
    clock=time.perf_counter, count_tokens=lambda s: len(s.split()), probes=m.GENERAL_PROBES,
) -> SuiteResult:
    roles = [s.role for s in subjects]
    if "base" not in roles or "specialist" not in roles:
        raise ValueError("need base and specialist subjects")
    if not items:
        raise ValueError("no evaluation items")
    reports = {s.role: SubjectReport() for s in subjects}
    for s in subjects:
        rep = reports[s.role]
        P = rep.pairs
        for it in items:  # closed-book: no sources provided
            ans = _timed(s, it.question, rep, clock, count_tokens)
            if it.kind == "concept":
                n, d = m.terminology_use(ans, list(it.required_terms))
                P.setdefault("terminology_coverage", []).append((n, d))
            else:
                P.setdefault("closed_book_exact_fact_accuracy", []).append((float(m.fact_correct(ans, it.expected)), 1.0))
        for q, pat in probes:
            P.setdefault("general_probe_pass_rate", []).append((float(m.probe_pass(_timed(s, q, rep, clock, count_tokens), pat)), 1.0))
        if retrieve is not None:
            for it in items:
                got = retrieve(it.question, k)
                src = "\n".join(f"[{c['ref']}] {c['text']}" for c in got)
                ans = _timed(s, RAG_PROMPT.format(sources=src, question=it.question), rep, clock, count_tokens)
                bad, tot = m.unsupported_claims(ans, " ".join(c["text"] for c in got))
                P.setdefault("rag_unsupported_claim_rate", []).append((bad, tot))
                texts = {c["ref"]: c["text"] for c in got}
                cited = m.parse_citations(ans)
                if it.kind == "fact":
                    P.setdefault("rag_exact_fact_accuracy", []).append((float(m.fact_correct(ans, it.expected)), 1.0))
                    good = lambda r: r in texts and any(q.matches(it.expected) for q in m.extract_quantities(texts[r]))  # noqa: E731
                else:
                    good = lambda r: r in texts and r in it.gold_refs  # noqa: E731
                ok = bool(cited) and all(r in texts for r in cited) and any(good(r) for r in cited)
                P.setdefault("citation_accuracy", []).append((float(ok), 1.0))

    names = [n for n in METRIC_SPECS if n in reports["base"].pairs]
    # an all-zero-denominator metric (e.g. no claims at all) is still reported, as 0.0 with n=0
    results, intervals = [], {}
    for role, rep in reports.items():
        for n in names:
            pr = rep.pairs[n]
            cat, hib = METRIC_SPECS[n]
            results.append(MetricResult(metric=n, subject_role=role, value=round(m.ratio(pr), 6), n=sum(1 for _, d in pr if d), higher_is_better=hib, category=cat))
            intervals[(n, role)] = m.bootstrap_ci(pr, seed=seed)
    specs = tuple(ev.MetricSpec(n, METRIC_SPECS[n][0], METRIC_SPECS[n][1]) for n in names)
    comps = ev.compare(results, specs)
    delta_ci = {n: m.paired_delta_ci(reports["base"].pairs[n], reports["specialist"].pairs[n], seed=seed) for n in names}
    caveats = [
        "Metrics are deterministic lexical/numeric heuristics (see evalsuite/metrics.py limits), not human or model judgements.",
        f"{len(items)} held-out items ({sum(i.kind == 'fact' for i in items)} fact, {sum(i.kind == 'concept' for i in items)} concept); "
        f"{len(probes)} general probes (a canary, not a benchmark).",
        "95% intervals are item-level bootstrap (seeded); intervals for different metrics are not corrected for multiple comparisons.",
    ]
    for r in results:
        if r.subject_role == "specialist" and r.n < SMALL_N:
            caveats.append(f"{r.metric}: n={r.n} < {SMALL_N}: statistically weak.")
    if retrieve is None:
        caveats.append("No retrieval callable given: grounding/citation/unsupported-claim metrics were not measured.")
    caveats.append("closed_book_exact_fact_accuracy on genuinely unseen documents is expected to be near 0 (exact facts belong to retrieval).")
    return SuiteResult(items, reports, results, specs, comps, delta_ci, intervals, caveats, retrieve is not None)


def to_evaluation_run(
    suite: SuiteResult, *, project_id: str, dataset: DatasetManifest, train_eval_group_overlap: int, evaluated_on: str,
    subjects: list[CallableSubject], synthetic: bool | None = None,
) -> EvaluationRun:
    """Build the sealed EvaluationRun. improvement_claim_allowed is False for synthetic data, too few items, a
    non-significant domain delta, or any regression (existing rules)."""
    synthetic = any(i.synthetic for i in suite.items) if synthetic is None else synthetic
    allowed = (
        not synthetic
        and ev.improvement_claim_allowed(EVALUATOR, suite.comparisons, len(suite.items), MIN_CLAIM_ITEMS)
        and suite.deltas_significant()
    )
    notes = list(suite.caveats)
    if synthetic:
        notes.append("Evaluation data comes from synthetic sources: improvement_claim_allowed is forced to false.")
    for c in suite.comparisons:
        lo, hi = suite.delta_ci[c.metric]
        notes.append(f"{c.metric}: base={c.base:.3f} specialist={c.specialist:.3f} delta={c.specialist_minus_base:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}]")
    for role, rep in suite.reports.items():
        notes.append(f"performance[{role}]: {rep.performance()}")
    return EvaluationRun(
        eval_id="eval-" + short(hash_obj([dataset.content_hash, [s.ref for s in subjects], [i.item_id for i in suite.items], evaluated_on])),
        project_id=project_id, dataset_hash=dataset.content_hash, eval_split="test", n_eval_examples=len(suite.items),
        train_eval_group_overlap=train_eval_group_overlap, evaluator=EVALUATOR,
        subjects=[EvalSubject(role=s.role, model_ref=s.ref, artifact_hash=s.artifact_hash) for s in subjects],  # type: ignore[arg-type]
        metrics=suite.metrics, comparisons=suite.comparisons, improvement_claim_allowed=allowed, notes=notes, evaluated_on=evaluated_on,
    ).seal()


class LexicalRetriever:
    """Tiny lexical retriever over {ref,text} chunks (same tokenization as the package reference index)."""

    def __init__(self, chunks: list[dict]):
        from ..packaging import tokenize

        self._tok = tokenize
        self.chunks = chunks
        self._sets = [set(tokenize(c["text"])) for c in chunks]

    def __call__(self, query: str, k: int = 3) -> list[dict]:
        q = set(self._tok(query))
        scored = sorted(((-len(q & s), c["ref"], i) for i, s in enumerate(self._sets) for c in [self.chunks[i]]))
        return [self.chunks[i] for sc, _, i in scored[:k] if sc < 0]
