"""Deterministic, transparent metric primitives for the evaluation suite.

LIMITS (read before trusting a number):

* Everything here is lexical/numeric matching. It does not understand meaning.
* ``extract_quantities`` knows a fixed unit table; facts in other units are not scored (the item
  generator skips them rather than guessing).
* ``unsupported_claims`` splits an answer into sentences and calls a sentence supported if >= 60%
  of its content words occur in the provided source text and every number in it occurs there too.
  It therefore MISSES wrong-but-lexically-similar claims (negations, swapped entities, wrong
  relations) and FLAGS correct paraphrases that use different words. Treat the rate as a coarse
  screening signal, not a truth judgement.
* Terminology coverage rewards using the domain's vocabulary; it does not check the vocabulary is
  used correctly.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

STOPWORDS = frozenset(
    "the a an of to in and or is are was were be been it its for on with as by at from that this these those which not can may "
    "should must will when where how what why do does did if then than also each all any more most other use used using before "
    "after during between into over under about one two three first second would could has have had their there they them you "
    "your we our i he she his her but so such only same both either neither per via being".split()
)

# unit alias -> (dimension, factor to the dimension's base unit); temperature handled separately
_UNITS: dict[str, tuple[str, float]] = {}


def _reg(dim: str, factor: float, *aliases: str) -> None:
    for a in aliases:
        _UNITS[a] = (dim, factor)


_reg("length", 1.0, "mm", "millimeter", "millimeters", "millimetre", "millimetres")
_reg("length", 10.0, "cm", "centimeter", "centimeters")
_reg("length", 1000.0, "m", "meter", "meters", "metre", "metres")
_reg("length", 25.4, "in", "inch", "inches")
_reg("torque", 1.0, "n·m", "n-m", "nm", "n.m", "n m", "newton-meter", "newton-meters")
_reg("torque", 1.3558179, "lb-ft", "lb·ft", "ft-lb", "ft·lb", "ft-lbs", "lb-ft.", "lbf-ft", "ft-lbf")
_reg("torque", 0.1129848, "in-lb", "in·lb", "lb-in", "in-lbs", "lbf-in")
_reg("pressure", 1.0, "kpa")
_reg("pressure", 100.0, "bar")
_reg("pressure", 6.894757, "psi")
_reg("pressure", 1000.0, "mpa")
_reg("voltage", 1.0, "v", "volt", "volts")
_reg("current", 1.0, "a", "amp", "amps", "ampere", "amperes")
_reg("current", 0.001, "ma", "milliamp", "milliamps")
_reg("resistance", 1.0, "ohm", "ohms", "ω")
_reg("resistance", 1000.0, "kohm", "kω")
_reg("frequency", 1.0, "hz")
_reg("rpm", 1.0, "rpm")
_reg("mass", 1.0, "g", "gram", "grams")
_reg("mass", 1000.0, "kg", "kilogram", "kilograms")
_reg("mass", 453.59237, "lb", "lbs", "pound", "pounds")
_reg("volume", 1.0, "ml", "milliliter", "milliliters")
_reg("volume", 1000.0, "l", "liter", "liters", "litre", "litres")
_reg("volume", 946.353, "qt", "quart", "quarts")
_reg("time", 1.0, "s", "sec", "second", "seconds")
_reg("time", 60.0, "min", "mins", "minute", "minutes")
_reg("time", 3600.0, "h", "hr", "hrs", "hour", "hours")
_reg("time", 86400.0, "day", "days")
_reg("percent", 1.0, "%", "percent")
_reg("temp_c", 1.0, "°c", "degc", "celsius")
_reg("temp_f", 1.0, "°f", "degf", "fahrenheit")

_UNIT_RE = "|".join(sorted((re.escape(u) for u in _UNITS), key=len, reverse=True))
_QTY = re.compile(rf"(?<![\w.])([-−]?\d+(?:[.,]\d+)?|\.\d+)\s*({_UNIT_RE})(?![a-zA-Z])", re.IGNORECASE)
_NUM = re.compile(r"[-−]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    dim: str
    base_value: float  # in the dimension's base unit

    def matches(self, other: "Quantity") -> bool:
        if self.dim != other.dim:
            return False
        if self.unit == other.unit:
            return math.isclose(self.value, other.value, rel_tol=1e-6, abs_tol=1e-9)
        return math.isclose(self.base_value, other.base_value, rel_tol=0.02, abs_tol=1e-9)  # manuals round conversions


def _to_base(value: float, unit: str) -> tuple[str, float]:
    dim, f = _UNITS[unit]
    if dim == "temp_f":
        return "temp_c", (value - 32) * 5 / 9
    return dim, value * f


def extract_quantities(text: str) -> list[Quantity]:
    out = []
    for m in _QTY.finditer(text):
        raw = m.group(1).replace("−", "-").replace(",", ".")
        unit = m.group(2).lower()
        try:
            v = float(raw)
        except ValueError:
            continue
        dim, base = _to_base(v, unit)
        out.append(Quantity(v, unit, dim, base))
    return out


def fact_correct(answer: str, expected: Quantity) -> bool:
    """Right iff the expected quantity appears and no *different* quantity of the same dimension is also asserted."""
    same_dim = [q for q in extract_quantities(answer) if q.dim == expected.dim]
    if not any(q.matches(expected) for q in same_dim):
        return False
    return all(q.matches(expected) for q in same_dim)


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) >= 3]


def _numbers(text: str) -> set[str]:
    return {n.replace("−", "-").rstrip("0").rstrip(".") if "." in n else n for n in _NUM.findall(text)}


_HEDGES = re.compile(r"\b(i do not know|i don't know|not specified|no information|cannot determine|unable to find|not mentioned)\b", re.I)


def split_claims(answer: str) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()):
        s = re.sub(r"\[[^\[\]]*\]", "", s).strip()  # citations are not claims
        if not s or s.endswith("?") or _HEDGES.search(s) or len(content_words(s)) < 3:
            continue
        out.append(s)
    return out


def unsupported_claims(answer: str, support_text: str, *, min_support: float = 0.6) -> tuple[int, int]:
    """Return (unsupported, total) claims in ``answer`` relative to ``support_text``."""
    support_words = set(content_words(support_text))
    support_nums = _numbers(support_text)
    claims = split_claims(answer)
    bad = 0
    for c in claims:
        words = content_words(c)
        ratio = sum(w in support_words for w in words) / len(words)
        nums_ok = _numbers(c) <= support_nums
        if ratio < min_support or not nums_ok:
            bad += 1
    return bad, len(claims)


def terminology_use(answer: str, required_terms: list[str]) -> tuple[int, int]:
    words = set(re.findall(r"[a-z][a-z0-9]+", answer.lower()))
    return sum(t in words for t in required_terms), len(required_terms)


_CITE = re.compile(r"\[([^\[\]]+)\]")


def parse_citations(answer: str) -> list[str]:
    refs = []
    for m in _CITE.finditer(answer):
        for part in m.group(1).split(","):
            part = part.strip()
            if part:
                refs.append(part)
    return refs


# --------------------------------------------------------------------------- #
# Fixed general-capability retention probes (tiny on purpose; a canary, not a benchmark)
# --------------------------------------------------------------------------- #

GENERAL_PROBES: tuple[tuple[str, str], ...] = (
    ("What is 12 plus 15? Answer with just the number.", r"\b27\b"),
    ("What is 9 times 8? Answer with just the number.", r"\b72\b"),
    ("What is the capital of France? One word.", r"\bparis\b"),
    ("What is the chemical symbol for water? Answer with the formula.", r"\bh2o\b|h₂o"),
    ("Which planet is known as the Red Planet? One word.", r"\bmars\b"),
    ("What is the opposite of 'hot'? One word.", r"\bcold\b"),
    ("Spell the plural of the word 'mouse'.", r"\bmice\b"),
    ("How many days are in a week? Answer with just the number.", r"\b7\b|\bseven\b"),
    ("Complete the sentence: The sun rises in the ...", r"\beast\b"),
    ("What is 100 divided by 4? Answer with just the number.", r"\b25\b"),
    ("Is the number 17 prime? Answer yes or no.", r"\byes\b"),
    ("Write the word 'banana' in all capital letters.", r"\bBANANA\b"),
)


def probe_pass(answer: str, pattern: str) -> bool:
    flags = 0 if pattern == r"\bBANANA\b" else re.IGNORECASE
    return re.search(pattern, answer, flags) is not None


# --------------------------------------------------------------------------- #
# Uncertainty: deterministic bootstrap over items of ratio metrics
# --------------------------------------------------------------------------- #


def ratio(pairs: list[tuple[float, float]]) -> float:
    den = sum(d for _, d in pairs)
    return sum(n for n, _ in pairs) / den if den else 0.0


def bootstrap_ci(pairs: list[tuple[float, float]], *, seed: int = 0, n_boot: int = 1000) -> tuple[float, float]:
    if not pairs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(pairs)
    vals = sorted(ratio([pairs[rng.randrange(k)] for _ in range(k)]) for _ in range(n_boot))
    return (vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1])


def paired_delta_ci(
    a: list[tuple[float, float]], b: list[tuple[float, float]], *, seed: int = 0, n_boot: int = 1000
) -> tuple[float, float]:
    """95% bootstrap CI of ratio(b) - ratio(a) with items resampled jointly (same items for both subjects)."""
    if not a or len(a) != len(b):
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(a)
    ds = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        ds.append(ratio([b[i] for i in idx]) - ratio([a[i] for i in idx]))
    ds.sort()
    return (ds[int(0.025 * n_boot)], ds[int(0.975 * n_boot) - 1])
