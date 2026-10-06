"""Golden vectors for device qualification: the Python spec regenerates them; the Kotlin port (qualify/) must match them."""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "gen_golden_device.py"
spec = importlib.util.spec_from_file_location("gen_golden_device", TOOL)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_golden_vectors_have_not_drifted():
    committed = gen.GOLDEN_PATH.read_text()
    assert committed == gen.render(gen.build()), (
        "device_qualification.v1.json is out of date vs llmtrainer.device: run `python factory/tools/gen_golden_device.py` "
        "and port any behavior change to qualify/ (Kotlin)"
    )


def test_golden_vectors_cover_the_required_situations():
    doc = json.loads(gen.GOLDEN_PATH.read_text())
    names = {c["name"] for c in doc["cases"]}
    for needed in ("flagship12/estimates-only", "phone8/estimates-only", "phone6/estimates-only", "phone4/with-retrieval-nothing-fits",
                   "flagship12/low-memory-state", "flagship12/tiny-free-storage", "snapshot/thermal-severe-withheld",
                   "snapshot/missing-avail", "measured/7b-passes-recommended"):
        assert needed in names
    assert len(doc["cases"]) >= 40


def test_no_estimate_only_case_is_ever_recommended():
    doc = json.loads(gen.GOLDEN_PATH.read_text())
    for c in doc["cases"]:
        if c["name"].startswith(("measured/", "negative")):
            continue
        assert all(p["tier"] != "recommended" for p in c["expected"]["picks"]), c["name"]


def test_snapshot_without_core_fields_is_withheld():
    prof, notes, withhold = gen.profile_from_snapshot({})
    assert prof is None and withhold == ["missing_core_fields"]


def test_hot_snapshot_is_withheld_not_recommended():
    cands = gen.catalog_candidates(gen.reference_catalog())
    _, _, w, picks = gen.recommend_for_snapshot(gen.snap(8, 4, 50, thermalStatus=3), cands)
    assert w == ["thermal_throttled"] and all(p["tier"] == "none" for p in picks)
