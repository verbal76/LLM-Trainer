from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from llmtrainer import pipeline as pl
from llmtrainer.schemas import DeviceProfile, RightsInfo

FIXED_NOW = "2026-10-05T12:00:00+00:00"


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setenv("LLMTRAINER_FIXED_NOW", FIXED_NOW)


def fixture_files() -> list[Path]:
    root = Path(str(resources.files("llmtrainer").joinpath("fixtures/synthetic_corpus")))
    return sorted(root.glob("*.md"))


SYNTH = RightsInfo(
    status="synthetic",
    license_id="LLM-Trainer-synthetic-fixture",
    permitted_training="yes",
    permitted_commercial="yes",
    permitted_redistribution="yes",
    evidence="in-repo synthetic fixture",
)


@pytest.fixture
def ws(tmp_path) -> pl.Workspace:
    pl.init_project(tmp_path / "proj", "Synthetic Fixture", "pipeline-validation")
    w = pl.Workspace(tmp_path / "proj")
    for f in fixture_files():
        pl.add_source(w, f, title=f.stem, origin="in-repo synthetic fixture", rights=SYNTH)
    return w


@pytest.fixture
def built(ws):
    return ws, pl.build_dataset_for_project(ws)


def make_device(**kw) -> DeviceProfile:
    base = dict(device_id="d", name="d", device_class="c", os="android", typical_available_ram_mb=None)
    base.update(kw)
    return DeviceProfile(**base)


def flagship12(**kw) -> DeviceProfile:
    return make_device(
        device_id="flagship-12gb", name="12GB flagship", device_class="flagship", total_ram_mb=12288, os_reserve_mb=3000,
        background_reserve_mb=1500, host_app_mb=600, storage_total_mb=256000, storage_free_mb=120000, mem_bandwidth_gbps=68, **kw,
    )


def phone8(**kw) -> DeviceProfile:
    return make_device(
        device_id="phone-8gb", name="8GB phone", device_class="upper-mid", total_ram_mb=8192, os_reserve_mb=2600,
        background_reserve_mb=1200, host_app_mb=500, storage_total_mb=128000, storage_free_mb=40000, mem_bandwidth_gbps=40, **kw,
    )


def phone4(**kw) -> DeviceProfile:
    return make_device(
        device_id="phone-4gb", name="4GB phone", device_class="entry", total_ram_mb=4096, os_reserve_mb=1800,
        background_reserve_mb=800, host_app_mb=400, storage_total_mb=64000, storage_free_mb=12000, mem_bandwidth_gbps=15, **kw,
    )
