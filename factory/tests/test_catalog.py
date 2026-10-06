import json

import pytest

from llmtrainer import catalog as cat
from llmtrainer import cli
from llmtrainer.licenses import LicenseRegistry
from llmtrainer.schemas import BaseModelLicenseEntry

GOOD_V = {
    "state": "VERIFIED", "evidence_level": "primary_license_text_read", "source_urls": ["https://x/LICENSE"],
    "verified_on": "2026-10-05", "license_text_sha256": "sha256:" + "ab" * 32, "verified_scope": "test",
}


def mk(size=None, verification=None, variants="default", **over) -> BaseModelLicenseEntry:
    if variants == "default":
        variants = [{"variant_id": "q4", "format": "gguf", "quantization": "Q4_K_M", "source_url": "https://example.invalid/m.gguf", "size_bytes": size}]
    d = dict(
        schema_version=2, model_family="m", exact_version="1", license_id="L", license_url="u", commercial_use="yes",
        fine_tuning_permitted="yes", derivative_adapter_permitted="yes", redistribution_permitted="yes",
        attribution_required="no", verification=verification or GOOD_V, variants=variants, parameter_count="7B (nominal)",
        architecture={"layers": 32, "kv_heads": 8, "head_dim": 128, "hidden_size": 4096},
    )
    d.update(over)
    return BaseModelLicenseEntry.model_validate(d)


def test_shipped_catalog_lists_and_shows_everything():
    reg = cat.load_registry()
    assert len(cat.catalog_rows(reg)) == 22
    for e in reg.entries.values():
        rec = cat.catalog_entry(e)
        json.dumps(rec)
        assert rec["android_inference"]["claim"] == "unverified"
        assert all(v["size_bytes"] is None for v in rec["variants"])  # never guessed


def test_resolve_model_exact_substring_ambiguous():
    reg = cat.load_registry()
    assert cat.resolve_model(reg, "Qwen/Qwen3-8B").exact_version.startswith("Qwen3-8B")
    assert cat.resolve_model(reg, "smollm3").model_family == "SmolLM3"
    with pytest.raises(KeyError, match="ambiguous"):
        cat.resolve_model(reg, "qwen3")
    with pytest.raises(KeyError, match="no catalog"):
        cat.resolve_model(reg, "nonexistent")


def test_training_feasibility_uses_estimator_and_skips_effective_counts():
    f = cat.training_feasibility(mk())
    assert f["methods"]["qlora"]["gpu_memory_gib"] < f["methods"]["lora"]["gpu_memory_gib"] < f["methods"]["full"]["gpu_memory_gib"]
    eff = cat.training_feasibility(mk(parameter_count="2.3B effective (x)"))
    assert eff["params_b"] is None and eff["methods"] == {}


def test_kv_cache_only_with_full_architecture():
    assert cat.android_feasibility(mk())["kv_cache_mb_f16"] == 512.0  # 2*32*8*128*4096 tokens*2 B
    assert cat.android_feasibility(mk(architecture={"layers": 32}))["kv_cache_mb_f16"] is None


def test_unverified_license_blocks_plan_and_confirmation_cannot_override():
    e = mk(size=100, verification={"state": "UNVERIFIED", "evidence_level": "secondary_source_only", "source_urls": ["u"]})
    plan = cat.plan_acquire(e)
    assert not plan.download_permitted and "LICENSE_BLOCKED" in plan.blocked_codes and plan.performs_download is False
    with pytest.raises(cat.AcquisitionRefused):
        cat.authorize_download(plan, confirm_download=True)


def test_verified_needs_explicit_confirmation():
    plan = cat.plan_acquire(mk(size=5 * 1024**3))
    assert plan.download_permitted and plan.large_download and "5.00 GiB" in plan.size_human
    with pytest.raises(cat.AcquisitionRefused) as ei:
        cat.authorize_download(plan, confirm_download=False)
    assert "CONFIRMATION_REQUIRED" in ei.value.why
    assert cat.authorize_download(plan, confirm_download=True) is plan


def test_unknown_size_is_treated_as_large_and_small_known_size_is_not():
    assert cat.plan_acquire(mk(size=None)).large_download
    assert not cat.plan_acquire(mk(size=1024**2)).large_download


def test_verified_but_conditional_use_blocks():
    e = mk(commercial_use="conditional")
    from llmtrainer.licenses import RequestedUse

    assert not cat.plan_acquire(e, use=RequestedUse(commercial=True)).download_permitted
    assert cat.plan_acquire(e).download_permitted


def test_no_variant_or_url_blocks():
    plan = cat.plan_acquire(mk(variants=[]))
    assert "NO_SOURCE_URL" in plan.blocked_codes and not plan.download_permitted
    with pytest.raises(KeyError, match="unknown variant"):
        cat.plan_acquire(mk(), "nope")


def test_catalog_module_has_no_network_or_download_code():
    src = open(cat.__file__).read()
    for word in ("urllib", "requests", "socket", "subprocess", "urlopen"):
        assert word not in src


def test_cli_list_show_plan(capsys):
    assert cli.main(["catalog", "list"]) == 0
    assert "UNVERIFIED" in capsys.readouterr().out
    assert cli.main(["catalog", "show", "qwen3-8b"]) == 0
    assert json.loads(capsys.readouterr().out)["license"]["claims_are_binding"] is False
    assert cli.main(["catalog", "plan-acquire", "qwen3-8b"]) == 0
    assert json.loads(capsys.readouterr().out)["download_permitted"] is False
    assert cli.main(["catalog", "plan-acquire", "qwen3-8b", "--confirm-download"]) == 1
    assert json.loads(capsys.readouterr().out)["download_authorized"] is False


def test_cli_confirm_with_verified_registry(tmp_path, capsys):
    (tmp_path / "m.json").write_text(json.dumps(mk(size=10).model_dump(mode="json")))
    args = ["catalog", "plan-acquire", "m@1", "--registry", str(tmp_path)]
    assert cli.main(args + ["--confirm-download"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["download_authorized"] is True and out["performs_download"] is False
    assert LicenseRegistry.load(tmp_path)
