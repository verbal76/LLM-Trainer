"""Command line interface: ``python -m llmtrainer <command>`` or ``llmtrainer <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import device as dv
from . import pipeline as pl
from . import schema_export
from .clock import now_iso
from .licenses import LicenseGateError, LicenseRegistry, RequestedUse, evaluate_gate
from .packaging import validate_package
from .schemas import DeviceProfile, RightsInfo, SafetyPolicy
from .training import ExperimentConfig, estimate_resources


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _perm(s: str) -> str:
    if s not in ("yes", "no", "conditional", "unverified"):
        raise argparse.ArgumentTypeError("must be yes|no|conditional|unverified")
    return s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmtrainer", description="LLM Trainer factory CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a specialist project workspace")
    s.add_argument("dir")
    s.add_argument("--name", required=True)
    s.add_argument("--domain", required=True)
    s.add_argument("--seed", type=int, default=1234)

    s = sub.add_parser("add-source", help="ingest a text/markdown source with explicit rights metadata")
    s.add_argument("project")
    s.add_argument("file")
    s.add_argument("--title", required=True)
    s.add_argument("--origin", required=True)
    s.add_argument("--rights-status", default="unverified",
                   choices=["owned", "licensed", "public_domain", "open_license", "synthetic", "unverified", "restricted"])
    s.add_argument("--license-id")
    s.add_argument("--permit-training", type=_perm, default="unverified")
    s.add_argument("--permit-commercial", type=_perm, default="unverified")
    s.add_argument("--permit-redistribution", type=_perm, default="unverified")
    s.add_argument("--evidence")

    s = sub.add_parser("remove-source", help="remove sources and print the rebuild plan")
    s.add_argument("project")
    s.add_argument("source_ids", nargs="+")
    s.add_argument("--reason", required=True)
    s = sub.add_parser("plan-removal", help="show what removing sources would invalidate (no changes)")
    s.add_argument("project")
    s.add_argument("source_ids", nargs="+")

    s = sub.add_parser("build-dataset", help="build leakage-protected train/validation/test dataset")
    s.add_argument("project")

    s = sub.add_parser("run-experiment", help="run the PIPELINE-VALIDATION STUB experiment end to end")
    s.add_argument("project")
    s.add_argument("--base-model", default=pl.STUB_MODEL_ID)
    s.add_argument("--seed", type=int)
    s.add_argument("--no-export", action="store_true")

    s = sub.add_parser("validate-package", help="validate an export package")
    s.add_argument("package")

    s = sub.add_parser("check-license", help="evaluate the license gate for a registry entry")
    s.add_argument("registry")
    s.add_argument("model_id")
    s.add_argument("--commercial", action="store_true")
    s.add_argument("--redistribute", action="store_true")
    s.add_argument("--format")

    s = sub.add_parser("qualify-device", help="recommend deployment configs for a device profile JSON")
    s.add_argument("device", help="DeviceProfile JSON (hash fields optional)")
    s.add_argument("--no-retrieval", action="store_true")
    s.add_argument("--out")

    s = sub.add_parser("estimate-resources", help="resource estimate for a training configuration")
    s.add_argument("--params-b", type=float, required=True)
    s.add_argument("--method", choices=["lora", "qlora", "full"], default="qlora")
    s.add_argument("--tokens", type=int, required=True)
    s.add_argument("--context", type=int, default=2048)
    s.add_argument("--epochs", type=int, default=1)

    s = sub.add_parser("catalog", help="model catalog and acquisition planning (never downloads automatically)")
    csub = s.add_subparsers(dest="catalog_cmd", required=True)
    c = csub.add_parser("list", help="list catalog models with license verification state")
    c.add_argument("--registry")
    c.add_argument("--json", action="store_true")
    c = csub.add_parser("show", help="show one catalog entry (id, repo id or unique substring)")
    c.add_argument("model")
    c.add_argument("--registry")
    c = csub.add_parser("plan-acquire", help="plan a model download; transfers nothing")
    c.add_argument("model")
    c.add_argument("--variant")
    c.add_argument("--registry")
    c.add_argument("--commercial", action="store_true")
    c.add_argument("--redistribute", action="store_true")
    c.add_argument("--confirm-download", action="store_true",
                   help="authorize the download (needs a VERIFIED license for the intended use); this command still transfers nothing")

    s = sub.add_parser("export-schemas", help="write JSON Schemas")
    s.add_argument("out_dir")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (pl.WorkspaceError, LicenseGateError, KeyError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _dispatch(a) -> int:
    if a.cmd == "init":
        proj = pl.init_project(Path(a.dir), a.name, a.domain, seed=a.seed)
        _print({"project_id": proj.project_id, "content_hash": proj.content_hash})
    elif a.cmd == "add-source":
        rights = RightsInfo(
            status=a.rights_status, license_id=a.license_id, permitted_training=a.permit_training,
            permitted_commercial=a.permit_commercial, permitted_redistribution=a.permit_redistribution, evidence=a.evidence,
        )
        m = pl.add_source(pl.Workspace(Path(a.project)), Path(a.file), title=a.title, origin=a.origin, rights=rights)
        s = m.sources[-1]
        _print({"source_id": s.source_id, "chunks": len(s.chunks), "manifest_hash": m.content_hash})
    elif a.cmd == "plan-removal":
        _print(pl.removal_plan(pl.Workspace(Path(a.project)), a.source_ids).as_dict())
    elif a.cmd == "remove-source":
        _print(pl.remove_source(pl.Workspace(Path(a.project)), a.source_ids, a.reason).as_dict())
    elif a.cmd == "build-dataset":
        d = pl.build_dataset_for_project(pl.Workspace(Path(a.project)))
        _print({"dataset_id": d.dataset_id, "hash": d.content_hash, "splits": {k: v.n_examples for k, v in d.splits.items()},
                "leakage": d.leakage_report.model_dump()})
    elif a.cmd == "run-experiment":
        ws = pl.Workspace(Path(a.project))
        cfg = ExperimentConfig(base_model_id=a.base_model, method="stub", seed=a.seed if a.seed is not None else ws.project().seed)
        out = pl.run_experiment(ws, cfg, export=not a.no_export)
        _print({
            "NOTICE": "PIPELINE-VALIDATION STUB: plumbing check only, no real training, no quality claim",
            "run": out["run"].run_id, "evaluation": out["evaluation"].eval_id,
            "improvement_claim_allowed": out["evaluation"].improvement_claim_allowed,
            "comparisons": [c.model_dump() for c in out["evaluation"].comparisons],
            "model_identity": out["model"].model_identity,
            "package": str(out["package_dir"]) if out["package_dir"] else None,
        })
    elif a.cmd == "validate-package":
        rep = validate_package(a.package)
        _print(rep.as_dict())
        return 0 if rep.ok else 1
    elif a.cmd == "check-license":
        reg = LicenseRegistry.load(a.registry)
        res = evaluate_gate(reg.get(a.model_id), RequestedUse(commercial=a.commercial, redistribute=a.redistribute, export_format=a.format))
        _print(res.model_dump(mode="json"))
        return 0 if res.allowed else 1
    elif a.cmd == "qualify-device":
        dev = DeviceProfile.model_validate_json(Path(a.device).read_text(encoding="utf-8"))
        rec = dv.recommend(dev.seal(), dv.reference_candidates(0 if a.no_retrieval else 128, 0 if a.no_retrieval else 256),
                           SafetyPolicy(), created_on=now_iso())
        if a.out:
            Path(a.out).write_text(rec.to_json(), encoding="utf-8")
        _print({"device": rec.device_id, "picks": [
            {"profile": p.label, "tier": p.tier, "config": p.config.config_id if p.config else None,
             "confidence": p.confidence, "warnings": p.warnings} for p in rec.picks]})
    elif a.cmd == "estimate-resources":
        cfg = ExperimentConfig(base_model_id="n/a", method=a.method, base_params_b=a.params_b, context_tokens=a.context, epochs=a.epochs)
        _print(estimate_resources(cfg, a.tokens).model_dump())
    elif a.cmd == "catalog":
        return _catalog(a)
    elif a.cmd == "export-schemas":
        _print([str(p) for p in schema_export.write_all(a.out_dir)])
    return 0


def _catalog(a) -> int:
    from . import catalog as cat

    reg = cat.load_registry(a.registry)
    if a.catalog_cmd == "list":
        rows = cat.catalog_rows(reg)
        if a.json:
            _print(rows)
        else:
            for r in rows:
                print(f"{r['license_state']:<10} {r['evidence_level']:<24} android={r['android']:<10} {r['id']}  [{r['parameter_count']}]")
        return 0
    entry = cat.resolve_model(reg, a.model)
    if a.catalog_cmd == "show":
        _print(cat.catalog_entry(entry))
        return 0
    plan = cat.plan_acquire(entry, a.variant, RequestedUse(commercial=a.commercial, redistribute=a.redistribute))
    out = plan.as_dict()
    out["download_authorized"] = False
    if a.confirm_download:
        try:
            cat.authorize_download(plan, True)
            out["download_authorized"] = True
            out["messages"] = out["messages"] + ["authorized; this command transfers nothing - run the download as a separate explicit step"]
        except cat.AcquisitionRefused as e:
            _print(out | {"refused": e.why})
            return 1
    _print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
