"""Generate JSON Schemas (v1) for every artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import ARTIFACT_TYPES, ENTRY_SCHEMA_VERSION, SCHEMA_VERSION


def generate() -> dict[str, dict]:
    out = {}
    for kind, cls in ARTIFACT_TYPES.items():
        schema = cls.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://hotatticgames.example/llmtrainer/schemas/v{SCHEMA_VERSION}/{kind}.schema.json"
        schema["x-schema-version"] = ENTRY_SCHEMA_VERSION if kind == "base_model_license_entry" else SCHEMA_VERSION
        out[kind] = schema
    return out


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def write_all(out_dir: str | Path) -> list[Path]:
    d = Path(out_dir) / f"v{SCHEMA_VERSION}"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, schema in generate().items():
        p = d / f"{kind}.schema.json"
        p.write_text(render(schema), encoding="utf-8")
        paths.append(p)
    return paths
