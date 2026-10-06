# ADR 0014: Versioned, content-addressed contracts (pydantic v2 + JSON Schema)

- Status: proposed (schemas themselves are owned by the schema agent)
- Date: 2026-10-05

## Decision
Contracts are JSON documents with `schema` and `schema_version` (start at 1, no fabricated migration history), validated by JSON Schema (jsonschema, MIT) and mirrored as pydantic v2 models (MIT) in Python and generated/hand-checked Kotlin data classes. Identity of an artifact = sha256 of its canonical JSON (sorted keys, no insignificant whitespace, UTF-8) for manifests and sha256 of bytes for blobs; manifests reference inputs by hash. Two materially different artifacts must never share an id; id collisions with differing bytes are a hard error.

## Rationale
Reproducibility, tamper evidence, and cross-language consumption (the Kotlin app and external apps validate packages without Python). JSON Schema is the language-neutral spec; pydantic gives ergonomic validation inside the factory.

## Alternatives rejected
Protobuf/FlatBuffers (binary, harder to diff/inspect), YAML-only configs for artifacts (ambiguity in typing), unversioned ad-hoc JSON.

## Consequences
CI validates every example artifact and registry file against schemas; canonicalization must be specified identically for Kotlin and Python and covered by golden hash vectors.
