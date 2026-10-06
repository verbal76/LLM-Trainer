# ADR 0013: Document ingestion tooling and dependency-license policy

- Status: proposed
- Date: 2026-10-05

## Decision
Primary parser: Docling (MIT, 2.133.0 on PyPI; IBM Research Zurich origin, LF AI & Data hosted; "individual models may carry their own separate license terms" per PyPI text, so model weights Docling downloads must go through our license check). Fallbacks: pypdf/trafilatura-class permissive libraries and optional OCR engines, all license-recorded in a dependency table. PyMuPDF (AGPL-3.0 or Artifex commercial, per PyPI 1.28.2) is **not** a default dependency; it may be enabled only by an explicit workstation-only flag because linking AGPL code into a distributed product has copyleft consequences (decision for the owner/legal, no conclusion asserted).

## Rationale
Ingested sources may be private/proprietary: parsing runs locally; no cloud OCR/parsing without explicit authorization (`CLAUDE.md`). Every extraction records parser name+version, page/offset maps, confidence, and warnings (tables, OCR).

## Alternatives rejected
Cloud document APIs (data leaves machine), AGPL parsers as default (license risk), ad-hoc regex PDF extraction (unreliable for tables).

## Uncertainty
Parser quality on motorcycle/HVAC-style manuals (tables, diagrams, scans) not evaluated; first experiment uses synthetic text only.
