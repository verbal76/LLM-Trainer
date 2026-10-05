"""Pydantic models for the v1 training-job package (see docs/studio/PACKAGE_FORMATS.md). Unknown fields are rejected."""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, Field, StringConstraints, model_validator

from ..schemas import Base, RightsInfo, Sha
from . import FORMAT_VERSION, JOB_FORMAT

IDENT_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
Ident = Annotated[str, StringConstraints(pattern=IDENT_RE)]
Perm3 = Literal["yes", "no", "conditional"]
Split = Literal["train", "validation", "test"]


def _iso(v: str) -> str:
    try:
        _dt.datetime.fromisoformat(v)
    except (ValueError, TypeError) as e:
        raise ValueError(f"not an ISO-8601 timestamp: {v!r}") from e
    return v


def _date(v: str) -> str:
    try:
        _dt.date.fromisoformat(v[:10])
    except (ValueError, TypeError) as e:
        raise ValueError(f"not an ISO date: {v!r}") from e
    return v


Timestamp = Annotated[str, AfterValidator(_iso)]
DateStr = Annotated[str, AfterValidator(_date)]


class IntendedUseJob(Base):
    commercial: bool = False
    redistribute_model: bool = False


class JobProject(Base):
    id: Ident
    name: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    purpose: str = ""
    created_at: Timestamp
    intended_use: IntendedUseJob = Field(default_factory=IntendedUseJob)


class JobBaseModel(Base):
    registry_id: str = Field(min_length=3)
    family: str = Field(min_length=1)
    exact_version: str = Field(min_length=1)
    variant: str | None = None
    source_url: str | None = None

    @model_validator(mode="after")
    def _id(self) -> Self:
        if self.registry_id != f"{self.family}@{self.exact_version}":
            raise ValueError("base_model.registry_id must equal '<family>@<exact_version>'")
        return self


class Permissions(Base):
    commercial_use: Perm3
    fine_tuning_permitted: Perm3
    derivative_adapter_permitted: Perm3
    redistribution_permitted: Perm3
    attribution_required: Perm3


class LicenseEvidence(Base):
    """Owner attestation. Every field optional at parse time: completeness is judged separately (fail closed to UNVERIFIED)."""

    model_id: str | None = None
    license_url: str | None = None
    fetched_at: str | None = None
    text_sha256: str | None = None
    evidence_level: str | None = None
    attested_by: str | None = None
    attested_on: str | None = None
    permissions: dict[str, Any] | None = None
    scope_note: str | None = None


class Method(Base):
    requested: str = Field(min_length=1)
    rationale: str = ""


class SourceIssue(Base):
    code: str
    detail: str = ""
    severity: Literal["info", "warning", "error"] = "warning"
    page: int | None = None


class JobSource(Base):
    id: Ident
    file_name: str = Field(min_length=1)
    title: str | None = None
    sha256: Sha
    size: int = Field(ge=0)
    mime: str
    ingested_at: Timestamp
    extractor: str
    extractor_version: str
    origin: str | None = None
    source_version: str | None = None
    rights: RightsInfo
    issues: list[SourceIssue] = Field(default_factory=list)


class JobManifest(Base):
    format: Literal["llmtrainer-training-job"] = JOB_FORMAT  # type: ignore[assignment]
    version: Literal[1] = FORMAT_VERSION  # type: ignore[assignment]
    job_id: Ident
    created_at: Timestamp
    created_by: dict[str, Any] | None = None
    project: JobProject
    device: dict[str, Any] | None = None
    base_model: JobBaseModel
    license_evidence: LicenseEvidence | None = None
    method: Method
    sources: list[JobSource] = Field(min_length=1)
    extensions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_sources(self) -> Self:
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate source ids")
        return self


class ChunkRow(Base):
    id: Ident
    source_id: Ident
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    role: Literal["train", "reference", "excluded"]
    exclude_reason: str | None = None
    text: str
    sha256: Sha
    origin: Literal["extracted", "ocr", "table", "edited", "synthetic"] = "extracted"
    group_id: str | None = None

    @model_validator(mode="after")
    def _ok(self) -> Self:
        if self.char_end < self.char_start:
            raise ValueError("char_end < char_start")
        if self.role == "excluded" and not self.exclude_reason:
            raise ValueError("excluded chunks need exclude_reason")
        return self


class ChunkRefRow(Base):
    source_id: Ident
    chunk_id: Ident


class ExampleRow(Base):
    example_id: Ident
    group_id: str = Field(min_length=1)
    prompt: str
    response: str
    origin: Literal["source_derived", "synthetic"] = "source_derived"
    derived_from: list[ChunkRefRow] = Field(min_length=1)
    task: str = "imported"
    quality: float = Field(default=0.5, ge=0, le=1)


class SplitConfigJob(Base):
    ratios: dict[str, float]
    group_by: Literal["document", "section"] = "document"
    near_duplicate_threshold: float = 0.8
    shingle_size: int = 5


class BuilderJob(Base):
    name: str
    version: str
    type: Literal["template", "llm", "human"]


class SplitAssignment(Base):
    group_by: Literal["document", "section"]
    groups: dict[str, Split]


class DatasetManifestJob(Base):
    dataset_id: str = Field(min_length=1)
    dataset_version: int = 1
    seed: int = 1234
    split_config: SplitConfigJob
    builder: BuilderJob
    split_assignment: SplitAssignment
    counts: dict[str, int] | None = None


class ExpectedQty(Base):
    value: float
    unit: str


class HeldoutItem(Base):
    item_id: Ident
    kind: Literal["fact", "concept"]
    question: str = Field(min_length=1)
    gold_refs: list[str] = Field(min_length=1)
    expected: ExpectedQty | None = None
    required_terms: list[str] = Field(default_factory=list)
    origin: Literal["source_derived"] = "source_derived"

    @model_validator(mode="after")
    def _kind(self) -> Self:
        if self.kind == "fact" and self.expected is None:
            raise ValueError("fact items need expected")
        if self.kind == "concept" and not self.required_terms:
            raise ValueError("concept items need required_terms")
        return self
