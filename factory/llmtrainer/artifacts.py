"""Downloadable-artifact registry (``registry/artifacts/<id>.json``): loader, invariants, license gate, profiler specs.

The files are written by ``scripts/catalog/refresh_catalog.py`` (CI; needs huggingface.co). Until the first refresh they are
``unrefreshed`` placeholders with NO hash, size or revision, and an unrefreshed artifact is never downloadable.

License rule (fail closed, mirrored by the app): an artifact license is VERIFIED only if the license text fetched by CI is
byte-identical (sha256) to a canonical reference text stored under ``registry/licenses`` (Apache-2.0, or MIT with its single
copyright line normalised) AND the model-card license id agrees. Everything else is UNVERIFIED and goes through the
owner-attestation flow; secondary-source claims never become VERIFIED.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .device import ArtifactSpec, full_precision_class_for
from .schemas import Base

REPO_ROOT = Path(__file__).resolve().parents[2]
SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REV_RE = re.compile(r"^[0-9a-f]{40}$")
SKIP_FILES = {"sources.json", "index.json"}

TrainingClass = Literal["local_full", "local_partial", "inference_only", "external_only"]


class ArtifactSource(Base):
    repo: str
    repo_url: str
    file: str | None = None
    file_glob: str | None = None
    revision: str | None = None
    download_url: str | None = None


class ArtifactArch(Base):
    name: str | None = None
    layers: int | None = None
    kv_heads: int | None = None
    heads: int | None = None
    head_dim: int | None = None
    embd: int | None = None
    ffn: int | None = None
    ctx_train: int | None = None
    vocab: int | None = None


class RuntimeRow(Base):
    architecture: str | None = None
    supported: Literal["yes", "no", "unverified"] = "unverified"
    min_build: str | None = None


class ArtifactTraining(Base):
    # NOTE: JSON key is "class"; exposed as ``klass``.
    klass: TrainingClass = Field(alias="class")
    reasons: list[str]
    model_class_if_full_precision: TrainingClass
    training_source_artifact_id: str | None = None
    assumptions: str | None = None

    model_config = {"extra": "forbid", "populate_by_name": True}


class ArtifactLicenseEvidence(Base):
    expected_spdx: str
    license_text_url: str | None = None
    license_text_sha256: str | None = None
    license_text_bytes: int | None = None
    canonical_spdx: str | None = None
    canonical_sha256: str | None = None
    canonical_source_url: str | None = None
    match: Literal["exact", "mit_copyright_line", "mismatch", "unavailable"]
    model_card_licenses: dict[str, str | None]
    model_card_agrees: bool
    fetched_at: str


class ArtifactLicense(Base):
    state: Literal["VERIFIED", "UNVERIFIED"]
    expected_spdx: str
    spdx_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    evidence: ArtifactLicenseEvidence | None = None


class DownloadableArtifact(Base):
    schema_version: Literal[1] = 1
    artifact_id: str
    refresh_state: Literal["unrefreshed", "refreshed"]
    refreshed_at: str | None = None
    optional: bool = False
    published: bool | None = None
    family: str
    base_repo: str
    source: ArtifactSource
    format: Literal["gguf"] = "gguf"
    quantization: str
    precision: Literal["quantized", "f32", "f16", "bf16"]
    size_bytes: int | None = Field(default=None, ge=1)
    sha256: str | None = None
    parameter_count_nominal_b: float
    parameter_count: int | None = None
    architecture: ArtifactArch
    chat_template_present: bool | None = None
    chat_template_sha256: str | None = None
    runtime_compat: dict[str, RuntimeRow]
    training: ArtifactTraining
    license: ArtifactLicense
    notes: list[str] = Field(default_factory=list)

    @property
    def downloadable(self) -> bool:
        return (
            self.refresh_state == "refreshed" and self.published is not False and bool(self.source.file)
            and bool(self.source.revision and REV_RE.match(self.source.revision)) and bool(self.sha256 and SHA_RE.match(self.sha256))
            and bool(self.size_bytes) and bool(self.source.download_url and self.source.download_url.startswith("https://")
                                               and f"/resolve/{self.source.revision}/" in self.source.download_url
                                               and self.source.download_url.endswith("/" + (self.source.file or "")))
        )

    @model_validator(mode="after")
    def _invariants(self) -> "DownloadableArtifact":
        problems: list[str] = []
        if self.refresh_state == "unrefreshed":
            for name, v in (("size_bytes", self.size_bytes), ("sha256", self.sha256), ("source.file", self.source.file), ("source.revision", self.source.revision),
                            ("source.download_url", self.source.download_url), ("parameter_count", self.parameter_count)):
                if v is not None:
                    problems.append(f"unrefreshed artifact must not carry {name} (never ship guessed values)")
            if self.license.state != "UNVERIFIED" or self.license.evidence is not None:
                problems.append("unrefreshed artifact cannot have license evidence or a VERIFIED license")
        elif self.published is not False:
            if self.sha256 is None or not SHA_RE.match(self.sha256):
                problems.append("refreshed artifact needs a sha256:<64 hex> (LFS oid)")
            if self.source.revision is None or not REV_RE.match(self.source.revision):
                problems.append("refreshed artifact needs the 40-hex commit revision")
            if not self.downloadable:
                problems.append("refreshed artifact must have an immutable resolve/<revision>/<file> https URL, a size and a hash")
        if self.license.state == "VERIFIED":
            ev = self.license.evidence
            if ev is None or ev.match not in ("exact", "mit_copyright_line") or not ev.model_card_agrees or ev.canonical_sha256 is None or ev.license_text_sha256 is None:
                problems.append("VERIFIED license needs a canonical-text match, an agreeing model card and the recorded hashes")
            elif ev.match == "exact" and ev.license_text_sha256 != ev.canonical_sha256:
                problems.append("exact match requires license_text_sha256 == canonical_sha256")
            if self.license.spdx_id != self.license.expected_spdx:
                problems.append("VERIFIED license spdx_id must equal the curated expected_spdx")
        if self.precision == "quantized" and self.training.klass != "inference_only":
            problems.append("quantized weights cannot be trained: training.class must be inference_only")
        if self.precision != "quantized" and self.training.klass == "inference_only":
            problems.append("a full-precision artifact is not inference_only")
        if self.precision != "quantized":
            if self.training.klass != full_precision_class_for(self.parameter_count / 1e9 if self.parameter_count else self.parameter_count_nominal_b):
                problems.append("training.class does not follow the parameter-count rule")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def to_spec(self, license_state: Literal["VERIFIED", "UNVERIFIED", "DISALLOWED"] | None = None) -> ArtifactSpec:
        a = self.architecture
        hag = self.runtime_compat.get("hag-engine")
        llama = self.runtime_compat.get("llama.cpp")
        rt: Literal["yes", "unverified", "no"] = "unverified"
        if (hag and hag.supported == "no") or (llama and llama.supported == "no"):
            rt = "no"
        elif hag and hag.supported == "yes":
            rt = "yes"
        return ArtifactSpec(
            artifact_id=self.artifact_id, model_id=self.base_repo, quantization=self.quantization, precision=self.precision, sha256=self.sha256,
            size_bytes=self.size_bytes, params=float(self.parameter_count) if self.parameter_count else None, nominal_params_b=self.parameter_count_nominal_b,
            layers=a.layers, kv_heads=a.kv_heads, head_dim=a.head_dim, embd=a.embd, ffn=a.ffn, vocab=a.vocab, ctx_train=a.ctx_train,
            training_class=self.training.klass, training_source_artifact_id=self.training.training_source_artifact_id, downloadable=self.downloadable,
            license_state=license_state or self.license.state, runtime_supported=rt)


def load_canonical(root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    import hashlib

    doc = json.loads((root / "registry/licenses/canonical.json").read_text(encoding="utf-8"))
    out = {}
    for lic in doc["licenses"]:
        text = (root / "registry/licenses" / lic["file"]).read_bytes()
        if hashlib.sha256(text).hexdigest() != lic["sha256"]:
            raise ValueError(f"canonical license {lic['spdx_id']} does not match its recorded sha256")
        out[lic["spdx_id"]] = lic
    return out


def load_artifacts(directory: Path | None = None, root: Path = REPO_ROOT) -> list[DownloadableArtifact]:
    """Load and validate every artifact file; cross-checks VERIFIED licenses against the canonical index."""
    d = directory or (root / "registry/artifacts")
    canon = load_canonical(root)
    out: list[DownloadableArtifact] = []
    for f in sorted(d.glob("*.json")):
        if f.name in SKIP_FILES:
            continue
        a = DownloadableArtifact.model_validate(json.loads(f.read_text(encoding="utf-8")))
        if a.artifact_id != f.stem:
            raise ValueError(f"{f.name}: artifact_id {a.artifact_id!r} does not match the file name")
        if a.license.state == "VERIFIED":
            ev = a.license.evidence
            assert ev is not None
            c = canon.get(ev.canonical_spdx or "")
            if c is None or ev.canonical_sha256 != "sha256:" + c["sha256"]:
                raise ValueError(f"{f.name}: VERIFIED license references a canonical text this repository does not hold")
            if ev.model_card_licenses and not all((v or "").lower() in c["card_ids"] for v in ev.model_card_licenses.values()):
                raise ValueError(f"{f.name}: VERIFIED license but a model card disagrees")
        out.append(a)
    ids = [a.artifact_id for a in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate artifact ids")
    return out


def license_gate(artifact: DownloadableArtifact, use_fine_tune: bool = True, use_commercial: bool = False, use_redistribute: bool = False,
                 root: Path = REPO_ROOT) -> tuple[bool, list[str]]:
    """Fail-closed gate on an artifact's CI-derived license. Returns (allowed, blocking reasons)."""
    if artifact.license.state != "VERIFIED":
        return False, ["LICENSE_UNVERIFIED: " + ("; ".join(artifact.license.reasons) or "no CI evidence yet")]
    c = load_canonical(root).get(artifact.license.spdx_id or "")
    if c is None:
        return False, ["LICENSE_EVIDENCE_INVALID: unknown canonical license"]
    perms = c["permissions"]
    blocking = []
    if use_fine_tune and perms["fine_tuning_permitted"] != "yes":
        blocking.append("PERMISSION_NOT_GRANTED: fine-tuning")
    if use_commercial and perms["commercial_use"] != "yes":
        blocking.append("PERMISSION_NOT_GRANTED: commercial use")
    if use_redistribute and perms["redistribution_permitted"] != "yes":
        blocking.append("PERMISSION_NOT_GRANTED: redistribution")
    return not blocking, blocking
