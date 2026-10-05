"""Versioned artifact contracts (schema_version 1).

Every artifact derives from :class:`Artifact`: it carries ``schema_version``,
a ``kind`` discriminator and a ``content_hash`` computed over the canonical
JSON of every other field. Call :meth:`Artifact.seal` before persisting and
:meth:`Artifact.verify` after loading.

Schema versions start honestly at 1; there is no migration history. The one
exception is the base-model registry entry, which is a separately versioned
contract at schema_version 2 (see ``ENTRY_SCHEMA_VERSION`` below); v1 entries
are rejected.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .hashing import hash_obj

SCHEMA_VERSION = 1

Sha = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
Permission = Literal["yes", "no", "conditional", "unverified"]
Verdict = Literal["fit", "no_fit", "unverified"]
Confidence = Literal["measured", "partial", "estimated"]
IsoDate = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}")]


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class Artifact(Base):
    schema_version: Literal[1] = SCHEMA_VERSION
    kind: str
    content_hash: str = ""

    def compute_hash(self) -> str:
        return hash_obj(self.model_dump(mode="json", exclude={"content_hash"}))

    def seal(self) -> Self:
        return self.model_copy(update={"content_hash": self.compute_hash()})

    def verify(self) -> bool:
        return bool(self.content_hash) and self.content_hash == self.compute_hash()

    @property
    def identity(self) -> str:
        if not self.content_hash:
            raise ValueError("artifact is not sealed")
        return self.content_hash

    def to_json(self) -> str:
        import json

        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- #
# Base-model license registry entry (entry schema_version 2)
# --------------------------------------------------------------------------- #
#
# Entry format history:
#   v1  permissions were trusted at face value ("yes" passed the gate).
#   v2  BREAKING. Adds ``verification.state`` / ``verification.evidence_level``.
#       The permission fields are now *claims*: the license gate ignores them
#       unless ``verification.state == "VERIFIED"``, which itself requires a
#       primary license/model-card read and the sha256 of the text inspected.
#       v1 entries are rejected by the loader (never silently accepted).

ENTRY_SCHEMA_VERSION = 2

VerificationState = Literal["VERIFIED", "UNVERIFIED", "DISALLOWED"]
EvidenceLevel = Literal["primary_license_text_read", "primary_model_card_read", "secondary_source_only", "none"]
PRIMARY_EVIDENCE: frozenset[str] = frozenset({"primary_license_text_read", "primary_model_card_read"})


class LicenseVerification(Base):
    state: VerificationState = Field(
        description="Gate-relevant verification state. Only VERIFIED lets the permission claims count; "
        "UNVERIFIED and DISALLOWED fail closed."
    )
    evidence_level: EvidenceLevel
    source_urls: list[str] = Field(default_factory=list)
    verified_on: IsoDate | None = None
    license_text_sha256: Sha | None = Field(
        default=None, description="sha256 of the exact license text that was inspected (required for VERIFIED)."
    )
    license_text_url: str | None = None
    verified_scope: str | None = Field(
        default=None, description="What exactly was verified (which text, which uses). Required for VERIFIED."
    )
    disallowed_reason: str | None = Field(default=None, description="Required for DISALLOWED.")
    uncertainties: list[str] = Field(default_factory=list)

    def evidence_problems(self) -> list[str]:
        """Why this verification block cannot support its declared state (empty when consistent)."""
        problems: list[str] = []
        if self.state == "VERIFIED":
            if self.evidence_level not in PRIMARY_EVIDENCE:
                problems.append(f"VERIFIED requires primary evidence, got evidence_level={self.evidence_level}")
            if not self.license_text_sha256:
                problems.append("VERIFIED requires license_text_sha256")
            if not self.verified_on:
                problems.append("VERIFIED requires verified_on")
            if not self.source_urls:
                problems.append("VERIFIED requires source_urls")
            if not self.verified_scope:
                problems.append("VERIFIED requires verified_scope")
        elif self.state == "DISALLOWED":
            if self.evidence_level not in PRIMARY_EVIDENCE:
                problems.append(f"DISALLOWED is not allowed with evidence_level={self.evidence_level}; must be UNVERIFIED")
            if not self.disallowed_reason:
                problems.append("DISALLOWED requires disallowed_reason")
        return problems

    @model_validator(mode="after")
    def _state_matches_evidence(self) -> Self:
        problems = self.evidence_problems()
        if problems:
            raise ValueError("; ".join(problems))
        return self


class ArchitectureInfo(Base):
    layers: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    hidden_size: int | None = None
    attention: str | None = Field(default=None, description="e.g. 'GQA'; null when not evidenced.")
    notes: str | None = None


class ModelVariant(Base):
    """One downloadable form of a model. Sizes/hashes are null unless actually known."""

    variant_id: str
    format: str
    quantization: str | None = None
    source_url: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: Sha | None = None
    size_evidence: str | None = None


class AndroidInference(Base):
    state: Literal["unverified", "evidenced", "infeasible"] = "unverified"
    runtime_candidates: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _evidenced_needs_evidence(self) -> Self:
        if self.state != "unverified" and not self.evidence:
            raise ValueError("android_inference state other than 'unverified' requires evidence")
        return self


class BaseModelLicenseEntry(Base):
    """Machine-readable license facts for one exact base-model version.

    The ``commercial_use`` .. ``attribution_required`` fields are *claims as
    recorded*; they carry no force unless ``verification.state == "VERIFIED"``.
    """

    schema_version: Literal[2] = ENTRY_SCHEMA_VERSION
    model_family: str
    exact_version: str
    license_id: str
    license_url: str
    commercial_use: Permission = Field(description="Claim; ignored by the gate unless VERIFIED.")
    fine_tuning_permitted: Permission = Field(description="Claim; ignored by the gate unless VERIFIED.")
    derivative_adapter_permitted: Permission = Field(description="Claim; ignored by the gate unless VERIFIED.")
    redistribution_permitted: Permission = Field(description="Claim; ignored by the gate unless VERIFIED.")
    attribution_required: Permission = Field(description="Claim; ignored by the gate unless VERIFIED.")
    restrictions: list[str] = Field(default_factory=list)
    supported_formats: list[str] = Field(default_factory=list)
    verification: LicenseVerification
    # catalog facts (never guessed; null when unknown)
    hf_repo_or_source: str | None = None
    parameter_count: str | None = None
    context_length: int | None = None
    architecture: ArchitectureInfo | None = None
    variants: list[ModelVariant] = Field(default_factory=list)
    android_inference: AndroidInference = Field(default_factory=AndroidInference)

    @property
    def entry_id(self) -> str:
        return f"{self.model_family}@{self.exact_version}"

    @model_validator(mode="before")
    @classmethod
    def _reject_old_entry_formats(cls, data: Any) -> Any:
        if isinstance(data, dict):
            v = data.get("schema_version", 1)
            if v != ENTRY_SCHEMA_VERSION:
                raise ValueError(
                    f"base-model entry schema_version {v!r} is not supported (need {ENTRY_SCHEMA_VERSION}); "
                    "v1 entries trusted permission claims without verification state and are rejected, "
                    "not silently upgraded - re-review and add verification.state/evidence_level"
                )
        return data

    @model_validator(mode="after")
    def _claims_need_sources(self) -> Self:
        claims = (
            self.commercial_use,
            self.fine_tuning_permitted,
            self.derivative_adapter_permitted,
            self.redistribution_permitted,
            self.attribution_required,
        )
        if any(c != "unverified" for c in claims) and not self.verification.source_urls:
            raise ValueError("recording a permission claim requires verification.source_urls")
        return self


class GateCheck(Base):
    requirement: str
    field: str
    value: Permission | str
    passed: bool
    reason: str
    code: str = Field(default="OK", description="Machine-readable reason code.")


class GateResult(Base):
    entry_id: str
    entry_hash: Sha
    requested_use: dict[str, Any]
    allowed: bool
    checks: list[GateCheck]
    blocking: list[str]
    attribution_required: Permission
    restrictions: list[str] = Field(default_factory=list)
    verification_state: VerificationState = "UNVERIFIED"
    reason_codes: list[str] = Field(default_factory=list, description="Machine-readable codes for every blocking check.")
    remediation: list[str] = Field(default_factory=list, description="What the owner must do to unblock.")


# --------------------------------------------------------------------------- #
# 1. SpecialistProject
# --------------------------------------------------------------------------- #


class IntendedUse(Base):
    commercial: bool = False
    redistribute_model: bool = False
    produce_adapter: bool = True
    export_formats: list[str] = Field(default_factory=list)


class SplitConfig(Base):
    ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "validation": 0.15, "test": 0.15}
    )
    group_by: Literal["document", "section"] = "document"
    near_duplicate_threshold: float = 0.8
    shingle_size: int = 5

    @model_validator(mode="after")
    def _ratios(self) -> Self:
        if set(self.ratios) != {"train", "validation", "test"}:
            raise ValueError("ratios must define exactly train, validation, test")
        if abs(sum(self.ratios.values()) - 1.0) > 1e-6 or min(self.ratios.values()) <= 0:
            raise ValueError("ratios must be positive and sum to 1")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        return self


class SpecialistProject(Artifact):
    kind: Literal["specialist_project"] = "specialist_project"
    project_id: str
    name: str
    domain: str
    description: str = ""
    created_on: str
    seed: int = 1234
    intended_use: IntendedUse = Field(default_factory=IntendedUse)
    split_config: SplitConfig = Field(default_factory=SplitConfig)
    base_model_candidates: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 2. SourceManifest
# --------------------------------------------------------------------------- #


class RightsInfo(Base):
    status: Literal[
        "owned", "licensed", "public_domain", "open_license", "synthetic", "unverified", "restricted"
    ] = "unverified"
    license_id: str | None = None
    permitted_training: Permission = "unverified"
    permitted_commercial: Permission = "unverified"
    permitted_redistribution: Permission = "unverified"
    evidence: str | None = None
    notes: str | None = None


class Transformation(Base):
    name: str
    version: str
    params: dict[str, Any] = Field(default_factory=dict)
    performed_on: str | None = None
    output_sha256: Sha | None = None


class ChunkRecord(Base):
    chunk_id: str
    section: str
    group_id: str
    page_start: int | None = None
    page_end: int | None = None
    char_start: int
    char_end: int
    text_sha256: Sha


class RemovalRecord(Base):
    removed_on: str
    reason: str
    removed_chunk_ids: list[str] = Field(default_factory=list)


class SourceRecord(Base):
    source_id: str
    title: str
    source_version: str | None = None
    source_date: str | None = None
    origin: str
    media_type: str
    original_filename: str
    size_bytes: int
    sha256: Sha
    ingested_on: str
    rights: RightsInfo
    transformations: list[Transformation] = Field(default_factory=list)
    chunks: list[ChunkRecord] = Field(default_factory=list)
    status: Literal["active", "removed"] = "active"
    removal: RemovalRecord | None = None


class SourceManifest(Artifact):
    kind: Literal["source_manifest"] = "source_manifest"
    project_id: str
    manifest_version: int = 1
    sources: list[SourceRecord] = Field(default_factory=list)

    def active_sources(self) -> list[SourceRecord]:
        return [s for s in self.sources if s.status == "active"]

    def get(self, source_id: str) -> SourceRecord:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        raise KeyError(source_id)


# --------------------------------------------------------------------------- #
# 3. DatasetManifest
# --------------------------------------------------------------------------- #


class ChunkRef(Base):
    source_id: str
    chunk_id: str


class GeneratorInfo(Base):
    name: str
    version: str
    type: Literal["template", "llm", "human"]


class ExampleRecord(Base):
    example_id: str
    split: Literal["train", "validation", "test"]
    group_id: str
    task: str
    derived_from: list[ChunkRef]
    generator: GeneratorInfo
    text_sha256: Sha
    quality_score: float


class SplitInfo(Base):
    file: str
    sha256: Sha
    n_examples: int
    n_groups: int


class LeakageReport(Base):
    group_overlap_pairs: int
    near_duplicate_pairs_found: int
    dropped_example_ids: list[str] = Field(default_factory=list)
    residual_cross_split_near_duplicates: int
    effective_group_by: Literal["document", "section"]
    notes: list[str] = Field(default_factory=list)


class DatasetManifest(Artifact):
    kind: Literal["dataset_manifest"] = "dataset_manifest"
    dataset_id: str
    dataset_version: int = 1
    project_id: str
    source_manifest_hash: Sha
    source_manifest_version: int
    seed: int
    split_config: SplitConfig
    builder: GeneratorInfo
    examples: list[ExampleRecord]
    splits: dict[str, SplitInfo]
    leakage_report: LeakageReport
    excluded_sources: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 4. TrainingRun
# --------------------------------------------------------------------------- #


class ResourceEstimate(Base):
    basis: str
    trainable_params_m: float
    gpu_memory_gib: float
    train_tokens: int
    gpu_hours: float
    notes: list[str] = Field(default_factory=list)


class TrainingRun(Artifact):
    kind: Literal["training_run"] = "training_run"
    run_id: str
    project_id: str
    dataset_hash: Sha
    source_manifest_hash: Sha
    base_model_id: str
    base_model_license_hash: Sha
    license_gate: GateResult
    method: Literal["stub", "lora", "qlora", "full"]
    executor: Literal["stub", "external"]
    is_pipeline_validation_stub: bool
    seed: int
    hyperparameters: dict[str, Any]
    environment: dict[str, str] = Field(default_factory=dict)
    resource_estimate: ResourceEstimate | None = None
    status: Literal["completed", "failed", "planned"]
    started_on: str
    finished_on: str | None = None
    train_metrics: dict[str, float] = Field(default_factory=dict)
    output_artifacts: dict[str, Sha] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 5. EvaluationRun
# --------------------------------------------------------------------------- #


class EvalSubject(Base):
    role: Literal["base", "specialist", "quantized"]
    model_ref: str
    artifact_hash: Sha | None = None


class MetricResult(Base):
    metric: str
    subject_role: Literal["base", "specialist", "quantized"]
    value: float
    n: int
    higher_is_better: bool = True
    category: Literal["domain", "general_regression", "grounding", "safety"] = "domain"


class MetricComparison(Base):
    metric: str
    category: str
    higher_is_better: bool
    base: float
    specialist: float
    quantized: float | None = None
    specialist_minus_base: float
    quantized_minus_specialist: float | None = None
    regression: bool
    regression_tolerance: float
    quantization_regression: bool = False


class EvaluatorInfo(Base):
    name: str
    version: str
    is_stub: bool


class EvaluationRun(Artifact):
    kind: Literal["evaluation_run"] = "evaluation_run"
    eval_id: str
    project_id: str
    dataset_hash: Sha
    eval_split: Literal["validation", "test"] = "test"
    n_eval_examples: int
    train_eval_group_overlap: int
    evaluator: EvaluatorInfo
    subjects: list[EvalSubject]
    metrics: list[MetricResult]
    comparisons: list[MetricComparison]
    improvement_claim_allowed: bool
    notes: list[str] = Field(default_factory=list)
    evaluated_on: str

    @model_validator(mode="after")
    def _roles(self) -> Self:
        roles = [s.role for s in self.subjects]
        if len(set(roles)) != len(roles):
            raise ValueError("duplicate subject roles")
        if "base" not in roles or "specialist" not in roles:
            raise ValueError("an evaluation must include both base and specialist subjects")
        known = set(roles)
        if any(m.subject_role not in known for m in self.metrics):
            raise ValueError("metric references a subject role that is not evaluated")
        if self.evaluator.is_stub and self.improvement_claim_allowed:
            raise ValueError("a stub evaluator can never allow an improvement claim")
        return self


# --------------------------------------------------------------------------- #
# 6. ModelManifest
# --------------------------------------------------------------------------- #


class BaseModelRef(Base):
    model_family: str
    exact_version: str
    license_id: str
    license_entry_hash: Sha


class TokenizerInfo(Base):
    name: str
    files: dict[str, Sha] = Field(default_factory=dict)


class ContextAssumptions(Base):
    trained_context_tokens: int
    max_supported_context_tokens: int
    recommended_context_tokens: int


class QuantizationInfo(Base):
    method: str
    bits_per_weight: float
    format: str


class RuntimeRequirements(Base):
    runtimes: list[str]
    formats: list[str]
    min_ram_mb_estimate: int | None = None
    notes: list[str] = Field(default_factory=list)


class ArtifactFile(Base):
    path: str
    sha256: Sha
    size_bytes: int
    role: str


class ModelManifest(Artifact):
    kind: Literal["model_manifest"] = "model_manifest"
    specialist_name: str
    specialist_version: str
    base_model: BaseModelRef
    adapter_method: Literal["stub", "lora", "qlora", "full", "merged"]
    dataset_version: int
    dataset_hash: Sha
    source_manifest_version: int
    source_manifest_hash: Sha
    training_run_hash: Sha
    training_config: dict[str, Any]
    tokenizer: TokenizerInfo
    context: ContextAssumptions
    quantization: QuantizationInfo | None = None
    runtime: RuntimeRequirements
    evaluation_hashes: list[Sha]
    export_date: str
    artifacts: list[ArtifactFile]
    target_profile: str
    known_limitations: list[str]
    is_pipeline_validation_stub: bool = False

    @property
    def model_identity(self) -> str:
        """Unambiguous identity: name@version+<hash prefix>. Changes if anything material changes."""
        from .hashing import short

        return f"{self.specialist_name}@{self.specialist_version}+{short(self.content_hash)}"


# --------------------------------------------------------------------------- #
# 7. DeviceProfile (+ candidate/measurement types)
# --------------------------------------------------------------------------- #


class Measurement(Base):
    """On-device benchmark results for ONE candidate config. Unmeasured fields stay null."""

    config_id: str
    ttft_ms: float | None = None
    tokens_per_s: float | None = None  # sustained decode throughput
    peak_ram_mb: float | None = None  # model-side RSS: weights+runtime+KV+retrieval
    sustained_ram_mb: float | None = None
    thermal_throttle_ratio: float | None = None  # sustained tps / initial tps, (0, 1]
    ui_jank_pct: float | None = None
    crashes: int | None = None
    anrs: int | None = None
    background_kills: int | None = None
    sustained_minutes: float | None = None
    measured_on: str | None = None


class DeviceProfile(Artifact):
    kind: Literal["device_profile"] = "device_profile"
    device_id: str
    name: str
    device_class: str
    soc: str | None = None
    os: str
    total_ram_mb: int
    typical_available_ram_mb: int | None = None
    os_reserve_mb: int
    background_reserve_mb: int
    host_app_mb: int
    host_app_name: str = "host"
    storage_total_mb: int
    storage_free_mb: int
    mem_bandwidth_gbps: float | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    source: Literal["declared", "probed"] = "declared"


class ModelSpec(Base):
    model_id: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int
    quality_score: float | None = None
    max_context: int = 8192


class QuantSpec(Base):
    name: str
    bits_per_weight: float
    quality_penalty: float


class RuntimeSpec(Base):
    name: str
    overhead_mb: float
    scratch_mb_per_1k_ctx: float = 0.0
    kv_dtype: Literal["f16", "q8_0", "q4_0"] = "f16"


class RetrievalBudget(Base):
    index_ram_mb: float = 0.0
    index_storage_mb: float = 0.0
    top_k: int = 0


class CandidateConfig(Base):
    """A complete deployable configuration: model + quant + context + runtime + retrieval."""

    config_id: str
    model: ModelSpec
    quant: QuantSpec
    context_tokens: int
    runtime: RuntimeSpec
    retrieval: RetrievalBudget = Field(default_factory=RetrievalBudget)
    weights_file_mb: float | None = None  # exact size if known; else derived from params*bpw


class SafetyPolicy(Base):
    safety_reserve_frac: float = 0.08
    safety_reserve_min_mb: float = 256
    storage_reserve_frac: float = 0.10
    storage_reserve_min_mb: float = 2048
    estimate_inflation: float = 1.25  # applied to runtime, KV cache and retrieval estimates
    weights_inflation: float = 1.05  # weights size is known; allow for alignment/page-cache slack
    fallback_bandwidth_gbps: float = 25.0
    bandwidth_efficiency: float = 0.4
    estimate_thermal_derate: float = 0.7
    min_tps: float = 3.0
    max_ttft_ms: float = 8000
    min_thermal_ratio: float = 0.7
    max_ui_jank_pct: float = 5.0
    min_sustained_minutes: float = 10.0
    max_quant_penalty: float = 3.0
    performance_max_utilization: float = 0.6
    performance_min_tps: float = 10.0
    balanced_max_utilization: float = 0.92
    balanced_min_tps: float = 4.0
    balanced_min_context: int = 2048
    max_quality_max_utilization: float = 1.0


class RamBreakdown(Base):
    weights_mb: float
    runtime_mb: float
    kv_cache_mb: float
    retrieval_mb: float
    estimate_inflation: float
    weights_inflation: float = 1.0
    required_model_side_mb: float
    measured_override: bool
    total_ram_mb: float
    os_reserve_mb: float
    background_reserve_mb: float
    host_app_mb: float
    safety_reserve_mb: float
    ram_budget_mb: float
    headroom_mb: float
    utilization: float


class StorageBreakdown(Base):
    model_file_mb: float
    retrieval_mb: float
    required_mb: float
    free_mb: float
    reserve_mb: float
    available_after_reserve_mb: float
    headroom_mb: float


class SustainedAssessment(Base):
    verdict: Verdict
    basis: Confidence
    est_tokens_per_s: float | None = None
    est_ttft_ms: float | None = None
    tokens_per_s: float | None = None
    ttft_ms: float | None = None
    reasons: list[str] = Field(default_factory=list)
    missing_measurements: list[str] = Field(default_factory=list)


class CandidateAssessment(Base):
    config: CandidateConfig
    storage_verdict: Verdict
    ram_verdict: Verdict
    sustained_verdict: Verdict
    storage: StorageBreakdown
    ram: RamBreakdown
    sustained: SustainedAssessment
    confidence: Confidence
    quality_score: float
    safe_to_deploy: bool  # all three verdicts == fit (i.e. measured)
    eligible: bool  # not no_fit anywhere; may be provisional on estimates
    blocking_reasons: list[str] = Field(default_factory=list)


class ProfilePick(Base):
    profile: Literal["performance", "balanced", "max_quality"]
    label: str
    config: CandidateConfig | None
    tier: Literal["recommended", "provisional", "none"]
    recommended: bool
    confidence: Confidence | None = None
    reason: str
    warnings: list[str] = Field(default_factory=list)
    verified_fallback_config_id: str | None = None


class DeploymentRecommendation(Artifact):
    kind: Literal["deployment_recommendation"] = "deployment_recommendation"
    algorithm_version: str
    device_profile_hash: Sha
    device_id: str
    policy: SafetyPolicy
    picks: list[ProfilePick]
    assessed: list[CandidateAssessment]
    created_on: str


# --------------------------------------------------------------------------- #
# 9. ExportPackage
# --------------------------------------------------------------------------- #

ROLES = (
    "model_weights",
    "adapter",
    "tokenizer",
    "model_manifest",
    "evaluation_report",
    "provenance_source_manifest",
    "provenance_dataset_manifest",
    "license_bundle",
    "license_gate",
    "attribution",
    "runtime_manifest",
    "reference_chunks",
    "reference_index",
    "readme",
)


class PackageFile(Base):
    path: str
    sha256: Sha
    size_bytes: int
    role: Literal[
        "model_weights",
        "adapter",
        "tokenizer",
        "model_manifest",
        "evaluation_report",
        "provenance_source_manifest",
        "provenance_dataset_manifest",
        "license_bundle",
        "license_gate",
        "attribution",
        "runtime_manifest",
        "reference_chunks",
        "reference_index",
        "readme",
    ]


class ReferencePackageInfo(Base):
    chunk_count: int
    chunks_path: str
    index_path: str
    excluded_source_ids: list[str] = Field(default_factory=list)  # sources lacking redistribution rights


class ExportPackage(Artifact):
    kind: Literal["export_package"] = "export_package"
    package_name: str
    package_version: str
    export_grade: str
    model_identity: str
    model_manifest_hash: Sha
    source_manifest_hash: Sha
    dataset_hash: Sha
    evaluation_hashes: list[Sha]
    license_gate: GateResult
    files: list[PackageFile]
    reference: ReferencePackageInfo | None = None
    runtime_formats: list[str]
    is_pipeline_validation_stub: bool
    created_on: str


# --------------------------------------------------------------------------- #

ARTIFACT_TYPES: dict[str, type[BaseModel]] = {
    "specialist_project": SpecialistProject,
    "source_manifest": SourceManifest,
    "dataset_manifest": DatasetManifest,
    "training_run": TrainingRun,
    "evaluation_run": EvaluationRun,
    "model_manifest": ModelManifest,
    "device_profile": DeviceProfile,
    "deployment_recommendation": DeploymentRecommendation,
    "export_package": ExportPackage,
    "base_model_license_entry": BaseModelLicenseEntry,
}


def load_artifact(data: dict[str, Any]) -> BaseModel:
    kind = data.get("kind")
    if kind not in ARTIFACT_TYPES:
        raise ValueError(f"unknown artifact kind: {kind!r}")
    return ARTIFACT_TYPES[kind].model_validate(data)
