"""Contratos versionados nas fronteiras do sistema de simplificação."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, ClassVar, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

SCHEMA_VERSION = "1.0"


class ContractModel(BaseModel):
    """Base comum para contratos persistidos e validados."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^1\.0$")


class BranchId(StrEnum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ModelRole(StrEnum):
    ANALYZER = "analyzer"
    SIMPLIFIER = "simplifier"
    EVALUATOR = "evaluator"


class ModelProvider(StrEnum):
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai_compatible"
    SIMULATED = "simulated"


class TargetAudience(StrEnum):
    DOMAIN_EXPERTS = "Estudantes, acadêmicos e profissionais da área."
    COMMUNICATION_PROFESSIONALS = "Jornalistas e profissionais de comunicação."
    GENERAL_PUBLIC = "Público geral"


class SimplificationIntensity(StrEnum):
    LIGHT = "light"
    MODERATE = "moderate"
    STRONG = "strong"


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class QualityVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NOT_EVALUATED = "not_evaluated"


class TerminationReason(StrEnum):
    APPROVAL = "approval"
    REVISION_LIMIT = "revision_limit"
    BUDGET = "budget"
    TECHNICAL_ERROR = "technical_error"
    INVALID_RESPONSE = "invalid_response"
    CANCELLATION = "cancellation"
    CONTENT_POLICY = "content_policy"


class AnalysisIssueType(StrEnum):
    LONG_SENTENCE = "long_sentence"
    COMPLEX_SYNTAX = "complex_syntax"
    JARGON = "jargon"
    AMBIGUITY = "ambiguity"
    HIGH_DENSITY = "high_density"
    POOR_COHESION = "poor_cohesion"
    NOMINALIZATION = "nominalization"
    PASSIVE_VOICE = "passive_voice"
    REDUNDANCY = "redundancy"
    OTHER = "other"


class EvaluationIssueType(StrEnum):
    OMISSION = "omission"
    DISTORTION = "distortion"
    HALLUCINATION = "hallucination"
    LEVEL_MISMATCH = "level_mismatch"
    TECHNICAL_TERM_LOSS = "technical_term_loss"
    READABILITY = "readability"
    COHERENCE = "coherence"


class BranchConfig(ContractModel):
    branch_id: BranchId
    target_audience: TargetAudience
    intensity: SimplificationIntensity


class ModelSettings(ContractModel):
    name: str = Field(min_length=1)
    provider: ModelProvider = ModelProvider.OPENROUTER
    base_url: str = Field(min_length=1)
    api_key_env: str | None = Field(
        default=None,
        pattern=r"^[A-Z_][A-Z0-9_]*$",
    )
    request_timeout_seconds: PositiveFloat = 60.0
    technical_retries: int = Field(default=0, ge=0)
    retry_backoff_seconds: float = Field(default=0.5, ge=0)
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: PositiveInt | None = None
    top_p: float | None = Field(default=None, ge=0, le=1)
    supports_structured_output: bool = False

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> Self:
        if self.provider == ModelProvider.OPENROUTER and not self.api_key_env:
            raise ValueError("OpenRouter exige api_key_env.")
        return self


class RoleModels(ContractModel):
    analyzer: ModelSettings
    simplifier: ModelSettings
    evaluator: ModelSettings

    def for_role(self, role: ModelRole) -> ModelSettings:
        return getattr(self, role.value)


class PromptPaths(ContractModel):
    analyzer: Path = Path("prompts/analisador.txt")
    simplifier: Path = Path("prompts/simplificador.txt")
    evaluator: Path = Path("prompts/avaliador.txt")


class RunConfig(ContractModel):
    models: RoleModels | None = None
    prompts: PromptPaths = Field(default_factory=PromptPaths)
    max_revisions: PositiveInt = 3

    # Compatibilidade para chamadas diretas antigas. A configuração externa
    # deve preferir `models`, que permite um modelo diferente por papel.
    model_name: str | None = Field(default=None, min_length=1, exclude=True)
    model_provider: str = Field(default="openrouter", min_length=1, exclude=True)
    temperature: float = Field(default=0.3, ge=0, le=2, exclude=True)
    max_retries: int = Field(default=0, ge=0, exclude=True)

    branches: tuple[BranchConfig, ...] = Field(
        default_factory=lambda: (
            BranchConfig(
                branch_id=BranchId.SIMPLE,
                target_audience=TargetAudience.DOMAIN_EXPERTS,
                intensity=SimplificationIntensity.LIGHT,
            ),
            BranchConfig(
                branch_id=BranchId.MODERATE,
                target_audience=TargetAudience.COMMUNICATION_PROFESSIONALS,
                intensity=SimplificationIntensity.MODERATE,
            ),
            BranchConfig(
                branch_id=BranchId.AGGRESSIVE,
                target_audience=TargetAudience.GENERAL_PUBLIC,
                intensity=SimplificationIntensity.STRONG,
            ),
        )
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        if self.models is None:
            if self.model_name is None:
                raise ValueError("Informe models ou model_name.")
            provider = ModelProvider(self.model_provider)
            shared_model = ModelSettings(
                name=self.model_name,
                provider=provider,
                base_url=(
                    "https://openrouter.ai/api/v1"
                    if provider == ModelProvider.OPENROUTER
                    else "http://localhost:8000/v1"
                ),
                api_key_env=(
                    "OPENROUTER_API_KEY"
                    if provider == ModelProvider.OPENROUTER
                    else None
                ),
                temperature=self.temperature,
                technical_retries=self.max_retries,
            )
            object.__setattr__(
                self,
                "models",
                RoleModels(
                    analyzer=shared_model,
                    simplifier=shared_model,
                    evaluator=shared_model,
                ),
            )

        branch_ids = [branch.branch_id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("Cada ramo deve possuir um branch_id único.")
        if set(branch_ids) != set(BranchId):
            raise ValueError("A configuração deve definir simple, moderate e aggressive.")
        return self

    def branch(self, branch_id: BranchId) -> BranchConfig:
        return next(branch for branch in self.branches if branch.branch_id == branch_id)

    def model(self, role: ModelRole) -> ModelSettings:
        if self.models is None:
            raise RuntimeError("Configuração validada sem modelos por papel.")
        return self.models.for_role(role)


class DocumentInput(ContractModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_or_validate_content_hash(self) -> Self:
        expected_hash = sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_hash is not None and self.content_hash != expected_hash:
            raise ValueError("content_hash não corresponde ao conteúdo do documento.")
        object.__setattr__(self, "content_hash", expected_hash)
        return self

    @classmethod
    def from_corpus_item(
        cls,
        item: dict[str, Any],
        *,
        id_field: str = "id",
        text_field: str = "juridico",
    ) -> DocumentInput:
        if not isinstance(item, dict):
            raise TypeError("Cada item do corpus deve ser um objeto.")
        document_id = item.get(id_field)
        text = item.get(text_field)
        if not isinstance(document_id, str):
            raise ValueError(f"O campo {id_field!r} deve ser uma string.")
        if not isinstance(text, str):
            raise ValueError(f"O campo {text_field!r} deve ser uma string.")
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {id_field, text_field}
        }
        return cls(document_id=document_id, text=text, metadata=metadata)


class AnalysisIssue(ContractModel):
    type: AnalysisIssueType
    excerpt: str = Field(min_length=1, max_length=500)
    problem: str = Field(min_length=1, max_length=500)
    recommended_action: str = Field(min_length=1, max_length=500)


class AnalysisResult(ContractModel):
    blocked: bool
    block_reason: str | None = Field(default=None, max_length=500)
    issues: list[AnalysisIssue] = Field(default_factory=list, max_length=10)
    must_keep: list[str] = Field(default_factory=list)
    semantic_preservation_risks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_blocked_analysis(self) -> Self:
        if self.blocked:
            if not self.block_reason:
                raise ValueError("Uma análise bloqueada deve informar block_reason.")
            if self.issues or self.must_keep or self.semantic_preservation_risks:
                raise ValueError("Uma análise bloqueada não pode orientar simplificação.")
        elif self.block_reason is not None:
            raise ValueError("block_reason deve ser nulo quando blocked for false.")
        return self


class EvaluationIssue(ContractModel):
    type: EvaluationIssueType
    excerpt: str = Field(min_length=1, max_length=500)
    problem: str = Field(min_length=1, max_length=500)
    required_fix: str = Field(min_length=1, max_length=500)


class EvaluationResult(ContractModel):
    verdict: QualityVerdict
    issues: list[EvaluationIssue] = Field(default_factory=list, max_length=6)

    @model_validator(mode="before")
    @classmethod
    def accept_prompt_status(cls, data: Any) -> Any:
        if isinstance(data, dict) and "status" in data and "verdict" not in data:
            data = {**data, "verdict": data["status"]}
            del data["status"]
        return data

    @model_validator(mode="after")
    def validate_verdict(self) -> Self:
        if self.verdict == QualityVerdict.APPROVED and self.issues:
            raise ValueError("Uma avaliação aprovada não pode conter issues.")
        if self.verdict == QualityVerdict.REJECTED and not self.issues:
            raise ValueError("Uma avaliação rejeitada deve conter ao menos uma issue.")
        return self


class TokenUsage(ContractModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    llm_calls: int = Field(default=0, ge=0)


class AttemptResult(ContractModel):
    attempt_id: UUID = Field(default_factory=uuid4)
    branch_id: BranchId
    attempt_number: PositiveInt
    simplification_call_id: UUID = Field(default_factory=uuid4)
    simplification_request_id: str | None = None
    simplification_finish_reason: str | None = None
    evaluation_call_id: UUID | None = None
    evaluation_request_id: str | None = None
    evaluation_finish_reason: str | None = None
    execution_status: ExecutionStatus
    quality_verdict: QualityVerdict = QualityVerdict.NOT_EVALUATED
    simplified_text: str | None = None
    evaluation: EvaluationResult | None = None
    termination_reason: TerminationReason | None = None
    error_message: str | None = Field(default=None, max_length=2000)
    usage: TokenUsage = Field(default_factory=TokenUsage)

    @model_validator(mode="after")
    def validate_attempt_state(self) -> Self:
        if self.evaluation is not None and self.evaluation.verdict != self.quality_verdict:
            raise ValueError("evaluation e quality_verdict devem possuir o mesmo veredito.")
        if self.execution_status == ExecutionStatus.COMPLETED and not self.simplified_text:
            raise ValueError("Uma tentativa concluída deve possuir simplified_text.")
        if self.execution_status == ExecutionStatus.FAILED and not self.error_message:
            raise ValueError("Uma tentativa com falha deve informar error_message.")
        return self


class BranchResult(ContractModel):
    run_id: UUID
    document_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_id: BranchId
    target_audience: TargetAudience
    intensity: SimplificationIntensity
    execution_status: ExecutionStatus
    quality_verdict: QualityVerdict
    termination_reason: TerminationReason
    final_text: str | None = None
    attempts: list[AttemptResult] = Field(default_factory=list)
    error_message: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_branch_state(self) -> Self:
        if self.execution_status == ExecutionStatus.COMPLETED and not self.final_text:
            if self.termination_reason != TerminationReason.CONTENT_POLICY:
                raise ValueError("Um ramo concluído deve possuir final_text.")
        if self.execution_status == ExecutionStatus.FAILED and not self.error_message:
            raise ValueError("Um ramo com falha deve informar error_message.")
        return self


class WorkflowResult(ContractModel):
    run_id: UUID = Field(default_factory=uuid4)
    document_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_text: str = Field(min_length=1)
    execution_status: ExecutionStatus
    termination_reason: TerminationReason
    analysis_call_id: UUID
    analysis_request_id: str | None = None
    analysis_finish_reason: str | None = None
    analysis: AnalysisResult | None = None
    branches: list[BranchResult] = Field(min_length=3, max_length=3)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    started_at: datetime
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_result_identity(self) -> Self:
        expected_hash = sha256(self.original_text.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("content_hash não corresponde ao texto original.")

        expected_ids = set(BranchId)
        actual_ids = {branch.branch_id for branch in self.branches}
        if actual_ids != expected_ids:
            raise ValueError("O resultado deve conter exatamente os três ramos.")
        for branch in self.branches:
            if (
                branch.run_id != self.run_id
                or branch.document_id != self.document_id
                or branch.content_hash != self.content_hash
            ):
                raise ValueError("A identidade de cada ramo deve coincidir com o resultado.")
        return self
