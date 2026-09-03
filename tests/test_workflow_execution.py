from __future__ import annotations

from core.providers import ModelClients, SimulatedModelClient
from core.schemas import (

    DocumentInput,
    ExecutionStatus,
    ModelProvider,
    ModelSettings,
    QualityVerdict,
    RoleModels,
    RunConfig,
    TerminationReason,
)
from core.simplificacao import LinguagemSimplesGraph


ANALYSIS_OK = (
    '{"blocked":false,"block_reason":null,"issues":[],"must_keep":[],'
    '"semantic_preservation_risks":[]}'
)
APPROVED = '{"status":"approved","issues":[]}'
REJECTED = (
    '{"status":"rejected","issues":[{"type":"readability",'
    '"excerpt":"Texto","problem":"Ainda está complexo.",'
    '"required_fix":"Use palavras mais comuns."}]}'
)


def make_config(max_revisions: int = 3) -> RunConfig:
    def settings(name: str, temperature: float) -> ModelSettings:
        return ModelSettings(
            name=name,
            provider=ModelProvider.SIMULATED,
            base_url="simulated://local",
            temperature=temperature,
        )

    return RunConfig(
        models=RoleModels(
            analyzer=settings("analyzer-model", 0.0),
            simplifier=settings("simplifier-model", 0.4),
            evaluator=settings("evaluator-model", 0.1),
        ),
        max_revisions=max_revisions,
    )


def make_clients(
    *,
    analysis: str = ANALYSIS_OK,
    simplifications: list[str] | None = None,
    evaluations: list[str] | None = None,
) -> ModelClients:
    return ModelClients(
        analyzer=SimulatedModelClient(responses=[analysis]),
        simplifier=SimulatedModelClient(
            responses=simplifications or ["Texto simplificado."] * 3
        ),
        evaluator=SimulatedModelClient(responses=evaluations or [APPROVED] * 3),
    )


def call_count(client: object) -> int:
    assert isinstance(client, SimulatedModelClient)
    return len(client.calls)


def test_workflow_runs_all_roles_without_api() -> None:
    clients = make_clients()

    result = LinguagemSimplesGraph.run(
        DocumentInput(document_id="DOC-1", text="Texto original."),
        config=make_config(),
        clients=clients,
    )

    assert result.execution_status == ExecutionStatus.COMPLETED
    assert result.termination_reason == TerminationReason.APPROVAL
    assert all(
        branch.quality_verdict == QualityVerdict.APPROVED
        for branch in result.branches
    )
    assert call_count(clients.analyzer) == 1
    assert call_count(clients.simplifier) == 3
    assert call_count(clients.evaluator) == 3
    assert result.usage.llm_calls == 7
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None


def test_rejected_branch_stops_at_revision_limit() -> None:
    clients = make_clients(
        simplifications=["Texto simplificado."] * 6,
        evaluations=[REJECTED] * 6,
    )

    result = LinguagemSimplesGraph.run(
        DocumentInput(document_id="DOC-2", text="Texto original."),
        config=make_config(max_revisions=2),
        clients=clients,
    )

    assert result.execution_status == ExecutionStatus.COMPLETED
    assert result.termination_reason == TerminationReason.REVISION_LIMIT
    for branch in result.branches:
        assert branch.execution_status == ExecutionStatus.COMPLETED
        assert branch.quality_verdict == QualityVerdict.REJECTED
        assert branch.termination_reason == TerminationReason.REVISION_LIMIT
        assert len(branch.attempts) == 2


def test_invalid_evaluator_response_produces_partial_result() -> None:
    clients = make_clients(
        evaluations=[APPROVED, "resposta inválida", APPROVED],
    )

    result = LinguagemSimplesGraph.run(
        DocumentInput(document_id="DOC-3", text="Texto original."),
        config=make_config(),
        clients=clients,
    )

    assert result.execution_status == ExecutionStatus.PARTIAL
    failed = [
        branch
        for branch in result.branches
        if branch.execution_status == ExecutionStatus.FAILED
    ]
    assert len(failed) == 1
    assert failed[0].termination_reason == TerminationReason.INVALID_RESPONSE
    assert failed[0].attempts[0].usage.input_tokens is None


def test_blocked_analysis_never_calls_simplifier_or_evaluator() -> None:
    blocked = (
        '{"blocked":true,"block_reason":"conteúdo discriminatório",'
        '"issues":[],"must_keep":[],"semantic_preservation_risks":[]}'
    )
    clients = make_clients(analysis=blocked)

    result = LinguagemSimplesGraph.run(
        DocumentInput(document_id="DOC-4", text="Texto bloqueado."),
        config=make_config(),
        clients=clients,
    )

    assert call_count(clients.analyzer) == 1
    assert call_count(clients.simplifier) == 0
    assert call_count(clients.evaluator) == 0
    assert result.execution_status == ExecutionStatus.COMPLETED
    assert all(
        branch.termination_reason == TerminationReason.CONTENT_POLICY
        for branch in result.branches
    )
    assert all(not branch.attempts for branch in result.branches)
