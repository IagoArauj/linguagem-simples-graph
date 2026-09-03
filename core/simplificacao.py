from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast
from uuid import UUID, uuid4

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from .schemas import (
    AnalysisResult,
    AttemptResult,
    BranchId,
    BranchResult,
    DocumentInput,
    EvaluationResult,
    ExecutionStatus,
    QualityVerdict,
    RunConfig,
    SCHEMA_VERSION,
    TerminationReason,
    TokenUsage,
    WorkflowResult,
)
from .providers import (
    InvalidResponseError,
    ModelClientError,
    ModelClients,
    ModelResponse,
)
from .workflow.state import WorkflowState


BRANCH_PREFIX = {
    BranchId.SIMPLE: "simple",
    BranchId.MODERATE: "moderate",
    BranchId.AGGRESSIVE: "aggressive",
}


def _validated_response(
    response: ModelResponse,
    schema: type[BaseModel],
) -> BaseModel:
    if isinstance(response.parsed, schema):
        return response.parsed
    try:
        return schema.model_validate_json(response.content)
    except ValidationError as exc:
        raise InvalidResponseError(
            "A resposta não corresponde ao schema solicitado."
        ) from exc


def _usage_updates(response: ModelResponse) -> dict[str, int]:
    usage = response.usage
    input_tokens = usage.input_tokens if usage is not None else None
    output_tokens = usage.output_tokens if usage is not None else None
    total_tokens = usage.total_tokens if usage is not None else None

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    unknown = input_tokens is None or output_tokens is None or total_tokens is None

    return {
        "llm_calls": 1,
        "input_tokens": input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "total_tokens": total_tokens or 0,
        "usage_unknown_calls": int(unknown),
    }


def _unknown_usage_updates() -> dict[str, int]:
    return {
        "llm_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_unknown_calls": 1,
    }


def _token_usage(response: ModelResponse) -> TokenUsage:
    usage = response.usage
    if usage is None:
        return TokenUsage(llm_calls=1)
    total_tokens = usage.total_tokens
    if total_tokens is None and usage.input_tokens is not None and usage.output_tokens is not None:
        total_tokens = usage.input_tokens + usage.output_tokens
    return TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=total_tokens,
        llm_calls=1,
    )


def _sum_known(first: int | None, second: int | None) -> int | None:
    if first is None or second is None:
        return None
    return first + second


class LinguagemSimplesGraph:
    @staticmethod
    def _load_prompt(filename: str | os.PathLike[str]) -> str:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()

    @staticmethod
    def _validate_boundary(
        document: DocumentInput | str,
        config: RunConfig,
        document_id: str | None,
    ) -> tuple[DocumentInput, RunConfig]:
        if isinstance(document, str):
            generated_id = document_id or f"ad-hoc:{uuid4()}"
            validated_document = DocumentInput(
                document_id=generated_id,
                text=document,
            )
        else:
            validated_document = DocumentInput.model_validate(document)

        return validated_document, RunConfig.model_validate(config)

    @staticmethod
    def run(
        document: DocumentInput | str,
        *,
        config: RunConfig,
        clients: ModelClients,
        document_id: str | None = None,
        run_id: UUID | None = None,
    ) -> WorkflowResult:
        validated_document, validated_config = LinguagemSimplesGraph._validate_boundary(
            document,
            config,
            document_id,
        )
        current_run_id = run_id or uuid4()
        started_at = datetime.now(timezone.utc)


        def llm_analisador_node(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt(
                validated_config.prompts.analyzer
            )
            analysis_call_id = uuid4()
            try:
                response = clients.analyzer.invoke(
                    [
                        SystemMessage(content=prompt),
                        HumanMessage(content=state["text"]),
                    ],
                    response_schema=AnalysisResult,
                )
                analysis = cast(
                    AnalysisResult,
                    _validated_response(response, AnalysisResult),
                )
            except InvalidResponseError as exc:
                return {
                    "analysis_call_id": str(analysis_call_id),
                    "analysis_error": str(exc),
                    "analysis_termination_reason": TerminationReason.INVALID_RESPONSE,
                    **_unknown_usage_updates(),
                }
            except ModelClientError as exc:
                return {
                    "analysis_call_id": str(analysis_call_id),
                    "analysis_error": str(exc),
                    "analysis_termination_reason": TerminationReason.TECHNICAL_ERROR,
                    **_unknown_usage_updates(),
                }
            except Exception as exc:
                return {
                    "analysis_call_id": str(analysis_call_id),
                    "analysis_error": str(exc),
                    "analysis_termination_reason": TerminationReason.TECHNICAL_ERROR,
                    **_unknown_usage_updates(),
                }

            return {
                "analysis_call_id": str(analysis_call_id),
                "analysis_request_id": response.request_id,
                "analysis_finish_reason": response.finish_reason,
                "analysis": analysis.model_dump(mode="json"),
                **_usage_updates(response),
            }

        def make_simplifier(
            branch_id: BranchId,
        ) -> Callable[[WorkflowState], dict[str, Any]]:
            prefix = BRANCH_PREFIX[branch_id]
            branch_config = validated_config.branch(branch_id)
            attempts_key = f"{prefix}_attempts"
            records_key = f"{prefix}_attempt_records"
            text_key = f"{prefix}_simplification"
            feedback_key = f"{prefix}_simplification_feedback"

            def simplifier(state: WorkflowState) -> dict[str, Any]:
                if "analysis_error" in state:
                    return {
                        f"{prefix}_execution_status": ExecutionStatus.FAILED,
                        f"{prefix}_quality_verdict": QualityVerdict.NOT_EVALUATED,
                        f"{prefix}_termination_reason": state.get(
                            "analysis_termination_reason",
                            TerminationReason.TECHNICAL_ERROR,
                        ),
                        f"{prefix}_error": state["analysis_error"],
                    }

                analysis = AnalysisResult.model_validate(state.get("analysis"))
                if analysis.blocked:
                    return {
                        f"{prefix}_execution_status": ExecutionStatus.COMPLETED,
                        f"{prefix}_quality_verdict": QualityVerdict.NOT_EVALUATED,
                        f"{prefix}_termination_reason": TerminationReason.CONTENT_POLICY,
                    }

                attempt_number = state.get(attempts_key, 0) + 1
                prompt = LinguagemSimplesGraph._load_prompt(
                    validated_config.prompts.simplifier
                )
                messages: list[Any] = [SystemMessage(content=prompt)]
                request = f"""
Análise do texto original: {json.dumps(state.get('analysis'), ensure_ascii=False)}
Público-alvo: {branch_config.target_audience.value}
Intensidade da simplificação: {branch_config.intensity.value}
Texto original: {state['text']}
"""
                messages.append(HumanMessage(content=request))

                previous_feedback = state.get(feedback_key)
                previous_text = state.get(text_key)
                if previous_feedback is not None and previous_text is not None:
                    messages.extend(
                        [
                            AIMessage(
                                content=f"Simplificação realizada: {previous_text}"
                            ),
                            HumanMessage(
                                content=(
                                    "Feedback para sua simplificação: "
                                    f"{json.dumps(previous_feedback, ensure_ascii=False)}. "
                                    "Melhore a simplificação com base nele."
                                )
                            ),
                        ]
                    )

                try:
                    response = clients.simplifier.invoke(messages)
                    simplified_text = response.content.strip()
                    if not simplified_text:
                        raise InvalidResponseError(
                            "O simplificador retornou conteúdo vazio."
                        )
                except InvalidResponseError as exc:
                    attempt = AttemptResult(
                        branch_id=branch_id,
                        attempt_number=attempt_number,
                        execution_status=ExecutionStatus.FAILED,
                        termination_reason=TerminationReason.INVALID_RESPONSE,
                        error_message=str(exc),
                        usage=TokenUsage(llm_calls=1),
                    )
                    return {
                        attempts_key: 1,
                        records_key: [
                            *state.get(records_key, []),
                            attempt.model_dump(mode="json"),
                        ],
                        f"{prefix}_execution_status": ExecutionStatus.FAILED,
                        f"{prefix}_quality_verdict": QualityVerdict.NOT_EVALUATED,
                        f"{prefix}_termination_reason": TerminationReason.INVALID_RESPONSE,
                        f"{prefix}_error": str(exc),
                        **_unknown_usage_updates(),
                    }
                except Exception as exc:
                    attempt = AttemptResult(
                        branch_id=branch_id,
                        attempt_number=attempt_number,
                        execution_status=ExecutionStatus.FAILED,
                        termination_reason=TerminationReason.TECHNICAL_ERROR,
                        error_message=str(exc),
                        usage=TokenUsage(llm_calls=1),
                    )
                    return {
                        attempts_key: 1,
                        records_key: [
                            *state.get(records_key, []),
                            attempt.model_dump(mode="json"),
                        ],
                        f"{prefix}_execution_status": ExecutionStatus.FAILED,
                        f"{prefix}_quality_verdict": QualityVerdict.NOT_EVALUATED,
                        f"{prefix}_termination_reason": TerminationReason.TECHNICAL_ERROR,
                        f"{prefix}_error": str(exc),
                        **_unknown_usage_updates(),
                    }

                attempt = AttemptResult(
                    branch_id=branch_id,
                    attempt_number=attempt_number,
                    simplification_request_id=response.request_id,
                    simplification_finish_reason=response.finish_reason,
                    execution_status=ExecutionStatus.RUNNING,
                    simplified_text=simplified_text,
                    usage=_token_usage(response),
                )
                return {
                    attempts_key: 1,
                    records_key: [
                        *state.get(records_key, []),
                        attempt.model_dump(mode="json"),
                    ],
                    text_key: simplified_text,
                    f"{prefix}_execution_status": ExecutionStatus.RUNNING,
                    f"{prefix}_quality_verdict": QualityVerdict.NOT_EVALUATED,
                    **_usage_updates(response),
                }

            return simplifier

        def make_evaluator(
            branch_id: BranchId,
        ) -> Callable[[WorkflowState], dict[str, Any]]:
            prefix = BRANCH_PREFIX[branch_id]
            branch_config = validated_config.branch(branch_id)
            attempts_key = f"{prefix}_attempts"
            records_key = f"{prefix}_attempt_records"
            text_key = f"{prefix}_simplification"
            feedback_key = f"{prefix}_simplification_feedback"

            def evaluator(state: WorkflowState) -> dict[str, Any]:
                prompt = LinguagemSimplesGraph._load_prompt(
                    validated_config.prompts.evaluator
                )
                request = f"""
Análise do texto original: {json.dumps(state.get('analysis'), ensure_ascii=False)}
Público-alvo: {branch_config.target_audience.value}
Intensidade da simplificação: {branch_config.intensity.value}
Texto original: {state['text']}
Texto simplificado: {state.get(text_key, '')}
"""
                evaluation_call_id = uuid4()
                try:
                    response = clients.evaluator.invoke(
                        [
                            SystemMessage(content=prompt),
                            HumanMessage(content=request),
                        ],
                        response_schema=EvaluationResult,
                    )
                    evaluation = cast(
                        EvaluationResult,
                        _validated_response(response, EvaluationResult),
                    )
                except InvalidResponseError as exc:
                    return finish_failed_evaluation(
                        state,
                        branch_id,
                        prefix,
                        records_key,
                        evaluation_call_id,
                        TerminationReason.INVALID_RESPONSE,
                        str(exc),
                    )
                except Exception as exc:
                    return finish_failed_evaluation(
                        state,
                        branch_id,
                        prefix,
                        records_key,
                        evaluation_call_id,
                        TerminationReason.TECHNICAL_ERROR,
                        str(exc),
                    )

                records = list(state.get(records_key, []))
                attempt = AttemptResult.model_validate(records[-1])
                attempt.evaluation_call_id = evaluation_call_id
                attempt.evaluation_request_id = response.request_id
                attempt.evaluation_finish_reason = response.finish_reason
                attempt.execution_status = ExecutionStatus.COMPLETED
                attempt.quality_verdict = evaluation.verdict
                attempt.evaluation = evaluation
                evaluation_usage = _token_usage(response)
                attempt.usage.input_tokens = _sum_known(
                    attempt.usage.input_tokens,
                    evaluation_usage.input_tokens,
                )
                attempt.usage.output_tokens = _sum_known(
                    attempt.usage.output_tokens,
                    evaluation_usage.output_tokens,
                )
                attempt.usage.total_tokens = _sum_known(
                    attempt.usage.total_tokens,
                    evaluation_usage.total_tokens,
                )
                attempt.usage.llm_calls += 1

                if evaluation.verdict == QualityVerdict.APPROVED:
                    reason = TerminationReason.APPROVAL
                    attempt.termination_reason = reason
                elif state.get(attempts_key, 0) >= validated_config.max_revisions:
                    reason = TerminationReason.REVISION_LIMIT
                    attempt.termination_reason = reason
                else:
                    reason = None

                records[-1] = attempt.model_dump(mode="json")
                result: dict[str, Any] = {
                    records_key: records,
                    feedback_key: evaluation.model_dump(mode="json"),
                    f"{prefix}_execution_status": ExecutionStatus.COMPLETED,
                    f"{prefix}_quality_verdict": evaluation.verdict,
                    **_usage_updates(response),
                }
                if reason is not None:
                    result[f"{prefix}_termination_reason"] = reason
                return result

            return evaluator

        def finish_failed_evaluation(
            state: WorkflowState,
            branch_id: BranchId,
            prefix: str,
            records_key: str,
            evaluation_call_id: UUID,
            reason: TerminationReason,
            error_message: str,
        ) -> dict[str, Any]:
            records = list(state.get(records_key, []))
            attempt = AttemptResult.model_validate(records[-1])
            attempt.evaluation_call_id = evaluation_call_id
            attempt.error_message = error_message
            attempt.termination_reason = reason
            attempt.usage.input_tokens = None
            attempt.usage.output_tokens = None
            attempt.usage.total_tokens = None
            attempt.usage.llm_calls += 1
            attempt.execution_status = ExecutionStatus.FAILED
            records[-1] = attempt.model_dump(mode="json")
            return {
                records_key: records,
                f"{prefix}_execution_status": ExecutionStatus.FAILED,
                f"{prefix}_quality_verdict": QualityVerdict.NOT_EVALUATED,
                f"{prefix}_termination_reason": reason,
                f"{prefix}_error": error_message,
                **_unknown_usage_updates(),
            }

        def make_post_simplifier_router(
            branch_id: BranchId,
        ) -> Callable[[WorkflowState], str]:
            prefix = BRANCH_PREFIX[branch_id]

            def route(state: WorkflowState) -> str:
                if state[f"{prefix}_execution_status"] == ExecutionStatus.RUNNING:
                    return "evaluate"
                return "finished"

            return route

        def make_feedback_router(
            branch_id: BranchId,
        ) -> Callable[[WorkflowState], str]:
            prefix = BRANCH_PREFIX[branch_id]

            def route(state: WorkflowState) -> str:
                if state[f"{prefix}_execution_status"] == ExecutionStatus.FAILED:
                    return "finished"
                if state[f"{prefix}_quality_verdict"] == QualityVerdict.APPROVED:
                    return "finished"
                if state.get(f"{prefix}_attempts", 0) < validated_config.max_revisions:
                    return "revise"
                return "finished"

            return route

        def branch_finished(state: WorkflowState) -> dict[str, Any]:
            return {}

        def aggregator(state: WorkflowState) -> dict[str, Any]:
            return {}

        builder = StateGraph(WorkflowState)
        builder.add_node("llm_analisador", cast(Any, llm_analisador_node))
        builder.add_node("aggregator", cast(Any, aggregator))

        finished_nodes: list[str] = []
        for branch_id in BranchId:
            prefix = BRANCH_PREFIX[branch_id]
            simplifier_node = f"llm_{prefix}_simplificator"
            evaluator_node = f"llm_{prefix}_avaliator"
            finished_node = f"{prefix}_finished"
            finished_nodes.append(finished_node)

            builder.add_node(
                simplifier_node,
                cast(Any, make_simplifier(branch_id)),
            )
            builder.add_node(
                evaluator_node,
                cast(Any, make_evaluator(branch_id)),
            )
            builder.add_node(finished_node, cast(Any, branch_finished))
            builder.add_edge("llm_analisador", simplifier_node)
            builder.add_conditional_edges(
                simplifier_node,
                make_post_simplifier_router(branch_id),
                {
                    "evaluate": evaluator_node,
                    "finished": finished_node,
                },
            )
            builder.add_conditional_edges(
                evaluator_node,
                make_feedback_router(branch_id),
                {
                    "revise": simplifier_node,
                    "finished": finished_node,
                },
            )

        builder.add_edge(START, "llm_analisador")
        builder.add_edge(finished_nodes, "aggregator")
        builder.add_edge("aggregator", END)
        workflow = builder.compile()

        graph_path = Path(__file__).resolve().parent / "graph.png"
        if not graph_path.exists():
            graph_image = workflow.get_graph().draw_mermaid_png()
            graph_path.write_bytes(graph_image)

        content_hash = validated_document.content_hash
        if content_hash is None:
            raise RuntimeError("Documento validado sem content_hash.")

        initial_state = cast(
            WorkflowState,
            cast(object, {
                "schema_version": SCHEMA_VERSION,
                "run_id": str(current_run_id),
                "document_id": validated_document.document_id,
                "content_hash": content_hash,
                "text": validated_document.text,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "usage_unknown_calls": 0,
            }),
        )
        state = cast(WorkflowState, workflow.invoke(initial_state))
        result = LinguagemSimplesGraph._build_result(
            state,
            validated_document,
            validated_config,
            current_run_id,
            started_at,
        )
        return result

    @staticmethod
    def _build_result(
        state: WorkflowState,
        document: DocumentInput,
        config: RunConfig,
        run_id: UUID,
        started_at: datetime,
    ) -> WorkflowResult:
        analysis = (
            AnalysisResult.model_validate(state["analysis"])
            if "analysis" in state
            else None
        )
        analysis_call_id = state.get("analysis_call_id")
        if analysis_call_id is None:
            raise RuntimeError("Workflow concluído sem analysis_call_id.")
        content_hash = document.content_hash
        if content_hash is None:
            raise RuntimeError("Documento validado sem content_hash.")
        branches: list[BranchResult] = []

        for branch_id in BranchId:
            prefix = BRANCH_PREFIX[branch_id]
            branch_config = config.branch(branch_id)
            execution_status = ExecutionStatus(
                state.get(f"{prefix}_execution_status", ExecutionStatus.FAILED)
            )
            quality_verdict = QualityVerdict(
                state.get(
                    f"{prefix}_quality_verdict",
                    QualityVerdict.NOT_EVALUATED,
                )
            )
            termination_reason = TerminationReason(
                state.get(
                    f"{prefix}_termination_reason",
                    TerminationReason.TECHNICAL_ERROR,
                )
            )
            error_message = state.get(f"{prefix}_error")
            if execution_status == ExecutionStatus.FAILED and not error_message:
                error_message = "O ramo terminou sem produzir um resultado válido."

            branches.append(
                BranchResult(
                    run_id=run_id,
                    document_id=document.document_id,
                    content_hash=content_hash,
                    branch_id=branch_id,
                    target_audience=branch_config.target_audience,
                    intensity=branch_config.intensity,
                    execution_status=execution_status,
                    quality_verdict=quality_verdict,
                    termination_reason=termination_reason,
                    final_text=state.get(f"{prefix}_simplification"),
                    attempts=[
                        AttemptResult.model_validate(attempt)
                        for attempt in state.get(
                            f"{prefix}_attempt_records",
                            [],
                        )
                    ],
                    error_message=error_message,
                )
            )

        completed = sum(
            branch.execution_status == ExecutionStatus.COMPLETED
            for branch in branches
        )
        if completed == len(branches):
            execution_status = ExecutionStatus.COMPLETED
        elif completed > 0:
            execution_status = ExecutionStatus.PARTIAL
        elif all(
            branch.execution_status == ExecutionStatus.CANCELLED
            for branch in branches
        ):
            execution_status = ExecutionStatus.CANCELLED
        else:
            execution_status = ExecutionStatus.FAILED

        reasons = {branch.termination_reason for branch in branches}
        if reasons == {TerminationReason.APPROVAL}:
            termination_reason = TerminationReason.APPROVAL
        elif TerminationReason.TECHNICAL_ERROR in reasons:
            termination_reason = TerminationReason.TECHNICAL_ERROR
        elif TerminationReason.INVALID_RESPONSE in reasons:
            termination_reason = TerminationReason.INVALID_RESPONSE
        elif TerminationReason.REVISION_LIMIT in reasons:
            termination_reason = TerminationReason.REVISION_LIMIT
        else:
            termination_reason = next(iter(reasons))

        usage_is_complete = state["usage_unknown_calls"] == 0

        return WorkflowResult(
            run_id=run_id,
            document_id=document.document_id,
            content_hash=content_hash,
            original_text=document.text,
            execution_status=execution_status,
            termination_reason=termination_reason,
            analysis_call_id=UUID(analysis_call_id),
            analysis_request_id=state.get("analysis_request_id"),
            analysis_finish_reason=state.get("analysis_finish_reason"),
            analysis=analysis,
            branches=branches,
            usage=TokenUsage(
                input_tokens=(state["input_tokens"] if usage_is_complete else None),
                output_tokens=(state["output_tokens"] if usage_is_complete else None),
                total_tokens=(state["total_tokens"] if usage_is_complete else None),
                llm_calls=state["llm_calls"],
            ),
            started_at=started_at,
        )
