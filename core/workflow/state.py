"""Estado interno do LangGraph.

Os dicionários são validados por schemas Pydantic ao entrar e sair do workflow.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any

from typing_extensions import NotRequired, TypedDict


class WorkflowState(TypedDict):
    schema_version: str
    run_id: str
    document_id: str
    content_hash: str
    text: str

    llm_calls: Annotated[int, add]
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]
    usage_unknown_calls: Annotated[int, add]

    analysis_call_id: NotRequired[str]
    analysis_request_id: NotRequired[str]
    analysis_finish_reason: NotRequired[str]
    analysis: NotRequired[dict[str, Any]]
    analysis_error: NotRequired[str]
    analysis_termination_reason: NotRequired[str]

    simple_attempts: NotRequired[Annotated[int, add]]
    moderate_attempts: NotRequired[Annotated[int, add]]
    aggressive_attempts: NotRequired[Annotated[int, add]]

    simple_attempt_records: NotRequired[list[dict[str, Any]]]
    moderate_attempt_records: NotRequired[list[dict[str, Any]]]
    aggressive_attempt_records: NotRequired[list[dict[str, Any]]]

    simple_simplification: NotRequired[str]
    moderate_simplification: NotRequired[str]
    aggressive_simplification: NotRequired[str]

    simple_simplification_feedback: NotRequired[dict[str, Any]]
    moderate_simplification_feedback: NotRequired[dict[str, Any]]
    aggressive_simplification_feedback: NotRequired[dict[str, Any]]

    simple_execution_status: NotRequired[str]
    moderate_execution_status: NotRequired[str]
    aggressive_execution_status: NotRequired[str]

    simple_quality_verdict: NotRequired[str]
    moderate_quality_verdict: NotRequired[str]
    aggressive_quality_verdict: NotRequired[str]

    simple_termination_reason: NotRequired[str]
    moderate_termination_reason: NotRequired[str]
    aggressive_termination_reason: NotRequired[str]

    simple_error: NotRequired[str]
    moderate_error: NotRequired[str]
    aggressive_error: NotRequired[str]
