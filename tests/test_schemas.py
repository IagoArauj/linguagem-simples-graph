from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.schemas import (
    BranchId,
    BranchResult,
    DocumentInput,
    ExecutionStatus,
    QualityVerdict,
    RunConfig,
    SimplificationIntensity,
    TargetAudience,
    TerminationReason,
    WorkflowResult,
)


def test_document_preserves_corpus_id_and_generates_hash() -> None:
    document = DocumentInput.from_corpus_item(
        {"id": "DOC-1", "juridico": "Texto jurídico.", "tema": "teste"}
    )

    assert document.document_id == "DOC-1"
    assert len(document.content_hash or "") == 64
    assert document.metadata == {"tema": "teste"}


@pytest.mark.parametrize(
    "item",
    [
        {"id": "", "juridico": "Texto"},
        {"id": "DOC", "juridico": ""},
        {"id": 1, "juridico": "Texto"},
        {"id": "DOC", "juridico": None},
    ],
)
def test_malformed_document_is_rejected(item: dict[str, object]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        DocumentInput.from_corpus_item(item)


def test_unknown_structured_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate(
            {
                "model_name": "fake",
                "branches": [
                    {
                        "branch_id": "unknown",
                        "target_audience": "qualquer público",
                        "intensity": "maximum",
                    }
                ],
            }
        )


def test_completed_execution_can_have_rejected_quality() -> None:
    document = DocumentInput(document_id="DOC", text="Texto original.")
    run_id = uuid4()
    content_hash = document.content_hash
    assert content_hash is not None

    branches = [
        BranchResult(
            run_id=run_id,
            document_id=document.document_id,
            content_hash=content_hash,
            branch_id=branch_id,
            target_audience=audience,
            intensity=intensity,
            execution_status=ExecutionStatus.COMPLETED,
            quality_verdict=(
                QualityVerdict.REJECTED
                if branch_id == BranchId.SIMPLE
                else QualityVerdict.APPROVED
            ),
            termination_reason=(
                TerminationReason.REVISION_LIMIT
                if branch_id == BranchId.SIMPLE
                else TerminationReason.APPROVAL
            ),
            final_text="Texto simplificado.",
        )
        for branch_id, audience, intensity in (
            (
                BranchId.SIMPLE,
                TargetAudience.DOMAIN_EXPERTS,
                SimplificationIntensity.LIGHT,
            ),
            (
                BranchId.MODERATE,
                TargetAudience.COMMUNICATION_PROFESSIONALS,
                SimplificationIntensity.MODERATE,
            ),
            (
                BranchId.AGGRESSIVE,
                TargetAudience.GENERAL_PUBLIC,
                SimplificationIntensity.STRONG,
            ),
        )
    ]

    result = WorkflowResult(
        run_id=run_id,
        document_id=document.document_id,
        content_hash=content_hash,
        original_text=document.text,
        execution_status=ExecutionStatus.COMPLETED,
        termination_reason=TerminationReason.REVISION_LIMIT,
        analysis_call_id=uuid4(),
        branches=branches,
        started_at=datetime.now(timezone.utc),
    )

    assert result.branches[0].execution_status == ExecutionStatus.COMPLETED
    assert result.branches[0].quality_verdict == QualityVerdict.REJECTED
