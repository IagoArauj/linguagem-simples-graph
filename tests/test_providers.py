from __future__ import annotations

from typing import Any

import pytest
from langchain.messages import HumanMessage

import core.providers.openai_compatible as adapter_module
from core.providers import (
    AuthenticationError,
    ModelClientFactory,
    ModelResponse,
    OpenAICompatibleModelClient,
    OpenRouterModelClient,
    SimulatedModelClient,
    TokenUsage,
    TransientError,
)
from core.schemas import (
    AnalysisResult,
    DocumentInput,
    ExecutionStatus,
    ModelProvider,
    ModelSettings,
    QualityVerdict,
    RoleModels,
    RunConfig,
)
from core.simplificacao import LinguagemSimplesGraph


ANALYSIS_OK = (
    '{"blocked":false,"block_reason":null,"issues":[],"must_keep":[],'
    '"semantic_preservation_risks":[]}'
)
APPROVED = '{"status":"approved","issues":[]}'


class FakeMessage:
    def __init__(
        self,
        content: str,
        *,
        with_usage: bool = True,
    ) -> None:
        self.content = content
        self.usage_metadata = (
            {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}
            if with_usage
            else None
        )
        self.response_metadata = {
            "request_id": "request-123",
            "finish_reason": "stop",
        }


class FakeChatOpenAI:
    initializations: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.initializations.append(kwargs)

    def invoke(self, messages: list[Any]) -> FakeMessage:
        request = str(messages[-1].content)
        if request == "Texto original.":
            content = ANALYSIS_OK
        elif "Texto simplificado:" in request:
            content = APPROVED
        else:
            content = "Texto simplificado."
        return FakeMessage(content)


@pytest.fixture(autouse=True)
def reset_fake_chat() -> None:
    FakeChatOpenAI.initializations.clear()


def test_simulated_provider_normalizes_metadata_and_preserves_unknown_usage() -> None:
    client = SimulatedModelClient(
        responses=[
            ModelResponse(
                content="Resposta.",
                request_id="sim-1",
                finish_reason="stop",
            )
        ]
    )

    response = client.invoke([HumanMessage(content="Entrada")])

    assert response.request_id == "sim-1"
    assert response.finish_reason == "stop"
    assert response.usage is None


def test_simulated_provider_retries_only_transient_failures() -> None:
    delays: list[float] = []
    client = SimulatedModelClient(
        script=[TransientError("temporário"), "Resposta."],
        technical_retries=1,
        retry_backoff_seconds=0.25,
        sleep=delays.append,
    )

    response = client.invoke([HumanMessage(content="Entrada")])

    assert response.content == "Resposta."
    assert len(client.calls) == 2
    assert delays == [0.25]


def test_authentication_failure_is_not_retried() -> None:
    client = SimulatedModelClient(
        script=[AuthenticationError("credencial recusada"), "não deve ser usada"],
        technical_retries=5,
    )

    with pytest.raises(AuthenticationError, match="credencial recusada"):
        client.invoke([HumanMessage(content="Entrada")])

    assert len(client.calls) == 1


def test_textual_json_fallback_validates_schema() -> None:
    client = SimulatedModelClient(responses=[f"```json\n{ANALYSIS_OK}\n```"])

    response = client.invoke(
        [HumanMessage(content="Texto original.")],
        response_schema=AnalysisResult,
    )

    assert isinstance(response.parsed, AnalysisResult)
    assert response.parsed.blocked is False


def test_openai_adapter_uses_explicit_settings_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "ChatOpenAI", FakeChatOpenAI)
    client = OpenAICompatibleModelClient(
        model="local-model",
        base_url="http://localhost:8000/v1",
        request_timeout_seconds=17,
        technical_retries=3,
        temperature=0.2,
        max_tokens=512,
        top_p=0.9,
    )

    response = client.invoke([HumanMessage(content="Texto simples")])
    initialization = FakeChatOpenAI.initializations[0]

    assert initialization["base_url"] == "http://localhost:8000/v1"
    assert initialization["timeout"] == 17
    assert initialization["max_retries"] == 0
    assert initialization["temperature"] == 0.2
    assert initialization["max_completion_tokens"] == 512
    assert initialization["top_p"] == 0.9
    assert response.usage == TokenUsage(4, 2, 6)
    assert response.request_id == "request-123"
    assert response.finish_reason == "stop"


def make_backend_config(provider: ModelProvider) -> RunConfig:
    api_key_env = "OPENROUTER_API_KEY" if provider == ModelProvider.OPENROUTER else None
    base_url = (
        "https://openrouter.ai/api/v1"
        if provider == ModelProvider.OPENROUTER
        else "http://localhost:8000/v1"
    )

    def settings(role: str) -> ModelSettings:
        return ModelSettings(
            name=f"{role}-model",
            provider=provider,
            base_url=base_url,
            api_key_env=api_key_env,
            technical_retries=1,
            supports_structured_output=False,
        )

    return RunConfig(
        models=RoleModels(
            analyzer=settings("analyzer"),
            simplifier=settings("simplifier"),
            evaluator=settings("evaluator"),
        )
    )


@pytest.mark.parametrize(
    ("provider", "expected_type"),
    [
        (ModelProvider.OPENROUTER, OpenRouterModelClient),
        (ModelProvider.OPENAI_COMPATIBLE, OpenAICompatibleModelClient),
    ],
)
def test_same_workflow_runs_with_both_endpoint_adapters(
    monkeypatch: pytest.MonkeyPatch,
    provider: ModelProvider,
    expected_type: type[OpenAICompatibleModelClient],
) -> None:
    monkeypatch.setattr(adapter_module, "ChatOpenAI", FakeChatOpenAI)
    config = make_backend_config(provider)
    clients = ModelClientFactory(
        environ={"OPENROUTER_API_KEY": "test-secret"}
    ).create_all(config)

    assert isinstance(clients.analyzer, expected_type)
    result = LinguagemSimplesGraph.run(
        DocumentInput(document_id=f"DOC-{provider.value}", text="Texto original."),
        config=config,
        clients=clients,
    )

    assert result.execution_status == ExecutionStatus.COMPLETED
    assert all(
        branch.quality_verdict == QualityVerdict.APPROVED
        for branch in result.branches
    )
    assert result.usage.input_tokens == 28
    assert result.usage.output_tokens == 14
    assert result.usage.total_tokens == 42
    assert result.usage.llm_calls == 7
