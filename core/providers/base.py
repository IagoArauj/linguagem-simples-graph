"""Contratos comuns e política de retries dos clientes de modelo."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

StructuredSchema = type[BaseModel] | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Uso de tokens informado pelo provedor; valores ausentes permanecem ``None``."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Resposta independente do formato específico de cada provedor."""

    content: str
    usage: TokenUsage | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parsed: Any | None = None


class ModelClientError(Exception):
    """Erro conhecido na fronteira com um provedor de modelos."""


class ConfigurationError(ModelClientError):
    """Configuração ausente, inválida ou incompatível."""


class AuthenticationError(ModelClientError):
    """Credencial ausente, recusada ou sem autorização."""


class TransientError(ModelClientError):
    """Falha temporária que pode ser repetida com segurança técnica."""


class InvalidResponseError(ModelClientError):
    """Resposta vazia, malformada ou incompatível com o schema solicitado."""


@runtime_checkable
class ModelClient(Protocol):
    """Interface síncrona mínima implementada por todos os providers."""

    def invoke(
        self,
        messages: Sequence[Any],
        *,
        response_schema: StructuredSchema | None = None,
        structured_output: StructuredSchema | None = None,
        text_fallback: bool = True,
    ) -> ModelResponse: ...

    def complete(
        self,
        messages: Sequence[Any],
        *,
        response_schema: StructuredSchema | None = None,
        structured_output: StructuredSchema | None = None,
        text_fallback: bool = True,
    ) -> ModelResponse: ...


class RetryingModelClient(ABC):
    """Centraliza retries técnicos; subclasses executam somente uma tentativa."""

    def __init__(
        self,
        *,
        technical_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if technical_retries < 0:
            raise ConfigurationError("technical_retries não pode ser negativo.")
        if retry_backoff_seconds < 0:
            raise ConfigurationError("retry_backoff_seconds não pode ser negativo.")
        self.technical_retries = technical_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep

    def invoke(
        self,
        messages: Sequence[Any],
        *,
        response_schema: StructuredSchema | None = None,
        structured_output: StructuredSchema | None = None,
        text_fallback: bool = True,
    ) -> ModelResponse:
        schema = _select_schema(response_schema, structured_output)
        for attempt in range(self.technical_retries + 1):
            try:
                return self._invoke_once(
                    messages,
                    response_schema=schema,
                    text_fallback=text_fallback,
                )
            except TransientError:
                if attempt >= self.technical_retries:
                    raise
                delay = self.retry_backoff_seconds * (2**attempt)
                if delay:
                    self._sleep(delay)
        raise AssertionError("loop de retry terminou sem resposta")

    def complete(
        self,
        messages: Sequence[Any],
        *,
        response_schema: StructuredSchema | None = None,
        structured_output: StructuredSchema | None = None,
        text_fallback: bool = True,
    ) -> ModelResponse:
        return self.invoke(
            messages,
            response_schema=response_schema,
            structured_output=structured_output,
            text_fallback=text_fallback,
        )

    @abstractmethod
    def _invoke_once(
        self,
        messages: Sequence[Any],
        *,
        response_schema: StructuredSchema | None,
        text_fallback: bool,
    ) -> ModelResponse:
        """Executa exatamente uma chamada ao endpoint."""


def parse_text_response(content: str, schema: StructuredSchema) -> Any:
    """Valida JSON textual, incluindo respostas cercadas por markdown."""

    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].lstrip()
    try:
        payload = json.loads(candidate)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(payload)
        return payload
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        raise InvalidResponseError(
            "A resposta textual não corresponde ao schema solicitado."
        ) from exc


def serialize_parsed(parsed: Any) -> str:
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json()
    return json.dumps(parsed, ensure_ascii=False)


def _select_schema(
    response_schema: StructuredSchema | None,
    structured_output: StructuredSchema | None,
) -> StructuredSchema | None:
    if response_schema is not None and structured_output is not None:
        raise ConfigurationError(
            "Informe apenas response_schema ou structured_output, não ambos."
        )
    return response_schema if response_schema is not None else structured_output


class SimulatedModelClient(RetryingModelClient):
    """Provider determinístico com respostas e falhas enfileiradas para testes."""

    def __init__(
        self,
        responses: Iterable[ModelResponse | str | Mapping[str, Any]] = (),
        failures: Iterable[Exception] = (),
        *,
        script: Iterable[ModelResponse | str | Mapping[str, Any] | Exception] | None = None,
        technical_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            technical_retries=technical_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            sleep=sleep,
        )
        scheduled = list(script) if script is not None else [*failures, *responses]
        self._script: deque[ModelResponse | str | Mapping[str, Any] | Exception] = deque(
            scheduled
        )
        self.calls: list[Sequence[Any]] = []

    def queue_response(self, response: ModelResponse | str | Mapping[str, Any]) -> None:
        self._script.append(response)

    def queue_failure(self, failure: Exception) -> None:
        self._script.append(failure)

    def _invoke_once(
        self,
        messages: Sequence[Any],
        *,
        response_schema: StructuredSchema | None,
        text_fallback: bool,
    ) -> ModelResponse:
        self.calls.append(messages)
        if not self._script:
            raise InvalidResponseError("O provider simulado não possui respostas programadas.")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ModelResponse):
            response = item
        elif isinstance(item, str):
            response = ModelResponse(content=item, metadata={"provider": "simulated"})
        else:
            content = str(item.get("content", ""))
            usage_value = item.get("usage")
            usage = usage_value if isinstance(usage_value, TokenUsage) else None
            if isinstance(usage_value, Mapping):
                usage = TokenUsage(
                    input_tokens=_optional_token(usage_value.get("input_tokens")),
                    output_tokens=_optional_token(usage_value.get("output_tokens")),
                    total_tokens=_optional_token(usage_value.get("total_tokens")),
                )
            response = ModelResponse(
                content=content,
                usage=usage,
                request_id=item.get("request_id"),
                finish_reason=item.get("finish_reason"),
                metadata=item.get("metadata", {"provider": "simulated"}),
                parsed=item.get("parsed"),
            )
        if response_schema is None or response.parsed is not None:
            return response
        if not text_fallback:
            raise InvalidResponseError("Saída estruturada indisponível no provider simulado.")
        return ModelResponse(
            content=response.content,
            usage=response.usage,
            request_id=response.request_id,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
            parsed=parse_text_response(response.content, response_schema),
        )


def _optional_token(value: object) -> int | None:
    return None if value is None else int(str(value))
