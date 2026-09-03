"""Adaptador para endpoints que implementam a API OpenAI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import openai
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .base import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    ModelResponse,
    RetryingModelClient,
    StructuredSchema,
    TokenUsage,
    TransientError,
    parse_text_response,
    serialize_parsed,
)


class OpenAICompatibleModelClient(RetryingModelClient):
    """Cliente OpenAI-compatible com normalização e retry controlado pelo adaptador."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        request_timeout_seconds: float = 60.0,
        technical_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        supports_structured_output: bool = False,
        provider_name: str = "openai_compatible",
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("O nome do modelo é obrigatório.")
        if not base_url.strip():
            raise ConfigurationError("base_url é obrigatória para endpoint OpenAI-compatible.")
        if request_timeout_seconds <= 0:
            raise ConfigurationError("request_timeout_seconds deve ser positivo.")
        super().__init__(
            technical_retries=technical_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self.provider_name = provider_name
        self.supports_structured_output = supports_structured_output
        try:
            self._chat = ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key=SecretStr(api_key or "not-needed"),
                timeout=request_timeout_seconds,
                max_retries=0,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                default_headers=default_headers,
            )
        except Exception as exc:
            raise ConfigurationError(f"Configuração inválida de {provider_name}: {exc}") from exc

    def _invoke_once(
        self,
        messages: Sequence[Any],
        *,
        response_schema: StructuredSchema | None,
        text_fallback: bool,
    ) -> ModelResponse:
        try:
            if response_schema is not None and self.supports_structured_output:
                schema = (
                    dict(response_schema)
                    if isinstance(response_schema, Mapping)
                    else response_schema
                )
                runnable = self._chat.with_structured_output(
                    schema,
                    include_raw=True,
                )
                result = runnable.invoke(messages)
                return self._normalize_structured(
                    result,
                    response_schema=response_schema,
                    text_fallback=text_fallback,
                )
            message = self._chat.invoke(messages)
            response = self._normalize_message(message)
            if response_schema is None:
                return response
            if not text_fallback:
                raise InvalidResponseError(
                    f"{self.provider_name} não oferece saída estruturada habilitada."
                )
            return ModelResponse(
                content=response.content,
                usage=response.usage,
                request_id=response.request_id,
                finish_reason=response.finish_reason,
                metadata=response.metadata,
                parsed=parse_text_response(response.content, response_schema),
            )
        except (AuthenticationError, ConfigurationError, InvalidResponseError, TransientError):
            raise
        except Exception as exc:
            raise self._translate_error(exc) from exc

    def _normalize_structured(
        self,
        result: Any,
        *,
        response_schema: StructuredSchema,
        text_fallback: bool,
    ) -> ModelResponse:
        if not isinstance(result, Mapping):
            raise InvalidResponseError("Resposta estruturada possui envelope inesperado.")
        raw = result.get("raw")
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
        response = self._normalize_message(raw) if raw is not None else ModelResponse("")
        if parsed is None:
            if text_fallback and response.content:
                parsed = parse_text_response(response.content, response_schema)
                return ModelResponse(
                    content=response.content,
                    usage=response.usage,
                    request_id=response.request_id,
                    finish_reason=response.finish_reason,
                    metadata=response.metadata,
                    parsed=parsed,
                )
            raise InvalidResponseError(
                f"Resposta estruturada inválida: {parsing_error or 'conteúdo ausente'}"
            )
        content = response.content or serialize_parsed(parsed)
        return ModelResponse(
            content=content,
            usage=response.usage,
            request_id=response.request_id,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
            parsed=parsed,
        )

    def _normalize_message(self, message: Any) -> ModelResponse:
        if message is None:
            raise InvalidResponseError("O endpoint retornou uma resposta vazia.")
        content = _content_as_text(getattr(message, "content", ""))
        response_metadata = dict(getattr(message, "response_metadata", None) or {})
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        usage_source = usage_metadata or token_usage
        usage = _normalize_usage(usage_source)
        headers = response_metadata.get("headers") or {}
        request_id = (
            response_metadata.get("request_id")
            or response_metadata.get("id")
            or headers.get("x-request-id")
        )
        finish_reason = response_metadata.get("finish_reason")
        metadata = {"provider": self.provider_name, **response_metadata}
        return ModelResponse(
            content=content,
            usage=usage,
            request_id=str(request_id) if request_id is not None else None,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            metadata=metadata,
        )

    @staticmethod
    def _translate_error(exc: Exception) -> Exception:
        if isinstance(exc, openai.AuthenticationError):
            return AuthenticationError(str(exc))
        if isinstance(
            exc,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        ):
            return TransientError(str(exc))
        status_code = getattr(exc, "status_code", None)
        if status_code in (401, 403):
            return AuthenticationError(str(exc))
        if status_code in (408, 409, 429) or (
            isinstance(status_code, int) and status_code >= 500
        ):
            return TransientError(str(exc))
        if isinstance(exc, (ValueError, TypeError)):
            return InvalidResponseError(str(exc))
        return InvalidResponseError(str(exc))


def _normalize_usage(source: Mapping[str, Any]) -> TokenUsage | None:
    if not source:
        return None
    input_tokens = source.get("input_tokens", source.get("prompt_tokens"))
    output_tokens = source.get("output_tokens", source.get("completion_tokens"))
    total_tokens = source.get("total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return TokenUsage(
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
    )


def _content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""
