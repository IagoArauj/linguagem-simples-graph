"""Adaptador do OpenRouter sobre sua API OpenAI-compatible."""

from __future__ import annotations

from collections.abc import Mapping

from .base import ConfigurationError
from .openai_compatible import OpenAICompatibleModelClient


class OpenRouterModelClient(OpenAICompatibleModelClient):
    """Cliente OpenRouter com URL e credencial recebidas externamente."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        request_timeout_seconds: float = 60.0,
        technical_retries: int = 0,
        retry_backoff_seconds: float = 0.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        supports_structured_output: bool = False,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ConfigurationError("A API key do OpenRouter é obrigatória.")
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            request_timeout_seconds=request_timeout_seconds,
            technical_retries=technical_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            supports_structured_output=supports_structured_output,
            provider_name="openrouter",
            default_headers=default_headers,
        )


OpenRouterAdapter = OpenRouterModelClient
