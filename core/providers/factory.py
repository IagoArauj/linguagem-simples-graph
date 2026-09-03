"""Fábrica de clientes de modelo selecionados por papel."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from core.schemas import ModelRole

from .base import ConfigurationError, ModelClient, SimulatedModelClient
from .openai_compatible import OpenAICompatibleModelClient
from .openrouter import OpenRouterModelClient


@dataclass(frozen=True, slots=True)
class ModelClients:
    analyzer: ModelClient
    simplifier: ModelClient
    evaluator: ModelClient

    def for_role(self, role: ModelRole) -> ModelClient:
        return getattr(self, role.value)


class ModelClientFactory:
    """Cria providers sem depender do módulo de carregamento de configuração."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        simulated_scripts: Mapping[ModelRole | str, Iterable[Any]] | None = None,
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._simulated_scripts = simulated_scripts or {}

    def create(self, run_config: Any, role: ModelRole) -> ModelClient:
        try:
            settings = run_config.model(role)
        except Exception as exc:
            raise ConfigurationError(
                f"Não foi possível obter a configuração do papel {role.value}."
            ) from exc

        provider_value = getattr(settings, "provider", "")
        provider = str(getattr(provider_value, "value", provider_value)).strip().lower()
        model = _required(settings, "name")
        request_timeout_seconds = float(
            getattr(settings, "request_timeout_seconds", 60.0)
        )
        technical_retries = int(getattr(settings, "technical_retries", 0))
        retry_backoff_seconds = float(
            getattr(settings, "retry_backoff_seconds", 0.0)
        )
        temperature = _optional_float(getattr(settings, "temperature", None))
        max_tokens = _optional_int(getattr(settings, "max_tokens", None))
        top_p = _optional_float(getattr(settings, "top_p", None))
        supports_structured_output = bool(
            getattr(settings, "supports_structured_output", False)
        )

        if provider == "openrouter":
            api_key = self._api_key(settings, required=True)
            if api_key is None:  # Garantia adicional para type checkers e mappings customizados.
                raise ConfigurationError("A API key do OpenRouter é obrigatória.")
            return OpenRouterModelClient(
                model=model,
                base_url=_required(settings, "base_url"),
                api_key=api_key,
                request_timeout_seconds=request_timeout_seconds,
                technical_retries=technical_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                supports_structured_output=supports_structured_output,
            )
        if provider in {"openai_compatible", "openai-compatible", "local"}:
            return OpenAICompatibleModelClient(
                model=model,
                base_url=_required(settings, "base_url"),
                api_key=self._api_key(settings, required=False),
                request_timeout_seconds=request_timeout_seconds,
                technical_retries=technical_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                supports_structured_output=supports_structured_output,
            )
        if provider in {"simulated", "mock", "fake"}:
            script = self._simulated_scripts.get(
                role, self._simulated_scripts.get(role.value, ())
            )
            return SimulatedModelClient(
                script=script,
                technical_retries=technical_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
        raise ConfigurationError(f"Provider de modelo desconhecido: {provider!r}.")

    def create_all(self, run_config: Any) -> ModelClients:
        return ModelClients(
            analyzer=self.create(run_config, ModelRole.ANALYZER),
            simplifier=self.create(run_config, ModelRole.SIMPLIFIER),
            evaluator=self.create(run_config, ModelRole.EVALUATOR),
        )

    def _api_key(self, settings: Any, *, required: bool) -> str | None:
        variable = getattr(settings, "api_key_env", None)
        if not variable:
            if required:
                raise ConfigurationError("api_key_env é obrigatório para este provider.")
            return None
        value = self._environ.get(str(variable))
        if not value and required:
            raise ConfigurationError(
                f"A variável de ambiente {variable!r} não está definida."
            )
        return value


def create_model_client(
    run_config: Any,
    role: ModelRole,
    *,
    environ: Mapping[str, str] | None = None,
    simulated_scripts: Mapping[ModelRole | str, Iterable[Any]] | None = None,
) -> ModelClient:
    """Atalho funcional para criar o cliente configurado para ``role``."""

    return ModelClientFactory(
        environ=environ,
        simulated_scripts=simulated_scripts,
    ).create(run_config, role)


def _required(settings: Any, field_name: str) -> str:
    value = getattr(settings, field_name, None)
    if value is None or not str(value).strip():
        raise ConfigurationError(f"{field_name} é obrigatório para este provider.")
    return str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
