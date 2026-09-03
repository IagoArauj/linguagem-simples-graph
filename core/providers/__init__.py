"""Clientes de modelos e contratos normalizados."""

from .base import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    ModelClient,
    ModelClientError,
    ModelResponse,
    RetryingModelClient,
    SimulatedModelClient,
    StructuredSchema,
    TokenUsage,
    TransientError,
)
from .factory import ModelClientFactory, ModelClients, create_model_client
from .openai_compatible import OpenAICompatibleModelClient
from .openrouter import OpenRouterAdapter, OpenRouterModelClient

__all__ = [
    "AuthenticationError",
    "ConfigurationError",
    "InvalidResponseError",
    "ModelClient",
    "ModelClientError",
    "ModelClientFactory",
    "ModelClients",
    "ModelResponse",
    "OpenAICompatibleModelClient",
    "OpenRouterAdapter",
    "OpenRouterModelClient",
    "RetryingModelClient",
    "SimulatedModelClient",
    "StructuredSchema",
    "TokenUsage",
    "TransientError",
    "create_model_client",
]
