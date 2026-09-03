"""Smoke tests das dependências necessárias ao workflow."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "yaml",
        "pydantic",
        "langchain",
        "langchain_openai",
        "langchain_openrouter",
        "langgraph",
        "nltk",
        "tqdm",
    ],
)
def test_runtime_dependency_can_be_imported(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None
