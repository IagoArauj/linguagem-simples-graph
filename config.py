"""Configuração externa, manifesto e retomada segura de execuções."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Mapping, Self
from uuid import UUID

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from core.schemas import (
    BranchConfig,
    BranchId,
    PromptPaths,
    RoleModels,
    RunConfig,
    SCHEMA_VERSION,
)

CONFIG_SCHEMA_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "openrouter.yaml"


class ConfigModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class CorpusSettings(ConfigModel):
    path: Path
    id_field: str = Field(default="id", min_length=1)
    text_field: str = Field(default="juridico", min_length=1)


class ExecutionSettings(ConfigModel):
    directory: Path
    results_filename: str = Field(default="results.jsonl", min_length=1)
    metadata_filename: str = Field(default="metadata_per_text.csv", min_length=1)

    @model_validator(mode="after")
    def validate_filenames(self) -> Self:
        for filename in (self.results_filename, self.metadata_filename):
            if Path(filename).name != filename:
                raise ValueError("Nomes de arquivos de execução não podem conter diretórios.")
        return self


class WorkflowSettings(ConfigModel):
    max_revisions: PositiveInt = 3
    branches: tuple[BranchConfig, ...]


class EffectiveConfig(ConfigModel):
    schema_version: str = Field(
        default=CONFIG_SCHEMA_VERSION,
        pattern=r"^1\.0$",
    )
    corpus: CorpusSettings
    execution: ExecutionSettings
    workflow: WorkflowSettings
    prompts: PromptPaths
    models: RoleModels

    def to_run_config(self) -> RunConfig:
        return RunConfig(
            models=self.models,
            prompts=self.prompts,
            max_revisions=self.workflow.max_revisions,
            branches=self.workflow.branches,
        )


class CliOverrides(ConfigModel):
    model: str | None = Field(default=None, min_length=1)
    corpus_path: Path | None = None
    execution_directory: Path | None = None
    max_revisions: PositiveInt | None = None


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolve_config_path(path: str | Path | None) -> Path:
    return resolve_project_path(path or DEFAULT_CONFIG_PATH)


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for key in path[:-1]:
        nested = target.get(key)
        if not isinstance(nested, dict):
            nested = {}
            target[key] = nested
        target = nested
    target[path[-1]] = value


def _environment_overrides(environ: Mapping[str, str]) -> list[tuple[tuple[str, ...], Any]]:
    overrides: list[tuple[tuple[str, ...], Any]] = []

    direct: dict[str, tuple[str, ...]] = {
        "LSG_CORPUS_PATH": ("corpus", "path"),
        "LSG_EXECUTION_DIR": ("execution", "directory"),
        "LSG_MAX_REVISIONS": ("workflow", "max_revisions"),
        "LSG_ANALYZER_PROMPT": ("prompts", "analyzer"),
        "LSG_SIMPLIFIER_PROMPT": ("prompts", "simplifier"),
        "LSG_EVALUATOR_PROMPT": ("prompts", "evaluator"),
    }
    for variable, path in direct.items():
        if variable in environ:
            value: Any = environ[variable]
            if variable == "LSG_MAX_REVISIONS":
                value = int(value)
            overrides.append((path, value))

    # O modelo geral é aplicado antes dos modelos específicos por papel.
    if "LSG_MODEL" in environ:
        for role in ("analyzer", "simplifier", "evaluator"):
            overrides.append((("models", role, "name"), environ["LSG_MODEL"]))

    for role in ("analyzer", "simplifier", "evaluator"):
        prefix = f"LSG_{role.upper()}"
        role_fields: dict[str, tuple[str, object]] = {
            "MODEL": ("name", str),
            "PROVIDER": ("provider", str),
            "BASE_URL": ("base_url", str),
            "API_KEY_ENV": ("api_key_env", str),
            "TIMEOUT_SECONDS": ("request_timeout_seconds", float),
            "TEMPERATURE": ("temperature", float),
            "TECHNICAL_RETRIES": ("technical_retries", int),
            "MAX_RETRIES": ("technical_retries", int),
            "RETRY_BACKOFF_SECONDS": ("retry_backoff_seconds", float),
            "MAX_TOKENS": ("max_tokens", int),
            "TOP_P": ("top_p", float),
            "SUPPORTS_STRUCTURED_OUTPUT": (
                "supports_structured_output",
                lambda value: value.strip().lower() in {"1", "true", "yes", "on"},
            ),
        }
        for suffix, (field_name, converter) in role_fields.items():
            variable = f"{prefix}_{suffix}"
            if variable in environ:
                if not callable(converter):
                    raise TypeError(f"Conversor inválido para {variable}.")
                overrides.append(
                    (
                        ("models", role, field_name),
                        converter(environ[variable]),
                    )
                )

    for branch_id in BranchId:
        variable = f"LSG_{branch_id.value.upper()}_TARGET_AUDIENCE"
        if variable not in environ:
            continue
        # A lista é reconstruída depois, pois branches são posicionais no YAML.
        overrides.append((("branch_audiences", branch_id.value), environ[variable]))

    return overrides


def _apply_branch_audience_overrides(
    data: dict[str, Any],
    audiences: dict[str, str],
) -> None:
    workflow = data.get("workflow")
    if not isinstance(workflow, dict):
        return
    branches = workflow.get("branches")
    if not isinstance(branches, (list, tuple)):
        return
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        branch_id = branch.get("branch_id")
        if isinstance(branch_id, str) and branch_id in audiences:
            branch["target_audience"] = audiences[branch_id]


def load_effective_config(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cli: CliOverrides | None = None,
) -> tuple[EffectiveConfig, Path]:
    """Carrega configuração com precedência CLI > ambiente > arquivo."""
    source_path = resolve_config_path(config_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {source_path}")

    with source_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("A raiz da configuração YAML deve ser um objeto.")

    # Valida o arquivo antes de aplicar overrides, rejeitando chaves desconhecidas.
    file_config = EffectiveConfig.model_validate(raw)
    data = file_config.model_dump(mode="python")

    branch_audiences: dict[str, str] = {}
    for path, value in _environment_overrides(environ or os.environ):
        if path[0] == "branch_audiences":
            branch_audiences[path[1]] = str(value)
        else:
            _set_nested(data, path, value)
    _apply_branch_audience_overrides(data, branch_audiences)

    if cli is not None:
        if cli.model is not None:
            for role in ("analyzer", "simplifier", "evaluator"):
                _set_nested(data, ("models", role, "name"), cli.model)
        if cli.corpus_path is not None:
            _set_nested(data, ("corpus", "path"), cli.corpus_path)
        if cli.execution_directory is not None:
            _set_nested(data, ("execution", "directory"), cli.execution_directory)
        if cli.max_revisions is not None:
            _set_nested(data, ("workflow", "max_revisions"), cli.max_revisions)

    effective = EffectiveConfig.model_validate(data)
    resolved_data = effective.model_dump(mode="python")
    resolved_data["corpus"]["path"] = resolve_project_path(effective.corpus.path)
    resolved_data["execution"]["directory"] = resolve_project_path(
        effective.execution.directory
    )
    for role in ("analyzer", "simplifier", "evaluator"):
        resolved_data["prompts"][role] = resolve_project_path(
            getattr(effective.prompts, role)
        )
    effective = EffectiveConfig.model_validate(resolved_data)

    if not effective.corpus.path.is_file():
        raise FileNotFoundError(f"Corpus não encontrado: {effective.corpus.path}")
    for role in ("analyzer", "simplifier", "evaluator"):
        prompt_path = getattr(effective.prompts, role)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt de {role} não encontrado: {prompt_path}")

    return effective, source_path


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "changed_files": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "changed_files": []}


def _dependency_versions() -> dict[str, str | None]:
    packages = (
        "ipython",
        "langchain",
        "langchain-openai",
        "langchain-openrouter",
        "langgraph",
        "nltk",
        "pydantic",
        "PyYAML",
        "tqdm",
    )
    dependencies: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return dependencies


def build_manifest(
    config: EffectiveConfig,
    *,
    config_source: Path,
    run_id: UUID,
) -> dict[str, Any]:
    corpus_hash = sha256_file(config.corpus.path)
    prompt_hashes = {
        role: sha256_file(getattr(config.prompts, role))
        for role in ("analyzer", "simplifier", "evaluator")
    }
    effective_config = config.model_dump(mode="json")
    compatibility_data = {
        "effective_config": effective_config,
        "corpus_hash": corpus_hash,
        "prompt_hashes": prompt_hashes,
        "schema_versions": {
            "config": CONFIG_SCHEMA_VERSION,
            "contracts": SCHEMA_VERSION,
        },
    }
    compatibility_hash = sha256(
        json.dumps(
            compatibility_data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "manifest_version": "1.0",
        "run_id": str(run_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_source),
        **compatibility_data,
        "compatibility_hash": compatibility_hash,
        "git": _git_metadata(),
        "dependencies": _dependency_versions(),
    }


def prepare_run_directory(
    config: EffectiveConfig,
    manifest: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    run_directory = config.execution.directory / str(manifest["run_id"])
    manifest_path = run_directory / "manifest.json"

    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Não há manifesto para retomar a execução: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            previous = json.load(file)
        if previous.get("compatibility_hash") != manifest["compatibility_hash"]:
            raise ValueError(
                "Retomada recusada: configuração, corpus, prompts ou schemas "
                "diferem da execução original."
            )
        return run_directory

    if run_directory.exists():
        raise FileExistsError(
            f"O diretório da execução já existe: {run_directory}. "
            "Use --resume com o mesmo --run-id para continuar."
        )

    run_directory.mkdir(parents=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return run_directory


def load_completed_documents(results_path: Path) -> dict[str, str]:
    completed: dict[str, str] = {}
    if not results_path.is_file():
        return completed
    with results_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                document_id = record["document_id"]
                content_hash = record["content_hash"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Registro inválido em {results_path}, linha {line_number}."
                ) from exc
            if not isinstance(document_id, str) or not isinstance(content_hash, str):
                raise ValueError(
                    f"Identidade inválida em {results_path}, linha {line_number}."
                )
            completed[document_id] = content_hash
    return completed
