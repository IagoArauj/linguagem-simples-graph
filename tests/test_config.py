import json
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from pydantic import ValidationError

from config import (
    CliOverrides,
    PROJECT_ROOT,
    build_manifest,
    load_effective_config,
    prepare_run_directory,
)
from core.schemas import ModelRole, TargetAudience


def test_paths_are_resolved_from_project_root() -> None:
    config, source = load_effective_config(
        "configs/openrouter.yaml",
        environ={},
    )

    assert source == PROJECT_ROOT / "configs/openrouter.yaml"
    assert config.corpus.path == PROJECT_ROOT / "input/corpus.json"
    assert config.prompts.analyzer == PROJECT_ROOT / "prompts/analisador.txt"


def test_precedence_is_cli_then_environment_then_yaml() -> None:
    config, _ = load_effective_config(
        "configs/openrouter.yaml",
        environ={
            "LSG_MODEL": "environment/general",
            "LSG_EVALUATOR_MODEL": "environment/evaluator",
            "LSG_MAX_REVISIONS": "4",
        },
        cli=CliOverrides(model="cli/all", max_revisions=5),
    )

    assert config.workflow.max_revisions == 5
    assert all(
        config.models.for_role(role).name == "cli/all"
        for role in ModelRole
    )


def test_environment_can_change_audience_without_code_change() -> None:
    config, _ = load_effective_config(
        "configs/openrouter.yaml",
        environ={"LSG_SIMPLE_TARGET_AUDIENCE": "Público geral"},
    )

    simple_branch = next(
        branch
        for branch in config.workflow.branches
        if branch.branch_id.value == "simple"
    )
    assert simple_branch.target_audience == TargetAudience.GENERAL_PUBLIC


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "configs/openrouter.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["api_key"] = "não deve ser aceito"
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_effective_config(invalid_path, environ={})


def test_manifest_does_not_include_environment_credentials() -> None:
    config, source = load_effective_config(
        "configs/openrouter.yaml",
        environ={"OPENROUTER_API_KEY": "segredo-que-nao-pode-vazar"},
    )
    manifest = build_manifest(config, config_source=source, run_id=uuid4())

    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "segredo-que-nao-pode-vazar" not in serialized
    assert {
        "effective_config",
        "corpus_hash",
        "prompt_hashes",
        "schema_versions",
        "dependencies",
        "git",
    } <= manifest.keys()


def test_resume_rejects_changed_configuration_and_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        '[{"id":"1","juridico":"Texto."}]',
        encoding="utf-8",
    )
    output_path = tmp_path / "runs"
    run_id = uuid4()

    config, source = load_effective_config(
        "configs/openrouter.yaml",
        environ={},
        cli=CliOverrides(
            corpus_path=corpus_path,
            execution_directory=output_path,
        ),
    )
    manifest = build_manifest(config, config_source=source, run_id=run_id)
    run_directory = prepare_run_directory(config, manifest, resume=False)
    assert prepare_run_directory(config, manifest, resume=True) == run_directory

    changed_config, _ = load_effective_config(
        "configs/openrouter.yaml",
        environ={},
        cli=CliOverrides(
            corpus_path=corpus_path,
            execution_directory=output_path,
            max_revisions=9,
        ),
    )
    changed_manifest = build_manifest(
        changed_config,
        config_source=source,
        run_id=run_id,
    )
    with pytest.raises(ValueError, match="Retomada recusada"):
        prepare_run_directory(changed_config, changed_manifest, resume=True)

    corpus_path.write_text(
        '[{"id":"1","juridico":"Texto alterado."}]',
        encoding="utf-8",
    )
    changed_corpus_manifest = build_manifest(
        config,
        config_source=source,
        run_id=run_id,
    )
    with pytest.raises(ValueError, match="Retomada recusada"):
        prepare_run_directory(config, changed_corpus_manifest, resume=True)
