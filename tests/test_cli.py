import json
from pathlib import Path

import pytest

from main import build_parser, summarize_results


def test_model_argument_remains_available() -> None:
    args = build_parser().parse_args(["--model", "qwen/qwen3-32b"])
    assert args.model == "qwen/qwen3-32b"


def test_resume_requires_run_id() -> None:
    parser = build_parser()
    args = parser.parse_args(["--resume"])
    assert args.resume is True
    assert args.run_id is None


def test_summary_preserves_unknown_token_usage(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    records = [
        {"usage": {"input_tokens": None, "output_tokens": 2, "llm_calls": 1}},
        {"usage": {"input_tokens": 3, "output_tokens": 4, "llm_calls": 2}},
    ]
    results.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )

    assert summarize_results(results) == (2, None, 6, 3)


def test_help_documents_precedence(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--model" in help_text
    assert "Precedência" in help_text
    assert "variáveis de ambiente" in help_text
