import json
from pathlib import Path

import pytest

import compute_metrics as metrics_module


def test_metrics_accept_jsonl_without_external_nilc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "simple_simplification": "Simples.",
                "moderate_simplification": "Moderada.",
                "aggressive_simplification": "Forte.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    processed_fields: list[str] = []

    def fake_compute(*args, **kwargs):
        processed_fields.append(args[3])
        return {"metric": 1}

    monkeypatch.setattr(metrics_module, "compute_nilc_metrix", fake_compute)

    metrics_module.compute_metrics(
        nilc_metrix_folder=tmp_path,
        input_file=input_path,
        workers=1,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["status"] == "success"
    assert "original" not in record
    assert set(processed_fields) == {
        "simple_simplification",
        "moderate_simplification",
        "aggressive_simplification",
    }


def test_skips_versioned_result_without_valid_simplifications(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": "run-1",
                "document_id": "4",
                "content_hash": "hash",
                "original_text": "Original.",
                "branches": [
                    {"branch_id": "simple", "final_text": None},
                    {"branch_id": "moderate", "final_text": None},
                    {"branch_id": "aggressive", "final_text": None},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("NILC-Metrix não deveria ser executado")

    monkeypatch.setattr(metrics_module, "compute_nilc_metrix", fail_if_called)

    metrics_module.compute_metrics(
        nilc_metrix_folder=tmp_path,
        input_file=input_path,
        output_file=output_path,
        workers=3,
        fail_fast=True,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["status"] == "skipped"
    assert record["reason"] == "no_valid_simplifications"
    assert len(record["missing_stages"]) == 3


def test_computes_only_available_simplifications(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "simple_simplification": "Simples.",
                "moderate_simplification": None,
                "aggressive_simplification": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    processed_fields: list[str] = []

    def fake_compute(*args, **kwargs):
        processed_fields.append(args[3])
        return {"metric": 1}

    monkeypatch.setattr(metrics_module, "compute_nilc_metrix", fake_compute)

    metrics_module.compute_metrics(
        nilc_metrix_folder=tmp_path,
        input_file=input_path,
        output_file=output_path,
        workers=3,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["status"] == "partial"
    assert processed_fields == ["simple_simplification"]
    assert set(record["missing_stages"]) == {
        "moderate_simplification",
        "aggressive_simplification",
    }


def test_resume_treats_non_error_statuses_as_completed(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.jsonl"
    output_path.write_text(
        "\n".join(
            json.dumps({"item_index": index, "status": status})
            for index, status in enumerate(
                ["success", "partial", "skipped", "error"],
                start=1,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert metrics_module._load_completed_indices(output_path) == {1, 2, 3}


def test_input_cli_argument_is_required(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["compute_metrics.py", "--nilc-metrix-folder", "/tmp/nilc"],
    )

    with pytest.raises(SystemExit) as exc_info:
        metrics_module.parse_arguments()

    assert exc_info.value.code == 2


def test_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="devem ser diferentes"):
        metrics_module.compute_metrics(
            nilc_metrix_folder=tmp_path,
            input_file=input_path,
        )
