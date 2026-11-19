import json
from pathlib import Path
from threading import Event, Lock, get_ident

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


def test_workers_process_whole_texts_without_waiting_for_batch(
    tmp_path: Path, monkeypatch,
) -> None:
    input_path = tmp_path / "results.json"
    output_path = tmp_path / "metrics.jsonl"
    input_path.write_text(json.dumps([
        {field: str(index) for field in metrics_module.REQUIRED_FIELDS}
        for index in range(1, 4)
    ]), encoding="utf-8")
    first_started = Event()
    third_saved = Event()
    lock = Lock()
    calls = {index: [] for index in range(1, 4)}
    active = 0
    max_active = 0
    writer_threads = []
    original_write = metrics_module._write_jsonl_record

    def fake_compute(text, folder, script, field):
        nonlocal active, max_active
        index = int(text)
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls[index].append((field, get_ident()))
        try:
            if index == 1 and field == "simple_simplification":
                first_started.set()
                assert third_saved.wait(5), "O terceiro texto ficou esperando o primeiro"
            elif index == 2:
                assert first_started.wait(5)
            return {"metric": index}
        finally:
            with lock:
                active -= 1

    def track_write(output, record):
        writer_threads.append(get_ident())
        original_write(output, record)
        if record["item_index"] == 3:
            third_saved.set()

    monkeypatch.setattr(metrics_module, "compute_nilc_metrix", fake_compute)
    monkeypatch.setattr(metrics_module, "_write_jsonl_record", track_write)
    metrics_module.compute_metrics(tmp_path, input_path, workers=2)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["item_index"] for record in records] == [2, 3, 1]
    assert all(record["status"] == "success" for record in records)
    assert max_active == 2
    assert set(writer_threads) == {get_ident()}
    for item_calls in calls.values():
        assert [field for field, _ in item_calls] == list(metrics_module.REQUIRED_FIELDS)
        assert len({thread for _, thread in item_calls}) == 1

    def fail_if_called(*args):
        raise AssertionError("Itens concluídos não devem ser reprocessados")

    monkeypatch.setattr(metrics_module, "compute_nilc_metrix", fail_if_called)
    metrics_module.compute_metrics(tmp_path, input_path, workers=4, resume=True)
    assert len(output_path.read_text().splitlines()) == 3


@pytest.mark.parametrize("fail_fast", [False, True])
def test_item_errors_and_fail_fast(tmp_path: Path, monkeypatch, fail_fast) -> None:
    input_path = tmp_path / "results.json"
    input_path.write_text(json.dumps([
        {field: text for field in metrics_module.REQUIRED_FIELDS}
        for text in ["bad", "good"]
    ]), encoding="utf-8")
    calls = []

    def fake_compute(text, *args):
        calls.append(text)
        if text == "bad":
            raise RuntimeError("NILC falhou")
        return {"metric": 1}

    monkeypatch.setattr(metrics_module, "compute_nilc_metrix", fake_compute)
    if fail_fast:
        with pytest.raises(RuntimeError, match="NILC falhou"):
            metrics_module.compute_metrics(
                tmp_path, input_path, workers=1, fail_fast=True
            )
        assert calls == ["bad"]
    else:
        metrics_module.compute_metrics(tmp_path, input_path, workers=1)
        assert calls == ["bad", "good", "good", "good"]
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert records[0]["status"] == "error"
    assert records[0]["error"] == "NILC falhou"
    assert [r["status"] for r in records] == (
        ["error"] if fail_fast else ["error", "success"]
    )


def test_workers_accepts_more_than_three(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", [
        "compute_metrics.py", "-f", str(tmp_path), "-i", "input.json", "-w", "4",
    ])
    assert metrics_module.parse_arguments().workers == 4
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps([
        {field: "Texto" for field in metrics_module.REQUIRED_FIELDS}
    ]), encoding="utf-8")
    monkeypatch.setattr(metrics_module, "compute_nilc_metrix", lambda *args: {"m": 1})
    metrics_module.compute_metrics(tmp_path, input_path, workers=4)
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["status"] == "success"


@pytest.mark.parametrize("workers", [0, -1])
def test_rejects_nonpositive_workers(tmp_path: Path, workers: int) -> None:
    with pytest.raises(ValueError, match="maior ou igual a 1"):
        metrics_module.compute_metrics(tmp_path, tmp_path / "input.json", workers=workers)


def test_rejects_same_input_and_output_path(tmp_path: Path) -> None:
    input_path = tmp_path / "metrics.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="devem ser diferentes"):
        metrics_module.compute_metrics(
            nilc_metrix_folder=tmp_path,
            input_file=input_path,
        )
