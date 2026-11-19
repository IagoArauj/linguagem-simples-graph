import csv
import json
import sys
from contextlib import contextmanager

from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

import main as cli


LEGACY_COLUMNS = [
    "run_id", "document_id", "content_hash", "execution_status",
    "termination_reason", "elapsed_time", "input_tokens", "output_tokens",
    "llm_calls",
]
COLUMNS = LEGACY_COLUMNS + ["retry_wait_seconds", "elapsed_without_backoff"]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.reader(file))


@pytest.mark.parametrize("empty_file", [False, True])
def test_new_metadata_header(tmp_path, empty_file):
    path = tmp_path / "metadata.csv"
    if empty_file:
        path.touch()
    cli.ensure_metadata_header(path)
    assert read_csv(path) == [COLUMNS]


def test_migration_preserves_values_and_is_idempotent(tmp_path):
    path = tmp_path / "metadata.csv"
    history = [
        ["run", 'documento, "ação"\nsegunda linha', "hash", "ok", "fim",
         "12.3400", "", "2", "3"],
        ["run", "outro", "hash2", "error", "falha", "0.1000", "0", "", "1"],
    ]
    write_csv(path, [LEGACY_COLUMNS, *history])
    cli.ensure_metadata_header(path)
    assert read_csv(path) == [COLUMNS, *[row + ["", ""] for row in history]]
    migrated = path.read_bytes()
    cli.ensure_metadata_header(path)
    assert path.read_bytes() == migrated
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("header", [
    ["unknown"], LEGACY_COLUMNS + ["custom"],
    LEGACY_COLUMNS + ["retry_wait_seconds"], list(reversed(COLUMNS)),
])
def test_unknown_headers_are_not_overwritten(tmp_path, header):
    path = tmp_path / "metadata.csv"
    write_csv(path, [header, ["historical data"]])
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Cabeçalho.*desconhecido"):
        cli.ensure_metadata_header(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("row", [["short"], ["extra"] * 10])
def test_malformed_legacy_rows_are_not_overwritten(tmp_path, row):
    path = tmp_path / "metadata.csv"
    write_csv(path, [LEGACY_COLUMNS, ["valid"] * 9, row])
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Linha.*inválida"):
        cli.ensure_metadata_header(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_migration_replaces_only_after_complete_write(tmp_path, monkeypatch):
    path = tmp_path / "metadata.csv"
    history = ["run", "doc", "hash", "ok", "fim", "1", "", "", "1"]
    write_csv(path, [LEGACY_COLUMNS, history])
    original = path.read_bytes()
    calls = []

    def failed_replace(source, destination):
        calls.append((source, destination))
        assert source.parent == path.parent
        assert destination == path
        assert path.read_bytes() == original
        assert read_csv(source) == [COLUMNS, history + ["", ""]]
        raise OSError("replace failed")

    monkeypatch.setattr(cli.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failed"):
        cli.ensure_metadata_header(path)
    assert len(calls) == 1
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("wait, expected", [(0.0, "10.0000"), (3.25, "6.7500"), (11.0, "0.0000")])
def test_main_csv_and_resume(tmp_path, monkeypatch, capsys, wait, expected):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([
        {"id": "doc1", "juridico": "Texto um."},
        {"id": "doc2", "juridico": "Texto dois."},
    ]), encoding="utf-8")
    run_id = uuid4()
    argv = [
        "main.py", "--corpus", str(corpus), "--output-dir", str(tmp_path / "runs"),
        "--run-id", str(run_id),
    ]
    load_config = cli.load_effective_config
    monkeypatch.setattr(cli, "load_effective_config", lambda path, cli: load_config(
        path, cli=cli, environ={},
    ))
    clients = SimpleNamespace(analyzer=object(), simplifier=object(), evaluator=object())
    monkeypatch.setattr(cli.ModelClientFactory, "create_all", lambda *_: clients)
    events = []
    active = False

    @contextmanager
    def track_document_retries(tracked):
        nonlocal active
        assert tracked == (clients.analyzer, clients.simplifier, clients.evaluator)
        assert not active
        active = True
        timing = SimpleNamespace(retry_wait_seconds=None)
        events.append("enter")
        try:
            yield timing
        finally:
            active = False
            timing.retry_wait_seconds = wait
            events.append("exit")

    # Isola somente o contrato de timing; não depende da implementação em core.
    timing_module = ModuleType("core.timing")
    timing_module.track_document_retries = track_document_retries
    monkeypatch.setitem(sys.modules, "core.timing", timing_module)

    def run(document, *, config, clients, run_id):
        assert active
        events.append(document.document_id)
        record = {
            "run_id": str(run_id), "document_id": document.document_id,
            "content_hash": document.content_hash,
            "usage": {"input_tokens": None, "output_tokens": 2, "llm_calls": 3},
        }
        return SimpleNamespace(
            run_id=run_id, document_id=document.document_id,
            content_hash=document.content_hash,
            execution_status=SimpleNamespace(value="success"),
            termination_reason=SimpleNamespace(value="completed"),
            usage=SimpleNamespace(**record["usage"]),
            model_dump=lambda **kwargs: record,
        )

    monkeypatch.setattr(cli.LinguagemSimplesGraph, "run", run)

    def execute(resume=False):
        clock = iter([0.0, 1.0, 11.0, 12.0, 22.0, 23.0])
        monkeypatch.setattr(cli.time, "perf_counter", lambda: next(clock))
        monkeypatch.setattr(sys, "argv", argv + (["--resume"] if resume else []))
        cli.main()
        return capsys.readouterr().out

    output = execute()
    directory = tmp_path / "runs" / str(run_id)
    metadata = directory / "metadata_per_text.csv"
    rows = read_csv(metadata)
    assert rows[0] == COLUMNS
    assert len(rows) == 3
    for row in rows[1:]:
        assert row[5:] == ["10.0000", "", "2", "3", f"{wait:.4f}", expected]
    assert events == ["enter", "doc1", "exit", "enter", "doc2", "exit"]
    assert f"Backoff sem chamadas ativas: {wait:.4f}s" in output
    assert f"Tempo sem esse backoff: {expected}s" in output
    results = directory / "results.jsonl"
    records = results.read_text(encoding="utf-8").splitlines()
    assert all("retry_wait_seconds" not in json.loads(record) for record in records)

    # Simula execução antiga parcialmente concluída, mantendo manifesto real.
    write_csv(metadata, [LEGACY_COLUMNS, rows[1][:-2]])
    results.write_text(records[0] + "\n", encoding="utf-8")
    events.clear()
    execute(resume=True)
    resumed = read_csv(metadata)
    assert resumed == [COLUMNS, rows[1][:-2] + ["", ""], rows[2]]
    assert events == ["enter", "doc2", "exit"]
    assert len(results.read_text(encoding="utf-8").splitlines()) == 2

    events.clear()
    before = metadata.read_bytes()
    execute(resume=True)
    assert not events
    assert metadata.read_bytes() == before
