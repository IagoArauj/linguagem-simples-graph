from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "input"
    / "PorSimplesSent"
    / "compute_corpus_metrics.py"
)


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compute_corpus_metrics", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_corpus(path: Path, production_ids: list[int]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for production_id in production_ids:
            record = {
                "production_id": production_id,
                "original_text": f"Original {production_id}",
                "natural_text": f"Natural {production_id}",
                "strong_text": f"Forte {production_id}",
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_loads_corpus_as_one_json_object_per_line(tmp_path: Path) -> None:
    module = load_script()
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, [10, 20])

    items = module._load_jsonl_items(corpus)

    assert [item["production_id"] for item in items] == [10, 20]
    assert items[0]["natural_text"] == "Natural 10"


def test_rejects_invalid_jsonl_with_line_number(tmp_path: Path) -> None:
    module = load_script()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"production_id":1,"original_text":"A",'
        '"natural_text":"B","strong_text":"C"}\n{inválido}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="linha 2"):
        module._load_jsonl_items(corpus)


def test_rejects_duplicate_production_id(tmp_path: Path) -> None:
    module = load_script()
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(corpus, [1, 1])

    with pytest.raises(ValueError, match="production_id duplicado"):
        module._load_jsonl_items(corpus)


def test_output_preserves_production_id_and_resume_uses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_script()
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "metrics.jsonl"
    write_corpus(corpus, [100, 200])
    calls: list[int] = []

    def fake_compute_item_metrics(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["item"]["production_id"])
        return {
            "original": {"metric": 1},
            "simple_simplification": {"metric": 2},
            "aggressive_simplification": {"metric": 3},
        }

    monkeypatch.setattr(module, "_compute_item_metrics", fake_compute_item_metrics)

    module.compute_metrics(
        nilc_metrix_folder=tmp_path,
        input_file=corpus,
        output_file=output,
        workers=1,
    )
    module.compute_metrics(
        nilc_metrix_folder=tmp_path,
        input_file=corpus,
        output_file=output,
        workers=1,
        resume=True,
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert calls == [100, 200]
    assert [record["production_id"] for record in records] == [100, 200]
    assert all(record["status"] == "success" for record in records)
