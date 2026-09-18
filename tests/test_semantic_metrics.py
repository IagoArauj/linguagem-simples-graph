import csv
import json
from pathlib import Path

import pytest

from metrics.semantic import BertCache, digest, evaluate_run, file_digest


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score(self, candidates, references):
        self.calls.append((candidates, references))
        return [0.8], [0.9], [0.85]


def fixture_run(tmp_path: Path, name="run-1"):
    corpus = tmp_path / "corpus.jsonl"
    source = "O texto original."
    corpus.write_text(json.dumps({"production_id": 1, "original_text": source,
                                  "natural_text": "Texto natural.", "strong_text": "Texto forte."}), encoding="utf-8")
    run = tmp_path / name
    run.mkdir()
    manifest = {"manifest_version": "1.0", "run_id": name, "corpus_hash": file_digest(corpus),
                "effective_config": {"corpus": {"path": str(corpus), "id_field": "production_id", "text_field": "original_text"},
                                     "execution": {"results_filename": "custom.jsonl"}}}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = {"schema_version": "1.0", "run_id": name, "document_id": "1", "original_text": source,
              "content_hash": digest(source), "branches": [
                  {"branch_id": "simple", "final_text": "Texto IA.", "quality_verdict": "rejected"},
                  {"branch_id": "moderate", "final_text": "Texto IA."},
                  {"branch_id": "aggressive", "final_text": None}]}
    (run / "custom.jsonl").write_text(json.dumps(result), encoding="utf-8")
    return run, corpus


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_manifest_outputs_and_cache_across_runs(tmp_path):
    run, corpus = fixture_run(tmp_path)
    scorer = FakeScorer()
    output = evaluate_run(run, settings={"model": "fake"}, scorer_factory=lambda: scorer)
    rows = read_csv(output)
    humans = read_csv(corpus.parent / "human_bertscore.csv")
    assert len(scorer.calls) == 3  # two humans and one distinct AI candidate
    assert all(ref == ["O texto original."] for _, ref in scorer.calls)
    assert len(humans) == 2
    assert {row["reference"] for row in humans} == {"natural", "strong"}
    assert "run_id" not in humans[0]
    assert len(rows) == 3
    assert {row["run_id"] for row in rows} == {"run-1"}
    assert rows[0]["quality_verdict"] == "rejected"
    assert rows[0]["bertscore_f1"] == "0.85"
    assert float(rows[0]["sari_multi_reference"]) >= 0
    assert rows[2]["metric_status"] == "skipped_no_text"
    assert rows[2]["bertscore_f1"] == ""
    run2, _ = fixture_run(tmp_path, "run-2")

    def should_not_initialize():
        pytest.fail("Cached pairs must not load model")

    evaluate_run(run2, settings={"model": "fake"}, scorer_factory=should_not_initialize)
    evaluate_run(run, settings={"model": "fake"}, scorer_factory=should_not_initialize)
    assert len(scorer.calls) == 3
    assert {row["run_id"] for row in read_csv(run2 / "semantic_metrics.csv")} == {"run-2"}


def test_cache_invalidates_on_text_or_settings_change(tmp_path):
    scorer = FakeScorer()
    path = tmp_path / "cache.sqlite3"
    cache = BertCache(path, {"model": "one"}, lambda: scorer)
    cache.score("A", "B")
    cache.score("A", "B")
    cache.score("A", "C")
    cache.close()
    cache = BertCache(path, {"model": "two"}, lambda: scorer)
    cache.score("A", "B")
    cache.close()
    assert len(scorer.calls) == 3


@pytest.mark.parametrize("corrupt", ["corpus", "run_id", "content_hash", "document_id", "final_text"])
def test_rejects_mismatches_before_inference(tmp_path, corrupt):
    run, corpus = fixture_run(tmp_path)
    if corrupt == "corpus":
        corpus.write_text("{}", encoding="utf-8")
    else:
        path = run / "custom.jsonl"
        result = json.loads(path.read_text())
        if corrupt == "final_text":
            result["branches"][0]["final_text"] = 12
        else:
            result[corrupt] = "wrong"
        path.write_text(json.dumps(result))
    scorer = FakeScorer()
    with pytest.raises(ValueError):
        evaluate_run(run, settings={}, scorer_factory=lambda: scorer)
    assert not scorer.calls
