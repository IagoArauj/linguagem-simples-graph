import csv
import hashlib
import json

import pytest

from metrics.paired_data import METRICS, load_paired_data


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def artifacts(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    jsonl(corpus, [{"production_id": ident, "original_text": "Original"} for ident in (9, 2)])
    corpus_hash = hashlib.sha256(corpus.read_bytes()).hexdigest()
    single = tmp_path / "single.jsonl"
    baseline = {"model": "provider/model", "original_text": "Original", "simplified_text": "Single",
                "status": "ok", "finish_reason": "stop"}
    single_rows = [{**baseline, "production_id": 2, "finish_reason": "length"},
                   {**baseline, "production_id": 9}]
    jsonl(single, single_rows)
    nilc = dict.fromkeys(METRICS[:5], 1)
    jsonl(tmp_path / "single_metrics.jsonl", [
        {**row, "status": "success", "simplified_text_metrics": nilc} for row in single_rows
    ])
    multi = tmp_path / "multi"
    multi.mkdir()
    manifest = {"manifest_version": "1.0", "run_id": "run", "corpus_hash": corpus_hash,
                "effective_config": {"models": {"simplifier": {"name": "provider/model"}},
                                     "corpus": {"id_field": "production_id", "text_field": "original_text"},
                                     "execution": {"results_filename": "results.jsonl"}}}
    (multi / "manifest.json").write_text(json.dumps(manifest))
    identity = {"document_id": "9", "run_id": "run", "content_hash": digest("Original")}
    branches = [
        {**identity, "branch_id": level, "final_text": level, "execution_status": status,
         "quality_verdict": verdict, "termination_reason": reason}
        for level, status, verdict, reason in (
            ("simple", "completed", "approved", "approval"),
            ("moderate", "completed", "rejected", "revision_limit"),
            ("aggressive", "failed", "not_evaluated", "invalid_response"),
        )
    ]
    jsonl(multi / "results.jsonl", [{**identity, "schema_version": "1.0", "original_text": "Original",
                                    "execution_status": "partial", "branches": branches}])
    metrics = {**identity, "status": "success", **{
        f"{b['branch_id']}_simplification": nilc for b in branches
    }}
    jsonl(multi / "metrics.jsonl", [{"status": "error", "item_index": 1}, metrics,
          {**metrics, "simple_simplification": {**nilc, "flesch": 0, "gunning_fox": "NaN"}}])
    common = {"metrics_version": "1.0", "corpus_hash": corpus_hash, "source_hash": digest("Original"),
              "scorer_config": '{"model":"xlm", "idf":false}', "metric_status": "success",
              "bertscore_f1": ".8", "sari_multi_reference": "25"}
    write_csv(tmp_path / "single_semantic_metrics.csv", [
        {**common, **row, "candidate_hash": digest("Single")} for row in single_rows
    ])
    write_csv(multi / "semantic_metrics.csv", [
        {**common, "scorer_config": '{"idf":false,"model":"xlm"}', "run_id": "run", "document_id": "9",
         "branch_id": b["branch_id"], "candidate_hash": digest(b["final_text"])} for b in branches
    ])
    return [single], [multi], corpus


def test_load_sparse_metrics_and_provenance(artifacts):
    models, ids, single, multi, approved, provenance = load_paired_data(*artifacts)
    model, = models
    assert model == "provider/model"
    assert ids == ["9", "2"]
    assert set(single) == {(model, "9", metric) for metric in METRICS}
    assert set(multi) == ({(model, level, "9", metric) for level in ("simple", "moderate")
                           for metric in METRICS} - {(model, "simple", "9", "gunning_fox")})
    assert multi[model, "simple", "9", "flesch"] == 0.0
    assert multi[model, "moderate", "9", "sari_multi_reference"] == 25.0
    assert approved == {(model, "simple", "9")}
    assert provenance["scorer_config"] == {"idf": False, "model": "xlm"}
    assert len(provenance["source_files"]) == 8
    assert all(len(source["sha256"]) == 64 for source in provenance["source_files"])
    assert "Retry NILC" in provenance["warnings"][0]
    json.dumps(provenance)


def test_optional_missing_sources(artifacts):
    singles, runs, _ = artifacts
    singles[0].with_name("single_semantic_metrics.csv").unlink()
    (runs[0] / "metrics.jsonl").unlink()
    _, _, single, multi, approved, provenance = load_paired_data(*artifacts)
    assert len(single) == 5
    assert len(multi) == 4
    assert len(approved) == 1
    assert len(provenance["warnings"]) == 2


@pytest.mark.parametrize(("field", "value", "message"), [
    ("source_hash", "wrong", "source_hash"),
    ("candidate_hash", "wrong", "candidate_hash"),
    ("scorer_config", '{"model":"other"}', "scorer_config incompatível"),
    ("model", "other", "model divergente"),
    ("original_text", "wrong", "original_text"),
    ("simplified_text", "wrong", "simplified_text"),
    ("production_id", "unknown", "Documento desconhecido"),
])
def test_reject_inconsistent_semantic_row(artifacts, field, value, message):
    path = artifacts[0][0].with_name("single_semantic_metrics.csv")
    with path.open() as file:
        rows = list(csv.DictReader(file))
    rows[0][field] = value  # Valida inclusive registros inelegíveis por finish_reason.
    write_csv(path, rows)
    with pytest.raises(ValueError, match=message):
        load_paired_data(*artifacts)


def test_reject_unpaired_model(artifacts):
    path = artifacts[1][0] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["effective_config"]["models"]["simplifier"]["name"] = "other"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Modelos sem par"):
        load_paired_data(*artifacts)


def test_reject_duplicate_document(artifacts):
    path = artifacts[0][0]
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicada"):
        load_paired_data(*artifacts)


def test_reject_changed_corpus_bytes(artifacts):
    path = artifacts[2]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="corpus_hash"):
        load_paired_data(*artifacts)
