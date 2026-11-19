import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from metrics import analysis_data
from metrics.analysis_data import COLUMNS, NILC_METRICS, load_analysis_data


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_records(path, rows, array=False):
    path.write_text(json.dumps(rows) if array else "\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture_run(tmp_path, name="run-1", array=False):
    corpus = tmp_path / "corpus.jsonl"
    humans = [{"production_id": ident, "original_text": f"Original {ident}.",
               "natural_text": f"Natural {ident}.", "strong_text": f"Forte {ident}."}
              for ident in (20, 10)]
    if not corpus.exists():
        write_records(corpus, humans, array=array)
    run = tmp_path / name
    run.mkdir()
    manifest = {"manifest_version": "1.0", "run_id": name,
                "corpus_hash": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "effective_config": {
                    "corpus": {"path": str(corpus), "id_field": "production_id", "text_field": "original_text"},
                    "execution": {"results_filename": "custom.jsonl", "metadata_filename": "operation.csv"},
                    "models": {"simplifier": {"name": "same-model"}}}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    results = [{"schema_version": "1.0", "run_id": name, "document_id": str(ident),
                "original_text": f"Original {ident}.", "content_hash": digest(f"Original {ident}."),
                "branches": [
                    {"branch_id": "simple", "final_text": f"IA {ident}", "quality_verdict": "rejected", "execution_status": "failed"},
                    {"branch_id": "moderate", "final_text": f"Moderado {ident}", "quality_verdict": "approved"},
                    {"branch_id": "aggressive", "final_text": None},
                    {"branch_id": "custom", "final_text": "Extra", "quality_verdict": "approved"}]}
               for ident in (10, 20)]
    write_records(run / "custom.jsonl", results)
    return run, corpus, manifest, results


def value(frame, run, doc, version, metric):
    rows = frame[(frame.run_id == run) & (frame.document_id == str(doc)) &
                 (frame.version == version) & (frame.metric == metric)]
    assert len(rows) == 1
    return rows.iloc[0].value


def semantic_row(manifest, result, branch=None, reference=None):
    candidate = branch["final_text"] if branch else f"Natural {result['document_id']}."
    row = {"metrics_version": "1.0", "corpus_hash": manifest["corpus_hash"],
           "document_id": result["document_id"], "source_hash": result["content_hash"],
           "candidate_hash": digest(candidate) if candidate else "",
           "scorer_config": json.dumps({"model": "fake-bert", "num_layers": 9}),
           "bertscore_f1": "0.8" if candidate else ""}
    if branch:
        row.update(run_id=manifest["run_id"], branch_id=branch["branch_id"],
                   metric_status="success" if candidate else "skipped_no_text",
                   rouge_l_natural_f1="0.5", sari_multi_reference="0.6")
    else:
        row["reference"] = reference or "natural"
    return row


@pytest.mark.parametrize("array", [False, True])
def test_lengths_missing_optional_sources_and_same_model_runs(tmp_path, array):
    run, corpus, _, _ = fixture_run(tmp_path, array=array)
    run2, _, _, _ = fixture_run(tmp_path, "run-2")
    frame, provenance = load_analysis_data([run, run2])
    assert list(frame.columns) == COLUMNS
    assert len(frame[frame.run_id == "corpus"]) == 6
    assert len(frame) == 18
    assert set(frame.run_name) == {"Corpus humano", "same-model [run-1]", "same-model [run-2]"}
    assert value(frame, "run-1", 10, "simple", "text_length") == len("IA 10")
    assert value(frame, "corpus", 20, "natural", "text_length") == len("Natural 20.")
    assert provenance["scorer_config"] is None
    assert len(provenance["warnings"]) == 8
    assert len(provenance["source_files"]) == 5
    for source in provenance["source_files"]:
        assert source["sha256"] == hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
    assert provenance["corpus"]["path"] == str(corpus)
    json.dumps(provenance, allow_nan=False)


def test_nilc_shuffled_ids_retries_partial_skipped_and_nonfinite(tmp_path):
    run, corpus, _, _ = fixture_run(tmp_path)
    write_records(corpus.parent / "metrics.jsonl", [
        {"production_id": 10, "status": "success", "original": {"flesch": 10},
         "simple_simplification": {"flesch": 11}, "aggressive_simplification": {"flesch": 12}},
        {"document_id": "20", "status": "partial", "original": {"flesch": 20}},
    ])
    write_records(run / "metrics.jsonl", [
        {"document_id": "20", "status": "success", "simple_simplification": {"flesch": 1}},
        {"document_id": "10", "status": "skipped", "simple_simplification": {"flesch": 999}},
        {"document_id": "20", "status": "partial", "simple_simplification": {"flesch": 2, "brunet": "inf", "cross_entropy": None, "ignored": 9}},
        {"document_id": "20", "status": "error"},
    ])
    frame, provenance = load_analysis_data([run])
    assert value(frame, "corpus", 10, "natural", "flesch") == 11
    assert value(frame, "corpus", 10, "strong", "flesch") == 12
    assert value(frame, "corpus", 20, "original", "flesch") == 20
    assert value(frame, "run-1", 20, "simple", "flesch") == 2
    assert pd.isna(value(frame, "run-1", 20, "simple", "brunet"))
    assert pd.isna(value(frame, "run-1", 20, "simple", "cross_entropy"))
    assert not ((frame.run_id == "run-1") & (frame.document_id == "10") & (frame.metric == "flesch")).any()
    assert "ignored" not in set(frame.metric)
    assert any("Retry NILC" in warning for warning in provenance["warnings"])


def test_legacy_indices_use_distinct_original_orders_and_warn(tmp_path):
    run, _, _, _ = fixture_run(tmp_path)
    custom = tmp_path / "legacy.jsonl"
    write_records(custom, [{"item_index": 1, "status": "success", "original": {"flesch": 20}}])
    write_records(run / "metrics.jsonl", [{"item_index": 1, "status": "partial", "simple_simplification": {"flesch": 10}}])
    frame, provenance = load_analysis_data([run], corpus_metrics=custom)
    assert value(frame, "corpus", 20, "original", "flesch") == 20
    assert value(frame, "run-1", 10, "simple", "flesch") == 10
    assert sum("Fallback legado" in warning for warning in provenance["warnings"]) == 2
    assert str(custom) in {source["path"] for source in provenance["source_files"]}


def test_semantic_and_metadata_join_by_id_and_approved_only(tmp_path):
    run, corpus, manifest, results = fixture_run(tmp_path)
    rows = [semantic_row(manifest, result, branch) for result in reversed(results) for branch in result["branches"]]
    write_csv(run / "semantic_metrics.csv", rows)
    write_csv(corpus.parent / "human_bertscore.csv", [semantic_row(manifest, result) for result in results])
    write_csv(run / "operation.csv", [
        {"document_id": "20", "run_id": "run-1", "content_hash": digest("Original 20."),
         "elapsed_time": 3, "input_tokens": "", "output_tokens": 8, "llm_calls": 4},
        {"document_id": "10", "run_id": "run-1", "content_hash": digest("Original 10."),
         "elapsed_time": 7, "input_tokens": 2, "output_tokens": "nan", "llm_calls": 5},
    ])
    full, _ = load_analysis_data([run])
    assert value(full, "run-1", 20, "simple", "bertscore_f1") == .8
    frame, provenance = load_analysis_data([run], approved_only=True)
    assert "simple" not in set(frame[frame.run_id != "corpus"].version)
    assert "aggressive" not in set(frame.version)
    assert value(frame, "corpus", 10, "natural", "bertscore_f1") == .8
    assert value(frame, "run-1", 20, "moderate", "sari_multi_reference") == .6
    assert value(frame, "run-1", 10, "execution", "elapsed_time") == 7
    assert pd.isna(value(frame, "run-1", 20, "execution", "input_tokens"))
    assert pd.isna(value(frame, "run-1", 10, "execution", "output_tokens"))
    assert provenance["scorer_config"]["model"] == "fake-bert"
    assert len(provenance["source_files"]) == 6


@pytest.mark.parametrize("field", ["metrics_version", "corpus_hash", "source_hash", "candidate_hash", "run_id", "document_id", "branch_id"])
def test_rejects_semantic_identity_and_hashes(tmp_path, field):
    run, _, manifest, results = fixture_run(tmp_path)
    row = semantic_row(manifest, results[0], results[0]["branches"][0])
    row[field] = "wrong"
    write_csv(run / "semantic_metrics.csv", [row])
    with pytest.raises(ValueError):
        load_analysis_data([run])


@pytest.mark.parametrize("field", ["source_hash", "candidate_hash", "scorer_config", "run_id"])
def test_semantic_required_fields(tmp_path, field):
    run, _, manifest, results = fixture_run(tmp_path)
    row = semantic_row(manifest, results[0], results[0]["branches"][0])
    del row[field]
    write_csv(run / "semantic_metrics.csv", [row])
    with pytest.raises(ValueError, match="obrigatórios"):
        load_analysis_data([run])


def test_scorer_mismatch_between_files(tmp_path):
    run, corpus, manifest, results = fixture_run(tmp_path)
    write_csv(corpus.parent / "human_bertscore.csv", [semantic_row(manifest, results[0])])
    row = semantic_row(manifest, results[0], results[0]["branches"][0])
    row["scorer_config"] = json.dumps({"model": "other-bert", "num_layers": 9})
    write_csv(run / "semantic_metrics.csv", [row])
    with pytest.raises(ValueError, match="scorer_config incompatível"):
        load_analysis_data([run])


@pytest.mark.parametrize("field", ["schema_version", "run_id", "content_hash", "original_text", "document_id"])
def test_result_contract(tmp_path, field):
    run, _, _, results = fixture_run(tmp_path)
    results[0][field] = "wrong"
    write_records(run / "custom.jsonl", results)
    with pytest.raises(ValueError):
        load_analysis_data([run])


@pytest.mark.parametrize("where", ["corpus", "results", "branches", "metadata", "semantic", "runs"])
def test_duplicates_rejected(tmp_path, where):
    run, corpus, manifest, results = fixture_run(tmp_path)
    if where == "corpus":
        rows = [json.loads(line) for line in corpus.read_text().splitlines()]
        write_records(corpus, rows + rows[:1])
        manifest["corpus_hash"] = hashlib.sha256(corpus.read_bytes()).hexdigest()
        (run / "manifest.json").write_text(json.dumps(manifest))
    elif where == "results":
        write_records(run / "custom.jsonl", results + results[:1])
    elif where == "branches":
        results[0]["branches"].append(results[0]["branches"][0])
        write_records(run / "custom.jsonl", results)
    elif where == "metadata":
        write_csv(run / "operation.csv", [{"document_id": "10", "elapsed_time": 1}] * 2)
    elif where == "semantic":
        write_csv(run / "semantic_metrics.csv", [semantic_row(manifest, results[0], results[0]["branches"][0])] * 2)
    with pytest.raises(ValueError, match="duplicado"):
        load_analysis_data([run, run] if where == "runs" else [run])


@pytest.mark.parametrize("field", ["corpus_hash", "id_field", "text_field"])
def test_incompatible_runs(tmp_path, field):
    run, _, _, _ = fixture_run(tmp_path)
    run2, _, manifest, _ = fixture_run(tmp_path, "run-2")
    target = manifest if field == "corpus_hash" else manifest["effective_config"]["corpus"]
    target[field] = "different"
    (run2 / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="compartilhar"):
        load_analysis_data([run, run2])


def test_repo_relative_corpus_and_override_hash(tmp_path, monkeypatch):
    run, corpus, manifest, _ = fixture_run(tmp_path)
    monkeypatch.setattr(analysis_data, "PROJECT_ROOT", tmp_path)
    manifest["effective_config"]["corpus"]["path"] = corpus.name
    (run / "manifest.json").write_text(json.dumps(manifest))
    frame, _ = load_analysis_data([run])
    assert not frame.empty
    override = tmp_path / "copy.jsonl"
    override.write_bytes(corpus.read_bytes())
    _, provenance = load_analysis_data([run], corpus_override=override)
    assert provenance["corpus"]["path"] == str(override)
    override.write_text("[]")
    with pytest.raises(ValueError, match="corpus_hash"):
        load_analysis_data([run], corpus_override=override)


@pytest.mark.parametrize("filename", ["../results.jsonl", "/tmp/result", "..", "sub\\result"])
@pytest.mark.parametrize("field", ["results_filename", "metadata_filename"])
def test_filenames_must_be_basename(tmp_path, filename, field):
    run, _, manifest, _ = fixture_run(tmp_path)
    manifest["effective_config"]["execution"][field] = filename
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="basename"):
        load_analysis_data([run])


@pytest.mark.parametrize("missing", ["manifest.json", "custom.jsonl", "corpus.jsonl"])
def test_required_sources(tmp_path, missing):
    run, corpus, _, _ = fixture_run(tmp_path)
    (corpus if missing == corpus.name else run / missing).unlink()
    with pytest.raises(FileNotFoundError):
        load_analysis_data([run])


@pytest.mark.parametrize("record", [
    {"document_id": "unknown", "status": "success"},
    {"item_index": 0, "status": "success"},
    {"item_index": 3, "status": "partial"},
    {"item_index": True, "status": "success"},
    {"document_id": "10", "run_id": "wrong", "status": "success"},
    {"document_id": "10", "content_hash": "wrong", "status": "partial"},
])
def test_nilc_invalid_identity(tmp_path, record):
    run, _, _, _ = fixture_run(tmp_path)
    write_records(run / "metrics.jsonl", [record])
    with pytest.raises(ValueError):
        load_analysis_data([run])


@pytest.mark.parametrize("field", ["document_id", "run_id", "content_hash", "corpus_hash"])
def test_metadata_invalid_identity(tmp_path, field):
    run, _, _, _ = fixture_run(tmp_path)
    row = {"document_id": "10", "elapsed_time": 1, field: "wrong"}
    write_csv(run / "operation.csv", [row])
    with pytest.raises(ValueError):
        load_analysis_data([run])


def test_all_thirteen_nilc_metrics_only(tmp_path):
    run, _, _, _ = fixture_run(tmp_path)
    metrics: dict[str, int] = dict.fromkeys(NILC_METRICS, 1)
    metrics["unrequested"] = 99
    write_records(run / "metrics.jsonl", [{"document_id": "10", "status": "success", "custom": metrics}])
    frame, _ = load_analysis_data([run])
    assert len(NILC_METRICS) == 13
    assert set(frame.metric) == {"text_length", *NILC_METRICS}


@pytest.mark.parametrize("field", ["source_hash", "candidate_hash", "corpus_hash"])
def test_human_semantic_hashes(tmp_path, field):
    run, corpus, manifest, results = fixture_run(tmp_path)
    row = semantic_row(manifest, results[0])
    row[field] = "wrong"
    write_csv(corpus.parent / "human_bertscore.csv", [row])
    with pytest.raises(ValueError, match=field):
        load_analysis_data([run])


def test_scorer_mismatch_across_runs(tmp_path):
    run, _, manifest, results = fixture_run(tmp_path)
    run2, _, manifest2, results2 = fixture_run(tmp_path, "run-2")
    write_csv(run / "semantic_metrics.csv", [semantic_row(manifest, results[0], results[0]["branches"][0])])
    row = semantic_row(manifest2, results2[0], results2[0]["branches"][0])
    row["scorer_config"] = json.dumps({"model": "different"})
    write_csv(run2 / "semantic_metrics.csv", [row])
    with pytest.raises(ValueError, match="scorer_config incompatível"):
        load_analysis_data([run, run2])


def test_nilc_approved_filter_and_last_valid_record_not_merge(tmp_path):
    run, _, _, _ = fixture_run(tmp_path)
    write_records(run / "metrics.jsonl", [
        {"document_id": "10", "status": "success", "simple_simplification": {"flesch": 1},
         "moderate_simplification": {"flesch": 2}},
        {"document_id": "10", "status": "partial", "simple_simplification": {"flesch": 3},
         "custom": {"brunet": 4}},
        {"document_id": "10", "status": "skipped"},
    ])
    full, _ = load_analysis_data([run])
    assert value(full, "run-1", 10, "simple", "flesch") == 3
    assert not ((full.version == "moderate") & (full.metric == "flesch")).any()
    filtered, _ = load_analysis_data([run], approved_only=True)
    assert "flesch" not in set(filtered.metric)
    assert value(filtered, "run-1", 10, "custom", "brunet") == 4


def test_multiline_scorer_config(tmp_path):
    run, _, manifest, results = fixture_run(tmp_path)
    row = semantic_row(manifest, results[0], results[0]["branches"][0])
    row["scorer_config"] = json.dumps({"model": "fake-bert", "num_layers": 9}, indent=2)
    write_csv(run / "semantic_metrics.csv", [row])
    frame, provenance = load_analysis_data([run])
    assert value(frame, "run-1", 10, "simple", "bertscore_f1") == .8
    assert provenance["scorer_config"]["num_layers"] == 9


def test_requires_runs():
    with pytest.raises(ValueError):
        load_analysis_data([])
