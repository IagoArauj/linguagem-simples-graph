import csv
import hashlib
import json
import sys

import pytest

from metrics import analyze_execution as module


@pytest.fixture
def run(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"id": "1", "text": "Original"}) + "\n")
    manifest = {
        "manifest_version": "1.0", "run_id": "run1",
        "corpus_hash": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "effective_config": {
            "corpus": {"path": str(corpus), "id_field": "id", "text_field": "text"},
            "execution": {"results_filename": "results.jsonl", "metadata_filename": "metadata.csv"},
            "models": {"simplifier": {"name": "modelo"}},
            "workflow": {"branches": [{"branch_id": "simple"}], "max_revisions": 3},
        },
    }
    document = {
        "schema_version": "1.0", "run_id": "run1", "document_id": "1",
        "content_hash": hashlib.sha256(b"Original").hexdigest(), "original_text": "Original",
        "execution_status": "failed", "termination_reason": "invalid_response", "analysis": None,
        "usage": {"llm_calls": 1, "input_tokens": None, "output_tokens": None},
        "branches": [{"branch_id": "simple", "execution_status": "failed",
                      "termination_reason": "invalid_response", "quality_verdict": "not_evaluated",
                      "attempts": [], "final_text": None, "error_message": "top_p is not supported"}],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "results.jsonl").write_text(json.dumps(document) + "\n")
    return folder


def test_standalone_without_corpus_or_metrics(tmp_path, monkeypatch, run):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "corpus.jsonl").unlink()
    original = {p.name: p.read_bytes() for p in run.iterdir()}
    output = module.create_execution_analysis([run])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["analysis_type"] == "execution"
    assert manifest["created_at"].endswith("+00:00")
    assert manifest["runs"][0]["directory"] == str(run)
    assert "execution_report.md" in manifest["artifacts"]
    assert manifest["analysis_code"]["execution_report.py"]
    assert {p.name: p.read_bytes() for p in run.iterdir()} == original
    assert not (output / "means.md").exists()
    with (output / "execution_resource_summary.csv").open() as stream:
        summaries = {r["resource"]: r for r in csv.DictReader(stream)}
    assert summaries["input_tokens"]["n"] == "0"
    assert summaries["llm_calls"]["mean"] == "1.0"
    with pytest.raises(FileExistsError):
        module.create_execution_analysis([run], output_dir=output)


def test_reuse_previous_analysis_manifest_and_track_changed_source(tmp_path, monkeypatch, run):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    previous = tmp_path / "old-analysis"
    previous.mkdir()
    parent = previous / "manifest.json"
    parent.write_text(json.dumps({"runs": [{"directory": str(run)}], "source_files": [
        {"path": str(run / "results.jsonl"), "sha256": "old-version"}]}))
    original = parent.read_bytes()
    output = module.create_execution_analysis(analysis_dir=previous)
    manifest = json.loads((output / "manifest.json").read_text())
    assert parent.read_bytes() == original
    assert manifest["parent_analysis"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["parent_analysis"] in manifest["source_files"]
    assert any("Fonte alterada" in w for w in manifest["warnings"])
    assert output != previous


def test_validation_and_failed_manifest(tmp_path, monkeypatch, run):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="exclusivamente"):
        module.create_execution_analysis()
    with pytest.raises(ValueError, match="exclusivamente"):
        module.create_execution_analysis([run], analysis_dir=run)
    with pytest.raises(ValueError, match="output/"):
        module.create_execution_analysis([run], output_dir=tmp_path / "outside")
    (run / "results.jsonl").write_text("invalid json")
    output = tmp_path / "output" / "failed-report"
    with pytest.raises(ValueError):
        module.create_execution_analysis([run], output_dir=output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_cli(tmp_path, monkeypatch, run, capsys):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "output" / "cli"
    monkeypatch.setattr(sys, "argv", ["analyze_execution", "--run-dir", str(run), "--output-dir", str(output)])
    module.main()
    assert "Análise operacional salva" in capsys.readouterr().out
    assert (output / "execution_report.md").exists()


def test_full_analysis_includes_failed_branches_even_with_approved_only(tmp_path, monkeypatch, run):
    from metrics import analyze
    monkeypatch.setattr(analyze, "PROJECT_ROOT", tmp_path)
    output = analyze.create_analysis([run], approved_only=True, plots=False)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["options"]["approved_only"] is True
    assert manifest["options"]["execution_all_outcomes"] is True
    assert len(manifest["source_files"]) == len({s["path"] for s in manifest["source_files"]})
    with (output / "execution_branch_summary.csv").open() as stream:
        branch = next(csv.DictReader(stream))
    assert branch["failed"] == "1"
    assert branch["propagated_errors"] == "1"
    assert (output / "means.md").exists()
