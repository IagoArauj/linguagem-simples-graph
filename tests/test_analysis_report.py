import json
from pathlib import Path

import pandas as pd
import pytest

import metrics.analyze as module


def values():
    rows = []
    for run, version, vals in [
        ("corpus", "original", [("1", 10), ("2", 100)]),
        ("corpus", "natural", [("1", 12), ("2", 101)]),
        ("run1", "simple", [("2", 105), ("1", float("nan"))]),
        ("run2", "simple", [("1", 15)]),
    ]:
        for ident, value in vals:
            rows.append(dict(run_id=run, run_name=run, document_id=ident,
                             version=version, metric="flesch", value=value))
    return pd.DataFrame(rows)


def test_means_and_deltas_use_available_paired_observations():
    summary, deltas, consistency = module.summarize(values())
    first = summary[summary.run_id == "run1"].iloc[0]
    assert first["n"] == 1
    assert first["mean"] == 105
    first_delta = deltas[deltas.run_id == "run1"].iloc[0]
    assert first_delta.original_mean == 100
    assert first_delta.delta_mean == 5
    assert first_delta.n == 1
    assert set(consistency.percentage) == {100}
    markdown = module.markdown_means(summary)
    assert "105.0000" in markdown
    assert "run1" in markdown and "run2" in markdown
    assert "mesmos documentos" in markdown


def test_report_manifest_and_no_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    provenance = {"runs": [{"run_id": "run1", "run_name": "modelo", "directory": "/runs/one", "models": {}}],
                  "corpus": {"path": "/input/corpus.jsonl", "hash": "abc"},
                  "source_files": [{"path": "/runs/one/results.jsonl", "sha256": "abc"}],
                  "warnings": [], "scorer_config": None}
    monkeypatch.setattr(module, "load_analysis_data", lambda *a: (values(), provenance))
    monkeypatch.setattr(module, "generate_plots", lambda *a: pytest.fail("plots disabled"))

    def operational(run_dirs, output):
        assert run_dirs == [Path("/runs/one")]
        (output / "execution_report.md").write_text("Relatório operacional")
        return {"runs": provenance["runs"], "source_files": provenance["source_files"].copy(), "warnings": []}

    monkeypatch.setattr(module, "write_execution_report", operational)
    output = module.create_analysis([Path("/runs/one")], plots=False)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["created_at"].endswith("+00:00")
    assert manifest["status"] == "completed"
    assert manifest["runs"][0]["directory"] == "/runs/one"
    assert manifest["source_files"][0]["sha256"] == "abc"
    assert (output / "execution_report.md").exists()
    assert manifest["options"]["execution_all_outcomes"] is True
    assert "execution_report.py" in manifest["analysis_code"]
    assert "execution_report.md" in manifest["artifacts"]
    assert (output / "means.md").exists()
    assert (output / "friedman.csv").exists()
    assert not (output / "combined").exists()
    with pytest.raises(FileExistsError):
        module.create_analysis([Path("/runs/one")], output_dir=output, plots=False)


def test_report_refuses_output_outside_project_output(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "load_analysis_data", lambda *a: (values(), {}))
    with pytest.raises(ValueError, match="output/"):
        module.create_analysis([], output_dir=tmp_path / "outside", plots=False)
