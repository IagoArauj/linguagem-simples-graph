import json
from pathlib import Path

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
                "text": "Original.",
                "simple_simplification": "Simples.",
                "moderate_simplification": "Moderada.",
                "aggressive_simplification": "Forte.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        metrics_module,
        "compute_nilc_metrix",
        lambda *args, **kwargs: {"metric": 1},
    )

    metrics_module.compute_metrics(
        nilc_metrix_folder=tmp_path,
        input_file=input_path,
        output_file=output_path,
        workers=1,
    )

    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["status"] == "success"
    assert record["original"] == {"metric": 1}
