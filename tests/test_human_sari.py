import csv
import hashlib
import json
from statistics import fmean

import pytest

from metrics.human_sari import EXPERIMENTS, compute_human_sari
from metrics.text_similarity import sari


def corpus(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path


def item(ident=1):
    return {"production_id": ident, "original_text": "A residência é muito grande e bonita.",
            "natural_text": "A casa é muito grande e bonita.", "strong_text": "A casa é grande."}


def test_four_comparisons_macro_means_and_provenance(tmp_path):
    docs = [item(), {**item(2), "strong_text": "A residência é grande e bonita."}]
    path = corpus(tmp_path, docs)
    before = path.read_bytes()
    md = compute_human_sari(path)
    rows = list(csv.DictReader((tmp_path / "human_sari.csv").open()))
    assert len(rows) == 8
    report = md.read_text()
    for key, label, prediction, reference, _ in EXPERIMENTS:
        expected = [sari(d["original_text"], d[prediction], [d[reference]]) for d in docs]
        assert [float(r["sari"]) for r in rows if r["experiment"] == key] == expected
        assert label in report
        assert f"{fmean(expected):.4f}" in report
    manifest = json.loads((tmp_path / "human_sari_manifest.json").read_text())
    assert manifest["corpus_sha256"] == hashlib.sha256(before).hexdigest()
    assert manifest["documents"] == 2
    assert path.read_bytes() == before


def test_singleton_and_overwrite(tmp_path):
    path = corpus(tmp_path, [item()])
    out = tmp_path / "new"
    md = compute_human_sari(path, output_dir=out)
    assert "| — |" in md.read_text()
    with pytest.raises(FileExistsError):
        compute_human_sari(path, output_dir=out)
    assert compute_human_sari(path, output_dir=out, overwrite=True) == md


@pytest.mark.parametrize("rows", [[], [item(), item()], [{**item(), "production_id": True}],
                                  [{**item(), "natural_text": None}], [{**item(), "strong_text": " "}],
                                  [{**item(), "original_text": 15}]])
def test_invalid_data_not_treated_as_zero(tmp_path, rows):
    with pytest.raises(ValueError):
        compute_human_sari(corpus(tmp_path, rows))
    assert not (tmp_path / "human_sari.md").exists()
