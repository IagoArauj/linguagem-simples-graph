import csv
import hashlib
import json
from copy import deepcopy

import pytest

from metrics.execution_report import write_execution_report


HASH = hashlib.sha256(b"Original").hexdigest()


def branch(bid="custom", status="completed", reason="approval", attempts=None, **extra):
    return {"branch_id": bid, "execution_status": status, "termination_reason": reason,
            "quality_verdict": "approved" if reason == "approval" else "not_evaluated",
            "attempts": [] if attempts is None else attempts, **extra}


def document(ident="private-id", branches=None, **extra):
    return {"run_id": "run1", "document_id": ident, "content_hash": HASH,
            "original_text": "Original", "execution_status": "completed",
            "termination_reason": "approval", "analysis": {"blocked": False},
            "branches": [branch()] if branches is None else branches, **extra}


def make_run(tmp_path, docs, metadata=None, run_id="run1", configured=("custom",), json_list=False):
    directory = tmp_path / run_id
    directory.mkdir()
    manifest = {"manifest_version": "1.0", "run_id": run_id,
                "effective_config": {"execution": {"results_filename": "renamed.data",
                                                    "metadata_filename": "renamed.csv"},
                                     "models": {"simplifier": {"name": "same-model"}},
                                     "workflow": {"branches": [{"branch_id": b} for b in configured],
                                                  "max_revisions": 3}}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "renamed.data").write_text(json.dumps(docs) if json_list else "\n".join(map(json.dumps, docs)))
    if metadata is not None:
        with (directory / "renamed.csv").open("w", newline="") as stream:
            fields = list(dict.fromkeys(k for row in metadata for k in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metadata)
    return directory


def rows(tmp_path, name):
    with (tmp_path / f"execution_{name}.csv").open() as stream:
        return list(csv.DictReader(stream))


def meta(ident="private-id", **extra):
    return {"run_id": "run1", "document_id": ident, "content_hash": HASH, **extra}


def test_resources_preference_missing_and_provenance(tmp_path):
    docs = [document(usage={"llm_calls": 2, "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
            document("second", usage={"input_tokens": None})]
    run = make_run(tmp_path, docs, [meta(input_tokens="99", elapsed_time="10", retry_wait_seconds="2",
                                       elapsed_without_backoff="8"),
                                  meta("second", input_tokens="20", elapsed_time="", retry_wait_seconds="",
                                       elapsed_without_backoff="")], json_list=True)
    provenance = write_execution_report([run], tmp_path)
    data = rows(tmp_path, "resources")
    assert data[0]["input_tokens"] == "10.0"
    assert data[1]["input_tokens"] == "20.0"
    assert data[1]["input_tokens_source"] == "metadata"
    assert data[1]["elapsed_time"] == ""
    summary = {r["resource"]: r for r in rows(tmp_path, "resource_summary")}
    assert summary["input_tokens"]["mean"] == "15.0"
    assert summary["elapsed_time"]["n"] == "1"
    assert summary["elapsed_time"]["missing"] == "1"
    assert summary["elapsed_time"]["coverage"] == "0.5"
    assert summary["elapsed_time"]["complete"] == "0"
    assert len(provenance["source_files"]) == 3
    for source in provenance["source_files"]:
        from pathlib import Path
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert any("Conflito" in w for w in provenance["warnings"])
    assert provenance["runs"][0]["max_revisions"] == 3
    markdown = (tmp_path / "execution_report.md").read_text()
    assert "private-id" not in markdown
    assert "nenhum erro registrado" in markdown
    assert "não causais" in markdown


def test_mixed_statuses_revision_limits_and_attempts(tmp_path):
    attempt = {"attempt_number": 1, "execution_status": "failed", "termination_reason": "technical_error",
               "evaluation_call_id": "call", "error_message": "429 private-person: too many requests"}
    docs = [document(branches=[branch("a", "failed", "technical_error", [attempt]),
                               branch("b", reason="revision_limit", quality_verdict="rejected",
                                      attempts=[{"attempt_number": 1, "execution_status": "completed", "quality_verdict": "rejected"}])],
                     execution_status="partial", termination_reason="technical_error"),
            document("second", branches=[branch("a", "cancelled", "cancellation")], execution_status="failed", termination_reason="budget")]
    run = make_run(tmp_path, docs, configured=("a", "b"))
    write_execution_report([run], tmp_path)
    summary = {r["branch_id"]: r for r in rows(tmp_path, "branch_summary")}
    assert summary["a"]["own_errors"] == "1"
    assert summary["a"]["own_error_rate"] == "0.5"
    assert summary["a"]["cancelled"] == "1"
    assert summary["b"]["revision_limit"] == "1"
    assert summary["b"]["own_errors"] == "0"
    assert summary["b"]["missing"] == "1"
    failures = rows(tmp_path, "failures")
    assert failures[0]["scope"] == "attempt"
    assert failures[0]["stage"] == "evaluator"
    assert failures[0]["category"] == "rate_limit"
    assert failures[1]["scope"] == "branch_final"
    assert failures[1]["event_count"] == "0"
    assert rows(tmp_path, "documents")[0]["execution_status"] == "partial"
    assert rows(tmp_path, "documents")[1]["missing_branches"] == "1"
    markdown = (tmp_path / "execution_report.md").read_text()
    assert "private-person" not in markdown
    assert r"revision\_limit" in markdown


def test_analyzer_propagation_one_event_three_impacts(tmp_path):
    branches = [branch(b, "failed", "technical_error", error_message="Unsupported parameter: top_p") for b in ("one", "two", "three")]
    run = make_run(tmp_path, [document(branches=branches, analysis=None, execution_status="failed",
                                      termination_reason="technical_error")], configured=("one", "two", "three"))
    write_execution_report([run], tmp_path)
    failures = rows(tmp_path, "failures")
    assert sum(int(f["event_count"]) for f in failures) == 1
    assert failures[0]["scope"] == "document"
    assert failures[0]["category"] == "unsupported_parameter"
    assert {f["stage"] for f in failures} == {"analyzer"}
    assert all(b["own_errors"] == "0" and b["propagated_errors"] == "1" for b in rows(tmp_path, "branch_summary"))
    markdown = (tmp_path / "execution_report.md").read_text()
    assert "empate" in markdown
    assert "impacto em 3 branches" in markdown


@pytest.mark.parametrize("message,reason,category", [
    ("429", "technical_error", "rate_limit"),
    ("top_p is not supported", "technical_error", "unsupported_parameter"),
    ("401 Unauthorized", "technical_error", "authentication"),
    ("request timed out", "technical_error", "timeout"),
    ("bad JSON", "invalid_response", "invalid_response"),
    ("opaque", "technical_error", "technical_error"),
    ("opaque", "budget", "unknown"),
])
@pytest.mark.parametrize("call_field,stage", [("simplification_call_id", "simplifier"), (None, "unknown")])
def test_error_categories_and_stage_evidence(tmp_path, message, reason, category, call_field, stage):
    attempt = {"execution_status": "failed", "termination_reason": reason, "error_message": message}
    if call_field:
        attempt[call_field] = "call-id"
    run = make_run(tmp_path, [document(branches=[branch(status="failed", reason=reason, attempts=[attempt])])])
    write_execution_report([run], tmp_path)
    failure = rows(tmp_path, "failures")[0]
    assert failure["category"] == category
    assert failure["stage"] == stage
    assert failure["termination_reason"] == reason
    assert failure["classification"] == "heuristic"
    assert sum(int(r["event_count"]) for r in rows(tmp_path, "failures")) == 1


def test_own_error_ties_and_recovered_attempt(tmp_path):
    bad = {"execution_status": "failed", "termination_reason": "technical_error"}
    docs = [document(branches=[branch("x", attempts=[bad]), branch("y", "failed", "invalid_response")])]
    run = make_run(tmp_path, docs, configured=("x", "y"))
    write_execution_report([run], tmp_path)
    assert "Ranking erros próprios por contagem: empate" in (tmp_path / "execution_report.md").read_text()
    assert {b["own_errors"] for b in rows(tmp_path, "branch_summary")} == {"1"}


@pytest.mark.parametrize("value", ["", None, "NaN", "inf", -1, True, "bad", 1.5])
def test_invalid_counts_never_become_zero(tmp_path, value):
    run = make_run(tmp_path, [document(usage={"input_tokens": value})])
    write_execution_report([run], tmp_path)
    assert rows(tmp_path, "resources")[0]["input_tokens"] == ""
    summary = next(r for r in rows(tmp_path, "resource_summary") if r["resource"] == "input_tokens")
    assert summary["mean"] == summary["known_sum"] == ""
    assert summary["n"] == "0"


def test_incoherent_totals_and_timing(tmp_path):
    run = make_run(tmp_path, [document(usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 9})],
                   [meta(elapsed_time="5", retry_wait_seconds="9", elapsed_without_backoff="1")])
    p = write_execution_report([run], tmp_path)
    r = rows(tmp_path, "resources")[0]
    assert r["total_tokens"] == r["retry_wait_seconds"] == r["elapsed_without_backoff"] == ""
    assert r["elapsed_time"] == "5.0"
    assert len([w for w in p["warnings"] if "incoerente" in w]) == 2


def test_historical_timing_and_zero_are_distinct(tmp_path):
    run = make_run(tmp_path, [document(usage={"input_tokens": 0, "output_tokens": 0})], [meta(elapsed_time="0")])
    write_execution_report([run], tmp_path)
    row = rows(tmp_path, "resources")[0]
    assert row["elapsed_time"] == row["input_tokens"] == "0.0"
    assert row["retry_wait_seconds"] == row["elapsed_without_backoff"] == ""
    assert row["total_tokens"] == "0"
    assert row["total_tokens_source"] == "derived_input_output"


@pytest.mark.parametrize("mutation,match", [
    (lambda d, m: d.update(run_id="other"), "run_id divergente"),
    (lambda d, m: m.update(run_id="other"), "run_id divergente"),
    (lambda d, m: m.update(content_hash="b" * 64), "content_hash divergente"),
    (lambda d, m: m.update(source_hash="b" * 64), "source_hash divergente"),
    (lambda d, m: m.update(document_id="other"), "ausente em results"),
    (lambda d, m: d.update(content_hash="b" * 64), "original_text"),
    (lambda d, m: d["branches"][0].update(run_id="other"), "run_id divergente"),
    (lambda d, m: d["branches"].append(deepcopy(d["branches"][0])), "branch_id duplicado"),
])
def test_identity_mismatches_rejected_before_writes(tmp_path, mutation, match):
    d, m = document(), meta()
    mutation(d, m)
    run = make_run(tmp_path, [d], [m])
    with pytest.raises(ValueError, match=match):
        write_execution_report([run], tmp_path)
    assert not list(tmp_path.glob("execution_*"))


@pytest.mark.parametrize("duplicate", ["run", "document", "metadata", "configuration"])
def test_duplicates(tmp_path, duplicate):
    run = make_run(tmp_path, [document()] * (2 if duplicate == "document" else 1),
                   [meta()] * (2 if duplicate == "metadata" else 1),
                   configured=("custom", "custom") if duplicate == "configuration" else ("custom",))
    with pytest.raises(ValueError, match="duplicado"):
        write_execution_report([run] * (2 if duplicate == "run" else 1), tmp_path)


def test_same_model_disambiguation_no_corpus_and_no_overwrite(tmp_path):
    first = make_run(tmp_path, [document()])
    second = make_run(tmp_path, [document(run_id="run2")], run_id="run2")
    sentinel = tmp_path / "existing.csv"
    sentinel.write_text("keep me")
    p = write_execution_report([first, second], tmp_path)
    assert [r["run_name"] for r in p["runs"]] == ["same-model [run1]", "same-model [run2]"]
    original = (tmp_path / "execution_report.md").read_bytes()
    with pytest.raises(FileExistsError):
        write_execution_report([first], tmp_path)
    assert (tmp_path / "execution_report.md").read_bytes() == original
    assert sentinel.read_text() == "keep me"
    assert all(r["event_count"] == "0" for r in rows(tmp_path, "failures"))


def test_empty_run_and_required_existing_output(tmp_path):
    run = make_run(tmp_path, [])
    with pytest.raises(NotADirectoryError):
        write_execution_report([run], tmp_path / "absent")
    write_execution_report([run], tmp_path)
    assert rows(tmp_path, "documents") == []
    assert all(r["coverage"] == "" for r in rows(tmp_path, "resource_summary"))
    assert rows(tmp_path, "failures") == []


def test_unknown_history_does_not_imply_propagation(tmp_path):
    b = branch(status="failed", reason="technical_error")
    del b["attempts"]
    run = make_run(tmp_path, [document(branches=[b], analysis=None,
                                      execution_status="failed", termination_reason="technical_error")])
    write_execution_report([run], tmp_path)
    assert rows(tmp_path, "documents")[0]["analyzer_propagated"] == "0"
    assert rows(tmp_path, "branches")[0]["attempts"] == ""
    assert rows(tmp_path, "failures")[0]["stage"] == "unknown"


def test_missing_cross_hash_is_explicit(tmp_path):
    m = meta()
    del m["content_hash"]
    run = make_run(tmp_path, [document()], [m])
    p = write_execution_report([run], tmp_path)
    assert any("Hash cruzado indisponível" in w for w in p["warnings"])


def test_finite_mean_when_known_sum_overflows(tmp_path):
    run = make_run(tmp_path, [document("a", elapsed_time=1e308), document("b", elapsed_time=1e308)])
    p = write_execution_report([run], tmp_path)
    summary = next(r for r in rows(tmp_path, "resource_summary") if r["resource"] == "elapsed_time")
    assert float(summary["mean"]) == 1e308
    assert summary["known_sum"] == ""
    assert any("Overflow" in w for w in p["warnings"])


@pytest.mark.parametrize("input_tokens,output_tokens", [(10, 5), (0, 0), (2**53 + 1, 2)])
@pytest.mark.parametrize("source", ["results", "metadata", "mixed"])
def test_exact_total_tokens_derivation(tmp_path, input_tokens, output_tokens, source):
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": None}
    metadata = meta()
    if source == "metadata":
        metadata.update(usage)
        usage = {}
    elif source == "mixed":
        metadata["output_tokens"] = usage.pop("output_tokens")
    run = make_run(tmp_path, [document(usage=usage)], [metadata])
    write_execution_report([run], tmp_path)
    resource = rows(tmp_path, "resources")[0]
    assert int(resource["total_tokens"]) == input_tokens + output_tokens
    assert resource["total_tokens_source"] == "derived_input_output"
    summary = next(r for r in rows(tmp_path, "resource_summary") if r["resource"] == "total_tokens")
    assert summary["n"] == "1"
    assert summary["coverage"] == "1.0"


@pytest.mark.parametrize("usage", [{"input_tokens": 4}, {"output_tokens": 5},
                                  {"input_tokens": 4, "output_tokens": 5, "total_tokens": "bad"}])
def test_total_not_derived_from_incomplete_or_invalid_data(tmp_path, usage):
    run = make_run(tmp_path, [document(usage=usage)])
    write_execution_report([run], tmp_path)
    assert rows(tmp_path, "resources")[0]["total_tokens"] == ""


def change_manifest(run, **updates):
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(updates)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("version", ["2.0", None, 1])
def test_incompatible_manifest_version_rejected_before_writes(tmp_path, version):
    run = make_run(tmp_path, [document()])
    change_manifest(run, manifest_version=version)
    with pytest.raises(ValueError, match="manifest_version incompatível"):
        write_execution_report([run], tmp_path)
    assert not list(tmp_path.glob("execution_*"))


def test_absent_manifest_version_warns(tmp_path):
    run = make_run(tmp_path, [document()])
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["manifest_version"]
    path.write_text(json.dumps(manifest))
    p = write_execution_report([run], tmp_path)
    assert any("manifest_version ausente" in w for w in p["warnings"])


def test_comparison_nonpaired_and_aggregate_csvs(tmp_path):
    first = make_run(tmp_path, [
        document("one", execution_status="partial", termination_reason="technical_error",
                 usage={"input_tokens": 10, "output_tokens": 5},
                 branches=[branch(status="failed", reason="technical_error", error_message="429")]),
        document("two", branches=[branch(reason="revision_limit")])])
    second = make_run(tmp_path, [document("different-id", run_id="run2", execution_status="failed",
                                          termination_reason="budget", branches=[branch(reason="budget")])], run_id="run2")
    change_manifest(first, corpus_hash="a" * 64)
    change_manifest(second, corpus_hash="b" * 64)
    p = write_execution_report([first, second], tmp_path)
    assert any("corpus_hash diferentes" in w for w in p["warnings"])
    assert len(p["source_files"]) == 4  # Nenhum corpus foi aberto.
    md = (tmp_path / "execution_report.md").read_text()
    joint = md.split("## Comparação conjunta entre runs (não pareada)")[1]
    assert "| same-model | run1 |" in joint
    assert "| same-model | run2 |" in joint
    assert r"| input\_tokens | 10 | 1 | 2 | 1 | 0.5 |" in joint
    assert "| same-model | run1 | 2 | 1 | 0 | 2 | 1 |" in joint
    assert "| same-model | run2 | 1 | 0 | 1 | 1 | 1 |" in joint
    assert "Cobertura histórico" in joint
    assert "corpus_hash diferentes" in joint
    assert "sem alinhamento de IDs" in md
    doc_counts = rows(tmp_path, "document_summary")
    assert sum(int(r["n"]) for r in doc_counts) == 3
    assert next(r for r in doc_counts if r["run_id"] == "run2")["termination_reason"] == "budget"
    failure_counts = rows(tmp_path, "failure_summary")
    assert sum(int(r["events"]) for r in failure_counts) == 2
    assert next(r for r in failure_counts if r["run_id"] == "run1")["category"] == "rate_limit"


def test_final_branch_reasons_and_coverage(tmp_path):
    docs = [document(str(index), branches=[branch(reason=reason)]) for index, reason in enumerate(
        ("content_policy", "budget", "cancellation", "revision_limit", "approval", None))]
    run = make_run(tmp_path, docs)
    write_execution_report([run], tmp_path)
    reasons = {r["termination_reason"]: r for r in rows(tmp_path, "branch_reasons")}
    for reason in ("content_policy", "budget", "cancellation", "revision_limit", "approval", "unknown"):
        assert reasons[reason]["n"] == "1"
        assert reasons[reason]["observed"] == "6"
        assert reasons[reason]["known_reasons"] == "5"
        assert float(reasons[reason]["coverage"]) == pytest.approx(5 / 6)
    md = (tmp_path / "execution_report.md").read_text()
    assert "### Motivos finais por branch" in md
    assert r"content\_policy" in md
    assert "budget" in md


@pytest.mark.parametrize("empty", [False, True])
def test_no_errors_claim_with_missing_history_or_empty_run(tmp_path, empty):
    b = branch()
    del b["attempts"]
    run = make_run(tmp_path, [] if empty else [document(branches=[b])])
    write_execution_report([run], tmp_path)
    md = (tmp_path / "execution_report.md").read_text()
    assert "sem erros" not in md
    assert "nenhum erro registrado" in md
    assert "Histórico incompleto" in md
    assert "0/0 branches observadas" in md if empty else "0/1 branches observadas" in md


def test_heading_sanitized_and_operational_definitions(tmp_path):
    run = make_run(tmp_path, [document()])
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["effective_config"]["models"]["simplifier"]["name"] = "model|\n# injected <script>[link](url)"
    path.write_text(json.dumps(manifest))
    write_execution_report([run], tmp_path)
    md = (tmp_path / "execution_report.md").read_text()
    assert "\n# injected" not in md
    assert "<script>" not in md
    assert "[link](url)" not in md
    assert r"## model\| \# injected &lt;script&gt;\[link\]\(url\)" in md
    introduction = md.split("## ")[0]
    assert "invocações lógicas" in introduction
    assert "retries técnicos não estão necessariamente contados" in introduction
    assert "Tempos em segundos" in introduction
    assert "somente sem outra chamada ativa" in introduction


def test_new_aggregate_files_are_not_overwritten(tmp_path):
    run = make_run(tmp_path, [document()])
    existing = tmp_path / "execution_document_summary.csv"
    existing.write_text("existing artifact")
    with pytest.raises(FileExistsError):
        write_execution_report([run], tmp_path)
    assert existing.read_text() == "existing artifact"
    assert not (tmp_path / "execution_report.md").exists()
