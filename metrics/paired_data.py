"""Carrega artefatos single e multi para comparação pareada, sem recalcular métricas."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path

LEVELS = ("simple", "moderate", "aggressive")
NILC_METRICS = ("flesch", "gunning_fox", "dalechall_adapted", "simple_word_ratio", "long_sentence_ratio")
SEMANTIC_METRICS = ("bertscore_f1", "sari_multi_reference")
METRICS = NILC_METRICS + SEMANTIC_METRICS
_USABLE = {"success", "partial"}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _id(value, where) -> str:
    if type(value) not in (str, int) or not str(value).strip():
        raise ValueError(f"ID inválido em {where}: {value!r}")
    return str(value)


def _check(row, field, expected, where, *, required=True):
    if not required and row.get(field) in (None, ""):
        return
    if field not in row or row[field] != expected:
        raise ValueError(f"{field} divergente em {where}: esperado {expected!r}")


def _unique(mapping, key, where):
    if key in mapping:
        raise ValueError(f"Identidade duplicada em {where}: {key!r}")


def _numbers(target, key, row, metrics):
    for metric in metrics:
        value = row.get(metric)
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            target[(*key, metric)] = number


class _Sources:
    def __init__(self):
        self.hashes = {}
        self.warnings = []
        self.scorer = None

    def read(self, path, *, optional=False):
        path = path.resolve()
        if optional and not path.exists():
            self.warnings.append(f"Fonte opcional ausente: {path}")
            return None
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if self.hashes.setdefault(path, digest) != digest:
            raise ValueError(f"Fonte mudou durante a leitura: {path}")
        return data.decode("utf-8-sig")

    def records(self, path, *, optional=False):
        data = self.read(path, optional=optional)
        if data is None:
            return []
        try:
            rows = json.loads(data) if data.lstrip().startswith("[") else [
                json.loads(line) for line in data.splitlines() if line.strip()
            ]
        except ValueError as exc:
            raise ValueError(f"JSON/JSONL inválido em {path}: {exc}") from exc
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"Esperados registros JSON em {path}")
        return rows

    def semantic(self, path, key_field, corpus_hash):
        data = self.read(path, optional=True)
        if data is None:
            return
        reader = csv.DictReader(io.StringIO(data))
        fields = reader.fieldnames or []
        required = {key_field, "metrics_version", "corpus_hash", "source_hash",
                    "candidate_hash", "scorer_config", "metric_status", *SEMANTIC_METRICS}
        if key_field == "branch_id":
            required |= {"document_id", "run_id"}
        if not required.issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError(f"Colunas semânticas inválidas em {path}: {sorted(required - set(fields))}")
        for row in reader:
            where = f"{path}:{reader.line_num}"
            if None in row or None in row.values():
                raise ValueError(f"Linha CSV incompleta ou excedente em {where}")
            _check(row, "metrics_version", "1.0", where)
            _check(row, "corpus_hash", corpus_hash, where)
            try:
                scorer = json.loads(row["scorer_config"])
            except ValueError as exc:
                raise ValueError(f"scorer_config inválido em {where}") from exc
            if not isinstance(scorer, dict) or not scorer:
                raise ValueError(f"scorer_config inválido em {where}")
            if self.scorer is None:
                self.scorer = scorer
            elif self.scorer != scorer:
                raise ValueError(f"scorer_config incompatível em {where}")
            yield row, where


def _hashes(row, source_hash, corpus_hash, where):
    for field, expected in (("source_hash", source_hash), ("content_hash", source_hash),
                            ("corpus_hash", corpus_hash)):
        _check(row, field, expected, where, required=False)


def _candidate_hash(row, text, where, *, required=False):
    if not _text(text) and row.get("candidate_hash") in (None, "", _digest(text or "")):
        return
    _check(row, "candidate_hash", _digest(text or ""), where, required=required)


def _single(sources, path, documents, corpus_hash, single):
    rows = sources.records(path)
    if not rows or not _text(rows[0].get("model")):
        raise ValueError(f"Modelo single ausente em {path}")
    model = rows[0]["model"]
    results = {}
    eligible = set()
    for row in rows:
        ident = _id(row.get("production_id"), path)
        where = f"{path}: {ident}"
        _unique(results, ident, where)
        if ident not in documents:
            raise ValueError(f"Documento desconhecido em {where}")
        original, source_hash = documents[ident]
        _check(row, "model", model, where)
        _check(row, "original_text", original, where)
        _hashes(row, source_hash, corpus_hash, where)
        text = row.get("simplified_text")
        if text is not None and not isinstance(text, str):
            raise ValueError(f"simplified_text inválido em {where}")
        _candidate_hash(row, text, where)
        results[ident] = row
        if row.get("status") == "ok" and row.get("finish_reason") == "stop" and _text(text):
            eligible.add(ident)

    def validate(row, where, *, semantic=False):
        ident = _id(row.get("production_id"), where)
        if ident not in results:
            raise ValueError(f"Documento desconhecido em {where}: {ident}")
        original, source_hash = documents[ident]
        result = results[ident]
        _check(row, "model", model, where)
        _check(row, "original_text", original, where)
        if (row.get("simplified_text") or "") != (result.get("simplified_text") or ""):
            raise ValueError(f"simplified_text divergente em {where}")
        if "document_id" in row and row["document_id"] not in (None, ""):
            if _id(row["document_id"], where) != ident:
                raise ValueError(f"document_id divergente em {where}")
        _hashes(row, source_hash, corpus_hash, where)
        if semantic:
            _check(row, "source_hash", source_hash, where)
        _candidate_hash(row, result.get("simplified_text"), where, required=semantic)
        return ident

    nilc_path = path.with_name(f"{path.stem}_metrics.jsonl")
    latest = {}
    for row in sources.records(nilc_path, optional=True):
        if row.get("status") not in _USABLE and row.get("production_id") is None:
            continue  # Erro do NILC pode conter apenas item_index e a mensagem.
        ident = validate(row, nilc_path)
        if row.get("status") in _USABLE:
            if ident in latest:
                sources.warnings.append(f"Retry NILC: último registro válido em {nilc_path}: {ident}")
            latest[ident] = row
    for ident, row in latest.items():
        if ident in eligible:
            stage = row.get("simplified_text_metrics")
            if isinstance(stage, dict):
                _numbers(single, (model, ident), stage, NILC_METRICS)

    seen = set()
    for row, where in sources.semantic(path.with_name(f"{path.stem}_semantic_metrics.csv"),
                                       "production_id", corpus_hash):
        ident = validate(row, where, semantic=True)
        _unique(seen, ident, where)
        seen.add(ident)
        if ident in eligible and row["metric_status"] in _USABLE:
            _numbers(single, (model, ident), row, SEMANTIC_METRICS)
    return model


def _multi(sources, directory, documents, corpus_hash, multi, approved, run_ids):
    path = directory / "manifest.json"
    manifest = json.loads(sources.read(path))
    _check(manifest, "manifest_version", "1.0", path)
    _check(manifest, "corpus_hash", corpus_hash, path)
    run_id = _id(manifest.get("run_id"), path)
    _unique(run_ids, run_id, path)
    run_ids.add(run_id)
    try:
        config = manifest["effective_config"]
        model = config["models"]["simplifier"]["name"]
        filename = config["execution"]["results_filename"]
        settings = config["corpus"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"effective_config inválido em {path}") from exc
    _check(settings, "id_field", "production_id", path)
    _check(settings, "text_field", "original_text", path)
    if not _text(model) or not _text(filename) or filename in {".", ".."} or any(c in filename for c in "/\\"):
        raise ValueError(f"Modelo ou results_filename inválido em {path}")

    def identity(row, ident, where):
        _check(row, "run_id", run_id, where)
        _check(row, "content_hash", documents[ident][1], where)
        _check(row, "model", model, where, required=False)
        _hashes(row, documents[ident][1], corpus_hash, where)

    path = directory / filename
    results = {}
    eligible = set()
    for row in sources.records(path):
        ident = _id(row.get("document_id"), path)
        where = f"{path}: {ident}"
        _unique(results, ident, where)
        if ident not in documents:
            raise ValueError(f"Documento desconhecido em {where}")
        identity(row, ident, where)
        _check(row, "schema_version", "1.0", where)
        _check(row, "original_text", documents[ident][0], where)
        branches = row.get("branches")
        if not isinstance(branches, list) or any(not isinstance(branch, dict) for branch in branches):
            raise ValueError(f"branches inválidos em {where}")
        results[ident] = by_level = {}
        for branch in branches:
            level = branch.get("branch_id")
            if level not in LEVELS:
                raise ValueError(f"branch_id inválido em {where}: {level!r}")
            _unique(by_level, level, where)
            identity(branch, ident, where)
            if _id(branch.get("document_id"), where) != ident:
                raise ValueError(f"document_id divergente no ramo em {where}")
            text = branch.get("final_text")
            if text is not None and not isinstance(text, str):
                raise ValueError(f"final_text inválido em {where}/{level}")
            _candidate_hash(branch, text, where)
            by_level[level] = branch
            reason = branch.get("termination_reason")
            if branch.get("execution_status") == "completed" and reason in {"approval", "revision_limit"} and _text(text):
                eligible.add((ident, level))
                if reason == "approval" and branch.get("quality_verdict") == "approved":
                    approved.add((model, level, ident))

    nilc_path = directory / "metrics.jsonl"
    latest = {}
    for row in sources.records(nilc_path, optional=True):
        if row.get("status") not in _USABLE and row.get("document_id") is None:
            continue
        ident = _id(row.get("document_id"), nilc_path)
        if ident not in results:
            raise ValueError(f"Documento desconhecido em {nilc_path}: {ident}")
        identity(row, ident, nilc_path)
        if row.get("status") in _USABLE:
            if ident in latest:
                sources.warnings.append(f"Retry NILC: último registro válido em {nilc_path}: {ident}")
            latest[ident] = row
    for ident, row in latest.items():
        for level in LEVELS:
            stage = row.get(f"{level}_simplification")
            if (ident, level) in eligible and isinstance(stage, dict):
                _numbers(multi, (model, level, ident), stage, NILC_METRICS)

    seen = set()
    for row, where in sources.semantic(directory / "semantic_metrics.csv", "branch_id", corpus_hash):
        ident, level = _id(row["document_id"], where), row["branch_id"]
        if ident not in results or level not in results[ident]:
            raise ValueError(f"Documento/ramo desconhecido em {where}: {ident}/{level}")
        _unique(seen, (ident, level), where)
        seen.add((ident, level))
        branch = results[ident][level]
        _check(row, "run_id", run_id, where)
        _check(row, "model", model, where, required=False)
        _check(row, "source_hash", documents[ident][1], where)
        _hashes(row, documents[ident][1], corpus_hash, where)
        _candidate_hash(row, branch.get("final_text"), where, required=True)
        for field in ("execution_status", "termination_reason", "quality_verdict"):
            if field in row:
                _check(row, field, branch.get(field) or "", where)
        if (ident, level) in eligible and row["metric_status"] in _USABLE:
            _numbers(multi, (model, level, ident), row, SEMANTIC_METRICS)
    return model, run_id


def load_paired_data(
    single_results: list[Path], multi_runs: list[Path], corpus: Path,
) -> tuple[
    list[str], list[str], dict[tuple[str, str, str], float],
    dict[tuple[str, str, str, str], float], set[tuple[str, str, str]], dict,
]:
    """Retorna modelos ordenados, IDs na ordem do corpus e métricas esparsas.

    Deve haver exatamente uma fonte de cada método por modelo. A unidade de
    elegibilidade multi é o ramo, mesmo quando o documento terminou parcial.
    Métricas ausentes/não finitas não são imputadas nem removem outras métricas.
    """
    if not single_results or not multi_runs:
        raise ValueError("Informe pelo menos um resultado single e uma run multi.")
    sources = _Sources()
    corpus = Path(corpus).resolve()
    documents = {}
    for row in sources.records(corpus):
        ident = _id(row.get("production_id"), corpus)
        _unique(documents, ident, corpus)
        original = row.get("original_text")
        if not _text(original):
            raise ValueError(f"original_text inválido em {corpus}: {ident}")
        documents[ident] = (original, _digest(original))
    if not documents:
        raise ValueError(f"Corpus vazio: {corpus}")
    corpus_hash = sources.hashes[corpus]
    single, multi, approved = {}, {}, set()
    singles, runs, run_ids = {}, {}, set()
    for path in map(Path, single_results):
        path = path.resolve()
        model = _single(sources, path, documents, corpus_hash, single)
        _unique(singles, model, path)
        singles[model] = str(path)
    for directory in map(Path, multi_runs):
        directory = directory.resolve()
        model, run_id = _multi(sources, directory, documents, corpus_hash, multi, approved, run_ids)
        _unique(runs, model, directory)
        runs[model] = {"directory": str(directory), "run_id": run_id}
    if singles.keys() != runs.keys():
        raise ValueError(f"Modelos sem par: só single={sorted(singles.keys() - runs.keys())}; "
                         f"só multi={sorted(runs.keys() - singles.keys())}")
    provenance = {
        "corpus": {"path": str(corpus), "sha256": corpus_hash},
        "single_results": singles, "multi_runs": runs,
        "source_files": [{"path": str(path), "sha256": digest} for path, digest in sources.hashes.items()],
        "scorer_config": sources.scorer, "warnings": sources.warnings,
    }
    return sorted(singles), list(documents), single, multi, approved, provenance
