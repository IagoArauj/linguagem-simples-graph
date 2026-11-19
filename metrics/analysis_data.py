"""Leitura local, validada e sem inferência dos dados de análise batch."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Literal, overload

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["run_id", "run_name", "document_id", "version", "metric", "value"]
NILC_METRICS = (
    "hard_conjunctions_ratio", "long_sentence_ratio", "medium_long_sentence_ratio",
    "simple_word_ratio", "lsa_adj_mean", "lsa_adj_std", "lsa_givenness_mean",
    "lsa_givenness_std", "cross_entropy", "brunet", "dalechall_adapted", "flesch",
    "gunning_fox",
)
OPERATION_METRICS = ("elapsed_time", "retry_wait_seconds", "elapsed_without_backoff",
                     "input_tokens", "output_tokens", "llm_calls")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identifier(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"Identificador inválido: {value!r}")
    return str(value).strip()


def _basename(value) -> str:
    if not isinstance(value, str) or value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Nome de arquivo deve ser basename: {value!r}")
    return value


def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _records(text: str, path: Path) -> list[dict]:
    try:
        records = json.loads(text) if text.lstrip().startswith("[") else [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    except ValueError as exc:
        raise ValueError(f"JSON/JSONL inválido: {path}: {exc}") from exc
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Esperada lista de objetos ou JSONL: {path}")
    return records


def load_analysis_data(
    run_dirs: list[Path], corpus_override: Path | None = None,
    corpus_metrics: Path | None = None, approved_only: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Retorna observações LONG e provenance JSON-serializável.

    Identidades conflitantes são erros, não joins parciais. Apenas NILC admite
    registros repetidos (retries): o último success/partial substitui o anterior.
    Fontes opcionais ausentes não impedem o cálculo de comprimentos.
    """
    if not run_dirs:
        raise ValueError("Informe pelo menos uma run.")
    provenance = {"runs": [], "corpus": {}, "source_files": [], "warnings": [], "scorer_config": None}
    sources = {}
    observations = []

    @overload
    def read(path: Path, optional: Literal[False] = False) -> str: ...

    @overload
    def read(path: Path, optional: Literal[True]) -> str | None: ...

    def read(path: Path, optional: bool = False) -> str | None:
        path = path.resolve()
        if optional and not path.exists():
            provenance["warnings"].append(f"Fonte opcional ausente: {path}")
            return None
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if path in sources and sources[path] != digest:
            raise ValueError(f"Fonte mudou durante a leitura: {path}")
        if path not in sources:
            sources[path] = digest
            provenance["source_files"].append({"path": str(path), "sha256": digest})
        return data.decode("utf-8-sig")

    def add(run_id, name, ident, version, metric, value):
        observations.append((run_id, name, ident, version, metric, _number(value)))

    runs = []
    run_ids = set()
    compatibility = None
    corpus = {}
    corpus_path = None
    for directory in run_dirs:
        directory = Path(directory).resolve()
        manifest = json.loads(read(directory / "manifest.json"))
        if manifest.get("manifest_version") != "1.0":
            raise ValueError(f"manifest_version incompatível: {directory}")
        run_id = _identifier(manifest.get("run_id"))
        if run_id == "corpus" or run_id in run_ids:
            raise ValueError(f"run_id duplicado ou reservado: {run_id}")
        run_ids.add(run_id)
        config = manifest["effective_config"]
        settings = config["corpus"]
        identity = (manifest["corpus_hash"], settings["id_field"], settings["text_field"])
        if compatibility is not None and identity != compatibility:
            raise ValueError("Runs devem compartilhar corpus_hash, id_field e text_field.")
        compatibility = identity
        path = Path(corpus_override) if corpus_override is not None else Path(settings["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        corpus_text = read(path)
        if sources[path] != identity[0]:
            raise ValueError(f"corpus_hash divergente do manifesto: {path}")
        if corpus_path is None:
            corpus_path = path
            provenance["corpus"] = {"path": str(path), "hash": identity[0]}
            for row in _records(corpus_text, path):
                ident = _identifier(row.get(settings["id_field"]))
                if ident in corpus:
                    raise ValueError(f"Documento duplicado no corpus: {ident}")
                original = row.get(settings["text_field"])
                if not isinstance(original, str) or not original.strip():
                    raise ValueError(f"Texto original inválido: {ident}")
                texts = {"original": original}
                for version, field in (("natural", "natural_text"), ("strong", "strong_text")):
                    text = row.get(field)
                    if text is not None and not isinstance(text, str):
                        raise ValueError(f"Texto humano inválido: {ident}/{field}")
                    if text and text.strip():
                        texts[version] = text
                corpus[ident] = texts
            if not corpus:
                raise ValueError("Corpus vazio.")
        models = config.get("models", {})
        simplifier = models.get("simplifier", {})
        name = simplifier.get("name") if isinstance(simplifier, dict) else simplifier
        name = name or run_id
        result_path = directory / _basename(config["execution"]["results_filename"])
        results = _records(read(result_path), result_path)
        runs.append({"run_id": run_id, "run_name": name, "directory": str(directory),
                     "models": models, "config": config, "results": results})

    assert compatibility is not None and corpus_path is not None
    corpus_hash, id_field, _ = compatibility
    counts = Counter(run["run_name"] for run in runs)
    for run in runs:
        if counts[run["run_name"]] > 1 or run["run_name"] == "Corpus humano":
            run["run_name"] = f'{run["run_name"]} [{run["run_id"]}]'
    # A model name could itself equal another run's decorated label.
    if len({run["run_name"] for run in runs}) != len(runs):
        for run in runs:
            run["run_name"] = f'{run["run_name"]} [{run["run_id"]}]'

    def validate_identity(row, ident, run_id=None, required=()):
        expected = {"corpus_hash": corpus_hash,
                    "content_hash": _digest(corpus[ident]["original"]),
                    "source_hash": _digest(corpus[ident]["original"])}
        if run_id is not None:
            expected["run_id"] = run_id
        for field, value in expected.items():
            if (field in required or row.get(field) not in (None, "")) and row.get(field) != value:
                raise ValueError(f"{field} divergente: {ident}")

    def nilc(path, run_id, name, texts, order, id_fields):
        data = read(path, optional=True)
        if data is None:
            return
        latest = {}
        for row in _records(data, path):
            ids = [_identifier(row[field]) for field in dict.fromkeys(id_fields)
                   if row.get(field) not in (None, "")]
            if ids:
                ident = ids[0]
                if any(value != ident for value in ids):
                    raise ValueError(f"IDs conflitantes: {path}")
            else:
                index = row.get("item_index")
                if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= len(order):
                    raise ValueError(f"item_index legado inválido: {path}: {index}")
                ident = order[index - 1]
                provenance["warnings"].append(
                    f"Fallback legado sem ID: {path}, item_index={index} -> document_id={ident} (ordem original).")
            if ident not in texts:
                raise ValueError(f"Documento desconhecido em {path}: {ident}")
            validate_identity(row, ident, None if run_id == "corpus" else run_id)
            if row.get("status") not in {"success", "partial"}:
                continue
            if ident in latest:
                provenance["warnings"].append(f"Retry NILC: último registro válido usado em {path}: {ident}")
            latest[ident] = row
        for ident, row in latest.items():
            for stage, metrics in row.items():
                if run_id == "corpus":
                    version = {"original": "original", "simple_simplification": "natural",
                               "aggressive_simplification": "strong"}.get(stage, stage)
                else:
                    version = stage if stage in texts[ident] else stage.removesuffix("_simplification")
                if version not in texts[ident] or not isinstance(metrics, dict):
                    continue
                for metric in NILC_METRICS:
                    if metric in metrics:
                        add(run_id, name, ident, version, metric, metrics[metric])

    def semantic(path, run_id, name, all_texts, selected):
        data = read(path, optional=True)
        if data is None:
            return
        reader = csv.DictReader(io.StringIO(data))
        required = {"metrics_version", "corpus_hash", "document_id", "source_hash",
                    "candidate_hash", "scorer_config", "reference" if run_id == "corpus" else "branch_id"}
        if run_id != "corpus":
            required |= {"run_id", "metric_status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Campos obrigatórios semantic 1.0 ausentes: {path}")
        seen = set()
        for row in reader:
            ident = _identifier(row.get("document_id"))
            version = row["reference" if run_id == "corpus" else "branch_id"]
            if ident not in all_texts or version not in all_texts[ident]:
                raise ValueError(f"Documento/versão desconhecido: {path}: {ident}/{version}")
            if (ident, version) in seen:
                raise ValueError(f"Documento/versão duplicado: {path}: {ident}/{version}")
            seen.add((ident, version))
            if row["metrics_version"] != "1.0":
                raise ValueError(f"metrics_version incompatível: {path}")
            validate_identity(row, ident, None if run_id == "corpus" else run_id,
                              required=("corpus_hash", "source_hash", "run_id"))
            try:
                scorer = json.loads(row["scorer_config"])
            except ValueError as exc:
                raise ValueError(f"scorer_config inválido: {path}") from exc
            if not isinstance(scorer, dict) or not scorer:
                raise ValueError(f"scorer_config inválido: {path}")
            if provenance["scorer_config"] is None:
                provenance["scorer_config"] = scorer
            elif provenance["scorer_config"] != scorer:
                raise ValueError(f"scorer_config incompatível entre fontes: {path}")
            candidate = all_texts[ident][version]
            if candidate and candidate.strip():
                if row["candidate_hash"] != _digest(candidate):
                    raise ValueError(f"candidate_hash divergente: {ident}/{version}")
            elif row["candidate_hash"] not in ("", _digest(candidate or "")):
                raise ValueError(f"candidate_hash divergente: {ident}/{version}")
            if row.get("metric_status", "success") not in {"success", "partial"}:
                continue
            if version not in selected.get(ident, {}):
                continue
            for metric, value in row.items():
                if metric and metric.startswith(("bertscore_", "rouge_l_", "sari_")):
                    add(run_id, name, ident, version, metric, value)

    for ident, texts in corpus.items():
        for version, text in texts.items():
            add("corpus", "Corpus humano", ident, version, "text_length", len(text))
    nilc(Path(corpus_metrics) if corpus_metrics is not None else corpus_path.parent / "metrics.jsonl",
         "corpus", "Corpus humano", corpus, list(corpus),
         (id_field, "production_id", "document_id"))
    semantic(corpus_path.parent / "human_bertscore.csv", "corpus", "Corpus humano", corpus, corpus)

    for run in runs:
        run_id, name = run["run_id"], run["run_name"]
        directory = Path(run["directory"])
        provenance["runs"].append({key: run[key] for key in ("run_id", "run_name", "directory", "models")})
        all_texts, selected = {}, {}
        for row in run["results"]:
            ident = _identifier(row.get("document_id"))
            if ident not in corpus or ident in all_texts:
                raise ValueError(f"Documento desconhecido ou duplicado nos results: {ident}")
            if row.get("schema_version") != "1.0":
                raise ValueError(f"schema_version incompatível: {ident}")
            validate_identity(row, ident, run_id, required=("content_hash", "run_id"))
            if row.get("original_text") != corpus[ident]["original"]:
                raise ValueError(f"original_text divergente: {ident}")
            if not isinstance(row.get("branches"), list):
                raise ValueError(f"branches inválidos: {ident}")
            all_texts[ident], selected[ident] = {}, {}
            for branch in row["branches"]:
                if not isinstance(branch, dict):
                    raise ValueError(f"Ramo inválido: {ident}")
                version = _identifier(branch.get("branch_id"))
                if version == "execution" or version in all_texts[ident]:
                    raise ValueError(f"branch_id duplicado ou reservado: {version}")
                validate_identity(branch, ident, run_id)
                if "document_id" in branch and _identifier(branch["document_id"]) != ident:
                    raise ValueError(f"document_id divergente no ramo: {ident}")
                text = branch.get("final_text")
                if text is not None and not isinstance(text, str):
                    raise ValueError(f"final_text inválido: {ident}/{version}")
                all_texts[ident][version] = text
                if text and text.strip() and (not approved_only or branch.get("quality_verdict") == "approved"):
                    selected[ident][version] = text
                    add(run_id, name, ident, version, "text_length", len(text))
        nilc(directory / "metrics.jsonl", run_id, name, selected, list(all_texts), ("document_id",))
        semantic(directory / "semantic_metrics.csv", run_id, name, all_texts, selected)
        filename = run["config"]["execution"].get("metadata_filename")
        if filename is None:
            provenance["warnings"].append(f"metadata_filename não configurado: {directory}")
            continue
        path = directory / _basename(filename)
        data = read(path, optional=True)
        if data is None:
            continue
        seen = set()
        for row in csv.DictReader(io.StringIO(data)):
            ident = _identifier(row.get("document_id"))
            if ident not in all_texts or ident in seen:
                raise ValueError(f"Documento desconhecido ou duplicado nos metadados: {ident}")
            seen.add(ident)
            validate_identity(row, ident, run_id)
            for metric in OPERATION_METRICS:
                if metric in row:
                    add(run_id, name, ident, "execution", metric, row[metric])

    return pd.DataFrame(observations, columns=COLUMNS), provenance
