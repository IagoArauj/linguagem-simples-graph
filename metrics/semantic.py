"""Métricas locais de experimentos, com cache persistente de BERTScore.

Execute: python -m metrics.semantic --help
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Callable

from tqdm import tqdm

from metrics.text_similarity import rouge_l, sari

VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    records = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Esperada lista de objetos ou JSONL: {path}")
    return records


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    # Publica somente CSVs completos; uma interrupção preserva a versão anterior.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


class BertCache:
    """Cache por fonte, candidata e identidade do scorer; persiste após cada par."""

    def __init__(self, path: Path, settings: dict, scorer_factory: Callable):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, p REAL, r REAL, f REAL)")
        self.settings = json.dumps(settings, sort_keys=True)
        self.factory = scorer_factory
        self.scorer = None

    def score(self, source: str, candidate: str) -> dict:
        key = digest(json.dumps([self.settings, source, candidate], ensure_ascii=False))
        cached = self.connection.execute("SELECT p, r, f FROM scores WHERE key = ?", (key,)).fetchone()
        if cached is None:
            if self.scorer is None:
                self.scorer = self.factory()
            precision, recall, f1 = self.scorer.score([candidate], [source])
            cached = tuple(float(values[0]) for values in (precision, recall, f1))
            self.connection.execute("INSERT INTO scores VALUES (?, ?, ?, ?)", (key, *cached))
            self.connection.commit()
        return dict(zip(("bertscore_precision", "bertscore_recall", "bertscore_f1"), cached))

    def close(self) -> None:
        self.connection.close()


def text_field(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Campo {field!r} deve conter texto não vazio.")
    return value


def document_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("Identificador de documento inválido.")
    return str(value).strip()


BERT_FIELDS = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
COMMON_FIELDS = ["metrics_version", "corpus_hash", "document_id", "source_hash", "candidate_hash", "scorer_config"]
HUMAN_FIELDS = COMMON_FIELDS + ["reference"] + BERT_FIELDS
EXPERIMENT_FIELDS = COMMON_FIELDS + ["run_id", "branch_id", "target_audience", "intensity", "execution_status", "quality_verdict", "termination_reason", "metric_status"] + BERT_FIELDS + [
    f"rouge_l_{reference}_{metric}" for reference in ("natural", "strong") for metric in ("precision", "recall", "f1")
] + ["sari_natural", "sari_strong", "sari_multi_reference"]


def evaluate_run(run_dir: Path, *, settings: dict, scorer_factory: Callable,
                 corpus_override: Path | None = None, natural_field: str = "natural_text",
                 strong_field: str = "strong_text") -> Path:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "1.0":
        raise ValueError("Versão de manifesto não suportada.")
    effective = manifest["effective_config"]
    corpus_config = effective["corpus"]
    corpus_path = corpus_override or Path(corpus_config["path"])
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path
    corpus_path = corpus_path.resolve()
    corpus_hash = file_digest(corpus_path)
    if corpus_hash != manifest["corpus_hash"]:
        raise ValueError("Hash do corpus difere do manifesto; comparação recusada.")
    filename = effective["execution"]["results_filename"]
    if Path(filename).name != filename:
        raise ValueError("results_filename deve ser um nome de arquivo.")
    results = load_records(run_dir / filename)
    corpus = {}
    for row in load_records(corpus_path):
        ident = document_id(row.get(corpus_config["id_field"]))
        if ident in corpus:
            raise ValueError(f"Identificador duplicado no corpus: {ident}")
        corpus[ident] = (
            text_field(row, corpus_config["text_field"]),
            text_field(row, natural_field), text_field(row, strong_field),
        )
    # Valida todas as associações antes de carregar os pesos ou calcular métricas.
    seen = set()
    for row in results:
        ident = document_id(row.get("document_id"))
        if ident in seen or ident not in corpus:
            raise ValueError(f"Documento duplicado ou ausente do corpus: {ident}")
        seen.add(ident)
        if row.get("schema_version") != "1.0" or row.get("run_id") != manifest["run_id"]:
            raise ValueError(f"Schema/run_id incompatível no documento {ident}")
        source = corpus[ident][0]
        if row.get("original_text") != source or row.get("content_hash") != digest(source):
            raise ValueError(f"Texto/hash divergente no documento {ident}")
        if not isinstance(row.get("branches"), list):
            raise ValueError(f"Ramos inválidos no documento {ident}")
        branch_ids = set()
        for branch in row["branches"]:
            if not isinstance(branch, dict):
                raise ValueError("Ramo deve ser um objeto.")
            branch_id = text_field(branch, "branch_id")
            if branch_id in branch_ids:
                raise ValueError(f"Ramo duplicado: {branch_id}")
            branch_ids.add(branch_id)
            if branch.get("final_text") is not None and not isinstance(branch["final_text"], str):
                raise ValueError("final_text deve ser string ou null.")

    cache = BertCache(corpus_path.parent / "bertscore_cache.sqlite3", settings, scorer_factory)
    scorer_config = json.dumps(settings, sort_keys=True)

    def common(ident: str, source: str, candidate: str) -> dict:
        return dict(metrics_version=VERSION, corpus_hash=corpus_hash, document_id=ident,
                    source_hash=digest(source), candidate_hash=digest(candidate), scorer_config=scorer_config)

    try:
        humans = []
        with tqdm(total=2 * len(corpus), desc="BERTScore humano", unit="texto", dynamic_ncols=True) as progress:
            for ident, (source, natural, strong) in corpus.items():
                for reference, candidate in (("natural", natural), ("strong", strong)):
                    progress.set_postfix(documento=ident, referencia=reference)
                    humans.append({**common(ident, source, candidate), "reference": reference,
                                   **cache.score(source, candidate)})
                    progress.update(1)
        write_csv(corpus_path.parent / "human_bertscore.csv", humans, HUMAN_FIELDS)
        rows = []
        branches = ((result, branch) for result in results for branch in result["branches"])
        with tqdm(total=sum(len(result["branches"]) for result in results), desc="Métricas IA", unit="ramo", dynamic_ncols=True) as progress:
            for result, branch in branches:
                ident = str(result["document_id"])
                source, natural, strong = corpus[ident]
                progress.set_postfix(documento=ident, ramo=branch["branch_id"])
                candidate = branch.get("final_text")
                row = {**common(ident, source, candidate or ""), "run_id": manifest["run_id"],
                       **{key: branch.get(key) for key in ("branch_id", "target_audience", "intensity", "execution_status", "quality_verdict", "termination_reason")}}
                if not candidate or not candidate.strip():
                    row.update(metric_status="skipped_no_text", candidate_hash=None)
                else:
                    row.update(metric_status="success", **cache.score(source, candidate))
                    for label, reference in (("natural", natural), ("strong", strong)):
                        row.update({f"rouge_l_{label}_{key}": value for key, value in rouge_l(candidate, reference).items()})
                        row[f"sari_{label}"] = sari(source, candidate, [reference])
                    row["sari_multi_reference"] = sari(source, candidate, [natural, strong])
                rows.append(row)
                progress.update(1)
        output = run_dir / "semantic_metrics.csv"
        write_csv(output, rows, EXPERIMENT_FIELDS)
        return output
    finally:
        cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BERTScore PT (XLM-R), ROUGE-L e SARI sem API de geração.")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True, help="Pastas dos experimentos com manifest.json.")
    parser.add_argument("--corpus", type=Path, help="Cópia local do corpus; deve corresponder ao hash do manifesto.")
    parser.add_argument("--natural-field", default="natural_text")
    parser.add_argument("--strong-field", default="strong_text")
    parser.add_argument("--model", default="xlm-roberta-base", help="Modelo Hugging Face ou caminho local imutável.")
    parser.add_argument("--num-layers", type=int, default=9, help="Camada do XLM-R (padrão: 9, para base).")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    if args.num_layers < 1:
        parser.error("--num-layers deve ser positivo")
    try:
        settings = {"model": args.model, "num_layers": args.num_layers, "lang": "pt",
                    "idf": False, "rescale_with_baseline": False, "device": args.device,
                    "bert_score_version": importlib.metadata.version("bert-score"),
                    "transformers_version": importlib.metadata.version("transformers"),
                    "torch_version": importlib.metadata.version("torch"), "metrics_version": VERSION}
    except importlib.metadata.PackageNotFoundError:
        parser.error("Instale as dependências locais: uv sync --extra metrics")

    scorer = None

    def factory():
        nonlocal scorer
        if scorer is None:
            tqdm.write(f"Carregando {args.model} em {args.device} (pode baixar os pesos no primeiro uso)...")
            from bert_score import BERTScorer
            scorer = BERTScorer(model_type=args.model, num_layers=args.num_layers, lang="pt",
                                idf=False, rescale_with_baseline=False, device=args.device)
            tqdm.write("Modelo carregado. Iniciando BERTScore.")
        return scorer

    for index, run_dir in enumerate(args.run_dir, start=1):
        tqdm.write(f"Experimento {index}/{len(args.run_dir)}: {run_dir}")
        output = evaluate_run(run_dir, settings=settings, scorer_factory=factory,
                              corpus_override=args.corpus, natural_field=args.natural_field,
                              strong_field=args.strong_field)
        tqdm.write(f"Métricas salvas em: {output}")


if __name__ == "__main__":
    main()
