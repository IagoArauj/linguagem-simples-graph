"""Baselines SARI humanos, offline: python -m metrics.human_sari --help."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev

from metrics.text_similarity import sari

EXPERIMENTS = (
    ("copy_natural", "Copy baseline", "original_text", "natural_text",
     "Não simplificar, contra a referência Natural"),
    ("copy_strong", "Copy baseline Strong", "original_text", "strong_text",
     "Não simplificar, contra a referência Strong"),
    ("natural_to_strong", "Human Natural → Strong", "natural_text", "strong_text",
     "Operações da Natural em relação às operações da Strong"),
    ("strong_to_natural", "Human Strong → Natural", "strong_text", "natural_text",
     "Operações da Strong em relação às operações da Natural"),
)
LABELS = {"original_text": "Original", "natural_text": "Natural", "strong_text": "Strong"}


def compute_human_sari(corpus_path: Path, *, output_dir: Path | None = None,
                       overwrite: bool = False) -> Path:
    corpus_path = Path(corpus_path).resolve()
    content = corpus_path.read_bytes()
    documents = []
    seen = set()
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"JSON inválido na linha {line_number}.") from exc
        if not isinstance(document, dict):
            raise ValueError(f"Esperado objeto na linha {line_number}.")
        ident = document.get("production_id")
        if isinstance(ident, bool) or not isinstance(ident, (str, int)) or not str(ident).strip():
            raise ValueError(f"production_id inválido na linha {line_number}.")
        ident = str(ident).strip()
        if ident in seen:
            raise ValueError(f"production_id duplicado: {ident}.")
        seen.add(ident)
        for field in LABELS:
            value = document.get(field)
            if not isinstance(value, str) or not value.strip():
                # Não converter ausência em texto vazio: a convenção SARI pode
                # atribuir crédito a operações vazias e distorcer a média.
                raise ValueError(f"Documento {ident}: {field} ausente, vazio ou não textual.")
        documents.append((ident, document))
    if not documents:
        raise ValueError("Corpus vazio.")

    output = Path(output_dir).resolve() if output_dir else corpus_path.parent
    paths = [output / name for name in ("human_sari.md", "human_sari.csv", "human_sari_manifest.json")]
    for path in paths:
        if path == corpus_path:
            raise ValueError("A saída não pode sobrescrever o corpus.")
        if path.is_symlink() or (path.exists() and not overwrite):
            raise FileExistsError(f"Saída já existe: {path}. Use --overwrite para recalcular.")
    rows = []
    scores = {experiment[0]: [] for experiment in EXPERIMENTS}
    for ident, document in documents:
        for key, label, prediction, reference, _ in EXPERIMENTS:
            score = sari(document["original_text"], document[prediction], [document[reference]])
            scores[key].append(score)
            rows.append({"document_id": ident, "experiment": key, "source": "original_text",
                         "prediction": prediction, "reference": reference, "sari": score})
    markdown = [
        "# SARI — baselines de cópia e simplificações humanas", "",
        f"Corpus: `{corpus_path.name}`. Documentos: **{len(documents)}**.", "",
        "Média aritmética dos escores individuais por documento, com peso igual. "
        "Escala de 0 a 100; uma referência por comparação. Não é agregação de n-gramas do corpus.", "",
        "| Experimento | Source | Prediction | Reference | n | SARI médio | Desvio-padrão amostral | O que mede |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for key, label, prediction, reference, meaning in EXPERIMENTS:
        values = scores[key]
        std = f"{stdev(values):.4f}" if len(values) > 1 else "—"
        markdown.append(f"| {label} | Original | {LABELS[prediction]} | {LABELS[reference]} | "
                        f"{len(values)} | {fmean(values):.4f} | {std} | {meaning} |")
    markdown += ["", "## Método e interpretação", "",
        "- Mesma função `metrics.text_similarity.sari` usada nas simplificações de IA; "
        "variante SARIsent/Evaluate documentada no código, com n-gramas de ordens 1 a 4.",
        r"- Tokenização: Unicode NFC, minúsculas e regex `\w+|[^\w\s]`, mantendo pontuação. "
        "Não usa stemming ou a tokenização inglesa padrão.",
        "- A convenção de divisão por zero é 1. Por isso, uma candidata que apenas copia "
        "o original não precisa receber SARI zero.",
        "- SARI avalia operações de manter, adicionar e remover em relação ao original. "
        "Não é uma medida simétrica: Natural → Strong pode diferir de Strong → Natural.",
        "- A referência é uma simplificação possível, não uma verdade única. Esses valores "
        "não são um teto humano nem comprovam preservação semântica ou superioridade entre níveis.",
        "- Compare com a IA apenas usando a mesma implementação, tokenização, referência "
        "e conjunto de documentos. Se houver falhas nas runs, use o CSV para parear os IDs.",
        "- `human_sari.csv` contém os valores sem arredondamento para cada documento/experimento. "
        "O manifesto registra hashes do corpus e do código. Nenhuma API ou modelo é utilizado.", ""]
    manifest = {
        "metrics_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_path": str(corpus_path), "corpus_sha256": hashlib.sha256(content).hexdigest(),
        "documents": len(documents), "scale": [0, 100], "aggregation": "macro mean per document",
        "experiments": [{"id": key, "label": label, "source": "original_text",
                         "prediction": prediction, "reference": reference}
                        for key, label, prediction, reference, _ in EXPERIMENTS],
        "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("human_sari.py", "text_similarity.py")},
        "artifacts": [path.name for path in paths[:2]],
    }
    output.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with paths[1].open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paths[0].open(mode, encoding="utf-8") as stream:
        stream.write("\n".join(markdown))
    with paths[2].open(mode, encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula quatro baselines SARI do corpus humano, sem modelos ou APIs.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Corpus JSONL: production_id, original_text, natural_text e strong_text.")
    parser.add_argument("--output-dir", type=Path, help="Pasta de saída; padrão: a mesma do corpus.")
    parser.add_argument("--overwrite", action="store_true", help="Substituir os três arquivos human_sari existentes.")
    args = parser.parse_args()
    try:
        output = compute_human_sari(args.input, output_dir=args.output_dir, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(f"Relatório salvo em: {output}")


if __name__ == "__main__":
    main()
