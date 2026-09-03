from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from uuid import UUID, uuid4

from config import (
    CliOverrides,
    EffectiveConfig,
    build_manifest,
    load_completed_documents,
    load_effective_config,
    prepare_run_directory,
)
from core.providers import ModelClientFactory
from core.schemas import DocumentInput
from core.simplificacao import LinguagemSimplesGraph


def build_parser() -> argparse.ArgumentParser:
    examples = """
Exemplos:
  uv run --env-file .env main.py --model qwen/qwen3-32b
  uv run --env-file .env main.py --config configs/openrouter.yaml
  uv run main.py --config configs/hpc.yaml --model qwen/qwen3-32b
  uv run --env-file .env main.py --run-id UUID --resume

Precedência da configuração, da maior para a menor:
  1. opções da CLI;
  2. variáveis de ambiente LSG_*;
  3. arquivo YAML informado por --config.

--model sobrescreve o modelo dos três papéis: analisador, simplificador e
avaliador. Para modelos diferentes por papel, configure o YAML ou use
LSG_ANALYZER_MODEL, LSG_SIMPLIFIER_MODEL e LSG_EVALUATOR_MODEL.

Credenciais não são aceitas no YAML nem em opções da CLI. Forneça-as por
variáveis de ambiente reconhecidas pelo provedor, por exemplo com
`uv run --env-file .env ...`. Elas não são gravadas no manifesto ou nos logs.

Caminhos relativos são resolvidos a partir da raiz do projeto, mesmo quando o
comando é iniciado em outro diretório.
"""
    parser = argparse.ArgumentParser(
        description="Executa o harness de simplificação de textos jurídicos.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/openrouter.yaml",
        help="Arquivo YAML de configuração. Padrão: configs/openrouter.yaml",
    )
    parser.add_argument(
        "--model",
        help=(
            "Sobrescreve o modelo dos três papéis. Mantido para compatibilidade "
            "com o comando de execução atual."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Sobrescreve o caminho do corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Sobrescreve o diretório-base das execuções.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        help="Sobrescreve o limite de tentativas por ramo.",
    )
    parser.add_argument(
        "--run-id",
        type=UUID,
        help="Identificador da execução. Obrigatório com --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Retoma uma execução existente após validar configuração, corpus, "
            "prompts e versões dos schemas."
        ),
    )
    return parser


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def ensure_metadata_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "run_id",
                "document_id",
                "content_hash",
                "execution_status",
                "termination_reason",
                "elapsed_time",
                "input_tokens",
                "output_tokens",
                "llm_calls",
            ]
        )


def load_corpus(config: EffectiveConfig) -> list[DocumentInput]:
    with config.corpus.path.open("r", encoding="utf-8") as file:
        raw_corpus = json.load(file)
    if not isinstance(raw_corpus, list):
        raise ValueError("O corpus deve ser uma lista de documentos.")
    return [
        DocumentInput.from_corpus_item(
            item,
            id_field=config.corpus.id_field,
            text_field=config.corpus.text_field,
        )
        for item in raw_corpus
    ]


def summarize_results(
    results_path: Path,
) -> tuple[int, int | None, int | None, int]:
    documents = 0
    input_tokens = 0
    output_tokens = 0
    input_usage_complete = True
    output_usage_complete = True
    llm_calls = 0
    if not results_path.is_file():
        return documents, input_tokens, output_tokens, llm_calls
    with results_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            usage = record.get("usage", {})
            documents += 1
            document_input_tokens = usage.get("input_tokens")
            document_output_tokens = usage.get("output_tokens")
            if document_input_tokens is None:
                input_usage_complete = False
            else:
                input_tokens += int(document_input_tokens)
            if document_output_tokens is None:
                output_usage_complete = False
            else:
                output_tokens += int(document_output_tokens)
            llm_calls += int(usage.get("llm_calls", 0))
    return (
        documents,
        input_tokens if input_usage_complete else None,
        output_tokens if output_usage_complete else None,
        llm_calls,
    )


def display_usage(value: int | None) -> str:
    return str(value) if value is not None else "desconhecido"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.resume and args.run_id is None:
        parser.error("--resume exige --run-id.")

    cli = CliOverrides(
        model=args.model,
        corpus_path=args.corpus,
        execution_directory=args.output_dir,
        max_revisions=args.max_revisions,
    )
    config, config_source = load_effective_config(args.config, cli=cli)
    run_config = config.to_run_config()
    run_id = args.run_id or uuid4()
    corpus = load_corpus(config)

    # Configuração e corpus são validados antes da criação dos clientes.
    manifest = build_manifest(
        config,
        config_source=config_source,
        run_id=run_id,
    )
    clients = ModelClientFactory().create_all(run_config)
    run_directory = prepare_run_directory(
        config,
        manifest,
        resume=args.resume,
    )
    results_path = run_directory / config.execution.results_filename
    metadata_path = run_directory / config.execution.metadata_filename
    ensure_metadata_header(metadata_path)

    completed = load_completed_documents(results_path)
    for document in corpus:
        previous_hash = completed.get(document.document_id)
        if previous_hash is not None and previous_hash != document.content_hash:
            raise ValueError(
                f"Retomada recusada: o documento {document.document_id!r} "
                "mudou de conteúdo."
            )

    pending = [
        document
        for document in corpus
        if document.document_id not in completed
    ]
    print(f"run_id: {run_id}")
    print(f"configuração: {config_source}")
    print(f"diretório: {run_directory}")
    print(f"documentos concluídos: {len(completed)}")
    print(f"documentos pendentes: {len(pending)}")

    session_start = time.perf_counter()
    for index, document in enumerate(pending, start=1):
        print(
            f"Iniciando simplificação do texto: {document.document_id} "
            f"[{index}/{len(pending)}]"
        )
        start_time = time.perf_counter()
        result = LinguagemSimplesGraph.run(
            document,
            config=run_config,
            clients=clients,
            run_id=run_id,
        )
        elapsed = time.perf_counter() - start_time

        append_jsonl(results_path, result.model_dump(mode="json"))
        with metadata_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    str(result.run_id),
                    result.document_id,
                    result.content_hash,
                    result.execution_status.value,
                    result.termination_reason.value,
                    f"{elapsed:.4f}",
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.usage.llm_calls,
                ]
            )

        print("Finalizado!")
        print(f"\tStatus: {result.execution_status.value}")
        print(f"\tMotivo: {result.termination_reason.value}")
        print(f"\tTempo: {elapsed:.4f}s")

    session_elapsed = time.perf_counter() - session_start
    documents, input_tokens, output_tokens, llm_calls = summarize_results(
        results_path
    )
    summary = {
        "run_id": str(run_id),
        "documents": documents,
        "corpus_size": len(corpus),
        "session_elapsed_time": round(session_elapsed, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "llm_calls": llm_calls,
    }
    with (run_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("---------------------")
    print("Dados gerais:")
    print(f"\tdocumentos={documents}/{len(corpus)}")
    print(
        "\ttokens: "
        f"input={display_usage(input_tokens)}, "
        f"output={display_usage(output_tokens)}"
    )
    print(f"\tchamadas ao modelo={llm_calls}")


if __name__ == "__main__":
    main()
