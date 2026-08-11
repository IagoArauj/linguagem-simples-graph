from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "A dependência 'tqdm' não está instalada.\n"
        "Instale com: pip install tqdm"
    ) from exc


REQUIRED_FIELDS = (
    "text",
    "simple_simplification",
    "moderate_simplification",
    "aggressive_simplification",
)

STAGES = (
    ("original", "text"),
    ("simple_simplification", "simple_simplification"),
    ("moderate_simplification", "moderate_simplification"),
    ("aggressive_simplification", "aggressive_simplification"),
)


def _convert_metric_value(value: str) -> int | float | str | None:
    """Converte um valor textual para int, float, None ou str."""
    normalized = value.strip()

    if normalized.lower() in {"none", "null", "nan"}:
        return None

    if re.fullmatch(r"[+-]?\d+", normalized):
        return int(normalized)

    try:
        return float(normalized)
    except ValueError:
        return normalized


def compute_nilc_metrix(
    text: str,
    nilc_metrix_folder: str | Path,
    script_to_run: str = "run_minimal.sh",
    field_name: str | None = None,
) -> dict[str, Any]:
    """
    Executa o script do NILC-Metrix e transforma a saída entre '++'
    em um dicionário Python.
    """
    folder = Path(nilc_metrix_folder).expanduser().resolve()
    script_path = folder / script_to_run

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Diretório do NILC-Metrix não encontrado: {folder}"
        )

    if not script_path.is_file():
        raise FileNotFoundError(f"Script não encontrado: {script_path}")

    if not isinstance(text, str):
        raise TypeError(
            f"O texto deve ser uma string, mas recebeu {type(text).__name__}."
        )

    if text.strip() == "":
        print(f"O texto está vazio: {field_name}")
        return {}

    try:
        result = subprocess.run(
            ["bash", str(script_path), text],
            cwd=folder,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Erro ao executar o NILC-Metrix para o texto \"{field_name}\".\n"
            f"Código de saída: {exc.returncode}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Não foi possível executar o script {script_path}: {exc}"
        ) from exc

    complete_output = f"{result.stdout}\n{result.stderr}"

    match = re.search(
        r"\+\+\s*(.*?)\s*,?\s*\+\+",
        complete_output,
        flags=re.DOTALL,
    )

    if not match:
        raise RuntimeError(
            "Não foi possível encontrar as métricas entre os delimitadores "
            "'++'.\n"
            f"Saída completa:\n{complete_output}"
        )

    metrics_text = match.group(1).strip().rstrip(",")
    metrics: dict[str, Any] = {}

    for item in metrics_text.split(","):
        item = item.strip()

        if not item:
            continue

        key, separator, raw_value = item.partition(":")

        if not separator:
            raise RuntimeError(f"Métrica em formato inválido: {item!r}")

        key = key.strip()
        raw_value = raw_value.strip()

        if not key:
            raise RuntimeError(f"Métrica sem nome encontrada: {item!r}")

        metrics[key] = _convert_metric_value(raw_value)

    if not metrics:
        raise RuntimeError(
            "Os delimitadores '++' foram encontrados, mas nenhuma métrica "
            "válida foi extraída."
        )

    return metrics


def format_duration(seconds: float) -> str:
    """Formata uma duração em segundos como HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _validate_item(item: Any, index: int) -> dict[str, Any]:
    """Valida um item do arquivo de entrada."""
    if not isinstance(item, dict):
        raise ValueError(f"O item {index} deve ser um objeto JSON.")

    missing_fields = [
        field for field in REQUIRED_FIELDS if field not in item
    ]

    if missing_fields:
        raise ValueError(
            f"Item {index} sem os campos obrigatórios: "
            f"{', '.join(missing_fields)}"
        )

    invalid_fields = [
        field
        for field in REQUIRED_FIELDS
        if not isinstance(item[field], str)
    ]

    if invalid_fields:
        raise ValueError(
            f"Item {index} possui campos que não são strings: "
            f"{', '.join(invalid_fields)}"
        )

    return item


def _write_jsonl_record(
    output_file: Any,
    record: dict[str, Any],
) -> None:
    """
    Grava um registro JSONL e força a sincronização com o disco.

    Dessa forma, todos os itens concluídos permanecem salvos mesmo se o
    processamento for interrompido depois.
    """
    json.dump(record, output_file, ensure_ascii=False)
    output_file.write("\n")
    output_file.flush()
    os.fsync(output_file.fileno())


def _load_completed_indices(output_path: Path) -> set[int]:
    """
    Lê o JSONL existente e retorna os índices concluídos com sucesso.
    Linhas incompletas ou inválidas são ignoradas.
    """
    completed: set[int] = set()

    if not output_path.is_file():
        return completed

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if (
                isinstance(record, dict)
                and record.get("status") == "success"
                and isinstance(record.get("item_index"), int)
            ):
                completed.add(record["item_index"])

    return completed


def _compute_item_metrics(
    item: dict[str, Any],
    nilc_metrix_folder: str | Path,
    script_to_run: str,
    workers: int,
    progress_bar: tqdm,
    item_index: int,
    total_items: int,
) -> dict[str, Any]:
    """
    Calcula as quatro variantes de um item, executando até ``workers``
    scripts simultaneamente.
    """
    metrics: dict[str, Any] = {}
    completed_stages = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_stage: dict[Future[dict[str, Any]], str] = {}

        for stage_name, field_name in STAGES:
            future = executor.submit(
                compute_nilc_metrix,
                item[field_name],
                nilc_metrix_folder,
                script_to_run,
                field_name
            )
            future_to_stage[future] = stage_name

        for future in as_completed(future_to_stage):
            stage_name = future_to_stage[future]
            metrics[stage_name] = future.result()
            completed_stages += 1

            progress_bar.set_description(
                f"Item {item_index}/{total_items} "
                f"| métricas {completed_stages}/{len(STAGES)}"
            )

    # Mantém a ordem estável das chaves no JSONL.
    return {
        stage_name: metrics[stage_name]
        for stage_name, _ in STAGES
    }


def compute_metrics(
    nilc_metrix_folder: str | Path,
    script_to_run: str = "run_minimal.sh",
    input_file: str | Path = "output/data.json",
    output_file: str | Path = "output/metrics.jsonl",
    *,
    workers: int = 2,
    append: bool = False,
    resume: bool = False,
    fail_fast: bool = False,
) -> None:
    """
    Calcula métricas para o texto original e para as três simplificações.

    O paralelismo ocorre dentro de cada item: original, simples, moderada e
    agressiva podem ser processadas simultaneamente.

    Cada item concluído é imediatamente gravado e sincronizado no JSONL.
    """
    if workers < 1 or workers > len(STAGES):
        raise ValueError(
            f"--workers deve estar entre 1 e {len(STAGES)}."
        )

    input_path = Path(input_file).expanduser().resolve()
    output_path = Path(output_file).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "O arquivo de entrada deve conter uma lista JSON."
        )

    total_items = len(data)

    if total_items == 0:
        print("Nenhum item encontrado no arquivo de entrada.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_indices = (
        _load_completed_indices(output_path)
        if resume
        else set()
    )

    if resume:
        output_mode = "a"
    else:
        output_mode = "a" if append else "w"

    pending_items = [
        (index, item)
        for index, item in enumerate(data, start=1)
        if index not in completed_indices
    ]

    if not pending_items:
        print("Todos os itens já foram processados com sucesso.")
        print(f"Arquivo de saída: {output_path}")
        return

    start_time = time.perf_counter()
    successful_items = 0
    failed_items = 0
    skipped_items = len(completed_indices)

    print(f"Itens encontrados: {total_items}")
    print(f"Itens já concluídos: {skipped_items}")
    print(f"Itens pendentes: {len(pending_items)}")
    print(f"Execuções paralelas por item: {workers}")
    print()

    with output_path.open(output_mode, encoding="utf-8") as output:
        progress_bar = tqdm(
            pending_items,
            total=len(pending_items),
            desc="Calculando métricas",
            unit="item",
            dynamic_ncols=True,
        )

        for processed_position, (index, raw_item) in enumerate(
            progress_bar,
            start=1,
        ):
            item_start_time = time.perf_counter()

            try:
                item = _validate_item(raw_item, index)

                metrics = _compute_item_metrics(
                    item=item,
                    nilc_metrix_folder=nilc_metrix_folder,
                    script_to_run=script_to_run,
                    workers=workers,
                    progress_bar=progress_bar,
                    item_index=index,
                    total_items=total_items,
                )

                item_elapsed = time.perf_counter() - item_start_time

                record = {
                    "item_index": index,
                    "status": "success",
                    "processing_seconds": round(item_elapsed, 3),
                    **metrics,
                }

                _write_jsonl_record(output, record)
                successful_items += 1

            except Exception as exc:
                failed_items += 1
                item_elapsed = time.perf_counter() - item_start_time

                error_record = {
                    "item_index": index,
                    "status": "error",
                    "processing_seconds": round(item_elapsed, 3),
                    "error": str(exc),
                }

                _write_jsonl_record(output, error_record)
                progress_bar.write(f"Erro no item {index}: {exc}")

                if fail_fast:
                    raise

            finally:
                remaining_items = len(pending_items) - processed_position
                elapsed_time = time.perf_counter() - start_time
                average_time = elapsed_time / processed_position
                estimated_remaining = average_time * remaining_items
                current_item_time = time.perf_counter() - item_start_time

                progress_bar.set_description("Calculando métricas")
                progress_bar.set_postfix(
                    {
                        "restantes": remaining_items,
                        "média/item": format_duration(average_time),
                        "último": format_duration(current_item_time),
                        "faltante": format_duration(estimated_remaining),
                        "erros": failed_items,
                    },
                    refresh=True,
                )

    total_elapsed = time.perf_counter() - start_time
    average_time = total_elapsed / len(pending_items)

    print()
    print("Processamento concluído.")
    print(f"Itens encontrados: {total_items}")
    print(f"Itens ignorados por já estarem concluídos: {skipped_items}")
    print(f"Itens processados com sucesso nesta execução: {successful_items}")
    print(f"Itens com erro nesta execução: {failed_items}")
    print(f"Tempo total desta execução: {format_duration(total_elapsed)}")
    print(f"Média por item: {format_duration(average_time)}")
    print(f"Arquivo de saída: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula métricas NILC-Metrix para textos originais e "
            "simplificados, com paralelismo e retomada."
        )
    )

    parser.add_argument(
        "--nilc-metrix-folder",
        "-f",
        required=True,
        help="Diretório onde está instalado o NILC-Metrix.",
    )

    parser.add_argument(
        "--script",
        "-s",
        default="run_minimal.sh",
        help="Nome do script de execução. Padrão: run_minimal.sh",
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Arquivo JSON de entrada. Padrão: output/data.json",
    )

    parser.add_argument(
        "--output",
        "-o",
        #default="output/metrics.jsonl",
        required=True,
        help="Arquivo JSONL de saída. Padrão: output/metrics.jsonl",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        choices=range(1, len(STAGES) + 1),
        default=2,
        metavar="{1,2,3,4}",
        help=(
            "Quantidade de métricas executadas simultaneamente por item. "
            "Padrão: 2"
        ),
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Adiciona os resultados ao arquivo existente.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continua um processamento anterior, ignorando os itens que já "
            "possuem status success no JSONL."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Interrompe o processamento no primeiro erro.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    try:
        compute_metrics(
            nilc_metrix_folder=args.nilc_metrix_folder,
            script_to_run=args.script,
            input_file=args.input,
            output_file=args.output,
            workers=args.workers,
            append=args.append,
            resume=args.resume,
            fail_fast=args.fail_fast,
        )
    except KeyboardInterrupt:
        print(
            "\nProcessamento interrompido pelo usuário. "
            "Os itens já concluídos permanecem salvos.",
            file=sys.stderr,
        )
        return 130
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
