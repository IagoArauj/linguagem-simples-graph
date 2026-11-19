"""Relatórios operacionais offline, sem corpus nem dependências de inferência.

CSV vazio significa desconhecido. ``known_sum`` nunca representa um total
completo sem cobertura 1. Os eventos de falha não são contagens de retries do
provider (estes não são persistidos como AttemptResult).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

RESOURCES = ("llm_calls", "input_tokens", "output_tokens", "total_tokens",
             "elapsed_time", "retry_wait_seconds", "elapsed_without_backoff")
STATUSES = ("completed", "failed", "partial", "cancelled", "unknown")
VERDICTS = ("approved", "rejected", "not_evaluated", "unknown")
REASONS = ("approval", "revision_limit", "budget", "technical_error",
           "invalid_response", "cancellation", "content_policy", "unknown")
OPERATIONAL = {"technical_error", "invalid_response"}


def _id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("Identificador ausente ou inválido")
    return str(value).strip()


def _basename(value):
    if not isinstance(value, str) or value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError("Arquivo da execução deve ser basename")
    return value


def _operational(row):
    return row.get("execution_status") == "failed" or row.get("termination_reason") in OPERATIONAL


def _category(row):
    """Heurística: mensagens são evidência, não são reproduzidas no relatório."""
    message = str(row.get("error_message") or "").lower()
    if re.search(r"\b429\b|rate.?limit|too many requests", message):
        return "rate_limit"
    if ("unsupported" in message or "not supported" in message or "not allowed" in message) and re.search(
        r"parameter|top_p|temperature|max_tokens|reasoning", message
    ):
        return "unsupported_parameter"
    if re.search(r"\b401\b|\b403\b|authentication|unauthorized|invalid api.?key", message):
        return "authentication"
    if re.search(r"timeout|timed out", message):
        return "timeout"
    if row.get("termination_reason") == "invalid_response":
        return "invalid_response"
    if row.get("termination_reason") == "technical_error":
        return "technical_error"
    return "unknown"


def _stage(row):
    for field in ("error_stage", "stage", "role"):
        if row.get(field) in {"analyzer", "simplifier", "evaluator"}:
            return row[field]
    if row.get("evaluation_call_id"):
        return "evaluator"
    if row.get("simplification_call_id"):
        return "simplifier"
    return "unknown"


def _cell(value):
    if value is None:
        return "desconhecido"
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = "".join("\\" + char if char in "\\`*_{}[]()#!|" else char for char in text)
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _table(headers, rows):
    return ["| " + " | ".join(map(_cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |", *[
                "| " + " | ".join(map(_cell, row)) + " |" for row in rows], ""]


def write_execution_report(run_dirs: list[Path], output_dir: Path) -> dict:
    """Escreve execution_*.csv/md em diretório existente, sem sobrescrever.

    Retorna provenance JSON-serializável com runs, source_files e warnings.
    Valida todas as fontes antes da escrita. IDs duplicados e joins conflitantes
    são erros; números inválidos/conflitantes geram avisos. Não consulta corpus.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if not run_dirs:
        raise ValueError("Informe pelo menos uma run")
    filenames = ["execution_report.md", "execution_resources.csv", "execution_documents.csv",
                 "execution_branches.csv", "execution_failures.csv",
                 "execution_resource_summary.csv", "execution_branch_summary.csv",
                                  "execution_document_summary.csv", "execution_failure_summary.csv",
                                  "execution_branch_reasons.csv"]
    for name in filenames:
        if (output_dir / name).exists() or (output_dir / name).is_symlink():
            raise FileExistsError(output_dir / name)
    provenance = {"runs": [], "source_files": [], "warnings": []}
    sources = {}

    def read(path):
        path = path.resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if path in sources and sources[path] != digest:
            raise ValueError("Fonte mudou durante leitura")
        if path not in sources:
            sources[path] = digest
            provenance["source_files"].append({"path": str(path), "sha256": digest})
        return data.decode("utf-8-sig")

    def warn(message):
        provenance["warnings"].append(message)

    def number(value, context):
        if value is None or value == "":
            return None
        try:
            n = float(value)
        except (ValueError, TypeError):
            n = float("nan")
        if isinstance(value, bool) or not math.isfinite(n) or n < 0:
            warn(f"Recurso inválido: {context}")
            return None
        if context.split(":")[-1] in RESOURCES[:4]:
            try:
                exact = Decimal(str(value))
            except InvalidOperation:
                warn(f"Contagem inválida: {context}")
                return None
            if exact != exact.to_integral_value():
                warn(f"Contagem não inteira: {context}")
                return None
            # Preserva inteiros além da precisão de float na soma derivada.
            if exact > 2**53:
                return int(exact)
        return n

    loaded = []
    seen_runs = set()
    for directory in run_dirs:
        directory = Path(directory).resolve()
        manifest = json.loads(read(directory / "manifest.json"))
        if "manifest_version" not in manifest:
            warn(f"manifest_version ausente: {directory}")
        elif manifest["manifest_version"] != "1.0":
            raise ValueError(f"manifest_version incompatível: {directory}")
        rid = _id(manifest.get("run_id"))
        if rid in seen_runs:
            raise ValueError("run_id duplicado")
        seen_runs.add(rid)
        config = manifest["effective_config"]
        execution = config["execution"]
        models = config.get("models", {})
        model = models.get("simplifier", {})
        name = (model.get("name") if isinstance(model, dict) else model) or rid
        workflow = config.get("workflow", {})
        configured = workflow.get("branches")
        expected = None if configured is None else [
            _id(b.get("branch_id") if isinstance(b, dict) else b) for b in configured]
        if expected is not None and len(expected) != len(set(expected)):
            raise ValueError("branch_id duplicado na configuração")
        run = {"run_id": rid, "run_name": name, "directory": str(directory),
               "models": models, "branches": expected, "max_revisions": workflow.get("max_revisions"),
                              "corpus_hash": manifest.get("corpus_hash")}
        provenance["runs"].append(run)
        text = read(directory / _basename(execution["results_filename"]))
        results = json.loads(text) if text.lstrip().startswith("[") else [
            json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(results, list) or any(not isinstance(r, dict) for r in results):
            raise ValueError("Results deve ser JSON lista ou JSONL de objetos")
        metadata = {}
        metadata_name = execution.get("metadata_filename")
        path = directory / _basename(metadata_name) if metadata_name else None
        if path is not None and path.exists():
            reader = csv.DictReader(io.StringIO(read(path)))
            if not {"run_id", "document_id"}.issubset(reader.fieldnames or []):
                raise ValueError("Metadata sem run_id/document_id")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError("Linha metadata inválida")
                ident = _id(row.get("document_id"))
                if ident in metadata:
                    raise ValueError("document_id duplicado em metadata")
                metadata[ident] = row
        else:
            warn(f"Metadata opcional ausente: run {rid}")
        loaded.append((run, manifest, results, metadata))
    corpus_hashes = {r["corpus_hash"] for r in provenance["runs"] if r["corpus_hash"]}
    if len(corpus_hashes) > 1:
        warn("corpus_hash diferentes entre runs: comparação operacional não pareada; corpus não consultado.")
    if len(loaded) > 1 and any(not r["corpus_hash"] for r in provenance["runs"]):
        warn("corpus_hash ausente em alguma run: identidade dos corpora não verificável; comparação não pareada.")
    names = Counter(r["run_name"] for r in provenance["runs"])
    for run in provenance["runs"]:
        if names[run["run_name"]] > 1:
            run["run_name"] += f' [{run["run_id"]}]'
    if len({r["run_name"] for r in provenance["runs"]}) != len(loaded):
        for run in provenance["runs"]:
            run["run_name"] = f'{run["run_id"]}: {run["run_name"]}'

    documents, resources, branches, failures, summaries, branch_summaries = [], [], [], [], [], []
    document_summaries, failure_summaries, branch_reasons = [], [], []
    markdown = ["# Relatório operacional offline", "",
                "llm_calls conta invocações lógicas; retries técnicos não estão necessariamente contados.",
                "Tempos em segundos. retry_wait_seconds mede backoff somente sem outra chamada ativa; elapsed_without_backoff exclui apenas esse backoff, não toda espera do provider.",
                "total_tokens ausente pode ser derivado pela soma exata de input_tokens + output_tokens conhecidos (source=derived_input_output), não uma estimativa.",
                "Estatísticas por documento observado nos results; não representam documentos ausentes da run.",
                "Campos CSV vazios = desconhecidos. Somas conhecidas não são totais reais quando incompletas.",
                "Categorias e estágios inferidos são heurísticos; mensagens brutas e IDs de documentos não são exibidos aqui.",
                "Tentativas são eventos persistidos, não retries internos do provider. Falhas finais e impacto propagado não somam novos eventos quando já representados.",
                "Comparações são descritivas, não causais e sem teste de significância. A comparação entre runs é não pareada: usa documentos observados em cada run, sem alinhamento de IDs ou amostras.", ""]
    for run, manifest, results, metadata in loaded:
        rid = run["run_id"]
        base = {"run_id": rid, "run_name": run["run_name"]}
        seen_docs = set()
        local_resources, local_docs, local_branches, local_failures = [], [], [], []

        def identity(row, doc, required=False):
            for key, expected in (("run_id", rid), ("document_id", doc["document_id"]),
                                  ("content_hash", doc.get("content_hash")),
                                  ("source_hash", doc.get("content_hash")),
                                  ("corpus_hash", manifest.get("corpus_hash"))):
                actual = row.get(key)
                if required and key in {"run_id", "document_id"} and actual in (None, ""):
                    raise ValueError(f"{key} ausente")
                if actual not in (None, "") and expected not in (None, "") and str(actual) != str(expected):
                    raise ValueError(f"{key} divergente")

        def failure(docbase, row, scope, branch_id="", attempt=None, propagated=False,
                    event=True, stage=None):
            local_failures.append({**docbase, "branch_id": branch_id,
                                   "attempt_number": attempt, "scope": scope,
                                   "execution_status": row.get("execution_status", "unknown"),
                                   "termination_reason": row.get("termination_reason"),
                                   "stage": stage or _stage(row), "category": _category(row),
                                   "classification": "heuristic", "analyzer_propagated": int(propagated),
                                   "event_count": int(event),
                                   "message_evidence": int(bool(row.get("error_message")))})

        for doc in results:
            ident = _id(doc.get("document_id"))
            if ident in seen_docs:
                raise ValueError("document_id duplicado em results")
            seen_docs.add(ident)
            identity(doc, doc, required=True)
            if doc.get("original_text") is not None and doc.get("content_hash"):
                if hashlib.sha256(doc["original_text"].encode()).hexdigest() != doc["content_hash"]:
                    raise ValueError("content_hash divergente do original_text")
            meta = metadata.get(ident, {})
            if meta:
                identity(meta, doc, required=True)
                for key in ("execution_status", "termination_reason"):
                    if meta.get(key) and doc.get(key) and meta[key] != doc[key]:
                        raise ValueError(f"{key} divergente entre results/metadata")
                if not doc.get("content_hash") or not (meta.get("content_hash") or meta.get("source_hash")):
                    warn(f"Hash cruzado indisponível: {rid}/{ident}")
            if not doc.get("content_hash"):
                warn(f"content_hash ausente em results: {rid}/{ident}")
            docbase = {**base, "document_id": ident}
            usage = doc.get("usage") or {}
            values, origins = {}, {}
            for metric in RESOURCES:
                primary = usage.get(metric) if metric in RESOURCES[:4] else doc.get(metric)
                a = number(primary, f"{rid}/{ident}/results:{metric}")
                b = number(meta.get(metric), f"{rid}/{ident}/metadata:{metric}")
                if a is not None and b is not None and not math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0002):
                    warn(f"Conflito numérico (preferido results): {rid}/{ident}:{metric}")
                values[metric] = a if a is not None else b
                origins[metric] = "results" if a is not None else "metadata" if b is not None else "missing"
            i, o, t = (values[k] for k in ("input_tokens", "output_tokens", "total_tokens"))
            if t is not None and ((i is not None and t < i) or (o is not None and t < o)
                                  or (i is not None and o is not None and t != i + o)):
                warn(f"total_tokens incoerente descartado: {rid}/{ident}")
                values["total_tokens"], origins["total_tokens"] = None, "incoherent"
            elif (t is None and i is not None and o is not None
                  and usage.get("total_tokens") in (None, "")
                  and meta.get("total_tokens") in (None, "")):
                values["total_tokens"] = int(i) + int(o)
                origins["total_tokens"] = "derived_input_output"
            elapsed, wait, net = (values[k] for k in RESOURCES[4:])
            if elapsed is not None and ((wait is not None and wait > elapsed + 0.0002)
                                        or (net is not None and net > elapsed + 0.0002)
                                        or (wait is not None and net is not None and
                                            not math.isclose(elapsed, wait + net, abs_tol=0.0002))):
                warn(f"Tempos incoerentes descartados: {rid}/{ident}")
                for metric in RESOURCES[5:]:
                    values[metric], origins[metric] = None, "incoherent"
            local_resources.append({**docbase, **values,
                                    **{f"{k}_source": v for k, v in origins.items()}})
            observed = doc.get("branches") or []
            ids = [_id(b.get("branch_id")) for b in observed]
            if len(ids) != len(set(ids)):
                raise ValueError("branch_id duplicado em results")
            expected = run["branches"]
            if expected is not None and set(ids) - set(expected):
                raise ValueError("branch_id não configurado")
            propagated = (doc.get("analysis") is None and bool(observed)
                          and all(b.get("attempts") == [] for b in observed)
                          and _operational(doc) and all(_operational(b) for b in observed))
            if propagated:
                evidence = next((b for b in observed if b.get("error_message")), doc)
                failure(docbase, {**doc, "error_message": evidence.get("error_message")},
                        "document", propagated=True, stage="analyzer")
            for branch in observed:
                identity(branch, doc)
                bid = branch["branch_id"]
                attempts = branch.get("attempts")
                failed_attempts = 0
                rejected_attempts = 0
                seen_attempts = set()
                for attempt in attempts or []:
                    identity(attempt, doc)
                    if attempt.get("branch_id", bid) != bid:
                        raise ValueError("branch_id divergente na tentativa")
                    number_id = attempt.get("attempt_number")
                    if number_id is not None and number_id in seen_attempts:
                        raise ValueError("attempt_number duplicado")
                    seen_attempts.add(number_id)
                    rejected_attempts += attempt.get("quality_verdict") == "rejected"
                    if _operational(attempt):
                        failed_attempts += 1
                        failure(docbase, attempt, "attempt", bid, number_id)
                final_error = _operational(branch)
                if final_error:
                    failure(docbase, branch, "branch_final", bid, propagated=propagated,
                            event=not propagated and not failed_attempts,
                            stage="analyzer" if propagated else (
                                _stage(attempts[-1]) if attempts else "unknown"))
                local_branches.append({**docbase, "branch_id": bid,
                                       "execution_status": branch.get("execution_status", "unknown"),
                                       "quality_verdict": branch.get("quality_verdict", "unknown"),
                                       "termination_reason": branch.get("termination_reason"),
                                       "attempts": len(attempts) if attempts is not None else None,
                                       "rejected_attempts": rejected_attempts if attempts is not None else None,
                                       "attempt_failures": failed_attempts,
                                       "final_operational_error": int(final_error),
                                       "own_error": int(bool(failed_attempts or (final_error and not propagated))),
                                       "propagated_error": int(propagated and final_error)})
            if _operational(doc) and not propagated and not any(
                _operational(b) or any(_operational(a) for a in b.get("attempts", []) or []) for b in observed
            ):
                failure(docbase, doc, "document")
            local_docs.append({**docbase, "execution_status": doc.get("execution_status", "unknown"),
                               "termination_reason": doc.get("termination_reason"),
                               "observed_branches": len(ids),
                               "expected_branches": len(expected) if expected is not None else None,
                               "missing_branches": len(set(expected) - set(ids)) if expected is not None else None,
                               "analyzer_propagated": int(propagated)})
        if set(metadata) - seen_docs:
            raise ValueError("document_id de metadata ausente em results")
        if run["branches"] is None:
            warn(f"Branches esperadas desconhecidas: run {rid}")
        documents.extend(local_docs)
        resources.extend(local_resources)
        branches.extend(local_branches)
        failures.extend(local_failures)
        n_docs = len(results)
        markdown += [f'## {_cell(run["run_name"])}', "", f'Documentos observados: {n_docs}; max_revisions configurado: {_cell(run["max_revisions"])}.', ""]
        status_reasons = Counter((d["execution_status"] if d["execution_status"] in STATUSES else "unknown",
                                  d["termination_reason"] if d["termination_reason"] in REASONS else "unknown") for d in local_docs)
        markdown += _table(["Status documento", "Motivo", "n", "Documentos"],
                           [(s, r, n, n_docs) for (s, r), n in sorted(status_reasons.items())])
        document_summaries.extend(
            {**base, "execution_status": status, "termination_reason": reason,
             "n": count, "total_documents": n_docs, "fraction": count / n_docs if n_docs else None}
            for (status, reason), count in sorted(status_reasons.items()))
        run_summaries = []
        for metric in RESOURCES:
            known = [r[metric] for r in local_resources if r[metric] is not None]
            n = len(known)
            try:
                known_sum = math.fsum(known) if n else None
            except OverflowError:
                known_sum = None
                warn(f"Overflow na soma conhecida: {rid}:{metric}")
            summary = {**base, "resource": metric, "mean": math.fsum(v / n for v in known) if n else None,
                       "n": n, "total_documents": n_docs, "missing": n_docs - n,
                       "coverage": n / n_docs if n_docs else None,
                       "known_sum": known_sum, "complete": int(n == n_docs and n_docs > 0)}
            run_summaries.append(summary)
        summaries.extend(run_summaries)
        markdown += ["### Recursos por documento", ""] + _table(
            ["Recurso", "Média", "n", "Documentos", "Ausentes", "Cobertura", "Soma conhecida"],
            [(s["resource"], s["mean"], s["n"], n_docs, s["missing"], s["coverage"], s["known_sum"]) for s in run_summaries])
        run_branches = []
        for bid in sorted(set(run["branches"] or []) | {b["branch_id"] for b in local_branches}):
            rows = [b for b in local_branches if b["branch_id"] == bid]
            count = len(rows)
            status = Counter(b["execution_status"] if b["execution_status"] in STATUSES else "unknown" for b in rows)
            quality = Counter(b["quality_verdict"] if b["quality_verdict"] in VERDICTS else "unknown" for b in rows)
            own = sum(b["own_error"] for b in rows)
            prop = sum(b["propagated_error"] for b in rows)
            summary = {**base, "branch_id": bid, "observed": count,
                       "expected": n_docs if run["branches"] is not None else None,
                       "missing": n_docs - count if run["branches"] is not None else None,
                       **{s: status[s] for s in STATUSES},
                       **{f"quality_{q}": quality[q] for q in VERDICTS},
                       "revision_limit": sum(b["termination_reason"] == "revision_limit" for b in rows),
                       "own_errors": own, "own_error_rate": own / count if count else None,
                       "propagated_errors": prop, "propagated_error_rate": prop / count if count else None,
                       "attempt_failures": sum(b["attempt_failures"] for b in rows),
                       "final_operational_errors": sum(b["final_operational_error"] for b in rows)}
            run_branches.append(summary)
        branch_summaries.extend(run_branches)
        markdown += ["### Branches (esperadas por documento observado)", ""] + _table(
            ["Branch", "Observadas", "Esperadas", "Ausentes", *STATUSES],
            [(b["branch_id"], b["observed"], b["expected"], b["missing"], *[b[s] for s in STATUSES]) for b in run_branches])
        markdown += _table(["Branch", *VERDICTS, "revision_limit", "Erros próprios", "Taxa própria", "Impacto propagado", "Taxa propagada"],
                           [(b["branch_id"], *[b[f"quality_{q}"] for q in VERDICTS], b["revision_limit"], b["own_errors"], b["own_error_rate"], b["propagated_errors"], b["propagated_error_rate"]) for b in run_branches])
        markdown += _table(
            ["Branch", "Tentativas conhecidas", "Rejeições em tentativas", "Branches com histórico", "Observadas", "Falhas em tentativas", "Falhas finais"],
            [(b["branch_id"],
              sum(r["attempts"] for r in local_branches if r["branch_id"] == b["branch_id"] and r["attempts"] is not None),
              sum(r["rejected_attempts"] for r in local_branches if r["branch_id"] == b["branch_id"] and r["rejected_attempts"] is not None),
              sum(r["attempts"] is not None for r in local_branches if r["branch_id"] == b["branch_id"]),
              b["observed"], b["attempt_failures"], b["final_operational_errors"]) for b in run_branches])
        reason_rows = []
        for b in run_branches:
            observed = [r for r in local_branches if r["branch_id"] == b["branch_id"]]
            reasons = Counter(r["termination_reason"] if r["termination_reason"] in REASONS else "unknown" for r in observed)
            known = sum(reasons[r] for r in REASONS if r != "unknown")
            for reason in REASONS:
                reason_rows.append({**base, "branch_id": b["branch_id"], "termination_reason": reason,
                                    "n": reasons[reason], "observed": b["observed"],
                                    "known_reasons": known,
                                    "coverage": known / b["observed"] if b["observed"] else None})
        branch_reasons.extend(reason_rows)
        markdown += ["### Motivos finais por branch", ""] + _table(
            ["Branch", *REASONS, "Motivos conhecidos", "Observadas", "Cobertura motivos"],
            [(b["branch_id"], *[next(r["n"] for r in reason_rows if r["branch_id"] == b["branch_id"] and r["termination_reason"] == reason) for reason in REASONS],
              next(r["known_reasons"] for r in reason_rows if r["branch_id"] == b["branch_id"]),
              b["observed"], next(r["coverage"] for r in reason_rows if r["branch_id"] == b["branch_id"])) for b in run_branches])
        categories = Counter((f["scope"], f["stage"], f["category"]) for f in local_failures if f["event_count"])
        failure_summaries.extend(
            {**base, "scope": scope, "stage": stage, "category": category,
             "classification": "heuristic", "events": count, "total_documents": n_docs}
            for (scope, stage, category), count in sorted(categories.items()))
        markdown += ["### Eventos operacionais (classificação heurística)", ""] + _table(
            ["Escopo", "Estágio", "Categoria", "Eventos"], [(*key, n) for key, n in sorted(categories.items())])
        markdown += ["### Conclusões descritivas", ""]
        history_n = sum(b["attempts"] is not None for b in local_branches)
        history_incomplete = (not local_branches or history_n < len(local_branches)
                              or any(b["execution_status"] not in STATUSES[:-1] for b in local_branches)
                              or run["branches"] is None
                              or any(d["missing_branches"] for d in local_docs))
        markdown.append(f'- Histórico de tentativas disponível: {history_n}/{len(local_branches)} branches observadas.'
                        + (" Histórico incompleto ou sem observações; ausência de registro não comprova ausência de erros." if history_incomplete else ""))
        for field, label in (("own_errors", "erros próprios"), ("propagated_errors", "impacto propagado")):
            maximum = max((b[field] for b in run_branches), default=0)
            if maximum == 0:
                markdown.append(f"- Ranking {label}: nenhum erro registrado ({len(local_branches)} branches observadas).")
            else:
                winners = [b for b in run_branches if b[field] == maximum]
                labels = "; ".join(f'{b["branch_id"]}: {maximum}/{b["observed"]} observadas ({maximum / b["observed"]:.1%})' for b in winners)
                markdown.append(f'- Ranking {label} por contagem: {"empate — " if len(winners) > 1 else ""}{labels}.')
                rates = [(b[field] / b["observed"], b) for b in run_branches if b["observed"]]
                peak = max(rate for rate, _ in rates)
                top = [b for rate, b in rates if rate == peak]
                markdown.append(f'- Maior taxa de {label}: {"empate — " if len(top) > 1 else ""}' + "; ".join(
                    f'{b["branch_id"]}: {b[field]}/{b["observed"]} ({peak:.1%})' for b in top) + ".")
        propagated_docs = sum(d["analyzer_propagated"] for d in local_docs)
        markdown.append(f"- Analyzer propagado (heurística): {propagated_docs}/{n_docs} documentos; impacto em {sum(b['propagated_errors'] for b in run_branches)} branches, não falhas independentes.")
        markdown.append(f"- Eventos operacionais: {sum(f['event_count'] for f in local_failures)}; tentativas com falha: {sum(b['attempt_failures'] for b in local_branches)}; falhas finais de branch: {sum(b['final_operational_error'] for b in local_branches)} (não somar essas contagens).")
        markdown.append(f"- Limite de revisões: {sum(b['revision_limit'] for b in run_branches)} branches; rejeições de qualidade não são erros técnicos.")
        for s in run_summaries:
            markdown.append(f'- Cobertura {s["resource"]}: {s["n"]}/{n_docs} documentos; média restrita aos valores conhecidos.')
        markdown.append("")
    markdown += ["## Comparação conjunta entre runs (não pareada)", "",
                 "Modelo simplifier e run_id identificam cada execução. n e cobertura dos recursos referem-se aos documentos com valores conhecidos, não ao corpus completo.",
                 "Erros de branches são contagens registradas, separadas entre próprios e propagados. Cobertura do histórico informa a disponibilidade das listas de tentativas, não garante que todos os retries tenham sido registrados.", ""]
    if len(corpus_hashes) > 1:
        markdown += ["Atenção: corpus_hash diferentes entre runs. As amostras podem ser diferentes; esta comparação não é pareada e não permite atribuir diferenças ao modelo.", ""]
    if len(loaded) > 1 and any(not r["corpus_hash"] for r in provenance["runs"]):
        markdown += ["corpus_hash ausente em alguma run: identidade dos corpora não verificável.", ""]
    models_by_run = {}
    for run in provenance["runs"]:
        model = run["models"].get("simplifier", {})
        models_by_run[run["run_id"]] = (model.get("name") if isinstance(model, dict) else model) or "unknown"
    markdown += ["### Médias de recursos por documento", ""] + _table(
        ["Modelo", "run_id", "Recurso", "Média", "n", "Documentos", "Ausentes", "Cobertura"],
        [(models_by_run[s["run_id"]], s["run_id"], s["resource"], s["mean"], s["n"], s["total_documents"], s["missing"], s["coverage"]) for s in summaries])
    comparison_docs, comparison_branches = [], []
    for run in provenance["runs"]:
        rid = run["run_id"]
        docs = [d for d in documents if d["run_id"] == rid]
        bs = [b for b in branches if b["run_id"] == rid]
        n_docs, n_branches = len(docs), len(bs)
        known_status = sum(d["execution_status"] in STATUSES[:-1] for d in docs)
        comparison_docs.append((models_by_run[rid], rid, n_docs,
                                sum(d["execution_status"] == "partial" for d in docs),
                                sum(d["execution_status"] == "failed" for d in docs),
                                known_status, known_status / n_docs if n_docs else None))
        expected = n_docs * len(run["branches"]) if run["branches"] is not None else None
        history_n = sum(b["attempts"] is not None for b in bs)
        reason_n = sum(b["termination_reason"] in REASONS[:-1] for b in bs)
        comparison_branches.append((models_by_run[rid], rid, n_branches, expected,
                                    n_branches / expected if expected else None,
                                    sum(b["own_error"] for b in bs),
                                    sum(b["propagated_error"] for b in bs),
                                    sum(b["termination_reason"] == "revision_limit" for b in bs),
                                    history_n, history_n / n_branches if n_branches else None,
                                    reason_n, reason_n / n_branches if n_branches else None))
    markdown += ["### Documentos: status registrados", ""] + _table(
        ["Modelo", "run_id", "Documentos", "partial", "failed", "n status conhecido", "Cobertura status"], comparison_docs)
    markdown += ["### Branches: erros registrados e limite de revisões", ""] + _table(
        ["Modelo", "run_id", "Observadas", "Esperadas", "Cobertura branches", "Erros próprios", "Impacto propagado", "revision_limit", "n histórico", "Cobertura histórico", "n motivos", "Cobertura motivos"], comparison_branches)
    markdown += ["## Limitações", "", f'Avisos de leitura/validação: {len(provenance["warnings"])} (detalhes no provenance retornado).',
                 "Ausências históricas de timing não são zeros. Não se estima tempo sem backoff nem tokens ausentes.",
                 "Denominadores esperados de branches abrangem somente documentos presentes em results, não o corpus completo.", ""]

    def write_csv(name, rows, columns):
        with (output_dir / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else columns)
            writer.writeheader()
            writer.writerows(rows)

    common = ["run_id", "run_name", "document_id"]
    write_csv("execution_resources.csv", resources, common + list(RESOURCES) + [f"{k}_source" for k in RESOURCES])
    write_csv("execution_documents.csv", documents, common + ["execution_status", "termination_reason", "observed_branches", "expected_branches", "missing_branches", "analyzer_propagated"])
    write_csv("execution_branches.csv", branches, common + ["branch_id", "execution_status", "quality_verdict", "termination_reason", "attempts", "rejected_attempts", "attempt_failures", "final_operational_error", "own_error", "propagated_error"])
    write_csv("execution_failures.csv", failures, common + ["branch_id", "attempt_number", "scope", "execution_status", "termination_reason", "stage", "category", "classification", "analyzer_propagated", "event_count", "message_evidence"])
    write_csv("execution_resource_summary.csv", summaries, ["run_id", "run_name", "resource", "mean", "n", "total_documents", "missing", "coverage", "known_sum", "complete"])
    write_csv("execution_branch_summary.csv", branch_summaries, ["run_id", "run_name", "branch_id", "observed", "expected", "missing", *STATUSES, *[f"quality_{q}" for q in VERDICTS], "revision_limit", "own_errors", "own_error_rate", "propagated_errors", "propagated_error_rate", "attempt_failures", "final_operational_errors"])
    write_csv("execution_document_summary.csv", document_summaries,
              ["run_id", "run_name", "execution_status", "termination_reason", "n", "total_documents", "fraction"])
    write_csv("execution_failure_summary.csv", failure_summaries,
              ["run_id", "run_name", "scope", "stage", "category", "classification", "events", "total_documents"])
    write_csv("execution_branch_reasons.csv", branch_reasons,
              ["run_id", "run_name", "branch_id", "termination_reason", "n", "observed", "known_reasons", "coverage"])
    with (output_dir / "execution_report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(markdown))
    return provenance
