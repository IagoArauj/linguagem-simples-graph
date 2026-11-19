"""Análise conjunta, offline: python -m metrics.analyze --help."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics.analysis_data import load_analysis_data
from metrics.execution_report import write_execution_report

VERSIONS = ("original", "natural", "strong", "simple", "moderate", "aggressive")


def generate_plots(values, output_dir):
    from metrics.analysis_plots import generate_plots as render
    return render(values, output_dir)


def statistical_tests(values):
    from metrics.analysis_plots import statistical_tests as calculate
    return calculate(values)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP = ["run_id", "run_name", "version", "metric"]
# Hipóteses exploratórias do notebook, não indicadores universais de qualidade.
DIRECTION = {
    "simple_word_ratio": 1, "flesch": 1, "lsa_adj_mean": 1,
    "lsa_givenness_mean": 1, "hard_conjunctions_ratio": -1,
    "long_sentence_ratio": -1, "medium_long_sentence_ratio": -1,
    "lsa_adj_std": -1, "lsa_givenness_std": -1, "cross_entropy": -1,
    "dalechall_adapted": -1, "gunning_fox": -1,
}


def summarize(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Médias disponíveis e deltas estritamente pareados por ID/métrica."""
    data = values.copy()
    data["value"] = pd.to_numeric(data.value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    summary = data.groupby(GROUP, sort=True).value.agg(
        n="count", mean="mean", median="median", std="std",
        q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
    ).reset_index()
    baselines = data[(data.run_id == "corpus") & (data.version == "original")][["document_id", "metric", "value"]]
    paired = data[~data.version.isin(["original", "execution"])].merge(
        baselines.rename(columns={"value": "original_value"}),
        on=["document_id", "metric"], how="inner", validate="many_to_one",
    ).dropna(subset=["value", "original_value"])
    paired["delta"] = paired.value - paired.original_value
    paired["directional_delta"] = paired.delta * paired.metric.map(DIRECTION)
    delta_summary = paired.groupby(GROUP, sort=True).agg(
        n=("delta", "count"), original_mean=("original_value", "mean"),
        simplified_mean=("value", "mean"), delta_mean=("delta", "mean"),
        delta_median=("delta", "median"), delta_std=("delta", "std"),
        directional_delta_mean=("directional_delta", "mean"),
    ).reset_index()
    directional = paired.dropna(subset=["directional_delta"]).copy()
    directional["change"] = np.select(
        [directional.directional_delta > 1e-10, directional.directional_delta < -1e-10],
        ["mudanca_favoravel", "mudanca_desfavoravel"], default="sem_mudanca",
    )
    consistency = directional.groupby(GROUP + ["change"]).size().rename("count").reset_index()
    consistency["percentage"] = consistency["count"] / consistency.groupby(GROUP)["count"].transform("sum") * 100
    return summary, delta_summary, consistency


def markdown_means(summary: pd.DataFrame) -> str:
    """Uma tabela por métrica, com identificação completa e cobertura."""
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    sections = [
        "# Comparação das execuções — médias\n",
        "Médias aritméticas das observações finitas disponíveis, por run e versão. "
        "`n` pode variar: as médias entre modelos não necessariamente usam os mesmos documentos. "
        "Ausências não são zero. Veja `values.csv` para análises de coortes pareadas.\n",
        "O nome é o modelo do simplificador; o manifesto preserva os modelos de todos os papéis. "
        "A versão `execution` contém recursos por documento, não por ramo.\n",
    ]
    series = summary[["run_id", "run_name", "version"]].drop_duplicates()
    for metric, metric_rows in summary.groupby("metric", sort=True):
        # Exibe também grupos sem observações, sem sugerir que ausência vale zero.
        table = series.merge(metric_rows, on=["run_id", "run_name", "version"], how="left")
        sections.extend([f"\n## {escape(metric)}\n", "| Modelo / referência | Run ID | Versão | n | Média | Mediana | Desvio-padrão |",
                         "|---|---|---|---:|---:|---:|---:|"])
        for row in table.itertuples():
            cells = [escape(row.run_name), escape(row.run_id), escape(row.version), str(int(row.n)) if pd.notna(row.n) else "0"]
            cells += [f"{value:.4f}" if pd.notna(value) else "—" for value in (row.mean, row.median, row.std)]
            sections.append("| " + " | ".join(cells) + " |")
    return "\n".join(sections) + "\n"


def create_analysis(run_dirs: list[Path], *, output_dir: Path | None = None,
                    corpus_override: Path | None = None, corpus_metrics: Path | None = None,
                    approved_only: bool = False, plots: bool = True) -> Path:
    values, provenance = load_analysis_data(run_dirs, corpus_override, corpus_metrics, approved_only)
    if values.empty:
        raise ValueError("Nenhuma observação disponível para análise.")
    created_at = datetime.now(timezone.utc)
    output_root = (PROJECT_ROOT / "output").resolve()
    output = output_dir or output_root / "analyses" / f"{created_at:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    output = output.resolve()
    if not output.is_relative_to(output_root) or output == output_root:
        raise ValueError("A pasta de análise deve ficar dentro de output/.")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "analysis_schema_version": "1.0", "created_at": created_at.isoformat(), "status": "running",
        **provenance,
        "options": {"approved_only": approved_only, "plots": plots,
                            "execution_report": True, "execution_all_outcomes": True},
        "aggregation": "available finite observations; means per run/version/metric; paired deltas by document_id",
        "statistical_policy": "Friedman on complete IDs, flesch/gunning_fox; Wilcoxon after p<0.05; Holm per run/metric; exploratory",
        "direction_hypotheses": DIRECTION,
        "dependencies": {name: importlib.metadata.version(name) for name in ("pandas", "numpy", "matplotlib", "scipy")},
        "analysis_code": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("analyze.py", "analysis_data.py", "analysis_plots.py", "execution_report.py")},
    }

    def save_manifest():
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_manifest()
    try:
        operational = write_execution_report(run_dirs, output)
        manifest["execution_report"] = operational
        sources = {source["path"]: source["sha256"] for source in manifest["source_files"]}
        for source in operational["source_files"]:
            if source["path"] in sources:
                if sources[source["path"]] != source["sha256"]:
                    raise ValueError(f"Fonte mudou durante a análise: {source['path']}")
            else:
                manifest["source_files"].append(source)
                sources[source["path"]] = source["sha256"]
        manifest["warnings"].extend(warning for warning in operational["warnings"]
                                    if warning not in manifest["warnings"])
        for warning in provenance["warnings"]:
            tqdm.write(f"Aviso: {warning}")
        values.to_csv(output / "values.csv", index=False)
        summary, deltas, consistency = summarize(values)
        summary.to_csv(output / "means.csv", index=False)
        (output / "means.md").write_text(markdown_means(summary), encoding="utf-8")
        deltas.to_csv(output / "deltas.csv", index=False)
        consistency.to_csv(output / "consistency.csv", index=False)
        comparable = values[values.version.isin(VERSIONS)]
        unknown = sorted(set(values.version) - set(VERSIONS) - {"execution"})
        if unknown:
            manifest["warnings"].append(f"Versões fora do notebook só aparecem nas tabelas: {unknown}")
        friedman, wilcoxon = statistical_tests(comparable)
        friedman.to_csv(output / "friedman.csv", index=False)
        wilcoxon.to_csv(output / "wilcoxon_holm.csv", index=False)
        if plots:
            groups = [("combined", comparable)] + [
                (f"runs/{index:03d}", comparable[comparable.run_id.isin(["corpus", run["run_id"]])])
                for index, run in enumerate(provenance["runs"], start=1)
            ]
            for folder, frame in tqdm(groups, desc="Gerando gráficos", unit="conjunto"):
                destination = output / folder
                destination.mkdir(parents=True)
                generate_plots(frame, destination)
        for index, run in enumerate(manifest["runs"], start=1):
            run["plots_directory"] = f"runs/{index:03d}" if plots else None
        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = [str(path.relative_to(output)) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"]
        save_manifest()
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest()
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Gráficos e tabelas conjuntos das runs, sem inferência ou NILC-Metrix.")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True, help="Pastas de execução, cada uma com manifest.json.")
    parser.add_argument("--output-dir", type=Path, help="Nova pasta dentro de output/; padrão: output/analyses/<timestamp>_<id>.")
    parser.add_argument("--corpus", type=Path, help="Cópia local do corpus; SHA256 deve coincidir com os manifestos.")
    parser.add_argument("--corpus-metrics", type=Path, help="NILC humano/original; padrão: metrics.jsonl junto ao corpus.")
    parser.add_argument("--approved-only", action="store_true", help="Incluir somente candidatas aprovadas nas métricas textuais; relatório operacional sempre inclui todos os desfechos.")
    parser.add_argument("--no-plots", action="store_true", help="Gerar somente tabelas e testes estatísticos.")
    args = parser.parse_args()
    try:
        output = create_analysis(args.run_dir, output_dir=args.output_dir,
                                 corpus_override=args.corpus, corpus_metrics=args.corpus_metrics,
                                 approved_only=args.approved_only, plots=not args.no_plots)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(f"Análise salva em: {output}")


if __name__ == "__main__":
    main()
