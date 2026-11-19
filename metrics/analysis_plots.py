"""Análise exploratória de métricas LONG, sem imputação de valores.

Entrada: run_id, run_name, document_id, version, metric, value. As versões são
original/natural/strong/simple/moderate/aggressive. Humanos podem estar na mesma
run ou, em análises conjuntas, uma única vez em run_id='corpus'. Chaves duplicadas
são rejeitadas (não se faz média silenciosa). NaN e infinitos são ausências.

``generate_plots(values, output_dir)`` requer uma pasta existente e retorna os
caminhos dos PNGs efetivamente escritos, como strings. Usa Agg; não escreve CSV.
Distribuições e radares usam os valores disponíveis, sem replicar o corpus por
modelo. Trajetórias usam, por métrica/run, somente documentos com as quatro
versões original/simple/moderate/aggressive; deltas usam pares disponíveis por
ID. O original compartilhado tem prioridade sobre o local para a mesma chave.
Dispersões juntam deltas pelo ID, run e versão, nunca pela posição das linhas.

Radares: escala conjunta por métrica (p5/p95 de todas as runs/versões), recortada
em [0, 1]. Eixos constantes finitos ficam em 0.5; eixos ausentes continuam NaN.
Não há inversão de direção: maior raio NÃO significa melhor qualidade. Há um
radar comparativo de medianas e radares por série (run/versão) com mediana/IQR.
Com menos de três métricas disponíveis no grupo, o radar é omitido. Grupos são
NILC, semânticas/outras e comprimento; painéis têm até seis métricas.

``statistical_tests`` retorna (Friedman, Wilcoxon), DataFrames com colunas fixas
mesmo vazios, prontos para ``to_csv(index=False)``. Para flesch e gunning_fox,
Friedman exige >=3 IDs completos nas quatro versões, com Kendall W corrigido
pelos empates via estatística de Friedman / (n * (k - 1)). Se p < .05, testa
os seis pares nessa mesma coorte, Wilcoxon bilateral (zero_method='wilcox',
method='auto'), e aplica Holm separadamente por run/métrica. Sem diferenças,
registra p=1, estatística=0 e status='all_equal'; insuficiência/indefinição têm
status explícito e resultados NaN. O portão de Friedman não é corrigido entre
runs/métricas; a inferência é exploratória e limitada em amostras pequenas.

Dependências: pandas, numpy, matplotlib; scipy apenas para statistical_tests.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

VERSIONS = ("original", "natural", "strong", "simple", "moderate", "aggressive")
PAIRED_VERSIONS = ("original", "simple", "moderate", "aggressive")
NILC_METRICS = (
    "hard_conjunctions_ratio", "long_sentence_ratio", "medium_long_sentence_ratio",
    "simple_word_ratio", "lsa_adj_mean", "lsa_adj_std", "lsa_givenness_mean",
    "lsa_givenness_std", "cross_entropy", "brunet", "dalechall_adapted", "flesch",
    "gunning_fox",
)
COLUMNS = ("run_id", "run_name", "document_id", "version", "metric", "value")
FRIEDMAN_COLUMNS = (
    "run_id", "run_name", "metric", "n_documents", "n_versions", "statistic",
    "p_value", "kendall_w", "status",
)
WILCOXON_COLUMNS = (
    "run_id", "run_name", "metric", "version_a", "version_b", "n_documents",
    "statistic", "p_value", "p_value_holm", "reject_holm", "status",
)


def _clean(values: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(values.columns)
    if missing:
        raise ValueError(f"Colunas LONG ausentes: {sorted(missing)}")
    data = values.loc[:, list(COLUMNS)].copy()
    if data[list(COLUMNS[:-1])].isna().any().any():
        raise ValueError("Identificadores, nomes, versões e métricas não podem ser nulos")
    unknown = set(data.version) - set(VERSIONS)
    if unknown:
        raise ValueError(f"Versões desconhecidas: {sorted(unknown)}")
    if data.duplicated(["run_id", "document_id", "version", "metric"]).any():
        raise ValueError("Chaves LONG duplicadas")
    if (data.groupby("run_id").run_name.nunique() > 1).any():
        raise ValueError("Cada run_id deve ter um único run_name")
    data["value"] = np.asarray(pd.to_numeric(data["value"], errors="coerce"), dtype=float)
    data.loc[~np.isfinite(data.value), "value"] = np.nan
    return data


def _runs(data: pd.DataFrame) -> list:
    return sorted((r for r in data.run_id.unique() if r != "corpus"), key=str)


def _run_values(data: pd.DataFrame, run_id) -> pd.DataFrame:
    local = data.loc[data["run_id"] == run_id, :]
    shared = data.loc[(data["run_id"] == "corpus") & (data["version"] == "original"), :]
    if run_id == "corpus" or shared.empty:
        return local.copy()
    # Prefer a measured corpus baseline, but do not discard a valid local value
    # merely because the shared measurement is absent.
    originals = pd.concat([
        shared.dropna(subset=["value"]),
        local.loc[local["version"] == "original", :],
    ]).drop_duplicates(["document_id", "metric"], keep="first")
    return pd.concat([local.loc[local["version"] != "original", :], originals], ignore_index=True)


def _complete(data: pd.DataFrame, run_id, metric: str) -> pd.DataFrame:
    rows = _run_values(data, run_id)
    rows = rows[(rows.metric == metric) & rows.version.isin(PAIRED_VERSIONS)]
    return rows.pivot(index="document_id", columns="version", values="value").reindex(
        columns=PAIRED_VERSIONS
    ).dropna()


def _deltas(data: pd.DataFrame) -> pd.DataFrame:
    results = []
    for run_id in sorted(data.run_id.unique(), key=str):
        rows = _run_values(data, run_id)
        original = rows.loc[rows["version"] == "original", ["document_id", "metric", "value"]]
        candidates = data[(data.run_id == run_id) & (data.version != "original")]
        joined = candidates.merge(
            original.rename(columns={"value": "baseline"}),
            on=["document_id", "metric"], how="inner", validate="many_to_one",
        )
        joined["value"] = joined.value - joined.baseline
        results.append(joined[list(COLUMNS)])
    if not results:
        return data.iloc[:0].copy()
    return pd.concat(results, ignore_index=True).replace(
        {"value": {np.inf: np.nan, -np.inf: np.nan}}
    ).dropna(subset=["value"])


def _holm(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def statistical_tests(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman/Kendall W e pós-testes Wilcoxon/Holm; veja convenções no módulo."""
    from scipy.stats import friedmanchisquare, wilcoxon

    data = _clean(values)
    omnibus, posthoc = [], []
    for run_id in _runs(data):
        run_name = data.loc[data.run_id == run_id, "run_name"].iloc[0]
        for metric in ("flesch", "gunning_fox"):
            complete = _complete(data, run_id, metric)
            n = len(complete)
            record = {"run_id": run_id, "run_name": run_name, "metric": metric,
                      "n_documents": n, "n_versions": 4, "statistic": np.nan,
                      "p_value": np.nan, "kendall_w": np.nan, "status": "insufficient_data"}
            omnibus.append(record)
            if n < 3:
                continue
            matrix = complete.to_numpy()
            if np.all(matrix == matrix[:, :1]):
                record.update(statistic=0.0, p_value=1.0, kendall_w=0.0, status="all_equal")
                continue
            statistic, p_value = friedmanchisquare(*matrix.T)
            if not np.isfinite([statistic, p_value]).all():
                record["status"] = "undefined"
                continue
            record.update(statistic=float(statistic), p_value=float(p_value),
                          kendall_w=float(statistic / (n * 3)), status="ok")
            if p_value >= .05:
                continue
            family = []
            for a, b in combinations(PAIRED_VERSIONS, 2):
                differences = complete[a].to_numpy() - complete[b].to_numpy()
                if np.all(differences == 0):
                    statistic, p_value, status = 0.0, 1.0, "all_equal"
                else:
                    result = np.asarray(wilcoxon(
                        differences, alternative="two-sided", zero_method="wilcox", method="auto"
                    ), dtype=float)
                    statistic, p_value = float(result[0]), float(result[1])
                    status = "ok" if np.isfinite([statistic, p_value]).all() else "undefined"
                family.append({
                    "run_id": run_id, "run_name": run_name, "metric": metric, "version_a": a,
                    "version_b": b, "n_documents": n, "statistic": float(statistic),
                    "p_value": float(p_value), "p_value_holm": np.nan,
                    "reject_holm": False, "status": status,
                })
            # Do not reduce the correction family silently if a test is undefined.
            if all(np.isfinite(row["p_value"]) for row in family):
                for row, adjusted in zip(family, _holm([r["p_value"] for r in family])):
                    row.update(p_value_holm=float(adjusted), reject_holm=bool(adjusted < .05))
            posthoc.extend(family)
    return (pd.DataFrame(omnibus, columns=FRIEDMAN_COLUMNS),
            pd.DataFrame(posthoc, columns=WILCOXON_COLUMNS))


def _groups(data: pd.DataFrame) -> dict[str, list[str]]:
    available = set(data.loc[data.value.notna(), "metric"])
    return {
        "lengths": [m for m in ("text_length",) if m in available],
        "nilc": [m for m in NILC_METRICS if m in available],
        "semantic_other": sorted(available - set(NILC_METRICS) - {"text_length"}),
    }


def _series(data: pd.DataFrame) -> list[tuple]:
    pairs = data[["run_id", "version"]].drop_duplicates().itertuples(index=False, name=None)
    return sorted(pairs, key=lambda p: (str(p[0]), VERSIONS.index(p[1])))


def _label(data: pd.DataFrame, run_id, version: str | None = None) -> str:
    name = data.loc[data.run_id == run_id, "run_name"].iloc[0]
    label = f"{name} [{run_id}]"
    return f"{label} / {version}" if version else label


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _panels(metrics: list[str]):
    plt = _pyplot()
    columns = min(3, len(metrics))
    rows = (len(metrics) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    return fig, list(axes.flat)[:len(metrics)]


def _save(fig, output_dir: Path, name: str, paths: list[str]) -> None:
    plt = _pyplot()
    try:
        fig.tight_layout()
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        paths.append(str(path))
    finally:
        plt.close(fig)


def _distribution(ax, arrays: list[np.ndarray], labels: list[str], kind: str) -> None:
    for position, array in enumerate(arrays, 1):
        if not len(array):
            continue
        if kind == "box":
            ax.boxplot([array], positions=[position], widths=.6)
        elif len(array) > 1 and np.ptp(array) > 0:
            ax.violinplot([array], positions=[position], showmedians=True, showextrema=False)
        else:
            ax.scatter([position] * len(array), array, s=18)
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=65, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=.25)


def _distributions(data, output_dir, paths, prefix="values") -> None:
    series = _series(data)
    labels = [_label(data, *s) for s in series]
    for group, metrics in _groups(data).items():
        for start in range(0, len(metrics), 6):
            panel = metrics[start:start + 6]
            for kind in (("box",) if group == "lengths" else ("box", "violin")):
                fig, axes = _panels(panel)
                for ax, metric in zip(axes, panel):
                    arrays = [data.loc[(data.run_id == r) & (data.version == v) &
                                       (data.metric == metric), "value"].dropna().to_numpy()
                              for r, v in series]
                    _distribution(ax, arrays, labels, kind)
                    ax.set_title(metric)
                    if prefix == "deltas":
                        ax.axhline(0, color="grey", linestyle="--", linewidth=.8)
                        ax.set_ylabel("Δ versão − original (pares por ID)")
                _save(fig, output_dir, f"{prefix}_{group}_{kind}_{start // 6 + 1:02d}", paths)


def _trajectories(data, output_dir, paths) -> None:
    for group, metrics in _groups(data).items():
        paired = {m: {r: _complete(data, r, m) for r in _runs(data)} for m in metrics}
        metrics = [m for m in metrics if any(not d.empty for d in paired[m].values())]
        for start in range(0, len(metrics), 6):
            panel = metrics[start:start + 6]
            fig, axes = _panels(panel)
            for ax, metric in zip(axes, panel):
                for run_id, complete in paired[metric].items():
                    if complete.empty:
                        continue
                    q = complete.quantile([.25, .5, .75]).to_numpy()
                    line, = ax.plot(range(4), q[1], marker="o",
                                    label=f"{_label(data, run_id)} (n={len(complete)})")
                    ax.fill_between(range(4), q[0], q[2], alpha=.15, color=line.get_color())
                ax.set_xticks(range(4), PAIRED_VERSIONS, rotation=25)
                ax.set_title(metric)
                ax.set_ylabel("Mediana / IQR — IDs completos")
                ax.legend(fontsize=7)
            _save(fig, output_dir, f"trajectories_{group}_{start // 6 + 1:02d}", paths)


def _normalize(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    for metric, rows in data.groupby("metric", sort=False):
        finite = rows.value.dropna()
        if finite.empty:
            continue
        low, high = finite.quantile([.05, .95])
        mask = (data.metric == metric) & data.value.notna()
        if high == low:
            normalized.loc[mask, "value"] = .5
        else:
            normalized.loc[mask, "value"] = ((data.loc[mask, "value"] - low) / (high - low)).clip(0, 1)
    return normalized


def _radar_axes(metrics):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    ax.set_xticks(angles, metrics, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Escala conjunta p5/p95 (não orientada por qualidade)", pad=30)
    return fig, ax, np.r_[angles, angles[0]]


def _radars(data, output_dir, paths) -> None:
    normalized = _normalize(data)
    for group, metrics in _groups(data).items():
        if len(metrics) < 3:
            continue
        summaries = []
        for run_id, version in _series(data):
            rows = normalized[(normalized.run_id == run_id) & (normalized.version == version)]
            q = rows.groupby("metric").value.quantile([.25, .5, .75]).unstack().reindex(
                index=metrics, columns=[.25, .5, .75]
            )
            if q.notna().any().any():
                summaries.append((run_id, version, q))
        fig, ax, angles = _radar_axes(metrics)
        for run_id, version, q in summaries:
            median = q[.5].to_numpy()
            ax.plot(angles, np.r_[median, median[0]], marker=".", label=_label(data, run_id, version))
        ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=7)
        _save(fig, output_dir, f"radar_{group}_median", paths)
        for index, (run_id, version, q) in enumerate(summaries, 1):
            fig, ax, angles = _radar_axes(metrics)
            low, median, high = (np.r_[q[c].to_numpy(), q[c].iloc[0]] for c in (.25, .5, .75))
            ax.plot(angles, median, marker="o", label=_label(data, run_id, version))
            ax.fill_between(angles, low, high, alpha=.25, label="IQR")
            ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=7)
            _save(fig, output_dir, f"radar_{group}_iqr_{index:03d}", paths)


def _scatter_pairs(deltas: pd.DataFrame, x_metric: str) -> pd.DataFrame:
    keys = ["run_id", "run_name", "document_id", "version"]
    x = deltas.loc[deltas["metric"] == x_metric, keys + ["value"]].rename(columns={"value": "x"})
    y = deltas.loc[deltas["metric"] == "lsa_adj_mean", keys + ["value"]].rename(columns={"value": "y"})
    return x.merge(y, on=keys, how="inner", validate="one_to_one").dropna(subset=["x", "y"])


def generate_plots(values: pd.DataFrame, output_dir: Path) -> list[str]:
    """Escreve apenas PNGs aplicáveis e retorna seus caminhos; não cria a pasta."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    data = _clean(values)
    if not data.value.notna().any():
        return []
    plt = _pyplot()
    paths: list[str] = []
    _distributions(data, output_dir, paths)
    _trajectories(data, output_dir, paths)
    _radars(data, output_dir, paths)
    deltas = _deltas(data)
    _distributions(deltas, output_dir, paths, prefix="deltas")
    for metric in ("flesch", "gunning_fox"):
        pairs = _scatter_pairs(deltas, metric)
        if pairs.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for run_id, version in _series(pairs):
            rows = pairs[(pairs.run_id == run_id) & (pairs.version == version)]
            ax.scatter(rows.x, rows.y, alpha=.7, label=_label(pairs, run_id, version))
        ax.axhline(0, color="grey", linewidth=.8)
        ax.axvline(0, color="grey", linewidth=.8)
        ax.set_xlabel(f"Δ {metric} (versão − original)")
        ax.set_ylabel("Δ lsa_adj_mean (versão − original)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
        _save(fig, output_dir, f"scatter_delta_{metric}_vs_lsa_adj_mean", paths)
    return paths
