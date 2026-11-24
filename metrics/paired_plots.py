"""Gráficos e tabela dos resultados da comparação pareada multi − single."""

import math
from collections import defaultdict
from pathlib import Path


FOREST_METRICS = {
    "flesch": "Flesch",
    "sari_multi_reference": "SARI (múltiplas referências)",
    "bertscore_f1": "BERTScore F1",
}
GROUP_FIELDS = ("analysis", "metric", "model", "level")
TABLE_FIELDS = (
    "analysis", "model", "level", "metric", "n_pairs", "mean_delta",
    "ci_mean_low", "ci_mean_high", "wilcoxon_w", "p_raw", "p_holm",
    "rank_biserial", "significant", "direction",
)
LATEX_ESCAPE = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "&": r"\&",
    "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _group(row):
    return tuple(row[field] for field in GROUP_FIELDS)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _setup_axes(plt, analysis, metric, labels, direction, xlabel):
    fig, ax = plt.subplots(figsize=(9, max(2.5, 1.7 + 0.38 * len(labels))), layout="constrained")
    ax.set_title(f"{FOREST_METRICS.get(metric, metric)} — {analysis}", parse_math=False)
    side = {"higher": "direita (Δ > 0)", "lower": "esquerda (Δ < 0)"}.get(direction)
    ax.set_xlabel(f"{xlabel} (multi − single)\nFavorável a multi: {side or 'não definido'}")
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set_yticks(range(1, len(labels) + 1), labels, parse_math=False)
    ax.set_ylim(len(labels) + 0.5, 0.5)
    return fig, ax


def write_visualizations(summary: list[dict], pairs: list[dict], output: Path) -> None:
    """Grava os PNGs em ``output``, omitindo painéis sem dados finitos."""
    output.mkdir(parents=True, exist_ok=True)
    forests = defaultdict(list)
    boxes = defaultdict(lambda: defaultdict(list))
    directions = defaultdict(set)
    for row in sorted(summary, key=_group):
        analysis, metric, model, level = _group(row)
        key = analysis, metric
        directions[key].add(row["direction"])
        mean = _finite(row["mean_delta"])
        if metric in FOREST_METRICS and mean is not None:
            forests[key].append((f"{model} / {level} (n={row['n_pairs']})", mean,
                                 _finite(row["ci_mean_low"]), _finite(row["ci_mean_high"])))
    for row in pairs:
        analysis, metric, model, level = _group(row)
        delta = _finite(row["delta"])
        if delta is not None:
            boxes[analysis, metric][model, level].append(delta)
    if not forests and not boxes:
        return

    from inspect import signature

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def direction_for(key):
        values = directions[key]
        return next(iter(values)) if len(values) == 1 else None

    for (analysis, metric), rows in sorted(forests.items()):
        fig, ax = _setup_axes(plt, analysis, metric, [row[0] for row in rows],
                              direction_for((analysis, metric)), "Δ médio (ponto) e IC (barra)")
        try:
            for position, (_, mean, low, high) in enumerate(rows, start=1):
                if low is not None and high is not None:
                    ax.hlines(position, low, high, color="C0")
                ax.plot(mean, position, "o", color="C0")
            fig.savefig(output / f"forest_{analysis}_{metric}.png", dpi=150)
        finally:
            plt.close(fig)

    for (analysis, metric), groups in sorted(boxes.items()):
        labels, values = zip(*((f"{model} / {level} (n={len(deltas)})", deltas)
                               for (model, level), deltas in sorted(groups.items())))
        fig, ax = _setup_axes(plt, analysis, metric, labels,
                              direction_for((analysis, metric)), "Δ bruto")
        try:
            # Matplotlib 3.9 uses vert; newer versions prefer orientation.
            horizontal = ({"orientation": "horizontal"} if "orientation" in signature(ax.boxplot).parameters
                          else {"vert": False})
            ax.boxplot(values, manage_ticks=False, **horizontal)
            fig.savefig(output / f"boxplot_{analysis}_{metric}.png", dpi=150)
        finally:
            plt.close(fig)


def _latex_cell(value):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "--"
    if isinstance(value, float):
        return format(value, ".6g")
    return "".join(LATEX_ESCAPE.get(char, char) for char in str(value))


def write_latex(summary: list[dict], output: Path) -> None:
    """Grava ``paired_tests.tex``, um fragmento LaTeX com todos os testes."""
    output.mkdir(parents=True, exist_ok=True)
    end_row = r" \\"
    lines = [r"\begin{tabular}{llllrrrrrrrrll}", r"\hline",
             " & ".join(map(_latex_cell, TABLE_FIELDS)) + end_row, r"\hline"]
    for row in sorted(summary, key=_group):
        lines.append(" & ".join(_latex_cell(row[field]) for field in TABLE_FIELDS) + end_row)
    lines.extend([r"\hline", r"\end{tabular}"])
    (output / "paired_tests.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
