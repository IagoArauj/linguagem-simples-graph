from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import friedmanchisquare, wilcoxon

from metrics import analysis_plots as plots


@pytest.fixture
def values() -> pd.DataFrame:
    """Six IDs, a single human corpus, and two models with different effects."""
    rows = []
    metrics = ("text_length", "flesch", "gunning_fox", "lsa_adj_mean",
               "simple_word_ratio", "bertscore_f1")
    for run_id, name, versions, multiplier in (
        ("corpus", "Corpus humano", plots.VERSIONS[:3], 1),
        ("model-a", "Modelo A", plots.PAIRED_VERSIONS[1:], 1),
        ("model-b", "Modelo B", plots.PAIRED_VERSIONS[1:], 2),
    ):
        for doc in range(6):
            for version in versions:
                level = plots.VERSIONS.index(version)
                for metric in metrics:
                    if metric == "bertscore_f1" and version == "original":
                        continue
                    baseline = {
                        "text_length": 1000 + 30 * doc,
                        "flesch": 30 + 2 * doc,
                        "gunning_fox": 20 + doc,
                        "lsa_adj_mean": .3 + .01 * doc,
                        "simple_word_ratio": .5,
                        "bertscore_f1": .9,
                    }[metric]
                    scale = {"text_length": -20, "flesch": 5, "gunning_fox": -1,
                             "lsa_adj_mean": .01, "simple_word_ratio": 0,
                             "bertscore_f1": -.01}[metric]
                    rows.append((run_id, name, f"doc-{doc}", version, metric,
                                 baseline + scale * level * multiplier))
    return pd.DataFrame(rows, columns=plots.COLUMNS).sample(frac=1, random_state=7)


def test_statistical_tests_match_scipy_and_correct_per_family(values):
    before = values.copy(deep=True)
    friedman, posthoc = plots.statistical_tests(values)
    assert tuple(friedman.columns) == plots.FRIEDMAN_COLUMNS
    assert tuple(posthoc.columns) == plots.WILCOXON_COLUMNS
    assert len(friedman) == 4
    assert len(posthoc) == 24
    assert set(friedman.status) == {"ok"}
    assert set(friedman.n_documents) == {6}
    for row in friedman.to_dict("records"):
        complete = plots._complete(plots._clean(values), row["run_id"], row["metric"])
        expected = friedmanchisquare(*complete.to_numpy().T)
        assert row["statistic"] == pytest.approx(expected.statistic)
        assert row["p_value"] == pytest.approx(expected.pvalue)
        assert row["kendall_w"] == pytest.approx(expected.statistic / 18)
        family = posthoc.loc[
                    (posthoc["run_id"] == row["run_id"]) & (posthoc["metric"] == row["metric"]), :
                ]
        assert len(family) == 6
        for pair in family.to_dict("records"):
            expected_pair = np.asarray(wilcoxon(
                complete[pair["version_a"]] - complete[pair["version_b"]]
            ), dtype=float)
            assert pair["p_value"] == pytest.approx(expected_pair[1])
        np.testing.assert_allclose(family.p_value_holm, plots._holm(family.p_value.tolist()))
        assert (family.p_value_holm >= family.p_value).all()
    pd.testing.assert_frame_equal(values, before)


def test_holm_known_values():
    np.testing.assert_allclose(plots._holm([.04, .001, .03, .8]), [.09, .004, .09, .8])


def test_missing_and_infinite_only_complete_ids(values):
    data = values.copy()
    data.loc[(data.run_id == "model-a") & (data.document_id == "doc-0") &
             (data.version == "simple") & (data.metric == "flesch"), "value"] = np.inf
    data = data[~((data.run_id == "model-a") & (data.document_id == "doc-1") &
                  (data.version == "aggressive") & (data.metric == "flesch"))]
    data.loc[(data.run_id == "corpus") & (data.document_id == "doc-2") &
             (data.version == "original") & (data.metric == "gunning_fox"), "value"] = np.nan
    friedman, _ = plots.statistical_tests(data)
    counts = friedman.set_index(["run_id", "metric"]).n_documents
    assert counts["model-a", "flesch"] == 4
    assert counts["model-b", "flesch"] == 6
    assert counts["model-a", "gunning_fox"] == 5
    assert counts["model-b", "gunning_fox"] == 5
    complete = plots._complete(plots._clean(data), "model-a", "flesch")
    assert set(complete.index) == {"doc-2", "doc-3", "doc-4", "doc-5"}


def test_same_run_baseline_and_no_significant_posthoc(values):
    data = values[values.run_id.isin(["corpus", "model-a"])].copy()
    data["run_id"], data["run_name"] = "local", "Local"
    data["value"] = data.document_id.map(lambda s: float(s[-1]))
    friedman, posthoc = plots.statistical_tests(data)
    assert set(friedman.status) == {"all_equal"}
    assert (friedman.p_value == 1).all()
    assert (friedman.kendall_w == 0).all()
    assert posthoc.empty
    assert len(plots._complete(plots._clean(data), "local", "flesch")) == 6


def test_nonsignificant_friedman_does_not_run_posthoc(values):
    data = values.copy()
    for doc in range(6):
        for i, version in enumerate(plots.PAIRED_VERSIONS):
            data.loc[(data.document_id == f"doc-{doc}") & (data.version == version), "value"] = (
                i if doc % 2 else 3 - i
            )
    friedman, posthoc = plots.statistical_tests(data)
    assert set(friedman.status) == {"ok"}
    assert (friedman.p_value >= .05).all()
    assert posthoc.empty


def test_all_equal_wilcoxon_pair(values):
    data = values.copy()
    data.loc[data.version == "moderate", "value"] = data.loc[
        data.version == "moderate", "value"
    ] - data.loc[data.version == "moderate", "metric"].map({
        "text_length": -20, "flesch": 5, "gunning_fox": -1,
        "lsa_adj_mean": .01, "simple_word_ratio": 0, "bertscore_f1": -.01,
    }) * data.loc[data.version == "moderate", "run_id"].map({"model-a": 1, "model-b": 2})
    _, posthoc = plots.statistical_tests(data)
    equal = posthoc[(posthoc.version_a == "simple") & (posthoc.version_b == "moderate")]
    assert len(equal) == 4
    assert set(equal.status) == {"all_equal"}
    assert (equal.p_value_holm == 1).all()


def test_empty_insufficient_and_csv_headers(values, tmp_path):
    empty = values.iloc[:0]
    friedman, posthoc = plots.statistical_tests(empty)
    for frame, columns in ((friedman, plots.FRIEDMAN_COLUMNS), (posthoc, plots.WILCOXON_COLUMNS)):
        assert frame.empty
        assert frame.to_csv(index=False).strip() == ",".join(columns)
    assert plots.generate_plots(empty, tmp_path) == []
    short = values[values.document_id.isin(["doc-0", "doc-1"])].copy()
    friedman, posthoc = plots.statistical_tests(short)
    assert set(friedman.status) == {"insufficient_data"}
    assert friedman.p_value.isna().all()
    assert posthoc.empty
    short["value"] = -np.inf
    assert plots.generate_plots(short, tmp_path) == []
    friedman, _ = plots.statistical_tests(short)
    assert (friedman.n_documents == 0).all()


def test_deltas_and_scatter_join_ids_not_indices(values):
    data = values.copy()
    data.index = [0] * len(data)
    data = data[~((data.run_id == "model-a") & (data.metric == "lsa_adj_mean") &
                  (data.document_id == "doc-0") & (data.version == "simple"))]
    deltas = plots._deltas(plots._clean(data))
    assert "bertscore_f1" not in set(deltas.metric)
    pairs = plots._scatter_pairs(deltas, "flesch")
    model = pairs[(pairs.run_id == "model-a") & (pairs.version == "simple")]
    assert set(model.document_id) == {f"doc-{i}" for i in range(1, 6)}
    np.testing.assert_allclose(model.x, 15)
    np.testing.assert_allclose(model.y, .03)
    assert not deltas.duplicated(["run_id", "document_id", "version", "metric"]).any()
    assert len(deltas[(deltas.run_id == "corpus") & (deltas.metric == "flesch")]) == 12


def test_shared_original_priority_with_local_fallback(values):
    local = values[(values.run_id == "corpus") & (values.version == "original")].copy()
    local["run_id"], local["run_name"], local["value"] = "model-a", "Modelo A", -100
    data = pd.concat([values, local], ignore_index=True)
    complete = plots._complete(plots._clean(data), "model-a", "flesch")
    assert complete.loc["doc-0", "original"] == 30
    data.loc[(data.run_id == "corpus") & (data.version == "original") &
             (data.metric == "flesch") & (data.document_id == "doc-0"), "value"] = np.nan
    complete = plots._complete(plots._clean(data), "model-a", "flesch")
    assert complete.loc["doc-0", "original"] == -100


def test_joint_robust_normalization_preserves_missing(values):
    data = plots._clean(values).reset_index(drop=True)
    missing_index = data.index[(data.metric == "lsa_adj_mean")][0]
    data.loc[missing_index, "value"] = np.nan
    norm = plots._normalize(data)
    assert pd.isna(norm.loc[missing_index, "value"])
    assert (norm.loc[norm.metric == "simple_word_ratio", "value"] == .5).all()
    for metric in ("flesch", "gunning_fox"):
        raw = data.loc[data.metric == metric, "value"]
        p5, p95 = raw.quantile([.05, .95])
        np.testing.assert_allclose(norm.loc[norm.metric == metric, "value"],
                                   ((raw - p5) / (p95 - p5)).clip(0, 1))
    assert norm.value.dropna().between(0, 1).all()


def test_violin_singleton_and_constant_fallback(monkeypatch):
    fig, ax = plt.subplots()
    def forbidden(*args, **kwargs):
        pytest.fail("KDE não deve receber singleton/constante")
    monkeypatch.setattr(ax, "violinplot", forbidden)
    try:
        plots._distribution(ax, [np.array([1.]), np.array([2., 2.]), np.array([])],
                            ["one", "constant", "missing"], "violin")
        assert len(ax.collections) == 2
    finally:
        plt.close(fig)


def test_radar_missing_axis_is_gap(values, tmp_path, monkeypatch):
    data = values[~((values.run_id == "model-a") & (values.version == "simple") &
                    (values.metric == "lsa_adj_mean"))]
    captured = []
    def capture(fig, output_dir, name, paths):
        if name == "radar_nilc_median":
            for line in fig.axes[0].lines:
                if "model-a" in line.get_label() and line.get_label().endswith("/ simple"):
                    captured.append(np.asarray(line.get_ydata()))
        plt.close(fig)
    monkeypatch.setattr(plots, "_save", capture)
    plots._radars(plots._clean(data), tmp_path, [])
    assert len(captured) == 1
    assert np.isnan(captured[0]).any()


def test_generate_comparative_pngs_no_human_replication(values, tmp_path, monkeypatch):
    original = values.copy(deep=True)
    observed = []
    distribution = plots._distribution
    def capture(ax, arrays, labels, kind):
        observed.append((labels, [len(a) for a in arrays]))
        return distribution(ax, arrays, labels, kind)
    monkeypatch.setattr(plots, "_distribution", capture)
    paths = plots.generate_plots(values, tmp_path)
    names = {Path(p).name for p in paths}
    assert {
        "values_lengths_box_01.png", "values_nilc_box_01.png",
        "values_nilc_violin_01.png", "values_semantic_other_box_01.png",
        "trajectories_nilc_01.png", "radar_nilc_median.png",
        "radar_nilc_iqr_001.png", "deltas_nilc_box_01.png",
        "deltas_nilc_violin_01.png", "scatter_delta_flesch_vs_lsa_adj_mean.png",
        "scatter_delta_gunning_fox_vs_lsa_adj_mean.png",
    } <= names
    assert len(names) == len(paths)
    for path in paths:
        assert Path(path).parent == tmp_path
        assert Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    labels, sizes = observed[0]
    assert len(labels) == 9  # Three human versions, three versions per model.
    assert sizes == [6] * 9
    assert len([label for label in labels if "Corpus humano" in label]) == 3
    assert len([n for n in names if n.startswith("radar_nilc_iqr_")]) == 9
    assert not plt.get_fignums()
    pd.testing.assert_frame_equal(values, original)


def test_generate_single_run_singleton_missing_and_constant(values, tmp_path):
    data = values[(values.document_id == "doc-0") &
                  values.run_id.isin(["corpus", "model-a"]) &
                  values.metric.isin(["flesch", "gunning_fox", "lsa_adj_mean"])].copy()
    data["run_id"], data["run_name"], data["value"] = "one", "One", 1.
    data.loc[data.version == "aggressive", "value"] = np.inf
    paths = plots.generate_plots(data, tmp_path)
    assert paths
    assert not any("trajectories" in p for p in paths)
    assert not plt.get_fignums()


def test_input_validation(values, tmp_path):
    with pytest.raises(ValueError, match="Colunas"):
        plots.generate_plots(values.drop(columns="metric"), tmp_path)
    with pytest.raises(ValueError, match="duplicadas"):
        plots.statistical_tests(pd.concat([values, values.iloc[:1]]))
    invalid = values.copy()
    invalid.iloc[0, invalid.columns.get_loc("version")] = "unknown"
    with pytest.raises(ValueError, match="Versões"):
        plots.generate_plots(invalid, tmp_path)
    with pytest.raises(NotADirectoryError):
        plots.generate_plots(values, tmp_path / "not-created")
