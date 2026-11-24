import numpy as np
import pytest
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from metrics.paired import SEED, compare, describe


def test_wilcoxon_and_signed_effect_with_ties_and_zeros():
    x = np.zeros(4)
    y = np.array([0., -1., 1., 3.])
    stats = describe(x, y, np.random.default_rng(SEED))
    expected = wilcoxon(y, alternative='two-sided', zero_method='wilcox', method='auto')
    assert stats['wilcoxon_w'] == expected.statistic
    assert stats['p_raw'] == expected.pvalue
    assert stats['rank_biserial'] == .5
    assert stats['n_nonzero'] == 3
    assert stats == describe(x, y, np.random.default_rng(SEED))


def test_degenerate_and_empty_samples():
    zero = describe(np.ones(2), np.ones(2), np.random.default_rng(SEED))
    assert zero['test_status'] == 'all_zero'
    assert zero['p_raw'] is zero['rank_biserial'] is None
    assert zero['ci_mean_low'] == zero['ci_median_high'] == 0
    empty = describe(np.array([]), np.array([]), np.random.default_rng(SEED))
    assert empty['test_status'] == 'no_pairs'
    assert empty['ci_mean_low'] is None


def test_pairs_coverage_direction_and_holm_by_family():
    ids = list('abcdefg')
    models = ['a', 'b']
    metric = 'gunning_fox'
    single = {(m, i, metric): 10. for m in models for i in ids[:-1]}
    multi = {(m, 'simple', i, metric): 9. if m == 'a' or i != 'a' else 11.
             for m in models for i in ids}
    approved = {(m, 'simple', i) for m in models for i in ids[:2]}
    summary, pairs, excluded = compare(models, ids, single, multi, approved)
    rows = [r for r in summary if r['metric'] == metric and r['level'] == 'simple']
    main = [r for r in rows if r['analysis'] == 'main']
    sensitivity = [r for r in rows if r['analysis'] == 'approved_only']
    assert [r['n_pairs'] for r in main] == [6, 6]
    assert [r['n_pairs'] for r in sensitivity] == [2, 2]
    assert main[0]['n_multi_valid'] == 7
    assert main[0]['improved_pct'] == 100 and main[0]['rank_biserial'] == -1
    assert [r['p_holm'] for r in main] == pytest.approx(
        multipletests([r['p_raw'] for r in main], method='holm')[1])
    assert any(r['document_id'] == 'g' and r['reason'] == 'single_no_valid_metric' for r in excluded)
    assert any(r['reason'] == 'multi_not_approved' for r in excluded)
    assert any(r['delta'] == -1 for r in pairs)
