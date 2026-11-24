"""Comparação offline Single-Agent × SIMPLA: python -m metrics.paired."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from metrics.paired_data import load_paired_data
from metrics.paired_plots import write_latex, write_visualizations

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ('simple', 'moderate', 'aggressive')
DIRECTIONS = dict(flesch='higher', gunning_fox='lower', dalechall_adapted='lower',
                  simple_word_ratio='higher', long_sentence_ratio='lower',
                  bertscore_f1='higher', sari_multi_reference='higher')
SEED = 2026
RESAMPLES = 10_000


def describe(x, y, rng):
    d = y - x
    n = len(d)
    nz = d[d != 0]
    result = dict.fromkeys('mean_single mean_multi mean_delta median_delta std_delta q1 q3 '
                           'ci_mean_low ci_mean_high ci_median_low ci_median_high '
                           'wilcoxon_w p_raw p_holm significant rank_biserial improved_pct '
                           'worsened_pct tied_pct bowley_skew p_sign_raw'.split())
    result.update(n_pairs=n, n_nonzero=len(nz), n_zero=n-len(nz),
                  tied_absolute_ranks=len(nz)-len(np.unique(abs(nz))),
                  test_status='no_pairs', bootstrap_status='no_pairs')
    if not n:
        return result
    q1, median, q3 = np.quantile(d, [.25, .5, .75])
    result.update(mean_single=float(x.mean()), mean_multi=float(y.mean()),
                  mean_delta=float(d.mean()), median_delta=float(median),
                  std_delta=float(d.std(ddof=1)) if n > 1 else None,
                  q1=float(q1), q3=float(q3),
                  bowley_skew=float((q3 + q1 - 2*median)/(q3-q1)) if q3 > q1 else None)
    if n == 1 or np.all(d == d[0]):
        bounds = [float(d[0])] * 4
        result['bootstrap_status'] = 'degenerate_constant'
    else:
        means, medians = [], []
        for start in range(0, RESAMPLES, 500):
            draws = d[rng.integers(n, size=(min(500, RESAMPLES-start), n))]
            means.extend(draws.mean(axis=1))
            medians.extend(np.median(draws, axis=1))
        bounds = [*np.quantile(means, [.025, .975]), *np.quantile(medians, [.025, .975])]
        result['bootstrap_status'] = 'percentile'
    for key, value in zip(('ci_mean_low', 'ci_mean_high', 'ci_median_low', 'ci_median_high'), bounds):
        result[key] = float(value)
    result['test_status'] = 'all_zero'
    if len(nz):
        test = wilcoxon(d, alternative='two-sided', zero_method='wilcox', method='auto')
        ranks = rankdata(abs(nz), method='average')
        result.update(wilcoxon_w=float(test.statistic), p_raw=float(test.pvalue),
                      test_status='tested', rank_biserial=float(np.dot(ranks, np.sign(nz))/ranks.sum()),
                      p_sign_raw=float(binomtest(int(np.sum(nz > 0)), len(nz), alternative='two-sided').pvalue))
    return result


def compare(models, ids, single, multi, approved):
    summary, pairs, excluded = [], [], []
    for analysis in ('main', 'approved_only'):
        for level in LEVELS:
            for metric, direction in DIRECTIONS.items():
                family = []
                for model in sorted(models):
                    group = dict(analysis=analysis, model=model, level=level, metric=metric, direction=direction)
                    x, y = [], []
                    n_single = n_multi = n_eligible = 0
                    for ident in ids:
                        a = single.get((model, ident, metric))
                        b = multi.get((model, level, ident, metric))
                        eligible = (model, level, ident) in approved
                        n_single += a is not None
                        n_multi += b is not None
                        n_eligible += b is not None and (analysis == 'main' or eligible)
                        row = dict(**group, document_id=ident, single=a, multi=b, multi_approved=eligible)
                        reasons = []
                        if a is None:
                            reasons.append('single_no_valid_metric')
                        if b is None:
                            reasons.append('multi_no_valid_metric')
                        elif analysis == 'approved_only' and not eligible:
                            reasons.append('multi_not_approved')
                        if reasons:
                            excluded.append(dict(**row, reason=';'.join(reasons)))
                        else:
                            x.append(a)
                            y.append(b)
                            pairs.append(dict(**row, delta=b-a))
                    x, y = np.asarray(x), np.asarray(y)
                    stats = describe(x, y, np.random.default_rng(SEED))
                    if len(x):
                        oriented = (y-x) * (1 if direction == 'higher' else -1)
                        stats.update(improved_pct=float(100*np.mean(oriented > 0)),
                                     worsened_pct=float(100*np.mean(oriented < 0)),
                                     tied_pct=float(100*np.mean(oriented == 0)))
                    family.append(dict(**group, n_corpus=len(ids), n_single_valid=n_single,
                                       n_multi_valid=n_multi, n_multi_eligible=n_eligible, **stats))
                tested = [r for r in family if r['p_raw'] is not None]
                if tested:
                    _, adjusted, _, _ = multipletests([r['p_raw'] for r in tested], alpha=.05, method='holm')
                    for row, p in zip(tested, adjusted):
                        row.update(p_holm=float(p), significant=bool(p < .05))
                for row in family:
                    row['holm_family_n'] = len(tested)
                summary.extend(family)
    return summary, pairs, excluded


METHOD = """# Single-Agent × SIMPLA — comparação pareada

Δ = Multi − Single. Os mesmos IDs entram nas duas médias de cada linha.
`higher` favorece Δ positivo, `lower` favorece Δ negativo. Para BERTScore F1,
“melhoria” significa maior similaridade ao original, não necessariamente melhor qualidade.
SARI usa as referências Natural e Strong conjuntamente.

- `main`: inclui ramos tecnicamente válidos encerrados por limite de revisão.
  `approved_only`: condicionada à aprovação do avaliador, sujeita a viés de seleção.
- `n_single_valid` e `n_multi_valid`: cobertura válida por métrica antes do filtro;
  `n_multi_eligible`: após o filtro; `n_pairs`: interseção. Sem imputação de falhas.
- Wilcoxon bilateral (`wilcox`, `auto`): H0 de simetria dos deltas em torno de zero.
  Zeros exatos não recebem postos; empates absolutos recebem postos médios.
  Grupos vazios ou só com zeros não têm teste nem efeito estimado.
- Para avaliar simetria, examine os boxplots e `bowley_skew` (assimetria quartílica,
  indefinida quando Q1=Q3). Assimetria forte enfraquece a interpretação do Wilcoxon
  como teste de localização; a medida descritiva não certifica simetria.
  `p_sign_raw` é um teste binomial bilateral exploratório, não corrigido, de
  equiprobabilidade de sinais entre deltas não nulos; testa outra hipótese.
- Holm controla o erro familiar entre os modelos com teste válido em cada
  (análise, nível, métrica), não globalmente entre métricas, níveis ou análises.
- IC 95% percentil, 10.000 reamostragens por documento, semente 2026.
  Os ICs são dos estimadores média/mediana, não de W, e não são simultâneos.
  Amostras constantes (incluindo n=1) têm intervalo degenerado, sem evidência
  de precisão populacional; o bootstrap pressupõe documentos independentes.
- `rank_biserial` usa postos dos |Δ| não nulos; o sinal segue Δ bruto,
  independentemente da direção favorável. Todos os percentuais incluem empates.
"""


def create_analysis(single_results, multi_runs, corpus, *, output_dir=None, plots=True):
    models, ids, single, multi, approved, provenance = load_paired_data(single_results, multi_runs, corpus)
    summary, pairs, excluded = compare(models, ids, single, multi, approved)
    now = datetime.now(timezone.utc)
    output = Path(output_dir or ROOT / 'output/analyses' / f'paired_{now:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}')
    output.mkdir(parents=True, exist_ok=False)
    for name, rows, fallback in [('paired_summary', summary, ()), ('paired_deltas', pairs, ('delta',)),
                                 ('paired_excluded', excluded, ('reason',))]:
        fields = list(rows[0]) if rows else ['analysis', 'model', 'level', 'metric', 'direction',
                                            'document_id', 'single', 'multi', 'multi_approved', *fallback]
        with (output / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / 'paired_method.md').write_text(METHOD, encoding='utf-8')
    write_latex(summary, output)
    if plots:
        write_visualizations(summary, pairs, output / 'plots')
    manifest = dict(created_at=now.isoformat(), provenance=provenance, bootstrap_seed=SEED,
                    bootstrap_resamples=RESAMPLES, bootstrap_method='percentile', models=models,
                    plots=plots, dependencies={k: version(k) for k in ('numpy','scipy','statsmodels','matplotlib')},
                    code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('paired.py','paired_data.py','paired_plots.py')},
                    artifacts=sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    for warning in provenance.get('warnings', []):
        print(f'Aviso: {warning}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-results', type=Path, nargs='+', required=True)
    parser.add_argument('--multi-run-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'input/PorSimplesSent/corpus.jsonl')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    try:
        output = create_analysis(args.single_results, args.multi_run_dir, args.corpus,
                                 output_dir=args.output_dir, plots=not args.no_plots)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Erro: {exc}\n')
    print(f'Análise pareada salva em: {output}')


if __name__ == '__main__':
    main()
