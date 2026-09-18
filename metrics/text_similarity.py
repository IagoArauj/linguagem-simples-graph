r"""Similaridade textual pura, sem modelos, stemmers ou recursos externos.

As duas métricas usam NFC + lower() e a regex Unicode ``\w+|[^\w\s]``:
sequências de letras (inclusive acentuadas), números e underscore formam
palavras; cada caractere não alfanumérico e não branco vira um token separado.
Hífens, apóstrofos, pontuação e símbolos são preservados como tokens. Não há
remoção de acentos, stopwords ou stemming. Não é a tokenização padrão inglesa
nem a normalização 13a do Evaluate; resultados só são comparáveis usando a
mesma tokenização.
"""

from collections import Counter
from collections.abc import Sequence
import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokeniza português/Unicode, normalizando equivalência canônica e caixa."""
    return _TOKEN_PATTERN.findall(unicodedata.normalize("NFC", text).lower())


def _ratio(numerator: float, denominator: float, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _f1(precision: float, recall: float) -> float:
    return _ratio(2 * precision * recall, precision + recall, empty=0.0)


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    """ROUGE-L por LCS de tokens, sem divisão em sentenças (não ROUGE-Lsum).

    Retorna precision=LCS/len(candidate), recall=LCS/len(reference) e F1
    harmônico (beta=1), todos em [0, 1]. Se qualquer texto não tiver tokens,
    retorna três zeros, inclusive quando ambos forem vazios. Repetições e
    pontuação participam da LCS. Memória O(min(C, R)), tempo O(C * R).
    Compare a candidata IA separadamente com natural e strong.
    """
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    longer, shorter = candidate_tokens, reference_tokens
    if len(longer) < len(shorter):
        longer, shorter = shorter, longer
    row = [0] * (len(shorter) + 1)
    for token in longer:
        diagonal = 0
        for index, other in enumerate(shorter, start=1):
            previous = row[index]
            row[index] = diagonal + 1 if token == other else max(row[index], row[index - 1])
            diagonal = previous

    precision = row[-1] / len(candidate_tokens)
    recall = row[-1] / len(reference_tokens)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1))


def _type_average_overlap(
    overlap: Counter[tuple[str, ...]], denominator: Counter[tuple[str, ...]]
) -> float:
    # Xu/evaluate macro-average por tipo, não razão das massas dos contadores.
    return _ratio(
        sum(overlap[gram] / count for gram, count in denominator.items()),
        len(denominator),
        empty=1.0,
    )


def sari(source: str, candidate: str, references: Sequence[str]) -> float:
    """SARI de uma sentença com múltiplas referências, escala [0, 100].

    Variante SARIsent/SARIgram de Xu usada pelo Hugging Face Evaluate, com
    correção do recall KEEP e convenção Evaluate 0/0=1. Não é média de SARIs
    calculados separadamente por referência nem agregação de contadores de
    um corpus. Para um corpus, pode-se fazer a média destes escores por item.
    A fonte é o original, a candidata é a IA e as referências são humanas
    (por exemplo, [natural, strong]).

    Definição exata para cada ordem n=1..4:
    * S e C são contadores de ocorrências de n-gramas da fonte e candidata,
      multiplicados por M (número de referências). R soma os contadores de
      todas as referências. Interseção usa mínimo; subtração trunca em zero.
    * KEEP: K=S&C, Kref=S&R, corretos=K&R. Precision é a média, por tipo
      em K, de corretos[g]/K[g]; recall=sum(corretos)/sum(Kref), isto é,
      razão das massas (correção Evaluate). Usa F1 de precision e recall.
    * DELETE: D=S-C, corretos=D-R. Usa somente precision: média por tipo
      em D de corretos[g]/D[g].
    * ADD usa conjuntos: A=tipos(C)-tipos(S), Aref=tipos(R)-tipos(S).
      Precision=|A&Aref|/|A|, recall=|A&Aref|/|Aref|; usa F1.
    O resultado é 100/12 vezes a soma de KEEP-F1, ADD-F1 e DELETE-precision
    nas quatro ordens. Ordens ausentes NÃO são descartadas ou reponderadas.

    Toda divisão com denominador zero vale 1 (F1 de P=R=0 vale 0).
    Assim, três textos vazios pontuam 100; textos curtos também recebem
    crédito nas ordens sem operações. Uma lista vazia de referências gera
    ValueError; referências individuais vazias são válidas e geram zero
    tokens (não o token vazio produzido por split(" ") no SARIsent original).
    Referência: https://github.com/huggingface/evaluate/blob/main/metrics/sari/sari.py
    Uma string
    passada no lugar da sequência de referências gera TypeError.
    """
    if isinstance(references, (str, bytes)):
        raise TypeError("references deve ser uma sequência de textos, não uma string")
    if not references:
        raise ValueError("references deve conter pelo menos uma referência")

    source_tokens = tokenize(source)
    candidate_tokens = tokenize(candidate)
    reference_tokens = [tokenize(reference) for reference in references]
    count = len(reference_tokens)
    score = 0.0
    for order in range(1, 5):
        source_counts = Counter({g: c * count for g, c in _ngrams(source_tokens, order).items()})
        candidate_counts = Counter({g: c * count for g, c in _ngrams(candidate_tokens, order).items()})
        reference_counts: Counter[tuple[str, ...]] = Counter()
        for tokens in reference_tokens:
            reference_counts.update(_ngrams(tokens, order))

        kept = source_counts & candidate_counts
        reference_kept = source_counts & reference_counts
        correct_kept = kept & reference_counts
        keep_f1 = _f1(
            _type_average_overlap(correct_kept, kept),
            _ratio(sum(correct_kept.values()), sum(reference_kept.values()), empty=1.0),
        )

        deleted = source_counts - candidate_counts
        delete_precision = _type_average_overlap(deleted - reference_counts, deleted)

        added = candidate_counts.keys() - source_counts.keys()
        reference_added = reference_counts.keys() - source_counts.keys()
        correct_added = len(added & reference_added)
        add_f1 = _f1(
            _ratio(correct_added, len(added), empty=1.0),
            _ratio(correct_added, len(reference_added), empty=1.0),
        )
        score += keep_f1 + add_f1 + delete_precision

    return score * (100.0 / 12.0)
