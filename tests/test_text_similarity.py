import unicodedata

import pytest

from metrics.text_similarity import rouge_l, sari, tokenize


def test_unicode_tokenization_preserves_accents_and_separates_punctuation() -> None:
    text = "AÇÃO, órgão! D'água — dá-se R$ 1.234,50; art_2 § 😀"
    assert tokenize(text) == [
        "ação", ",", "órgão", "!", "d", "'", "água", "—", "dá", "-", "se",
        "r", "$", "1", ".", "234", ",", "50", ";", "art_2", "§", "😀",
    ]
    assert tokenize(unicodedata.normalize("NFD", text)) == tokenize(text)
    assert tokenize("  \n\t") == []
    assert tokenize("ações ação action actions") == ["ações", "ação", "action", "actions"]


@pytest.mark.parametrize(
    ("candidate", "reference", "precision", "recall", "f1"),
    [
        ("a b c", "a b c", 1, 1, 1),
        ("a b", "a x b", 1, 2 / 3, 4 / 5),
        ("a x b", "a b", 2 / 3, 1, 4 / 5),
        ("a b c", "c b a", 1 / 3, 1 / 3, 1 / 3),
        ("a a b", "a b b", 2 / 3, 2 / 3, 2 / 3),
        ("a", "b", 0, 0, 0),
        ("", "", 0, 0, 0),
        (" \n", "ação", 0, 0, 0),
        ("ação", "", 0, 0, 0),
        ("AÇÃO!", "ação", 1 / 2, 1, 2 / 3),
        ("ação", "acao", 0, 0, 0),
        ("a\nb", "a b", 1, 1, 1),
    ],
)
def test_rouge_l_known_lcs(candidate, reference, precision, recall, f1) -> None:
    assert rouge_l(candidate, reference) == pytest.approx(
        {"precision": precision, "recall": recall, "f1": f1}
    )


def test_rouge_l_unicode_canonical_equivalence() -> None:
    assert rouge_l("AÇÃO do ÓRGÃO.", unicodedata.normalize("NFD", "ação do órgão.")) == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }


@pytest.mark.parametrize("text", ["", "a", "a b", "a b c", "A decisão do órgão é válida."])
def test_sari_perfect_copy_when_references_agree(text: str) -> None:
    assert sari(text, text, [text, text]) == pytest.approx(100)


def test_sari_copy_is_penalized_when_reference_simplifies() -> None:
    # KEEP F1 por ordem: 4/5, 2/3, 0, 0; ADD e DELETE = 1.
    assert sari("a a a b", "a a a b", ["a b"]) == pytest.approx(710 / 9)
    assert sari("a a a b", "a b", ["a b"]) > sari("a a a b", "a a a b", ["a b"])


def test_sari_occurrence_counts_and_delete_precision() -> None:
    # DELETE unigram a: D[a]=1, R[a]=1 -> zero, apesar de C=R.
    # Demais operações/ordens = 1; distingue contadores de conjuntos.
    assert sari("a a b", "a b", ["a b"]) == pytest.approx(275 / 3)


def test_sari_multiple_references_are_pooled_not_scored_separately() -> None:
    # Unigram KEEP: P=1/2, R=1, F1=2/3; ADD=0; DELETE=1.
    expected = 800 / 9
    assert sari("a", "a", ["a", "b"]) == pytest.approx(expected)
    pairwise_average = (sari("a", "a", ["a"]) + sari("a", "a", ["b"])) / 2
    assert expected != pytest.approx(pairwise_average)
    assert sari("a", "a", ["a", "a", "b"]) == pytest.approx(90)


def test_sari_keep_recall_uses_reference_mass_not_type_average() -> None:
    # Unigram KEEP: P=1, R=2/3 (não 1/2), F1=4/5; DELETE=1/2.
    # Bigram KEEP=0, DELETE=1/2; ADD=1 em todas as ordens.
    assert sari("a b", "a", ["a", "a b"]) == pytest.approx(245 / 3)


def test_sari_add_uses_union_of_reference_types() -> None:
    # ADD unigram F1=2/3 (a vs {a,b}), bigram F1=0 (aa vs vazio).
    assert sari("", "a a", ["a", "b"]) == pytest.approx(800 / 9)


def test_sari_evaluate_known_example() -> None:
    # Exemplo público de evaluate/sari; sua pontuação usa DELETE precision.
    assert sari(
        "About 95 species are currently accepted .",
        "About 95 you now get in .",
        [
            "About 95 species are currently known .",
            "About 95 species are now accepted .",
            "95 species are now accepted .",
        ],
    ) == pytest.approx(26.953601953601954)


@pytest.mark.parametrize(
    ("source", "candidate", "references", "expected"),
    [
        ("", "", [""], 100),
        ("a", "", [""], 100),
        ("", "a", ["a"], 100),
        ("", "", ["a"], 275 / 3),
        ("", "a", [""], 275 / 3),
        ("a", "", ["a"], 250 / 3),
        ("a", "a", [""], 275 / 3),
        ("", "", ["", "a"], 275 / 3),
        ("a b c d", "a b c d", [""], 200 / 3),
    ],
)
def test_sari_empty_conventions_and_fixed_four_orders(source, candidate, references, expected) -> None:
    assert sari(source, candidate, references) == pytest.approx(expected)


def test_sari_requires_at_least_one_reference() -> None:
    with pytest.raises(ValueError, match="pelo menos uma"):
        sari("a", "a", [])


@pytest.mark.parametrize("references", ["texto", "", b"texto"])
def test_sari_rejects_bare_string_references(references) -> None:
    with pytest.raises(TypeError, match="sequência"):
        sari("a", "a", references)


def test_sari_unicode_and_punctuation() -> None:
    source = "A DECISÃO do ÓRGÃO é válida!"
    candidate = "A decisão é válida."
    references = ["A decisão é válida.", "A decisão vale."]
    expected = sari(source, candidate, references)
    assert sari(
        unicodedata.normalize("NFD", source.lower()),
        unicodedata.normalize("NFD", candidate.upper()),
        [unicodedata.normalize("NFD", ref.upper()) for ref in references],
    ) == pytest.approx(expected)
    assert sari("ação", "ação", ["ação"]) > sari("ação", "acao", ["ação"])
    assert sari("ação!", "ação!", ["ação!"]) > sari("ação!", "ação", ["ação!"])


def test_sari_reference_order_duplication_and_input_immutability() -> None:
    references = ["O órgão decidiu.", "Houve uma decisão."]
    original = references.copy()
    source, candidate = "O órgão proferiu uma decisão.", "O órgão decidiu."
    expected = sari(source, candidate, references)
    assert sari(source, candidate, tuple(reversed(references))) == pytest.approx(expected)
    assert sari(source, candidate, references * 2) == pytest.approx(expected)
    assert references == original


def test_metric_ranges_on_small_exhaustive_inputs() -> None:
    texts = ["", "a", "b", "a a", "a b", "b a", "a b a b", "ação!"]
    for source in texts:
        for candidate in texts:
            for reference in texts:
                assert 0 <= sari(source, candidate, [reference, source]) <= 100
                assert all(0 <= value <= 1 for value in rouge_l(candidate, reference).values())
