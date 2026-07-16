import re
from typing import Dict

import nltk

VOWELS = set("aeiouáéíóúâêîôûàèìòùãõäëïöü")
SENTENCE_TERMINALS = re.compile(r"[.!?]+")


def count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:()[]{}""''«»-")
    if not word:
        return 0
    prev_is_vowel = False
    groups = 0
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_is_vowel:
            groups += 1
        prev_is_vowel = is_vowel
    return max(groups, 1)


def count_sentences(text: str) -> int:
    spans = list(nltk.sent_tokenize(text.strip(), language="portuguese"))
    return max(len(spans), 1)


def count_words(text: str) -> list[str]:
    tokens = nltk.wordpunct_tokenize(text.lower())
    return [t for t in tokens if any(c.isalpha() for c in t)]


def flesch_reading_ease(text: str) -> float:
    sentences = count_sentences(text)
    words = count_words(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    fre = 206.835 - 1.015 * (n_words / sentences) - 84.6 * (syllables / n_words)
    return round(fre, 2)


def gunning_fog(text: str) -> float:
    sentences = count_sentences(text)
    words = count_words(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    fog = 0.4 * (n_words / sentences + 100 * (complex_words / n_words))
    return round(fog, 2)


def compute_readability(text: str) -> Dict[str, float]:
    return {
        "flesch": flesch_reading_ease(text),
        "gunning_fog": gunning_fog(text),
    }
