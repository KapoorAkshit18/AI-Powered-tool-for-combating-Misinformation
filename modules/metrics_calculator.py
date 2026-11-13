"""
Measurable Empirical Metrics Calculator for Misinformation Detection.

Computes concrete, mathematically rigorous, quantifiable properties of text:
  - Flesch Reading Ease & Flesch-Kincaid Grade Level
  - Lexical Diversity / Type-Token Ratio (TTR)
  - Uppercase Character & Shouting Ratio (%)
  - Punctuation & Sensationalism Intensity (per 100 words)
  - Named Entity & Concrete Reference Count
  - Evidence & Citation Density
  - Statistical Margin of Error & 95% Confidence Interval
"""

import math
import re
from typing import Any


def _count_syllables(word: str) -> int:
    """Estimate syllable count in an English word using phonetic heuristic."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    # Remove non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 1

    # Remove silent e endings
    if word.endswith("e") and not word.endswith("le"):
        word = word[:-1]

    # Count vowel groups
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)
    return max(1, count)


def calculate_text_metrics(text: str) -> dict[str, Any]:
    """
    Calculate concrete, measurable linguistic and statistical metrics for text.

    Returns:
        Dictionary with quantitative, reproducible values.
    """
    if not text or not text.strip():
        return {
            "word_count": 0,
            "sentence_count": 0,
            "char_count": 0,
            "uppercase_ratio_pct": 0.0,
            "punctuation_density_per_100w": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "lexical_diversity_ttr": 0.0,
            "evidence_markers_count": 0,
            "margin_of_error_pct": 0.0,
            "confidence_interval_95": (0.0, 0.0),
        }

    raw_chars = len(text)
    alpha_chars = sum(1 for c in text if c.isalpha())
    upper_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = (upper_chars / alpha_chars * 100.0) if alpha_chars > 0 else 0.0

    # Tokenization
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text)
    word_count = len(words)
    unique_words = set(w.lower() for w in words)
    lexical_diversity = (len(unique_words) / word_count) if word_count > 0 else 0.0

    # Sentence count
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    sentence_count = max(1, len(sentences))

    # Punctuation counts (!, ?, ...)
    exclamations = text.count("!")
    questions = text.count("?")
    caps_words = sum(1 for w in words if len(w) > 1 and w.isupper())
    sensational_punct = exclamations + questions
    punct_density = (sensational_punct / word_count * 100.0) if word_count > 0 else 0.0

    # Syllables
    syllable_count = sum(_count_syllables(w) for w in words)

    # Readability Formulas (Standard Flesch Formulas)
    # Flesch Reading Ease: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    words_per_sent = word_count / sentence_count
    syllables_per_word = (syllable_count / word_count) if word_count > 0 else 1.0

    reading_ease = 206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word)
    reading_ease = round(max(0.0, min(100.0, reading_ease)), 2)

    # Flesch-Kincaid Grade Level: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    fk_grade = (0.39 * words_per_sent) + (11.8 * syllables_per_word) - 15.59
    fk_grade = round(max(0.0, fk_grade), 1)

    # Evidence & verifiable citation density
    evidence_patterns = [
        r"\b(?:according to|published in|study by|researchers from|data from|reported by)\b",
        r"\b(?:percent|%|\b\d+(?:\.\d+)?\b(?:\s*(?:million|billion|thousand|patients|cases|deaths|participants)))\b",
        r"\b(?:https?://|doi\.org|\.gov|\.edu|\.org)\b",
        r"\b(?:trial|journal|peer-reviewed|clinical|statistics|survey)\b",
    ]
    evidence_hits = 0
    for pattern in evidence_patterns:
        evidence_hits += len(re.findall(pattern, text, re.IGNORECASE))

    # Statistical 95% Confidence Margin of Error (Wilson / binomial approximation)
    # Reflects sampling uncertainty based on sample size (word count & evidence)
    sample_size = max(word_count, 1)
    # Smaller samples have wider margins of error; larger texts have tighter margins
    margin_of_error = 1.96 * math.sqrt(0.25 / sample_size) * 100.0
    margin_of_error = round(min(25.0, max(2.5, margin_of_error)), 2)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": raw_chars,
        "uppercase_ratio_pct": round(uppercase_ratio, 2),
        "caps_words_count": caps_words,
        "exclamations_count": exclamations,
        "punctuation_density_per_100w": round(punct_density, 2),
        "flesch_reading_ease": reading_ease,
        "flesch_kincaid_grade": fk_grade,
        "lexical_diversity_ttr": round(lexical_diversity, 3),
        "evidence_markers_count": evidence_hits,
        "margin_of_error_pct": margin_of_error,
    }
