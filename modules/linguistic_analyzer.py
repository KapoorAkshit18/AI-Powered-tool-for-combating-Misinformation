"""
Linguistic Analyzer Module for the Misinformation Detection Tool.

Detects misinformation-associated linguistic patterns:
  - Clickbait indicators (ALL CAPS, excessive punctuation, superlatives)
  - Sensationalism scoring
  - Logical fallacy patterns (hasty generalization, false dichotomy, appeal to emotion)
  - Hedging vs. assertive language balance
"""

import re
from typing import Any


# ── Pattern Dictionaries ────────────────────────────────────────────

SUPERLATIVES: set[str] = {
    "best", "worst", "greatest", "biggest", "most", "least",
    "fastest", "strongest", "deadliest", "largest", "smallest",
    "ultimate", "absolute", "supreme", "perfect", "total",
}

HEDGING_PHRASES: list[str] = [
    "according to", "reportedly", "allegedly", "it is believed",
    "sources say", "experts suggest", "may", "might", "could",
    "possibly", "perhaps", "it appears", "seems to", "likely",
    "it is possible", "there are indications",
]

ASSERTIVE_PHRASES: list[str] = [
    "without a doubt", "undeniably", "certainly", "absolutely",
    "definitively", "guaranteed", "proven", "confirmed fact",
    "the truth is", "it is certain", "make no mistake",
    "there is no question", "everyone knows",
]

# Regex-based logical fallacy detectors
FALLACY_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "Hasty Generalization": [
        re.compile(r"\b(all|every|everyone|always|never|nobody)\b.*\b(is|are|does|do|will|has|have)\b", re.I),
        re.compile(r"\bpeople always\b", re.I),
        re.compile(r"\beveryone knows\b", re.I),
    ],
    "False Dichotomy": [
        re.compile(r"\beither\b.*\bor\b", re.I),
        re.compile(r"\byou('re| are) (either|with us or against)", re.I),
        re.compile(r"\bthere (is|are) only two (options|choices|ways)\b", re.I),
    ],
    "Appeal to Emotion": [
        re.compile(r"\bthink of the children\b", re.I),
        re.compile(r"\bhow (could|can|dare) (you|they|we)\b", re.I),
        re.compile(r"\bwon'?t someone think\b", re.I),
        re.compile(r"\b(innocent|helpless|vulnerable) (people|victims|children)\b", re.I),
    ],
}


# ── Helper functions ────────────────────────────────────────────────

def _count_caps_words(text: str) -> int:
    """Count words that are fully uppercase and at least 2 characters long."""
    return sum(1 for word in text.split() if word.isupper() and len(word) >= 2)


def _excessive_punctuation_score(text: str) -> float:
    """Return a 0-1 score based on repeated exclamation/question marks."""
    patterns = re.findall(r"[!?]{2,}", text)
    return min(len(patterns) / 5.0, 1.0)


def _superlative_count(text: str) -> list[str]:
    """Return superlative words found in the text."""
    tokens = set(text.lower().split())
    return sorted(tokens & SUPERLATIVES)


def _detect_fallacies(text: str) -> dict[str, bool]:
    """Check text against known logical fallacy patterns."""
    results: dict[str, bool] = {}
    for fallacy_name, patterns in FALLACY_PATTERNS.items():
        results[fallacy_name] = any(p.search(text) for p in patterns)
    return results


def _phrase_hits(text_lower: str, phrases: list[str]) -> list[str]:
    """Return which phrases appear in the lowercase text."""
    return [p for p in phrases if p in text_lower]


# ── Public API ──────────────────────────────────────────────────────

def analyze_linguistics(text: str) -> dict[str, Any]:
    """
    Analyse the linguistic features of *text* for misinformation signals.

    Args:
        text: The article or claim text to analyse.

    Returns:
        A dictionary containing:
            - ``clickbait_score``       (float):       0-100 clickbait likelihood.
            - ``caps_words``            (int):         Number of ALL-CAPS words.
            - ``excessive_punctuation`` (float):       0-1 punctuation score.
            - ``superlatives``          (list[str]):   Superlative words found.
            - ``sensationalism_score``  (float):       0-100 sensationalism score.
            - ``fallacies_detected``    (dict):        Logical fallacy flags.
            - ``hedging_phrases``       (list[str]):   Hedging phrases found.
            - ``assertive_phrases``     (list[str]):   Assertive phrases found.
            - ``language_balance``      (str):         "hedging", "assertive", or "balanced".
            - ``linguistic_credibility``(float):       Derived credibility sub-score [0, 100].
    """
    if not text or not text.strip():
        return {
            "clickbait_score": 0.0,
            "caps_words": 0,
            "excessive_punctuation": 0.0,
            "superlatives": [],
            "sensationalism_score": 0.0,
            "fallacies_detected": {k: False for k in FALLACY_PATTERNS},
            "hedging_phrases": [],
            "assertive_phrases": [],
            "language_balance": "balanced",
            "linguistic_credibility": 50.0,
        }

    text_lower = text.lower()
    word_count = max(len(text.split()), 1)

    # ── Clickbait Indicators ────────────────────────────────────────
    caps_words = _count_caps_words(text)
    punct_score = _excessive_punctuation_score(text)
    superlatives = _superlative_count(text)

    caps_ratio = min(caps_words / word_count * 10, 1.0)
    superlative_ratio = min(len(superlatives) / word_count * 15, 1.0)

    clickbait_score = (
        caps_ratio * 35
        + punct_score * 35
        + superlative_ratio * 30
    )
    clickbait_score = min(clickbait_score, 100.0)

    # ── Sensationalism ──────────────────────────────────────────────
    sensationalism_score = (clickbait_score * 0.6) + (punct_score * 40 * 0.4)
    sensationalism_score = min(sensationalism_score, 100.0)

    # ── Logical Fallacies ───────────────────────────────────────────
    fallacies = _detect_fallacies(text)
    fallacy_count = sum(fallacies.values())

    # ── Hedging vs. Assertive ───────────────────────────────────────
    hedging_hits = _phrase_hits(text_lower, HEDGING_PHRASES)
    assertive_hits = _phrase_hits(text_lower, ASSERTIVE_PHRASES)

    if len(assertive_hits) > len(hedging_hits) + 1:
        language_balance = "assertive"
    elif len(hedging_hits) > len(assertive_hits) + 1:
        language_balance = "hedging"
    else:
        language_balance = "balanced"

    # ── Credibility Sub-Score ───────────────────────────────────────
    credibility = 100.0
    credibility -= clickbait_score * 0.30        # up to −30
    credibility -= sensationalism_score * 0.20   # up to −20
    credibility -= fallacy_count * 10            # up to −30
    if language_balance == "assertive":
        credibility -= 10
    credibility = max(0.0, min(100.0, credibility))

    return {
        "clickbait_score": round(clickbait_score, 2),
        "caps_words": caps_words,
        "excessive_punctuation": round(punct_score, 4),
        "superlatives": superlatives,
        "sensationalism_score": round(sensationalism_score, 2),
        "fallacies_detected": fallacies,
        "hedging_phrases": hedging_hits,
        "assertive_phrases": assertive_hits,
        "language_balance": language_balance,
        "linguistic_credibility": round(credibility, 2),
    }
