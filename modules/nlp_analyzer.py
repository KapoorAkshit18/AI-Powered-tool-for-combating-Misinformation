"""
NLP Analyzer Module for the Misinformation Detection Tool.

Uses TextBlob for:
  - Sentiment analysis (polarity and subjectivity)
  - Emotional / loaded language detection
  - Bias indicator scoring (one-sided language)

Returns a dictionary of scores and flags.
"""

import re
from textblob import TextBlob



# Words commonly associated with emotional manipulation in news
EMOTIONAL_WORDS: set[str] = {
    # Fear / outrage
    "shocking", "horrifying", "terrifying", "alarming", "outrageous",
    "devastating", "catastrophic", "nightmare", "panic", "crisis",
    "threat", "danger", "deadly", "fatal", "destroy", "annihilate",
    # Hype / exaggeration
    "incredible", "unbelievable", "amazing", "miracle", "breakthrough",
    "revolutionary", "stunning", "explosive", "bombshell", "unprecedented",
    "jaw-dropping", "mind-blowing", "game-changer",
    # Emotional manipulation
    "heartbreaking", "tragic", "disgusting", "shameful", "betrayal",
    "conspiracy", "coverup", "exposed", "corrupt", "evil",
    "radical", "extremist", "fanatic",
    # Urgency
    "urgent", "breaking", "emergency", "immediately", "now",
}

# Words that signal one-sided / biased framing
BIAS_WORDS: set[str] = {
    "always", "never", "everyone", "nobody", "all", "none",
    "obviously", "clearly", "undeniably", "certainly", "definitely",
    "proves", "exposed", "exposed", "admitted", "slammed",
    "destroyed", "obliterated", "owned", "crushed",
    "so-called", "alleged", "supposedly", "regime", "propaganda",
}


def _count_word_hits(text_lower: str, word_set: set[str]) -> list[str]:
    """Return the list of words from *word_set* that appear in the text."""
    # Strip punctuation and tokenize words
    tokens = set(re.findall(r"\b[a-z0-9_-]+\b", text_lower))
    return sorted(tokens & word_set)


def analyze_text(text: str) -> dict:
    """
    Run NLP analysis on the provided text.

    Args:
        text: The article or claim text to analyse.

    Returns:
        A dictionary containing:
            - ``polarity``          (float):  Sentiment polarity  [-1, 1].
            - ``subjectivity``      (float):  Subjectivity score  [0, 1].
            - ``sentiment_label``   (str):    Human-readable sentiment label.
            - ``emotional_words``   (list[str]): Emotional words found.
            - ``emotional_score``   (float):  Ratio of emotional words to total words [0, 1].
            - ``bias_words``        (list[str]): Bias indicator words found.
            - ``bias_score``        (float):  Ratio of bias words to total words [0, 1].
            - ``nlp_credibility``   (float):  Derived credibility component [0, 100].
    """
    if not text or not text.strip():
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment_label": "Neutral",
            "emotional_words": [],
            "emotional_score": 0.0,
            "bias_words": [],
            "bias_score": 0.0,
            "nlp_credibility": 50.0,
        }

    blob = TextBlob(text)
    polarity: float = blob.sentiment.polarity
    subjectivity: float = blob.sentiment.subjectivity

    # Sentiment label
    if polarity > 0.3:
        sentiment_label = "Positive"
    elif polarity < -0.3:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Word-level analysis
    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)

    emotional_hits = _count_word_hits(text_lower, EMOTIONAL_WORDS)
    emotional_score = min(len(emotional_hits) / word_count * 10, 1.0)  # amplified ratio, capped at 1

    bias_hits = _count_word_hits(text_lower, BIAS_WORDS)
    bias_score = min(len(bias_hits) / word_count * 10, 1.0)

    # ── Derive a credibility sub-score ──────────────────────────────
    # High subjectivity, extreme polarity, emotional/bias words → lower credibility.
    credibility = 100.0
    credibility -= subjectivity * 30           # up to −30
    credibility -= abs(polarity) * 20          # up to −20
    credibility -= emotional_score * 25        # up to −25
    credibility -= bias_score * 25             # up to −25
    credibility = max(0.0, min(100.0, credibility))

    return {
        "polarity": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "sentiment_label": sentiment_label,
        "emotional_words": emotional_hits,
        "emotional_score": round(emotional_score, 4),
        "bias_words": bias_hits,
        "bias_score": round(bias_score, 4),
        "nlp_credibility": round(credibility, 2),
    }
