"""
Source Credibility Module for the Misinformation Detection Tool.

Assesses the credibility of a news source domain by looking it up
against curated lists of reliable and unreliable sources stored in
``data/reliable_sources.json`` and ``data/unreliable_sources.json``.
"""

import json
from pathlib import Path
from typing import Optional

# Resolve paths relative to the project root (one level up from modules/)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_RELIABLE_PATH = _DATA_DIR / "reliable_sources.json"
_UNRELIABLE_PATH = _DATA_DIR / "unreliable_sources.json"

# Module-level caches (loaded on first use)
_reliable_sources: Optional[dict[str, str]] = None
_unreliable_sources: Optional[dict[str, str]] = None


def _load_sources() -> tuple[dict[str, str], dict[str, str]]:
    """Load and cache the reliable / unreliable source lists."""
    global _reliable_sources, _unreliable_sources

    if _reliable_sources is None:
        try:
            with open(_RELIABLE_PATH, "r", encoding="utf-8") as fh:
                _reliable_sources = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            _reliable_sources = {}

    if _unreliable_sources is None:
        try:
            with open(_UNRELIABLE_PATH, "r", encoding="utf-8") as fh:
                _unreliable_sources = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            _unreliable_sources = {}

    return _reliable_sources, _unreliable_sources


def assess_source(domain: Optional[str]) -> dict:
    """
    Assess the credibility of a news source based on its domain.

    Args:
        domain: The domain name to evaluate (e.g. ``"reuters.com"``).
                If ``None`` or empty, the source is treated as unknown.

    Returns:
        A dictionary with:
            - ``domain``             (str | None):  The evaluated domain.
            - ``credibility_score``  (int):         Score from 0 (unreliable) to 100 (reliable).
            - ``category``           (str):         Source category / type.
            - ``status``             (str):         ``"reliable"``, ``"unreliable"``, or ``"unknown"``.
            - ``explanation``        (str):         Human-readable assessment.
    """
    reliable, unreliable = _load_sources()

    if not domain:
        return {
            "domain": None,
            "credibility_score": 50,
            "category": "N/A",
            "status": "unknown",
            "explanation": "No source domain provided; credibility cannot be determined from the source alone.",
        }

    domain = domain.lower().strip()

    # ── Known reliable source ──────────────────────────────────────
    if domain in reliable:
        category = reliable[domain]
        return {
            "domain": domain,
            "credibility_score": 85,
            "category": category,
            "status": "reliable",
            "explanation": (
                f"'{domain}' is recognised as a reliable source "
                f"(category: {category}). It has a strong track record "
                "for accuracy and editorial standards."
            ),
        }

    # ── Known unreliable source ────────────────────────────────────
    if domain in unreliable:
        category = unreliable[domain]
        return {
            "domain": domain,
            "credibility_score": 15,
            "category": category,
            "status": "unreliable",
            "explanation": (
                f"'{domain}' is flagged as an unreliable source "
                f"(category: {category}). Content from this domain "
                "should be treated with significant scepticism."
            ),
        }

    # ── Unknown source ─────────────────────────────────────────────
    return {
        "domain": domain,
        "credibility_score": 50,
        "category": "Unknown",
        "status": "unknown",
        "explanation": (
            f"'{domain}' is not in our curated source lists. "
            "Its credibility is uncertain — verify the content through "
            "independent fact-checking sources."
        ),
    }
