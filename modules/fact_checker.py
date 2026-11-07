"""
Fact Checker Module for the Misinformation Detection Tool.

Integrates with the Google Fact Check Tools API to look up
existing fact-checks for a given claim.
"""

from typing import Optional

import requests


FACTCHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def check_claim(claim: str, api_key: Optional[str] = None) -> dict:
    """
    Query the Google Fact Check Tools API for matching fact-checks.

    Args:
        claim:   The claim text to search for.
        api_key: Google Fact Check Tools API key.  If ``None`` or empty,
                 the function returns an "unavailable" result.

    Returns:
        A dictionary with:
            - ``available``        (bool):         Whether the API was queried.
            - ``message``          (str | None):   Human-readable status message.
            - ``fact_checks``      (list[dict]):   List of matching fact-check entries.
            - ``match_found``      (bool):         True if at least one match was returned.
            - ``fact_check_score`` (float):        Credibility sub-score [0, 100].
    """
    # ── No API key ──────────────────────────────────────────────────
    if not api_key:
        return {
            "available": False,
            "message": "No API key configured. Skipping fact-check lookup.",
            "fact_checks": [],
            "match_found": False,
            "fact_check_score": 50.0,  # neutral when unavailable
        }

    # ── Query the API ───────────────────────────────────────────────
    try:
        params = {
            "query": claim[:1000],  # API has a query length limit
            "key": api_key,
            "languageCode": "en",
        }
        response = requests.get(FACTCHECK_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        return {
            "available": True,
            "message": "Fact-check API request timed out.",
            "fact_checks": [],
            "match_found": False,
            "fact_check_score": 50.0,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "available": True,
            "message": f"Fact-check API error: {exc}",
            "fact_checks": [],
            "match_found": False,
            "fact_check_score": 50.0,
        }

    # ── Parse results ───────────────────────────────────────────────
    claims = data.get("claims", [])
    if not claims:
        return {
            "available": True,
            "message": "No matching fact-checks found.",
            "fact_checks": [],
            "match_found": False,
            "fact_check_score": 50.0,  # neutral when no data
        }

    fact_checks: list[dict] = []
    for item in claims:
        for review in item.get("claimReview", []):
            fact_checks.append({
                "claim_text": item.get("text", ""),
                "claimant": item.get("claimant", "Unknown"),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
                "url": review.get("url", ""),
                "title": review.get("title", ""),
                "rating": review.get("textualRating", ""),
            })

    # ── Derive a credibility sub-score from ratings ─────────────────
    score = _score_from_ratings(fact_checks)

    return {
        "available": True,
        "message": f"Found {len(fact_checks)} fact-check(s).",
        "fact_checks": fact_checks,
        "match_found": True,
        "fact_check_score": round(score, 2),
    }


# Mapping common fact-check textual ratings to a 0-100 credibility value.
_RATING_SCORES: dict[str, float] = {
    "true": 95.0,
    "mostly true": 85.0,
    "half true": 55.0,
    "partly true": 55.0,
    "mostly false": 25.0,
    "false": 10.0,
    "pants on fire": 5.0,
    "incorrect": 10.0,
    "misleading": 20.0,
    "unproven": 40.0,
    "mixture": 50.0,
}


def _score_from_ratings(fact_checks: list[dict]) -> float:
    """Compute a credibility score by averaging the matched ratings."""
    scores: list[float] = []
    for fc in fact_checks:
        rating = fc.get("rating", "").lower().strip()
        for key, value in _RATING_SCORES.items():
            if key in rating:
                scores.append(value)
                break
        # If no known rating matched, treat as neutral
        else:
            scores.append(50.0)

    return sum(scores) / len(scores) if scores else 50.0
