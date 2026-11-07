"""
Scoring Engine for the Misinformation Detection Tool.

Aggregates results from all analysis modules and produces an overall
credibility score, verdict, component breakdown, and recommendations.
"""

from typing import Any, Optional


# ── Verdict thresholds ──────────────────────────────────────────────
_VERDICTS: list[tuple[int, str]] = [
    (75, "Likely Credible"),
    (50, "Needs Verification"),
    (25, "Likely Misleading"),
    (0,  "Highly Suspicious"),
]

# ── Component weights (must sum to 1.0) ─────────────────────────────
_WEIGHTS: dict[str, float] = {
    "nlp": 0.25,
    "fact_check": 0.25,
    "source": 0.25,
    "linguistic": 0.25,
}


def _determine_verdict(score: float) -> str:
    """Map a 0-100 credibility score to a human-readable verdict."""
    for threshold, label in _VERDICTS:
        if score >= threshold:
            return label
    return "Highly Suspicious"


def _build_recommendations(
    overall: float,
    nlp: dict,
    fact_check: dict,
    source: dict,
    linguistic: dict,
) -> list[str]:
    """Generate actionable recommendations based on the analysis."""
    recs: list[str] = []

    # Fact-check availability
    if not fact_check.get("available", False):
        recs.append(
            "Configure a Google Fact Check API key to enable automated fact-check lookups."
        )
    elif fact_check.get("match_found"):
        recs.append(
            "Existing fact-checks were found — review the listed sources for detailed verdicts."
        )
    else:
        recs.append(
            "No existing fact-checks were found. Consider manually verifying through "
            "Snopes, PolitiFact, or FactCheck.org."
        )

    # Source credibility
    source_status = source.get("status", "unknown")
    if source_status == "unreliable":
        recs.append(
            f"The source domain '{source.get('domain')}' is flagged as unreliable. "
            "Cross-reference this claim with trusted news outlets."
        )
    elif source_status == "unknown":
        recs.append(
            "The source domain is not in our database. Try verifying the publisher's "
            "editorial standards and track record."
        )

    # NLP / emotional language
    if nlp.get("emotional_score", 0) > 0.3:
        recs.append(
            "The text contains a high level of emotional language. "
            "Emotionally charged articles are more likely to be misleading."
        )
    if nlp.get("bias_score", 0) > 0.3:
        recs.append(
            "Significant bias indicators were detected. Look for multiple "
            "perspectives on this topic."
        )

    # Linguistic red flags
    if linguistic.get("clickbait_score", 0) > 50:
        recs.append(
            "The text shows clickbait characteristics (ALL CAPS, excessive punctuation, "
            "superlatives). Treat sensational headlines with caution."
        )
    fallacies = linguistic.get("fallacies_detected", {})
    detected = [name for name, found in fallacies.items() if found]
    if detected:
        recs.append(
            f"Possible logical fallacies detected: {', '.join(detected)}. "
            "Evaluate the underlying arguments carefully."
        )

    # Overall
    if overall < 25:
        recs.append(
            "⚠️  This content is highly suspicious. Do NOT share it before "
            "independent verification."
        )
    elif overall < 50:
        recs.append(
            "This content shows multiple warning signs. Verify with at least "
            "two independent, reputable sources before sharing."
        )

    return recs


def calculate_credibility_score(
    nlp_result: dict,
    fact_check_result: dict,
    source_result: dict,
    linguistic_result: dict,
) -> dict[str, Any]:
    """
    Aggregate all component analyses into an overall credibility assessment.

    Args:
        nlp_result:        Output of :func:`modules.nlp_analyzer.analyze_text`.
        fact_check_result:  Output of :func:`modules.fact_checker.check_claim`.
        source_result:      Output of :func:`modules.source_credibility.assess_source`.
        linguistic_result:  Output of :func:`modules.linguistic_analyzer.analyze_linguistics`.

    Returns:
        A dictionary with:
            - ``overall_score``   (float):       Weighted credibility score [0, 100].
            - ``verdict``         (str):         Human-readable verdict string.
            - ``component_scores``(dict):        Individual component scores.
            - ``recommendations`` (list[str]):   Actionable suggestions for the user.
    """
    # Safely extract component scores
    nlp_dict = nlp_result if isinstance(nlp_result, dict) else {}
    fc_dict = fact_check_result if isinstance(fact_check_result, dict) else {}
    src_dict = source_result if isinstance(source_result, dict) else None
    ling_dict = linguistic_result if isinstance(linguistic_result, dict) else {}

    nlp_score = float(nlp_dict.get("nlp_credibility", 50.0))
    fc_score = float(fc_dict.get("fact_check_score", 50.0))
    ling_score = float(ling_dict.get("linguistic_credibility", 50.0))

    # If source_result is provided, include it in weights; otherwise rebalance
    if src_dict is not None and "credibility_score" in src_dict:
        src_score = float(src_dict.get("credibility_score", 50.0))
        overall = (
            nlp_score * _WEIGHTS["nlp"]
            + fc_score * _WEIGHTS["fact_check"]
            + src_score * _WEIGHTS["source"]
            + ling_score * _WEIGHTS["linguistic"]
        )
    else:
        # Rebalance across the 3 remaining components
        src_score = None
        total_w = _WEIGHTS["nlp"] + _WEIGHTS["fact_check"] + _WEIGHTS["linguistic"]
        overall = (
            nlp_score * (_WEIGHTS["nlp"] / total_w)
            + fc_score * (_WEIGHTS["fact_check"] / total_w)
            + ling_score * (_WEIGHTS["linguistic"] / total_w)
        )

    overall = round(max(0.0, min(100.0, overall)), 2)
    verdict = _determine_verdict(overall)

    component_scores = {
        "NLP / Sentiment Analysis": round(nlp_score, 2),
        "Fact-Check Lookup": round(fc_score, 2),
        "Linguistic Analysis": round(ling_score, 2),
    }
    if src_score is not None:
        component_scores["Source Credibility"] = round(src_score, 2)

    recommendations = _build_recommendations(
        overall, nlp_result, fact_check_result, source_result, linguistic_result
    )

    return {
        "overall_score": overall,
        "verdict": verdict,
        "component_scores": component_scores,
        "recommendations": recommendations,
    }
