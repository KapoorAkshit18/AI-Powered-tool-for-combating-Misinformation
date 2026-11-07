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
    if source is not None and isinstance(source, dict):
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
    gemini_result: Optional[dict] = None,
    wiki_result: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Aggregate all component analyses into an overall credibility assessment.

    Args:
        nlp_result:        Output of :func:`modules.nlp_analyzer.analyze_text`.
        fact_check_result: Output of :func:`modules.fact_checker.check_claim`.
        source_result:     Output of :func:`modules.source_credibility.assess_source`.
        linguistic_result: Output of :func:`modules.linguistic_analyzer.analyze_linguistics`.
        gemini_result:     Optional output of :func:`modules.gemini_verifier.analyze_with_gemini`.
        wiki_result:       Optional output of :func:`modules.wiki_verifier.verify_with_wikipedia`.

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
    gem_dict = gemini_result if isinstance(gemini_result, dict) else None
    wiki_dict = wiki_result if isinstance(wiki_result, dict) else None

    nlp_score = float(nlp_dict.get("nlp_credibility", 50.0))
    fc_score = float(fc_dict.get("fact_check_score", 50.0))
    ling_score = float(ling_dict.get("linguistic_credibility", 50.0))

    components: list[tuple[str, float, float]] = [
        ("NLP / Tone Analysis", nlp_score, 0.20),
        ("Fact-Check Lookup", fc_score, 0.25),
        ("Linguistic Style", ling_score, 0.20),
    ]

    src_score = None
    if src_dict is not None and "credibility_score" in src_dict:
        src_score = float(src_dict.get("credibility_score", 50.0))
        components.append(("Source Credibility", src_score, 0.25))

    wiki_score = None
    if wiki_dict and wiki_dict.get("available"):
        wiki_score = float(wiki_dict.get("score", 50.0))
        components.append(("Wikipedia Knowledge Graph", wiki_score, 0.20))

    gem_score = None
    if gem_dict and gem_dict.get("available") and gem_dict.get("ai_credibility_score") is not None:
        try:
            gem_score = float(gem_dict["ai_credibility_score"])
            components.append(("Gemini Deep Reasoning", gem_score, 0.35))
        except (ValueError, TypeError):
            pass

    # Normalize weights across all active components
    total_weight = sum(w for _, _, w in components)
    raw_overall = sum(score * (w / total_weight) for _, score, w in components)

    # -------------------------------------------------------------------------
    # CRITICAL CALIBRATION RULE:
    # Calm writing style alone does NOT equal factual truth.
    # A claim with ZERO affirmative factual evidence (no fact-check confirmation,
    # no trusted publisher domain, no Gemini factual verification) can NEVER
    # be declared "Likely Credible" (>70) solely because it lacks exclamation points.
    # -------------------------------------------------------------------------
    has_affirmative_evidence = False
    if fc_dict.get("available") and fc_score >= 65:
        has_affirmative_evidence = True
    elif src_score is not None and src_score >= 70:
        has_affirmative_evidence = True
    elif gem_score is not None and gem_score >= 65:
        has_affirmative_evidence = True

    if not has_affirmative_evidence and raw_overall > 55.0:
        # Cap uncorroborated assertions at "Needs Verification"
        overall = 52.0
    else:
        overall = raw_overall

    overall = round(max(0.0, min(100.0, overall)), 2)
    verdict = _determine_verdict(overall)

    component_scores = {name: round(score, 2) for name, score, _ in components}

    recommendations = _build_recommendations(
        overall, nlp_result, fact_check_result, source_result, linguistic_result
    )

    if not has_affirmative_evidence:
        recommendations.insert(
            0,
            "⚠️ Unverified Claim: This statement uses calm, non-sensational phrasing, "
            "but lacks verified factual citations or database matches. Treat as unverified."
        )

    # Incorporate Wikipedia context if ambiguity was detected
    if wiki_dict and wiki_dict.get("detected_ambiguities"):
        recommendations.append(
            f"Knowledge Graph: Contextual ambiguity detected ({', '.join(wiki_dict['detected_ambiguities'])}). "
            "Clarify terminology or definitions."
        )

    # Incorporate Gemini reasoning points into recommendations if available
    if gem_dict and gem_dict.get("available"):
        gem_recs = gem_dict.get("reasoning_points", [])
        if gem_recs:
            recommendations.extend([f"AI Analysis: {r}" for r in gem_recs[:3]])

    return {
        "overall_score": overall,
        "verdict": verdict,
        "component_scores": component_scores,
        "recommendations": recommendations,
    }
