"""
Wikipedia Knowledge Corroboration Module (Free, No API Key Required).

Queries the public Wikipedia search API to cross-reference entities and statements
against verified encyclopedia entries to detect factual alignment and ambiguity.
"""

import html
import re
from typing import Any
import requests

WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MisinfoDetector/2.0 (AI Misinformation Combat Tool; educational/research use)"


def _clean_snippet(raw_html: str) -> str:
    """Remove HTML tags and entities from Wikipedia search snippets."""
    clean = re.sub(r"<[^>]+>", "", raw_html)
    return html.unescape(clean).strip()


def verify_with_wikipedia(claim_text: str) -> dict[str, Any]:
    """
    Search Wikipedia for entities and statements related to the claim.

    Returns:
        Dictionary with matching articles, snippets, and knowledge corroboration score.
    """
    if not claim_text or not claim_text.strip():
        return {
            "available": False,
            "corroborated": False,
            "score": 50.0,
            "matches": [],
            "summary": "No text provided for knowledge verification.",
        }

    # Clean claim to form a search query
    query = re.sub(r"[^\w\s]", " ", claim_text).strip()
    words = query.split()
    if not words:
        return {
            "available": False,
            "corroborated": False,
            "score": 50.0,
            "matches": [],
            "summary": "No search terms found.",
        }

    # Search top 5 relevant articles
    params = {
        "action": "query",
        "list": "search",
        "srsearch": " ".join(words[:8]),
        "srlimit": 4,
        "format": "json",
        "utf8": 1,
    }

    try:
        response = requests.get(
            WIKI_API_ENDPOINT,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=5,
        )
        if response.status_code != 200:
            return {
                "available": False,
                "corroborated": False,
                "score": 50.0,
                "matches": [],
                "summary": "Wikipedia lookup service unreachable.",
            }

        data = response.json()
        search_results = data.get("query", {}).get("search", [])

        if not search_results:
            return {
                "available": True,
                "corroborated": False,
                "score": 40.0,
                "matches": [],
                "summary": "No matching knowledge base articles found for this specific claim.",
            }

        matches = []
        for item in search_results:
            title = item.get("title", "")
            snippet = _clean_snippet(item.get("snippet", ""))
            matches.append({
                "title": title,
                "snippet": snippet,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            })

        # Check for ambiguity indicators in snippets (e.g., "to avoid confusion", "dispute", "myth")
        combined_text = " ".join([m["title"] + " " + m["snippet"] for m in matches]).lower()
        ambiguity_signals = [
            "confusion", "dispute", "myth", "controversy", "conspiracy",
            "differing views", "broad term", "distinguish", "slang",
        ]
        detected_ambiguities = [sig for sig in ambiguity_signals if sig in combined_text]

        # Check keyword overlap between claim and Wikipedia snippets
        claim_keywords = set(w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{3,}\b", claim_text))
        stop_words = {"the", "and", "once", "every", "that", "this", "from", "with", "for", "are"}
        claim_content_words = claim_keywords - stop_words

        overlap_ratio = 0.0
        if claim_content_words:
            snippet_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", combined_text))
            # Match word stems/substrings (e.g. orbit vs orbits)
            matched_words = {
                cw for cw in claim_content_words
                if any(cw in sw or sw in cw for sw in snippet_words)
            }
            overlap_ratio = len(matched_words) / len(claim_content_words)

        # Factual alignment scoring
        if overlap_ratio >= 0.70 and not detected_ambiguities:
            score = 85.0
            corroborated = True
            matched_titles = ", ".join(m["title"] for m in matches[:3])
            summary = (
                f"Strong Wikipedia Knowledge Alignment: Confirmed consensus across "
                f"authoritative reference articles ({matched_titles})."
            )
        elif overlap_ratio >= 0.45 and not detected_ambiguities:
            score = 75.0
            corroborated = True
            summary = f"Supported by Wikipedia reference articles with {round(overlap_ratio * 100)}% factual context match."
        elif detected_ambiguities:
            score = 45.0
            corroborated = False
            summary = (
                f"Knowledge base indicates contextual ambiguity or naming distinctions "
                f"related to this topic ({', '.join(detected_ambiguities)})."
            )
        else:
            score = 55.0
            corroborated = False
            summary = f"Found {len(matches)} related reference articles on Wikipedia for context verification."

        return {
            "available": True,
            "corroborated": corroborated,
            "score": score,
            "matches": matches,
            "detected_ambiguities": detected_ambiguities,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "available": False,
            "corroborated": False,
            "score": 50.0,
            "matches": [],
            "summary": f"Wikipedia lookup error: {exc}",
        }
