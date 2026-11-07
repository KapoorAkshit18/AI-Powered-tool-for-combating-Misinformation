"""
Input Processor Module for the Misinformation Detection Tool.

Handles three types of input:
  - Raw text claims (pass-through)
  - URLs (fetches and extracts article content via trafilatura)
  - Pasted article text

Returns a standardized dictionary with title, text, source URL, and domain.
"""

from urllib.parse import urlparse
from typing import Optional

import trafilatura


def _is_url(text: str) -> bool:
    """Check whether the given string looks like a URL."""
    text = text.strip()
    if not (text.startswith("http://") or text.startswith("https://")):
        return False
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def _extract_domain(url: str) -> Optional[str]:
    """Extract the base domain from a URL (e.g. 'www.example.com' -> 'example.com')."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Strip common 'www.' prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else None
    except Exception:
        return None


def _fetch_url(url: str) -> dict:
    """
    Fetch and extract content from a URL using trafilatura.

    Returns a dict with extracted title and text, or an error message.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return {
                "title": "",
                "text": "",
                "error": f"Could not download content from {url}",
            }

        # Extract main text content
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        )

        # Extract metadata for the title
        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else ""

        if not text:
            return {
                "title": title,
                "text": "",
                "error": "Could not extract meaningful text from the page.",
            }

        return {"title": title, "text": text, "error": None}

    except Exception as exc:
        return {
            "title": "",
            "text": "",
            "error": f"Error fetching URL: {exc}",
        }


def process_input(input_text: str, input_type: str) -> dict:
    """
    Process user input based on its declared type.

    Args:
        input_text: The raw user input (a claim, URL, or pasted article).
        input_type: One of ``"claim"``, ``"url"``, or ``"article"``.

    Returns:
        A dictionary with the following keys:
            - ``title``      (str):           Extracted or empty title.
            - ``text``       (str):           The main text to analyse.
            - ``source_url`` (str | None):    Original URL if applicable.
            - ``domain``     (str | None):    Extracted domain from the URL.
            - ``error``      (str | None):    Error message, if any.
    """
    input_text = input_text.strip()

    if not input_text:
        return {
            "title": "",
            "text": "",
            "source_url": None,
            "domain": None,
            "error": "No input provided.",
        }

    # ── Claim (raw text, short statement) ───────────────────────────
    if input_type == "claim":
        return {
            "title": "",
            "text": input_text,
            "source_url": None,
            "domain": None,
            "error": None,
        }

    # ── URL ─────────────────────────────────────────────────────────
    if input_type == "url":
        if not _is_url(input_text):
            return {
                "title": "",
                "text": "",
                "source_url": None,
                "domain": None,
                "error": "The provided input does not appear to be a valid URL.",
            }

        domain = _extract_domain(input_text)
        result = _fetch_url(input_text)

        return {
            "title": result["title"],
            "text": result["text"],
            "source_url": input_text,
            "domain": domain,
            "error": result["error"],
        }

    # ── Pasted article text ─────────────────────────────────────────
    if input_type == "article":
        # Attempt to use the first line as a title if it is short enough
        lines = input_text.split("\n", 1)
        if len(lines) > 1 and len(lines[0]) < 200:
            title = lines[0].strip()
            text = lines[1].strip()
        else:
            title = ""
            text = input_text

        return {
            "title": title,
            "text": text,
            "source_url": None,
            "domain": None,
            "error": None,
        }

    # ── Unknown input type ──────────────────────────────────────────
    return {
        "title": "",
        "text": "",
        "source_url": None,
        "domain": None,
        "error": f"Unknown input type: '{input_type}'. Use 'claim', 'url', or 'article'.",
    }
