"""
Gemini Deep Reasoning & Multi-Claim Verification Module.

Leverages Google Gemini via Google AI Studio API key to perform:
  - Atomic factual claim extraction
  - Deep logical fallacy & cognitive manipulation detection
  - Evidence verification & source credibility cross-examination
  - Explainable AI reasoning & confidence scoring

Produces structured analysis output, NOT conversational text.
"""

import json
import logging
import re
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Primary and robust fallback Gemini models (compatible with current Google AI Studio API)
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def _build_analysis_prompt(text: str) -> str:
    """Construct an objective, forensic analysis prompt enforcing strict JSON output."""
    return f"""You are an objective, rigorous AI Fact-Checking Engine and Misinformation Forensic Analyst.
Your task is to analyze the following content for credibility, misinformation, and factual integrity.
Do NOT respond conversationally. Do NOT speak in the first person. 
You MUST respond ONLY with a valid JSON object matching the exact schema specified below.

CONTENT TO ANALYZE:
\"\"\"{text}\"\"\"

SCHEMA:
{{
  "ai_credibility_score": <integer from 0 to 100, where 100 is completely verified/credible and 0 is complete fabrication/hoax>,
  "confidence": <"High" | "Medium" | "Low">,
  "verdict_summary": "<concise 1-2 sentence forensic verdict>",
  "atomic_claims": [
    {{
      "claim": "<specific verifiable statement extracted from the text>",
      "verifiability": "<High | Medium | Low>",
      "assessment": "<Likely True | Questionable | Misleading | False | Unverifiable>",
      "note": "<brief explanation of factual context>"
    }}
  ],
  "logical_fallacies": [
    "<name of specific fallacy detected, e.g. Hasty Generalization, False Causation, Strawman, Appeal to Fear, or None>"
  ],
  "manipulation_tactics": [
    "<specific tactic, e.g. Cherry-picking, Out-of-context framing, Fabricated authority, Sensationalist hyperbole, or None>"
  ],
  "evidence_quality": "<Strong | Moderate | Weak | Non-existent | Fabricated>",
  "reasoning_points": [
    "<specific analytical point explaining the credibility assessment>"
  ]
}}

Ensure valid JSON syntax only. Do not surround with markdown backticks if possible, or use standard ```json blocks."""


def analyze_with_gemini(text: str, api_key: Optional[str] = None) -> dict[str, Any]:
    """
    Run deep misinformation analysis and claim breakdown using Gemini.

    Args:
        text: The text claim or extracted article content to analyze.
        api_key: Google AI Studio Gemini API key.

    Returns:
        Structured dictionary with atomic claims, logical fallacies,
        credibility score, and evidence evaluation.
    """
    if not api_key or not api_key.strip():
        return {
            "available": False,
            "message": "Gemini API key not configured. Add your key from aistudio.google.com in Settings to enable deep AI reasoning.",
            "ai_credibility_score": None,
        }

    text = text.strip()
    if not text:
        return {
            "available": False,
            "message": "Empty text provided for analysis.",
            "ai_credibility_score": None,
        }

    # Limit text length to avoid token limits for very large inputs
    truncated_text = text[:8000]
    prompt = _build_analysis_prompt(truncated_text)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,  # Low temperature for deterministic, factual reasoning
            "topP": 0.95,
            "maxOutputTokens": 2048,
        }
    }

    headers = {"Content-Type": "application/json"}
    last_error = ""

    # Attempt calling models with automatic fallback
    for model_name in GEMINI_MODELS:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=18,
            )

            if response.status_code == 200:
                data = response.json()
                raw_text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )

                # Clean markdown formatting if present
                clean_json = raw_text.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                elif clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                clean_json = clean_json.strip()

                parsed = json.loads(clean_json)
                parsed["available"] = True
                parsed["model_used"] = model_name
                return parsed

            elif response.status_code in (400, 403):
                # Invalid key or permission error - stop trying other models
                err_data = response.json().get("error", {})
                return {
                    "available": False,
                    "error": f"API Error ({response.status_code}): {err_data.get('message', 'Invalid API key or unauthorized request.')}",
                    "ai_credibility_score": None,
                }
            else:
                last_error = f"{model_name} returned HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            last_error = f"Request timed out while contacting {model_name}"
        except json.JSONDecodeError:
            last_error = "Model response could not be parsed as structured JSON"
        except Exception as exc:
            last_error = str(exc)

    return {
        "available": False,
        "error": f"Gemini analysis unavailable: {last_error}",
        "ai_credibility_score": None,
    }
