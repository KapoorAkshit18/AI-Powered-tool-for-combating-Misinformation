"""Unit tests for modules/gemini_verifier.py."""
import unittest
from modules.gemini_verifier import analyze_with_gemini


class TestGeminiVerifier(unittest.TestCase):
    def test_no_api_key_graceful_fallback(self):
        result = analyze_with_gemini("Drinking bleach cures viruses.", api_key=None)
        self.assertFalse(result.get("available"))
        self.assertIn("not configured", result.get("message", ""))
        self.assertIsNone(result.get("ai_credibility_score"))

    def test_empty_text(self):
        result = analyze_with_gemini("", api_key="dummy_key")
        self.assertFalse(result.get("available"))


if __name__ == "__main__":
    unittest.main()
