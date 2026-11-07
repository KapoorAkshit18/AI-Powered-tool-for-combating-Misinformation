"""Unit tests for modules/nlp_analyzer.py."""
import unittest
from modules.nlp_analyzer import analyze_text


class TestNLPAnalyzer(unittest.TestCase):
    def test_neutral_text(self):
        text = "The committee held its regular meeting on Tuesday morning to discuss budget allocation."
        result = analyze_text(text)
        self.assertIn("nlp_credibility", result)
        self.assertIn("polarity", result)
        self.assertIn("subjectivity", result)
        # Neutral factual reporting should yield reasonable credibility
        self.assertGreaterEqual(result["nlp_credibility"], 50.0)

    def test_highly_emotional_text(self):
        text = "SHOCKING and OUTRAGEOUS tragedy! EVIL corrupt monsters destroy innocent lives in horrifying disaster!"
        result = analyze_text(text)
        self.assertIn("nlp_credibility", result)
        # Loaded emotional content should lower NLP credibility
        self.assertLess(result["nlp_credibility"], 70.0)
        self.assertGreater(len(result.get("emotional_words", [])), 0)

    def test_empty_text(self):
        result = analyze_text("")
        self.assertIn("nlp_credibility", result)


if __name__ == "__main__":
    unittest.main()
