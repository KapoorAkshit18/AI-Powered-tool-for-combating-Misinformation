"""Unit tests for modules/linguistic_analyzer.py."""
import unittest
from modules.linguistic_analyzer import analyze_linguistics


class TestLinguisticAnalyzer(unittest.TestCase):
    def test_clickbait_detection(self):
        text = "YOU WON'T BELIEVE WHAT HAPPENED NEXT!!! THIS SECRET WILL CHANGE YOUR LIFE FOREVER!!!"
        result = analyze_linguistics(text)
        self.assertIn("linguistic_credibility", result)
        self.assertGreater(result.get("clickbait_score", 0), 30.0)
        self.assertLess(result["linguistic_credibility"], 85.0)

    def test_objective_text(self):
        text = "According to data published by the National Meteorological Service, rainfall was 12 percent below average."
        result = analyze_linguistics(text)
        self.assertGreaterEqual(result["linguistic_credibility"], 70.0)


if __name__ == "__main__":
    unittest.main()
