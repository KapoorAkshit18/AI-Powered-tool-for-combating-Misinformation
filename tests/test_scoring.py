"""Unit tests for modules/scoring_engine.py."""
import unittest
from modules.scoring_engine import calculate_credibility_score


class TestScoringEngine(unittest.TestCase):
    def test_calculate_score_without_source(self):
        nlp = {"nlp_credibility": 80.0}
        fc = {"available": True, "fact_check_score": 75.0}
        ling = {"linguistic_credibility": 85.0}

        result = calculate_credibility_score(
            nlp_result=nlp,
            fact_check_result=fc,
            source_result=None,
            linguistic_result=ling,
        )
        self.assertIn("overall_score", result)
        self.assertIn("verdict", result)
        self.assertGreaterEqual(result["overall_score"], 70.0)

    def test_calculate_score_with_unreliable_source(self):
        nlp = {"nlp_credibility": 50.0}
        fc = {"fact_check_score": 30.0}
        source = {"credibility_score": 15.0}
        ling = {"linguistic_credibility": 40.0}

        result = calculate_credibility_score(
            nlp_result=nlp,
            fact_check_result=fc,
            source_result=source,
            linguistic_result=ling,
        )
        self.assertLess(result["overall_score"], 50.0)
        self.assertIn("Source Credibility", result["component_scores"])


if __name__ == "__main__":
    unittest.main()
