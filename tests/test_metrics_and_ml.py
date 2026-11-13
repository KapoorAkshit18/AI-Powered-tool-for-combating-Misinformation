"""Unit tests for modules/metrics_calculator.py and modules/ml_classifier.py."""
import unittest
from modules.metrics_calculator import calculate_text_metrics
from modules.ml_classifier import predict_misinformation_probability


class TestMetricsAndML(unittest.TestCase):
    def test_text_metrics_calculation(self):
        text = "This is a simple, clear test sentence to verify reading ease and grade level."
        metrics = calculate_text_metrics(text)
        self.assertIn("flesch_reading_ease", metrics)
        self.assertIn("flesch_kincaid_grade", metrics)
        self.assertIn("lexical_diversity_ttr", metrics)
        self.assertGreater(metrics["flesch_reading_ease"], 0.0)
        self.assertGreaterEqual(metrics["lexical_diversity_ttr"], 0.0)

    def test_ml_misinformation_detection(self):
        hoax = "Drinking bleach cures coronavirus and all viral diseases completely"
        pred = predict_misinformation_probability(hoax)
        self.assertTrue(pred["available"])
        self.assertGreater(pred["misinfo_probability_pct"], 60.0)
        self.assertLess(pred["ml_credibility_score"], 40.0)

    def test_ml_credible_fact_detection(self):
        fact = "Photosynthesis is the process by which green plants transform light energy into chemical energy"
        pred = predict_misinformation_probability(fact)
        self.assertTrue(pred["available"])
        self.assertGreater(pred["credible_probability_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
