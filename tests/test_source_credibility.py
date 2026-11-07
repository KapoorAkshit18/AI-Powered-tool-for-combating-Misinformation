"""Unit tests for modules/source_credibility.py."""
import unittest
from modules.source_credibility import assess_source


class TestSourceCredibility(unittest.TestCase):
    def test_reliable_source(self):
        result = assess_source("reuters.com")
        self.assertGreaterEqual(result.get("credibility_score", 0), 75.0)
        self.assertEqual(result.get("status"), "reliable")

    def test_unreliable_source(self):
        result = assess_source("infowars.com")
        self.assertLessEqual(result.get("credibility_score", 100), 40.0)
        self.assertEqual(result.get("status"), "unreliable")

    def test_unknown_source(self):
        result = assess_source("some-completely-new-blog-xyz99.org")
        self.assertEqual(result.get("status"), "unknown")
        # Unknown sources should receive a neutral baseline score
        self.assertEqual(result.get("credibility_score"), 50.0)


if __name__ == "__main__":
    unittest.main()
