"""Unit tests for modules/wiki_verifier.py."""
import unittest
from modules.wiki_verifier import verify_with_wikipedia


class TestWikiVerifier(unittest.TestCase):
    def test_empty_query(self):
        result = verify_with_wikipedia("")
        self.assertFalse(result.get("available"))

    def test_valid_query(self):
        result = verify_with_wikipedia("Albert Einstein physics Nobel Prize")
        self.assertTrue(result.get("available"))
        self.assertIn("score", result)
        self.assertGreater(len(result.get("matches", [])), 0)


if __name__ == "__main__":
    unittest.main()
