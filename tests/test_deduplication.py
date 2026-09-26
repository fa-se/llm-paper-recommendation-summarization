"""Run from the repo root: uv run python -m unittest tests.test_deduplication"""

import unittest

from core.services.deduplication import DuplicateFilter, has_usable_abstract, text_key


class DeduplicationTest(unittest.TestCase):
    def test_text_key_ignores_case_punctuation_and_whitespace(self):
        self.assertEqual(text_key("Coral Reefs: A Review"), text_key("coral reefs  a review."))
        self.assertNotEqual(text_key("Coral reefs"), text_key("Coral reef"))
        self.assertIsNone(text_key(None))
        self.assertIsNone(text_key(" -- "))

    def test_junk_abstracts(self):
        self.assertFalse(has_usable_abstract("International audience"))
        self.assertFalse(has_usable_abstract(None))
        self.assertTrue(has_usable_abstract(" ".join(["word"] * 20)))

    def test_duplicate_if_title_or_abstract_matches(self):
        duplicate_filter = DuplicateFilter(title_keys=[text_key("Stored title")])
        self.assertFalse(duplicate_filter.is_new("Stored Title", "some abstract"))
        self.assertTrue(duplicate_filter.is_new("New title", "An abstract."))
        self.assertFalse(duplicate_filter.is_new("Translated title", "an abstract"))
        self.assertTrue(duplicate_filter.is_new(None, None))


if __name__ == "__main__":
    unittest.main()
