"""Tests for fixed-seed balanced sampling across a gzip JSONL source."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ThreeClassSampleTests(unittest.TestCase):
    def test_rating_mapping_has_a_distinct_neutral_class(self):
        from gift_cards.three_class_sample import rating_to_sentiment

        expected = {
            1: "NEGATIVE",
            2: "NEGATIVE",
            3: "NEUTRAL",
            4: "POSITIVE",
            5: "POSITIVE",
        }
        self.assertEqual({rating: rating_to_sentiment(rating) for rating in expected}, expected)
        with self.assertRaises(ValueError):
            rating_to_sentiment(0)

    def test_balanced_sample_is_reproducible_and_scans_entire_source(self):
        from gift_cards.three_class_sample import balanced_sample

        records = []
        words = ["alpha", "bravo", "charlie", "delta", "echo"]
        for rating in (5, 3, 1):
            for word in words:
                records.append(
                    {"rating": rating, "title": f"Review {word}", "text": f"Comment {word}"}
                )
        records.append({"rating": 3, "title": "Three Stars", "text": "Leaky label"})

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "reviews.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
            sample_a, stats_a = balanced_sample(source, per_class=2, seed=6418)
            sample_b, stats_b = balanced_sample(source, per_class=2, seed=6418)
            sample_c, _ = balanced_sample(source, per_class=2, seed=9001)

        self.assertEqual(sample_a, sample_b)
        self.assertEqual(stats_a, stats_b)
        self.assertNotEqual(sample_a, sample_c)
        self.assertEqual(stats_a["source_records_scanned"], len(records))
        self.assertEqual(stats_a["eligible_by_class"], {
            "POSITIVE": 5,
            "NEUTRAL": 5,
            "NEGATIVE": 5,
        })
        self.assertEqual(stats_a["skipped_unusable"], 1)
        self.assertEqual(len(sample_a), 6)
        self.assertEqual(
            {label: sum(row["actual_sentiment"] == label for row in sample_a)
             for label in ("POSITIVE", "NEUTRAL", "NEGATIVE")},
            {"POSITIVE": 2, "NEUTRAL": 2, "NEGATIVE": 2},
        )
        self.assertEqual([row["sample_id"] for row in sample_a], list(range(1, 7)))
        self.assertTrue(all("source_line" in row for row in sample_a))


if __name__ == "__main__":
    unittest.main()
