"""Tests for API-free balanced sentiment metrics and NRC enrichment."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class OfflineBalancedAnalysisTests(unittest.TestCase):
    def test_analyze_validates_labels_and_adds_nrc_to_every_row(self):
        from gift_cards.offline_balanced_analysis import analyze_and_enrich

        rows = [
            {"sample_id": "1", "source_line": "10", "rating": "5.0", "title": "Good", "text": "Loved it", "actual_sentiment": "POSITIVE", "predicted_sentiment": "POSITIVE", "correct": "True"},
            {"sample_id": "2", "source_line": "20", "rating": "3.0", "title": "Okay", "text": "Average", "actual_sentiment": "NEUTRAL", "predicted_sentiment": "NEGATIVE", "correct": "False"},
            {"sample_id": "3", "source_line": "30", "rating": "1.0", "title": "Bad", "text": "Failed", "actual_sentiment": "NEGATIVE", "predicted_sentiment": "NEGATIVE", "correct": "True"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "nrc.csv"
            metrics = analyze_and_enrich(rows, lambda row: "TRUST", output)
            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))

        self.assertEqual(len(saved), 3)
        self.assertTrue(all(row["nrc_emotion"] == "TRUST" for row in saved))
        self.assertEqual(metrics["total_reviews"], 3)
        self.assertEqual(metrics["correct"], 2)
        self.assertAlmostEqual(metrics["overall_accuracy"], 2 / 3)
        self.assertEqual(metrics["star_rating_distribution"], {"1": 1, "2": 0, "3": 1, "4": 0, "5": 1})
        self.assertEqual(metrics["actual_sentiment_distribution"], {"POSITIVE": 1, "NEUTRAL": 1, "NEGATIVE": 1})
        self.assertEqual(metrics["predicted_sentiment_distribution"], {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": 2})
        self.assertEqual(metrics["confusion_matrix"]["NEUTRAL"]["NEGATIVE"], 1)
        self.assertEqual(metrics["accuracy_by_class"]["NEUTRAL"], 0.0)
        self.assertEqual(metrics["nrc_emotion_distribution"], {"TRUST": 3})

    def test_rejects_actual_sentiment_that_disagrees_with_rating(self):
        from gift_cards.offline_balanced_analysis import analyze_and_enrich

        row = {"sample_id": "1", "source_line": "10", "rating": "3.0", "title": "Okay", "text": "Average", "actual_sentiment": "POSITIVE", "predicted_sentiment": "POSITIVE", "correct": "True"}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "ground truth"):
                analyze_and_enrich([row], lambda value: "TRUST", Path(folder) / "out.csv")


if __name__ == "__main__":
    unittest.main()
