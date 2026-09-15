"""Tests for saved three-class predictions and metrics."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ThreeClassEvaluationTests(unittest.TestCase):
    def test_writes_results_and_calculates_requested_metrics(self):
        from gift_cards.three_class_evaluate import evaluate

        actual = ["POSITIVE"] * 3 + ["NEUTRAL"] * 3 + ["NEGATIVE"] * 3
        predicted = iter(
            ["POSITIVE", "POSITIVE", "NEGATIVE",
             "POSITIVE", "NEGATIVE", "NEUTRAL",
             "NEGATIVE", "NEGATIVE", "POSITIVE"]
        )
        records = [
            {
                "sample_id": number,
                "source_line": number + 100,
                "rating": {"POSITIVE": 5, "NEUTRAL": 3, "NEGATIVE": 1}[label],
                "title": f"Title {chr(96 + number)}",
                "text": f"Text {chr(96 + number)}",
                "actual_sentiment": label,
            }
            for number, label in enumerate(actual, 1)
        ]
        seen = []

        def predict(review):
            seen.append(review)
            return next(predicted)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "results.csv"
            metrics = evaluate(records, predict, output)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertTrue(all(set(review) == {"title", "text"} for review in seen))
        self.assertEqual(
            list(rows[0]),
            ["sample_id", "source_line", "rating", "title", "text",
             "actual_sentiment", "predicted_sentiment", "correct"],
        )
        self.assertEqual(metrics["total_reviews"], 9)
        self.assertEqual(metrics["correct"], 5)
        self.assertEqual(metrics["incorrect"], 4)
        self.assertEqual(metrics["overall_accuracy"], 5 / 9)
        self.assertEqual(metrics["confusion_matrix"], {
            "POSITIVE": {"POSITIVE": 2, "NEUTRAL": 0, "NEGATIVE": 1},
            "NEUTRAL": {"POSITIVE": 1, "NEUTRAL": 1, "NEGATIVE": 1},
            "NEGATIVE": {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": 2},
        })
        self.assertEqual(metrics["accuracy_by_class"], {
            "POSITIVE": 2 / 3,
            "NEUTRAL": 1 / 3,
            "NEGATIVE": 2 / 3,
        })
        self.assertEqual(metrics["neutral_predictions"], {
            "POSITIVE": 1,
            "NEUTRAL": 1,
            "NEGATIVE": 1,
        })

    def test_http_429_stops_immediately_with_partial_csv(self):
        from gift_cards.three_class_evaluate import EvaluationAPIError, evaluate

        records = [
            {"sample_id": 1, "source_line": 10, "rating": 5, "title": "Good",
             "text": "Worked", "actual_sentiment": "POSITIVE"},
            {"sample_id": 2, "source_line": 20, "rating": 3, "title": "Fine",
             "text": "Acceptable", "actual_sentiment": "NEUTRAL"},
            {"sample_id": 3, "source_line": 30, "rating": 1, "title": "Bad",
             "text": "Failed", "actual_sentiment": "NEGATIVE"},
        ]
        calls = 0

        class TooManyRequests(Exception):
            status_code = 429

        def predict(review):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TooManyRequests("usage limit")
            return "POSITIVE"

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "results.csv"
            with self.assertRaises(EvaluationAPIError) as caught:
                evaluate(records, predict, output)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(calls, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(caught.exception.sample_id, 2)
        self.assertEqual(caught.exception.source_line, 20)


if __name__ == "__main__":
    unittest.main()
