"""Tests for first-100 sentiment and emotion evaluation."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class EmotionEvaluationTests(unittest.TestCase):
    def test_writes_both_emotions_match_and_agreement(self):
        from gift_cards.emotion_evaluate import evaluate_emotions

        records = [
            {"rating": "5.0", "title": "Great", "text": "Loved it"},
            {"rating": "1.0", "title": "Bad", "text": "Failed"},
            {"rating": "3.0", "title": "Mixed", "text": "Acceptable"},
        ]
        seen = []
        llm_results = iter(
            [
                {"sentiment": "POSITIVE", "emotion": "JOY"},
                {"sentiment": "NEGATIVE", "emotion": "ANGER"},
                {"sentiment": "POSITIVE", "emotion": "TRUST"},
            ]
        )
        nrc_results = iter(["JOY", "SADNESS", "TRUST"])

        def predict(review):
            seen.append(review)
            return next(llm_results)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "results.csv"
            summary = evaluate_emotions(
                records, predict, lambda review: next(nrc_results), output
            )
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(
            seen,
            [
                {"title": "Great", "text": "Loved it"},
                {"title": "Bad", "text": "Failed"},
                {"title": "Mixed", "text": "Acceptable"},
            ],
        )
        self.assertEqual(
            list(rows[0]),
            [
                "rating",
                "title",
                "text",
                "actual_sentiment",
                "predicted_sentiment",
                "correct",
                "llm_emotion",
                "nrc_emotion",
                "emotion_match",
            ],
        )
        self.assertEqual([row["emotion_match"] for row in rows], ["True", "False", "True"])
        self.assertEqual(summary["emotion_matches"], 2)
        self.assertEqual(summary["emotion_mismatches"], 1)
        self.assertEqual(summary["emotion_agreement_rate"], 2 / 3)
        self.assertEqual(summary["accuracy"], 2 / 3)

    def test_http_429_stops_without_calling_nrc_for_failed_review(self):
        from gift_cards.emotion_evaluate import EmotionEvaluationAPIError, evaluate_emotions

        records = [
            {"rating": 5, "title": "First", "text": "Good", "_source_line": 10},
            {"rating": 1, "title": "Second", "text": "Bad", "_source_line": 12},
            {"rating": 5, "title": "Third", "text": "Good", "_source_line": 13},
        ]
        calls = {"llm": 0, "nrc": 0}

        class TooManyRequests(Exception):
            status_code = 429

        def predict(review):
            calls["llm"] += 1
            if calls["llm"] == 2:
                raise TooManyRequests("usage limit")
            return {"sentiment": "POSITIVE", "emotion": "JOY"}

        def nrc_predict(review):
            calls["nrc"] += 1
            return "JOY"

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "results.csv"
            with self.assertRaises(EmotionEvaluationAPIError) as caught:
                evaluate_emotions(records, predict, nrc_predict, output)
            with output.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(calls, {"llm": 2, "nrc": 1})
        self.assertEqual(len(rows), 1)
        self.assertEqual(caught.exception.review_number, 2)
        self.assertEqual(caught.exception.source_line, 12)


if __name__ == "__main__":
    unittest.main()
