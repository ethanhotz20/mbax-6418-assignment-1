"""Tests for adding two emotion methods to saved three-class rows."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class BalancedEmotionTests(unittest.TestCase):
    def test_enriches_rows_and_calculates_emotion_summary(self):
        from gift_cards.balanced_emotions import enrich

        rows = [
            {"sample_id": "1", "source_line": "10", "rating": "5.0",
             "title": "Great", "text": "Loved it", "actual_sentiment": "POSITIVE",
             "predicted_sentiment": "POSITIVE", "correct": "True"},
            {"sample_id": "2", "source_line": "20", "rating": "3.0",
             "title": "Mixed", "text": "Some problems", "actual_sentiment": "NEUTRAL",
             "predicted_sentiment": "NEGATIVE", "correct": "False"},
        ]
        llm = iter(["JOY", "SADNESS"])
        nrc = iter(["JOY", "ANGER"])
        seen = []

        def predict(review):
            seen.append(review)
            return next(llm)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "results_with_emotions.csv"
            summary = enrich(rows, predict, lambda review: next(nrc), output)
            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))

        self.assertEqual(seen, [
            {"title": "Great", "text": "Loved it"},
            {"title": "Mixed", "text": "Some problems"},
        ])
        self.assertEqual(list(saved[0])[-3:], ["llm_emotion", "nrc_emotion", "emotion_match"])
        self.assertEqual([row["emotion_match"] for row in saved], ["True", "False"])
        self.assertEqual(summary["emotion_matches"], 1)
        self.assertEqual(summary["emotion_mismatches"], 1)
        self.assertEqual(summary["emotion_agreement_rate"], 0.5)
        self.assertEqual(summary["llm_emotion_distribution"], {"JOY": 1, "SADNESS": 1})
        self.assertEqual(summary["nrc_emotion_distribution"], {"ANGER": 1, "JOY": 1})

    def test_http_429_stops_at_exact_sample(self):
        from gift_cards.balanced_emotions import EnrichmentAPIError, enrich

        rows = [
            {"sample_id": str(number), "source_line": str(number * 10), "rating": "5.0",
             "title": "Good", "text": "Worked", "actual_sentiment": "POSITIVE",
             "predicted_sentiment": "POSITIVE", "correct": "True"}
            for number in (1, 2, 3)
        ]
        calls = 0

        class TooManyRequests(Exception):
            status_code = 429

        def predict(review):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TooManyRequests("usage limit")
            return "JOY"

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "partial.csv"
            with self.assertRaises(EnrichmentAPIError) as caught:
                enrich(rows, predict, lambda review: "JOY", output)
            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))

        self.assertEqual(calls, 2)
        self.assertEqual(len(saved), 1)
        self.assertEqual(caught.exception.sample_id, 2)
        self.assertEqual(caught.exception.source_line, 20)

    def test_resume_preserves_completed_rows_and_calls_only_missing_rows(self):
        from gift_cards.balanced_emotions import enrich, resume_enrich

        base_rows = [
            {"sample_id": str(number), "source_line": str(number * 10), "rating": "5.0",
             "title": f"Title {chr(96 + number)}", "text": "Worked",
             "actual_sentiment": "POSITIVE", "predicted_sentiment": "POSITIVE",
             "correct": "True"}
            for number in (1, 2, 3)
        ]
        calls = []

        def predict(review):
            calls.append(review["title"])
            return "JOY"

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "partial.csv"
            enrich(base_rows[:1], lambda review: "ANGER", lambda review: "ANGER", output)
            summary = resume_enrich(
                base_rows, predict, lambda review: "JOY", output
            )
            with output.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))

        self.assertEqual(calls, ["Title b", "Title c"])
        self.assertEqual(len(saved), 3)
        self.assertEqual(saved[0]["llm_emotion"], "ANGER")
        self.assertEqual(summary["total_reviews"], 3)
        self.assertEqual(summary["emotion_matches"], 3)
        self.assertEqual(summary["emotion_agreement_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
