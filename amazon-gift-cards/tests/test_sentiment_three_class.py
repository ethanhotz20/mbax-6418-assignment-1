"""Tests for the rating-blind three-class sentiment experiment."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ThreeClassSentimentTests(unittest.TestCase):
    def test_parser_accepts_only_exact_three_class_schema(self):
        from gift_cards.sentiment_three_class import LABELS, parse_sentiment

        self.assertEqual(LABELS, ("POSITIVE", "NEUTRAL", "NEGATIVE"))
        for label in LABELS:
            self.assertEqual(parse_sentiment(json.dumps({"sentiment": label})), label)
        invalid = [
            '{"sentiment":"positive"}',
            '{"sentiment":"MIXED"}',
            '{"sentiment":"POSITIVE","reason":"good"}',
            '{"sentiment":"POSITIVE","sentiment":"NEGATIVE"}',
            '```json\n{"sentiment":"NEUTRAL"}\n```',
            '{}',
            '[]',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_sentiment(raw)

    def test_classifier_sends_only_title_and_text(self):
        from gift_cards.sentiment_three_class import classify

        client = MagicMock()
        response = SimpleNamespace(
            status="completed", id="fixture", model="fixture-model", usage=None
        )
        client.responses.create.return_value.__enter__.return_value = [
            SimpleNamespace(
                type="response.output_text.done",
                text='{"sentiment":"NEUTRAL"}',
            ),
            SimpleNamespace(type="response.completed", response=response),
        ]
        settings = {
            "model": "fixture-model",
            "reasoning_effort": "low",
            "timeout_seconds": 120,
        }
        record = {
            "title": "It was fine",
            "text": "Nothing especially good or bad.",
            "rating": 5,
            "user_id": "PRIVATE_SENTINEL",
        }

        result = classify(record, client, settings, "Instructions")

        self.assertEqual(result["sentiment"], "NEUTRAL")
        sent = client.responses.create.call_args.kwargs
        self.assertEqual(
            json.loads(sent["input"][0]["content"]),
            {"title": "It was fine", "text": "Nothing especially good or bad."},
        )
        self.assertNotIn("rating", json.dumps(sent))
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(sent))
        self.assertEqual(
            sent["text"]["format"]["schema"]["properties"]["sentiment"]["enum"],
            ["POSITIVE", "NEUTRAL", "NEGATIVE"],
        )
        self.assertTrue(sent["stream"])
        self.assertFalse(sent["store"])


if __name__ == "__main__":
    unittest.main()
