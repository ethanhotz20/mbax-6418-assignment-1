"""Tests for rating-blind emotion-only inference."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class EmotionOnlyTests(unittest.TestCase):
    def test_parser_accepts_only_one_allowed_emotion(self):
        from gift_cards.emotion_only import EMOTIONS, parse_emotion

        for emotion in EMOTIONS:
            self.assertEqual(parse_emotion(json.dumps({"emotion": emotion})), emotion)
        for raw in (
            '{"emotion":"joy"}',
            '{"emotion":"HAPPINESS"}',
            '{"emotion":"JOY","reason":"happy"}',
            '{"emotion":"JOY","emotion":"TRUST"}',
            '{}',
            '[]',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_emotion(raw)

    def test_classifier_sends_only_title_and_text(self):
        from gift_cards.emotion_only import classify

        client = MagicMock()
        response = SimpleNamespace(
            status="completed", id="fixture", model="fixture-model", usage=None
        )
        client.responses.create.return_value.__enter__.return_value = [
            SimpleNamespace(type="response.output_text.done", text='{"emotion":"JOY"}'),
            SimpleNamespace(type="response.completed", response=response),
        ]
        settings = {
            "model": "fixture-model",
            "reasoning_effort": "low",
            "timeout_seconds": 120,
        }
        record = {
            "title": "Wonderful",
            "text": "This made me happy.",
            "rating": 1,
            "actual_sentiment": "NEGATIVE",
            "user_id": "PRIVATE_SENTINEL",
        }

        result = classify(record, client, settings, "Instructions")

        self.assertEqual(result["emotion"], "JOY")
        sent = client.responses.create.call_args.kwargs
        self.assertEqual(
            json.loads(sent["input"][0]["content"]),
            {"title": "Wonderful", "text": "This made me happy."},
        )
        self.assertNotIn("rating", json.dumps(sent))
        self.assertNotIn("NEGATIVE", json.dumps(sent))
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(sent))
        self.assertTrue(sent["stream"])
        self.assertFalse(sent["store"])


if __name__ == "__main__":
    unittest.main()
