"""Tests for strict LLM emotion output and deterministic NRC scoring."""
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class EmotionTests(unittest.TestCase):
    def test_parser_accepts_exact_sentiment_and_primary_emotion(self):
        from gift_cards.emotion import EMOTIONS, parse_sentiment_emotion

        for emotion in EMOTIONS:
            raw = json.dumps({"sentiment": "POSITIVE", "emotion": emotion})
            self.assertEqual(
                parse_sentiment_emotion(raw),
                {"sentiment": "POSITIVE", "emotion": emotion},
            )

        invalid = [
            '{"sentiment":"NEUTRAL","emotion":"JOY"}',
            '{"sentiment":"POSITIVE","emotion":"HAPPINESS"}',
            '{"sentiment":"POSITIVE","emotion":"joy"}',
            '{"sentiment":"POSITIVE"}',
            '{"sentiment":"POSITIVE","emotion":"JOY","reason":"nice"}',
            '{"sentiment":"POSITIVE","emotion":"JOY","emotion":"TRUST"}',
            '```json\n{"sentiment":"POSITIVE","emotion":"JOY"}\n```',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_sentiment_emotion(raw)

    def test_classifier_sends_only_title_and_text(self):
        from gift_cards.emotion import classify_sentiment_emotion

        client = MagicMock()
        response = SimpleNamespace(
            status="completed", id="fixture", model="fixture-model", usage=None
        )
        events = [
            SimpleNamespace(
                type="response.output_text.done",
                text='{"sentiment":"NEGATIVE","emotion":"ANGER"}',
            ),
            SimpleNamespace(type="response.completed", response=response),
        ]
        client.responses.create.return_value.__enter__.return_value = events
        settings = {
            "model": "fixture-model",
            "reasoning_effort": "low",
            "timeout_seconds": 120,
        }
        record = {
            "title": "Awful",
            "text": "This failed immediately.",
            "rating": 5,
            "user_id": "PRIVATE_SENTINEL",
        }

        result = classify_sentiment_emotion(record, client, settings, "Instructions")

        self.assertEqual(result["sentiment"], "NEGATIVE")
        self.assertEqual(result["emotion"], "ANGER")
        sent = client.responses.create.call_args.kwargs
        self.assertEqual(
            json.loads(sent["input"][0]["content"]),
            {"title": "Awful", "text": "This failed immediately."},
        )
        self.assertNotIn("rating", json.dumps(sent))
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(sent))
        self.assertEqual(
            sent["text"]["format"]["schema"]["required"],
            ["sentiment", "emotion"],
        )
        self.assertEqual(sent["stream"], True)
        self.assertEqual(sent["store"], False)

    def test_nrc_primary_uses_counts_then_fixed_tie_break(self):
        from gift_cards.emotion import primary_emotion_from_scores

        self.assertEqual(primary_emotion_from_scores({"joy": 3, "trust": 1}), "JOY")
        self.assertEqual(
            primary_emotion_from_scores({"anger": 2, "anticipation": 2}),
            "ANGER",
        )
        self.assertEqual(primary_emotion_from_scores({}), "TRUST")
        with self.assertRaises(ValueError):
            primary_emotion_from_scores({"joy": -1})

    def test_nrc_analysis_uses_title_and_text_tokens_only(self):
        from gift_cards.emotion import nrc_primary_emotion

        class FakeAnalyzer:
            received = None

            def load_token_list(self, tokens):
                type(self).received = tokens
                self.raw_emotion_scores = {"joy": 2, "positive": 2}

        record = {
            "title": "Happy gift",
            "text": "Loved it!",
            "rating": 1,
            "user_id": "PRIVATE_SENTINEL",
        }
        result = nrc_primary_emotion(record, analyzer_factory=FakeAnalyzer)

        self.assertEqual(result, "JOY")
        self.assertEqual(FakeAnalyzer.received, ["happy", "gift", "loved", "it"])


if __name__ == "__main__":
    unittest.main()
