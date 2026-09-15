"""Offline input-boundary contracts; synthetic fixtures only."""
import importlib
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class SentimentTests(unittest.TestCase):
    def test_only_title_and_text_cross_input_boundary(self):
        self.assertTrue((ROOT / 'src/gift_cards/sentiment.py').exists(), 'Sentiment module must exist')
        module = importlib.import_module('gift_cards.sentiment')
        record = {'title': 'Excellent', 'text': 'Works perfectly.', 'rating': 1,
                  'user_id': 'PRIVATE_SENTINEL', 'helpful_vote': 999}
        payload = module.prepare_review(record)
        self.assertEqual(json.loads(payload), {'title': 'Excellent', 'text': 'Works perfectly.'})
        record['rating'] = 5
        self.assertEqual(payload, module.prepare_review(record))
        self.assertNotIn('PRIVATE_SENTINEL', payload)
        self.assertNotIn('rating', payload)


    def test_rating_like_text_is_held_before_network(self):
        module = importlib.import_module('gift_cards.sentiment')
        for text in ('Five Stars', 'I rate it highly', '5/5', '５ out of ５',
                     '&#53; stars', '★★★', 'score: ten', 'It cost $20',
                     's\u200btars', '', '   ', None, 5):
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.prepare_review({'title': '', 'text': text})
        self.assertEqual(json.loads(module.prepare_review({'title': 'Great!', 'text': ''})),
                         {'title': 'Great!', 'text': ''})


    def test_parser_accepts_only_exact_label_schema(self):
        module = importlib.import_module('gift_cards.sentiment')
        self.assertTrue(hasattr(module, 'parse_sentiment'), 'Strict parser must exist')
        for label in ('POSITIVE', 'NEGATIVE'):
            self.assertEqual(module.parse_sentiment(json.dumps({'sentiment': label})), label)
        for raw in ('{"sentiment":"NEUTRAL"}', '{"sentiment":"positive"}',
                    '{"sentiment":"POSITIVE","reason":"nice"}',
                    '{"sentiment":"POSITIVE","sentiment":"NEGATIVE"}',
                    '```json\n{"sentiment":"POSITIVE"}\n```', '{}', '[]',
                    '{"sentiment":null}', '{"sentiment":[]}', 'null',
                    '{"sentiment":"POSITIVE"} extra'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                module.parse_sentiment(raw)


    def test_classifier_sends_only_safe_input_and_validates_response(self):
        module = importlib.import_module('gift_cards.sentiment')
        self.assertTrue(hasattr(module, 'classify'), 'Classifier must exist')
        client = MagicMock()
        response = SimpleNamespace(status='completed', output=[],
                                   id='fixture', model='fixture-model', usage=None)
        events = [SimpleNamespace(type='response.output_text.done', text='{"sentiment":"POSITIVE"}'),
                  SimpleNamespace(type='response.completed', response=response)]
        client.responses.create.return_value.__enter__.return_value = events
        settings = {'model': 'fixture-model', 'reasoning_effort': 'low', 'timeout_seconds': 120}
        record = {'title': 'Great', 'text': 'Works perfectly.', 'rating': 1, 'expected': 'NEVER_SEND'}
        result = module.classify(record, client, settings, 'Test instructions')
        self.assertEqual(result['sentiment'], 'POSITIVE')
        sent = client.responses.create.call_args.kwargs
        self.assertEqual(json.loads(sent['input'][0]['content']), {'title': 'Great', 'text': 'Works perfectly.'})
        self.assertNotIn('NEVER_SEND', json.dumps(sent))
        self.assertTrue(sent['text']['format']['strict'])
        self.assertFalse(sent['store'])
        self.assertTrue(sent['stream'])
        client.reset_mock()
        with self.assertRaises(ValueError):
            module.classify({'title': 'Five Stars', 'text': 'Good'}, client, settings, 'Test')
        client.responses.create.assert_not_called()
        response.status = 'incomplete'
        with self.assertRaises(ValueError):
            module.classify(record, client, settings, 'Test')


if __name__ == '__main__':
    unittest.main()
