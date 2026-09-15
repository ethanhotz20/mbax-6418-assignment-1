"""Tests for selecting and scoring a small rating-blind evaluation sample."""
import csv
import gzip
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class EvaluationTests(unittest.TestCase):
    def test_selects_first_usable_reviews_in_source_order(self):
        self.assertTrue((ROOT / 'src/gift_cards/evaluate.py').exists(), 'Evaluation module must exist')
        module = importlib.import_module('gift_cards.evaluate')
        records = [
            {'rating': 5.0, 'title': '', 'text': ''},
            {'rating': 1.0, 'title': 'Five Stars', 'text': 'Bad'},
            {'rating': 5.0, 'title': 'Great', 'text': 'Loved it'},
            {'rating': 3.0, 'title': 'Poor', 'text': 'Did not work'},
            {'rating': 4.0, 'title': 'Later', 'text': 'Good'},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reviews.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record) + '\n')
            selected, skipped = module.select_first_usable(path, limit=2)
        self.assertEqual([r['title'] for r in selected], ['Great', 'Poor'])
        self.assertEqual([r['_source_line'] for r in selected], [3, 4])
        self.assertEqual(skipped, 2)

    def test_scores_after_prediction_and_writes_exact_csv_columns(self):
        module = importlib.import_module('gift_cards.evaluate')
        records = [
            {'rating': 5.0, 'title': 'Great', 'text': 'Loved it', '_source_line': 1},
            {'rating': 2.0, 'title': 'Bad', 'text': 'Failed', '_source_line': 2},
            {'rating': 3.0, 'title': 'Mixed', 'text': 'Acceptable', '_source_line': 3},
        ]
        seen = []
        predictions = iter(['POSITIVE', 'POSITIVE', 'NEGATIVE'])
        def predict(review):
            seen.append(review)
            return next(predictions)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.csv'
            summary = module.evaluate(records, predict, output)
            with output.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(seen, [
            {'title': 'Great', 'text': 'Loved it'},
            {'title': 'Bad', 'text': 'Failed'},
            {'title': 'Mixed', 'text': 'Acceptable'},
        ])
        self.assertEqual(list(rows[0]), ['rating', 'title', 'text', 'actual_sentiment',
                                        'predicted_sentiment', 'correct'])
        self.assertEqual(summary, {'total_reviews': 3, 'correct': 2, 'incorrect': 1,
                                   'accuracy': 2 / 3, 'actual_positive': 1,
                                   'actual_negative': 2})
        self.assertEqual(rows[1]['correct'], 'False')

    def test_http_429_stops_immediately_and_reports_exact_review(self):
        module = importlib.import_module('gift_cards.evaluate')
        records = [
            {'rating': 5.0, 'title': 'First', 'text': 'Good', '_source_line': 10},
            {'rating': 1.0, 'title': 'Second', 'text': 'Bad', '_source_line': 12},
            {'rating': 5.0, 'title': 'Third', 'text': 'Good', '_source_line': 13},
        ]
        calls = 0
        class TooManyRequests(Exception):
            status_code = 429
        def predict(review):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise TooManyRequests('usage limit')
            return 'POSITIVE'
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'results.csv'
            with self.assertRaises(module.EvaluationAPIError) as caught:
                module.evaluate(records, predict, output)
            with output.open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(calls, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(caught.exception.review_number, 2)
        self.assertEqual(caught.exception.source_line, 12)
        self.assertEqual(caught.exception.status_code, 429)


if __name__ == '__main__':
    unittest.main()
