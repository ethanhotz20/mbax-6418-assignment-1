"""Small, rating-blind sentiment evaluation helpers."""
import csv
import gzip
import json

from gift_cards.sentiment import prepare_review

COLUMNS = ['rating', 'title', 'text', 'actual_sentiment',
           'predicted_sentiment', 'correct']


class EvaluationAPIError(Exception):
    def __init__(self, review_number, source_line, status_code):
        super().__init__(f'API stopped at review {review_number}, source line {source_line}, HTTP {status_code}')
        self.review_number = review_number
        self.source_line = source_line
        self.status_code = status_code


def select_first_usable(path, limit=100):
    """Return the first usable reviews in source order and the number skipped."""
    selected = []
    skipped = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for source_line, line in enumerate(stream, 1):
            record = json.loads(line)
            rating = record.get('rating')
            if rating not in (1, 2, 3, 4, 5):
                skipped += 1
                continue
            try:
                prepare_review({'title': record.get('title'), 'text': record.get('text')})
            except ValueError:
                skipped += 1
                continue
            selected.append({'rating': rating, 'title': record['title'],
                             'text': record['text'], '_source_line': source_line})
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise ValueError(f'Found only {len(selected)} usable reviews; required {limit}')
    return selected, skipped


def rating_to_sentiment(rating):
    return 'POSITIVE' if rating in (4, 5) else 'NEGATIVE'


def evaluate(records, predict, output_path):
    """Predict from title/text only, then derive actual labels and append each row."""
    totals = {'total_reviews': 0, 'correct': 0, 'incorrect': 0,
              'accuracy': 0.0, 'actual_positive': 0, 'actual_negative': 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for review_number, record in enumerate(records, 1):
            try:
                predicted = predict({'title': record['title'], 'text': record['text']})
            except Exception as exc:
                if getattr(exc, 'status_code', None) == 429:
                    raise EvaluationAPIError(review_number, record['_source_line'], 429) from None
                raise
            if predicted not in ('POSITIVE', 'NEGATIVE'):
                raise ValueError(f'Invalid predicted sentiment: {predicted!r}')
            actual = rating_to_sentiment(record['rating'])
            correct = predicted == actual
            writer.writerow({'rating': record['rating'], 'title': record['title'],
                             'text': record['text'], 'actual_sentiment': actual,
                             'predicted_sentiment': predicted, 'correct': correct})
            stream.flush()
            totals['total_reviews'] += 1
            totals['correct' if correct else 'incorrect'] += 1
            totals['actual_positive' if actual == 'POSITIVE' else 'actual_negative'] += 1
    totals['accuracy'] = totals['correct'] / totals['total_reviews'] if totals['total_reviews'] else 0.0
    return totals
