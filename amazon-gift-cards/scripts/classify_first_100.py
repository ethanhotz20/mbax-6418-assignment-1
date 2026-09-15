"""Classify the first 100 usable Gift Cards reviews and report accuracy."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gift_cards.evaluate import EvaluationAPIError, evaluate, select_first_usable
from gift_cards.sentiment import classify

DATASET = ROOT / 'data/raw/Gift_Cards.jsonl.gz'
OUTPUT_DIR = ROOT / 'data/processed/first_100_sentiment_gpt-5.6-sol'
RESULTS = OUTPUT_DIR / 'results.csv'
INCORRECT = OUTPUT_DIR / 'incorrect_reviews.csv'
SUMMARY = OUTPUT_DIR / 'summary.json'


def main():
    settings = json.loads((ROOT / 'config/sentiment.json').read_text(encoding='utf-8'))
    prompt = (ROOT / 'prompts/sentiment_v1.txt').read_text(encoding='utf-8')
    reviews, skipped = select_first_usable(DATASET, limit=100)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f'Output directory is not empty: {OUTPUT_DIR}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import agent.auxiliary_client as auxiliary
    from hermes_cli.config import load_config

    hermes_model = load_config().get('model', {})
    expected = {'provider': settings['provider'], 'default': settings['model'],
                'base_url': settings['base_url']}
    if not isinstance(hermes_model, dict) or any(hermes_model.get(key) != value
                                                for key, value in expected.items()):
        raise RuntimeError('Hermes provider, model, or endpoint does not match config/sentiment.json')

    client, resolved_model = auxiliary.resolve_provider_client(
        settings['provider'], model=settings['model'], raw_codex=True)
    if client is None or resolved_model != settings['model']:
        raise RuntimeError('Hermes authentication or exact model resolution is unavailable')
    if str(client.base_url).rstrip('/') != settings['base_url'].rstrip('/'):
        client.close()
        raise RuntimeError('Resolved endpoint does not match config/sentiment.json')
    client = client.with_options(max_retries=0)

    def predict(review):
        return classify(review, client, settings, prompt)['sentiment']

    try:
        metrics = evaluate(reviews, predict, RESULTS)
    except EvaluationAPIError as exc:
        completed = exc.review_number - 1
        status = {'status': 'stopped', 'http_status': exc.status_code,
                  'sample_review_number': exc.review_number,
                  'source_line': exc.source_line, 'completed_reviews': completed,
                  'message': ('HTTP 429 stopped model calls immediately at sample review '
                              f'{exc.review_number} (dataset source line {exc.source_line}).')}
        SUMMARY.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        print(status['message'])
        print(f'Partial CSV contains {completed} completed reviews: {RESULTS}')
        return 1
    finally:
        client.close()

    with RESULTS.open(encoding='utf-8', newline='') as stream:
        incorrect = [row for row in csv.DictReader(stream) if row['correct'] == 'False']
    with INCORRECT.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rating', 'title', 'text',
                                                    'actual_sentiment',
                                                    'predicted_sentiment', 'correct'])
        writer.writeheader()
        writer.writerows(incorrect)

    summary = {'status': 'completed', **metrics, 'usable_reviews': 100,
               'skipped_before_selection_completed': skipped,
               'last_selected_source_line': reviews[-1]['_source_line'],
               'model': settings['model']}
    SUMMARY.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    print(f"Total reviews: {metrics['total_reviews']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Overall accuracy: {metrics['accuracy']:.2%}")
    print(f"Actual positive reviews: {metrics['actual_positive']}")
    print(f"Actual negative reviews: {metrics['actual_negative']}")
    print('\nIncorrectly classified reviews:')
    if not incorrect:
        print('None')
    for number, row in enumerate(incorrect, 1):
        print(f"\n{number}. Rating: {row['rating']}; actual: {row['actual_sentiment']}; "
              f"predicted: {row['predicted_sentiment']}")
        print(f"Title: {row['title']}")
        print(f"Text: {row['text']}")
    print(f'\nResults CSV: {RESULTS}')
    print(f'Incorrect reviews CSV: {INCORRECT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
