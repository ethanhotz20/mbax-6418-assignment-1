"""Build one self-contained HTML dashboard from the saved results CSV."""
import argparse
import csv
from html import escape
from pathlib import Path

COLUMNS = ['rating', 'title', 'text', 'actual_sentiment',
           'predicted_sentiment', 'correct', 'llm_emotion',
           'nrc_emotion', 'emotion_match']
EMOTIONS = {'ANGER', 'ANTICIPATION', 'DISGUST', 'FEAR', 'JOY',
            'SADNESS', 'SURPRISE', 'TRUST'}


def read_results(path):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f'Expected CSV columns: {COLUMNS}')
        rows = list(reader)
    if not rows:
        raise ValueError('Results CSV is empty')
    for row_number, row in enumerate(rows, 2):
        if row['actual_sentiment'] not in ('POSITIVE', 'NEGATIVE'):
            raise ValueError(f'Invalid actual sentiment on CSV line {row_number}')
        if row['predicted_sentiment'] not in ('POSITIVE', 'NEGATIVE'):
            raise ValueError(f'Invalid predicted sentiment on CSV line {row_number}')
        expected_correct = str(row['actual_sentiment'] == row['predicted_sentiment'])
        if row['correct'] != expected_correct:
            raise ValueError(f'Inconsistent correct value on CSV line {row_number}')
        if row['llm_emotion'] not in EMOTIONS or row['nrc_emotion'] not in EMOTIONS:
            raise ValueError(f'Invalid emotion on CSV line {row_number}')
        expected_match = str(row['llm_emotion'] == row['nrc_emotion'])
        if row['emotion_match'] != expected_match:
            raise ValueError(f'Inconsistent emotion_match on CSV line {row_number}')
    return rows


def build_html(rows):
    total = len(rows)
    correct = sum(row['correct'] == 'True' for row in rows)
    incorrect = total - correct
    accuracy = correct / total
    emotion_matches = sum(row['emotion_match'] == 'True' for row in rows)
    emotion_agreement = emotion_matches / total
    actual_positive = sum(row['actual_sentiment'] == 'POSITIVE' for row in rows)
    actual_negative = sum(row['actual_sentiment'] == 'NEGATIVE' for row in rows)
    predicted_positive = sum(row['predicted_sentiment'] == 'POSITIVE' for row in rows)
    predicted_negative = sum(row['predicted_sentiment'] == 'NEGATIVE' for row in rows)
    tp = sum(row['actual_sentiment'] == 'POSITIVE' and row['predicted_sentiment'] == 'POSITIVE' for row in rows)
    fn = sum(row['actual_sentiment'] == 'POSITIVE' and row['predicted_sentiment'] == 'NEGATIVE' for row in rows)
    fp = sum(row['actual_sentiment'] == 'NEGATIVE' and row['predicted_sentiment'] == 'POSITIVE' for row in rows)
    tn = sum(row['actual_sentiment'] == 'NEGATIVE' and row['predicted_sentiment'] == 'NEGATIVE' for row in rows)

    review_rows = []
    for number, row in enumerate(rows, 1):
        status = 'correct' if row['correct'] == 'True' else 'incorrect'
        review_rows.append(
            f'<tr class="review-row {status}">'
            f'<td>{number}</td><td>{escape(row["rating"])}</td>'
            f'<td>{escape(row["title"])}</td><td>{escape(row["text"])}</td>'
            f'<td><span class="label">{escape(row["actual_sentiment"])}</span></td>'
            f'<td><span class="label">{escape(row["predicted_sentiment"])}</span></td>'
            f'<td>{escape(row["llm_emotion"])}</td><td>{escape(row["nrc_emotion"])}</td>'
            f'<td><span class="result {status}">{"Yes" if status == "correct" else "No"}</span></td>'
            '</tr>')

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amazon Gift Cards Sentiment and Emotion Results</title>
<style>
:root {{ --ink:#17202a; --muted:#5f6b76; --line:#d9dee3; --surface:#f6f7f8; --good:#176b45; --good-bg:#eaf6ef; --bad:#9b2c2c; --bad-bg:#fff0f0; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:white; font:15px/1.5 Arial, Helvetica, sans-serif; }}
main {{ width:min(1180px, calc(100% - 32px)); margin:32px auto 56px; }}
h1 {{ margin:0 0 6px; font-size:28px; }}
h2 {{ margin:32px 0 12px; font-size:20px; }}
.subtitle {{ margin:0 0 24px; color:var(--muted); }}
.metrics {{ display:grid; grid-template-columns:repeat(5, minmax(140px, 1fr)); border:1px solid var(--line); }}
.metric {{ padding:18px; border-right:1px solid var(--line); }}
.metric:last-child {{ border-right:0; }}
.metric strong {{ display:block; font-size:26px; line-height:1.15; }}
.metric span {{ color:var(--muted); }}
.counts {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.count-box {{ padding:16px 18px; background:var(--surface); border:1px solid var(--line); }}
.count-box h3 {{ margin:0 0 8px; font-size:16px; }}
.count-box p {{ margin:4px 0; }}
table {{ width:100%; border-collapse:collapse; }}
th, td {{ padding:10px 12px; border:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:var(--surface); font-size:13px; }}
.matrix {{ width:auto; min-width:430px; }}
.matrix td {{ text-align:center; font-size:18px; font-weight:700; }}
.matrix th {{ text-align:center; }}
.table-wrap {{ overflow-x:auto; }}
.reviews {{ min-width:1000px; }}
.reviews td:nth-child(1), .reviews td:nth-child(2), .reviews td:nth-child(7) {{ white-space:nowrap; }}
.reviews tr.incorrect {{ background:var(--bad-bg); }}
.filters {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:0 0 12px; }}
.filters button {{ padding:8px 12px; border:1px solid var(--line); background:white; color:var(--ink); cursor:pointer; border-radius:3px; }}
.filters button[aria-pressed="true"] {{ background:var(--ink); color:white; border-color:var(--ink); }}
.visible-count {{ margin-left:auto; color:var(--muted); }}
.filtered-out {{ display:none; }}
.label {{ font-size:12px; font-weight:700; }}
.result {{ display:inline-block; min-width:42px; padding:2px 8px; text-align:center; font-weight:700; border-radius:3px; }}
.result.correct {{ color:var(--good); background:var(--good-bg); }}
.result.incorrect {{ color:var(--bad); background:var(--bad-bg); }}
@media (max-width:760px) {{ .metrics {{ grid-template-columns:1fr 1fr; }} .metric:nth-child(2) {{ border-right:0; }} .metric:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }} .counts {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<main data-total="{total}" data-correct="{correct}" data-incorrect="{incorrect}"
      data-actual-positive="{actual_positive}" data-actual-negative="{actual_negative}"
      data-predicted-positive="{predicted_positive}" data-predicted-negative="{predicted_negative}"
      data-tp="{tp}" data-fn="{fn}" data-fp="{fp}" data-tn="{tn}"
      data-emotion-agreement="{emotion_agreement:.6f}">
  <h1>Amazon Gift Cards Sentiment and Emotion Results</h1>
  <p class="subtitle">Sentiment and emotion classification from the saved CSV results.</p>

  <section class="metrics" aria-label="Evaluation summary">
    <div class="metric"><strong>{total}</strong><span>Total reviews</span></div>
    <div class="metric"><strong>{accuracy:.2%}</strong><span>Overall accuracy</span></div>
    <div class="metric"><strong>{correct}</strong><span>Correct</span></div>
    <div class="metric"><strong>{incorrect}</strong><span>Incorrect</span></div>
    <div class="metric"><strong>{emotion_agreement:.2%}</strong><span>Emotion agreement</span></div>
  </section>

  <h2>Sentiment counts</h2>
  <section class="counts">
    <div class="count-box"><h3>Actual</h3><p>Positive: <strong>{actual_positive}</strong></p><p>Negative: <strong>{actual_negative}</strong></p></div>
    <div class="count-box"><h3>Predicted</h3><p>Positive: <strong>{predicted_positive}</strong></p><p>Negative: <strong>{predicted_negative}</strong></p></div>
  </section>

  <h2>Confusion matrix</h2>
  <div class="table-wrap">
    <table class="matrix">
      <thead><tr><th>Actual ↓ / Predicted →</th><th>Positive</th><th>Negative</th></tr></thead>
      <tbody><tr><th>Positive</th><td>{tp}</td><td>{fn}</td></tr><tr><th>Negative</th><td>{fp}</td><td>{tn}</td></tr></tbody>
    </table>
  </div>

  <h2>Review results</h2>
  <div class="filters" aria-label="Filter reviews">
    <button type="button" data-filter="all" aria-pressed="true">All Reviews</button>
    <button type="button" data-filter="correct" aria-pressed="false">Correct Predictions</button>
    <button type="button" data-filter="incorrect" aria-pressed="false">Incorrect Predictions</button>
    <span class="visible-count" aria-live="polite">Visible rows: <strong id="visible-count">{total}</strong></span>
  </div>
  <div class="table-wrap">
    <table class="reviews">
      <thead><tr><th>#</th><th>Rating</th><th>Title</th><th>Review text</th><th>Actual</th><th>Predicted</th><th scope="col">LLM emotion</th><th scope="col">NRC emotion</th><th>Correct</th></tr></thead>
      <tbody>{''.join(review_rows)}</tbody>
    </table>
  </div>
</main>
<script>
const rows = Array.from(document.querySelectorAll('tr.review-row'));
const buttons = document.querySelectorAll('[data-filter]');
const visibleCount = document.getElementById('visible-count');
buttons.forEach(button => button.addEventListener('click', () => {{
  const filter = button.dataset.filter;
  let visible = 0;
  rows.forEach(row => {{
    const show = filter === 'all' || row.classList.contains(filter);
    row.classList.toggle('filtered-out', !show);
    if (show) visible += 1;
  }});
  visibleCount.textContent = visible;
  buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
}}));
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_results(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(rows), encoding='utf-8')
    print(f'Created {args.output} from {len(rows)} CSV rows')


if __name__ == '__main__':
    main()
