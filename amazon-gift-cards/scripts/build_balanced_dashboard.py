"""Build a self-contained dashboard from balanced three-class results with emotions."""
import argparse
import csv
from html import escape
from pathlib import Path

LABELS = ("POSITIVE", "NEUTRAL", "NEGATIVE")
EMOTIONS = (
    "ANGER", "ANTICIPATION", "DISGUST", "FEAR",
    "JOY", "SADNESS", "SURPRISE", "TRUST",
)
NRC_COLUMNS = [
    "sample_id", "source_line", "rating", "title", "text",
    "actual_sentiment", "predicted_sentiment", "correct",
    "nrc_emotion",
]
LLM_COLUMNS = NRC_COLUMNS[:-1] + ["llm_emotion", "nrc_emotion", "emotion_match"]


def read_results(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != NRC_COLUMNS:
            raise ValueError(f"Expected CSV columns: {NRC_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise ValueError("Results CSV is empty")
    for line, row in enumerate(rows, 2):
        if row["actual_sentiment"] not in LABELS or row["predicted_sentiment"] not in LABELS:
            raise ValueError(f"Invalid sentiment on CSV line {line}")
        if row["nrc_emotion"] not in EMOTIONS:
            raise ValueError(f"Invalid emotion on CSV line {line}")
        if row["correct"] != str(row["actual_sentiment"] == row["predicted_sentiment"]):
            raise ValueError(f"Inconsistent correct value on CSV line {line}")

        rating = float(row["rating"])
        if rating not in (1, 2, 3, 4, 5):
            raise ValueError(f"Invalid rating on CSV line {line}")
    return rows


def read_partial_llm(path, rows):
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != LLM_COLUMNS:
            raise ValueError(f"Expected partial LLM columns: {LLM_COLUMNS}")
        partial = list(reader)
    valid_ids = {row["sample_id"] for row in rows}
    if any(row["sample_id"] not in valid_ids or row["llm_emotion"] not in EMOTIONS for row in partial):
        raise ValueError("Invalid partial LLM emotion rows")
    return {row["sample_id"]: row["llm_emotion"] for row in partial}


def counts(rows, key, values):
    return {value: sum(row[key] == value for row in rows) for value in values}


def build_html(rows, llm_by_id=None):
    llm_by_id = llm_by_id or {}
    total = len(rows)
    correct = sum(row["correct"] == "True" for row in rows)
    accuracy = correct / total
    rating_counts = {
        rating: sum(float(row["rating"]) == rating for row in rows)
        for rating in range(1, 6)
    }
    actual = counts(rows, "actual_sentiment", LABELS)
    predicted = counts(rows, "predicted_sentiment", LABELS)
    confusion = {
        actual_label: {
            predicted_label: sum(
                row["actual_sentiment"] == actual_label
                and row["predicted_sentiment"] == predicted_label
                for row in rows
            )
            for predicted_label in LABELS
        }
        for actual_label in LABELS
    }
    class_accuracy = {
        label: confusion[label][label] / actual[label] if actual[label] else 0.0
        for label in LABELS
    }
    nrc_emotions = counts(rows, "nrc_emotion", EMOTIONS)
    largest_error = max(
        confusion[actual_label][predicted_label]
        for actual_label in LABELS for predicted_label in LABELS
        if actual_label != predicted_label
    )

    data = [
        f'data-total="{total}"',
        f'data-accuracy="{accuracy:.6f}"',
        f'data-correct="{correct}"',
        f'data-incorrect="{total - correct}"',
        f'data-llm-emotion-completed="{len(llm_by_id)}"',
    ]
    data.extend(f'data-rating-{rating}="{rating_counts[rating]}"' for rating in range(1, 6))
    for label in LABELS:
        lower = label.lower()
        data.extend((
            f'data-actual-{lower}="{actual[label]}"',
            f'data-predicted-{lower}="{predicted[label]}"',
            f'data-class-{lower}="{class_accuracy[label]:.6f}"',
        ))
        for predicted_label in LABELS:
            data.append(
                f'data-cm-{lower}-{predicted_label.lower()}="{confusion[label][predicted_label]}"'
            )
    for emotion in EMOTIONS:
        data.append(f'data-nrc-{emotion.lower()}="{nrc_emotions[emotion]}"')
    data.extend(
        f'data-neutral-predicted-{label.lower()}="{confusion["NEUTRAL"][label]}"'
        for label in LABELS
    )

    review_rows = []
    for row in rows:
        status = "correct" if row["correct"] == "True" else "incorrect"
        llm_emotion = llm_by_id.get(row["sample_id"], "Not available")
        review_rows.append(
            f'<tr class="review-row {status}">'
            f'<td>{escape(row["sample_id"])}</td><td>{escape(row["rating"])}</td>'
            f'<td>{escape(row["title"])}</td><td>{escape(row["text"])}</td>'
            f'<td>{escape(row["actual_sentiment"])}</td>'
            f'<td>{escape(row["predicted_sentiment"])}</td>'
            f'<td><span class="result {status}">{"Yes" if status == "correct" else "No"}</span></td>'
            f'<td>{escape(row["nrc_emotion"])}</td><td>{escape(llm_emotion)}</td></tr>'
        )

    rating_rows = "".join(
        f"<tr><th>{rating} star</th><td>{rating_counts[rating]}</td><td>{rating_counts[rating] / total:.2%}</td></tr>"
        for rating in range(1, 6)
    )
    sentiment_rows = "".join(
        f"<tr><th>{label.title()}</th><td>{actual[label]}</td><td>{predicted[label]}</td></tr>"
        for label in LABELS
    )
    matrix_header = "".join(f"<th>{label.title()}</th>" for label in LABELS)
    matrix_rows = "".join(
        f"<tr><th>{actual_label.title()}</th>"
        + "".join(
            '<td{}>{}</td>'.format(
                ' class="largest-problem"'
                if actual_label != pred and confusion[actual_label][pred] == largest_error
                else "",
                confusion[actual_label][pred],
            )
            for pred in LABELS
        )
        + "</tr>"
        for actual_label in LABELS
    )
    accuracy_rows = "".join(
        f"<tr><th>{label.title()}</th><td>{confusion[label][label]} / {actual[label]}</td>"
        f"<td>{class_accuracy[label]:.2%}</td></tr>"
        for label in LABELS
    )
    emotion_rows = "".join(
        f"<tr><th>{emotion.title()}</th><td>{nrc_emotions[emotion]}</td>"
        f"<td>{nrc_emotions[emotion] / total:.2%}</td></tr>"
        for emotion in EMOTIONS
    )
    neutral_outcomes = "".join(
        f"<tr><th>Predicted {label.title()}</th><td>{confusion['NEUTRAL'][label]}</td>"
        f"<td>{confusion['NEUTRAL'][label] / actual['NEUTRAL']:.2%}</td></tr>"
        for label in ("NEUTRAL", "POSITIVE", "NEGATIVE")
    )

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Balanced Amazon Gift Cards Analysis</title>
<style>
:root {{ --ink:#18212b; --muted:#66717d; --line:#d8dee5; --surface:#f5f7f9; --accent:#315d83; --good:#176b45; --good-bg:#eaf6ef; --bad:#9b2c2c; --bad-bg:#fff0f0; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:white; color:var(--ink); font:15px/1.5 Arial, Helvetica, sans-serif; }}
main {{ width:min(1240px, calc(100% - 32px)); margin:32px auto 56px; }}
h1 {{ margin:0 0 6px; font-size:28px; }} h2 {{ margin:30px 0 12px; font-size:20px; }}
.subtitle {{ margin:0 0 22px; color:var(--muted); }}
.metrics {{ display:grid; grid-template-columns:repeat(4, 1fr); border:1px solid var(--line); }}
.metric {{ padding:18px; border-right:1px solid var(--line); }} .metric:last-child {{ border-right:0; }}
.metric strong {{ display:block; font-size:26px; }} .metric span {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:18px; align-items:start; }}
.panel {{ border:1px solid var(--line); }} .panel h3 {{ margin:0; padding:12px 14px; background:var(--surface); font-size:16px; }}
table {{ width:100%; border-collapse:collapse; }} th, td {{ padding:9px 11px; border:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:var(--surface); font-size:13px; }} .panel table th, .panel table td {{ border-width:1px 0 0; }}
.matrix td, .matrix thead th {{ text-align:center; }} .matrix tbody td {{ font-size:18px; font-weight:700; }}
.matrix td.largest-problem {{ color:var(--bad); background:var(--bad-bg); outline:2px solid var(--bad); outline-offset:-2px; }}
.table-wrap {{ overflow-x:auto; }} .reviews {{ min-width:1260px; }} .reviews tr.incorrect {{ background:var(--bad-bg); }}
.filters {{ display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:12px; }}
.filters button {{ padding:8px 12px; border:1px solid var(--line); border-radius:3px; background:white; color:var(--ink); cursor:pointer; }}
.filters button[aria-pressed="true"] {{ background:var(--accent); border-color:var(--accent); color:white; }}
.visible-count {{ margin-left:auto; color:var(--muted); }} .filtered-out {{ display:none; }}
.result {{ display:inline-block; min-width:42px; padding:2px 8px; border-radius:3px; text-align:center; font-weight:700; }}
.result.correct {{ color:var(--good); background:var(--good-bg); }} .result.incorrect {{ color:var(--bad); background:var(--bad-bg); }}
.notice {{ margin:14px 0 0; padding:12px 14px; border:1px solid var(--line); background:var(--surface); color:var(--muted); }}
@media (max-width:760px) {{ .metrics, .grid {{ grid-template-columns:1fr; }} .metric {{ border-right:0; border-bottom:1px solid var(--line); }} .metric:last-child {{ border-bottom:0; }} .visible-count {{ width:100%; margin-left:0; }} }}
</style>
</head>
<body>
<main {' '.join(data)}>
  <h1>Balanced Amazon Gift Cards Analysis</h1>
  <p class="subtitle">Balanced 150-review evaluation · three-class sentiment · complete NRC emotion analysis.</p>
  <section class="metrics" aria-label="Summary">
    <div class="metric"><strong>{total}</strong><span>Total reviews</span></div>
    <div class="metric"><strong>{accuracy:.2%}</strong><span>Overall accuracy</span></div>
    <div class="metric"><strong>{correct}</strong><span>Correct predictions</span></div>
    <div class="metric"><strong>{total - correct}</strong><span>Incorrect predictions</span></div>
  </section>

  <div class="grid">
    <section><h2>Star rating distribution</h2><div class="panel"><table><thead><tr><th>Rating</th><th>Count</th><th>Share</th></tr></thead><tbody>{rating_rows}</tbody></table></div></section>
    <section><h2>Sentiment distribution</h2><div class="panel"><table><thead><tr><th>Class</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>{sentiment_rows}</tbody></table></div></section>
  </div>

  <div class="grid">
    <section><h2>Confusion matrix</h2><div class="panel table-wrap"><table class="matrix"><thead><tr><th>Actual ↓ / Predicted →</th>{matrix_header}</tr></thead><tbody>{matrix_rows}</tbody></table></div></section>
    <section><h2>Class accuracy</h2><div class="panel"><table><thead><tr><th>Class</th><th>Correct</th><th>Accuracy</th></tr></thead><tbody>{accuracy_rows}</tbody></table></div></section>
  </div>

  <div class="grid">
    <section><h2>Actual NEUTRAL review outcomes</h2><div class="panel"><table><thead><tr><th>Outcome</th><th>Count</th><th>Share</th></tr></thead><tbody>{neutral_outcomes}</tbody></table></div></section>
    <section><h2>NRC emotion distribution</h2><div class="panel"><table><thead><tr><th>Emotion</th><th>Count</th><th>Share</th></tr></thead><tbody>{emotion_rows}</tbody></table></div></section>
  </div>

  <h2>Review results</h2>
  <div class="filters" aria-label="Filter reviews">
    <button type="button" data-filter="all" aria-pressed="true">All Reviews</button>
    <button type="button" data-filter="correct" aria-pressed="false">Correct Predictions</button>
    <button type="button" data-filter="incorrect" aria-pressed="false">Incorrect Predictions</button>
    <span class="visible-count" aria-live="polite">Visible rows: <strong id="visible-count">{total}</strong></span>
  </div>
  <div class="table-wrap"><table class="reviews">
    <thead><tr><th>#</th><th>Rating</th><th>Title</th><th>Review text</th><th>Actual sentiment</th><th>Predicted sentiment</th><th>Correct</th><th scope="col">NRC emotion</th><th scope="col">LLM emotion</th></tr></thead>
    <tbody>{''.join(review_rows)}</tbody>
  </table></div>
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--llm-partial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_results(args.input)
    llm_by_id = read_partial_llm(args.llm_partial, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(rows, llm_by_id), encoding="utf-8")
    print(f"Created {args.output} from {len(rows)} CSV rows")


if __name__ == "__main__":
    main()
