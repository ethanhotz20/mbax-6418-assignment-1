"""Three-class sentiment evaluation and metrics."""
import csv

from gift_cards.sentiment_three_class import LABELS
from gift_cards.three_class_sample import rating_to_sentiment

COLUMNS = [
    "sample_id",
    "source_line",
    "rating",
    "title",
    "text",
    "actual_sentiment",
    "predicted_sentiment",
    "correct",
]


class EvaluationAPIError(Exception):
    def __init__(self, sample_id, source_line, status_code):
        super().__init__(
            f"API stopped at sample {sample_id}, source line {source_line}, HTTP {status_code}"
        )
        self.sample_id = sample_id
        self.source_line = source_line
        self.status_code = status_code


def evaluate(records, predict, output_path):
    """Predict from title/text only, append rows, and calculate three-class metrics."""
    confusion = {
        actual: {predicted: 0 for predicted in LABELS} for actual in LABELS
    }
    class_totals = {label: 0 for label in LABELS}
    class_correct = {label: 0 for label in LABELS}
    metrics = {
        "total_reviews": 0,
        "correct": 0,
        "incorrect": 0,
        "overall_accuracy": 0.0,
        "confusion_matrix": confusion,
        "accuracy_by_class": {label: 0.0 for label in LABELS},
        "neutral_predictions": {label: 0 for label in LABELS},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for record in records:
            actual = record["actual_sentiment"]
            if actual not in LABELS or actual != rating_to_sentiment(record["rating"]):
                raise ValueError(f"Invalid actual sentiment for sample {record['sample_id']}")
            review = {"title": record["title"], "text": record["text"]}
            try:
                predicted = predict(review)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429:
                    raise EvaluationAPIError(
                        record["sample_id"], record["source_line"], 429
                    ) from None
                raise
            if predicted not in LABELS:
                raise ValueError(f"Invalid predicted sentiment: {predicted!r}")

            correct = predicted == actual
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "source_line": record["source_line"],
                    "rating": record["rating"],
                    "title": record["title"],
                    "text": record["text"],
                    "actual_sentiment": actual,
                    "predicted_sentiment": predicted,
                    "correct": correct,
                }
            )
            stream.flush()

            metrics["total_reviews"] += 1
            metrics["correct" if correct else "incorrect"] += 1
            confusion[actual][predicted] += 1
            class_totals[actual] += 1
            class_correct[actual] += correct
            if actual == "NEUTRAL":
                metrics["neutral_predictions"][predicted] += 1

    total = metrics["total_reviews"]
    if total:
        metrics["overall_accuracy"] = metrics["correct"] / total
    metrics["accuracy_by_class"] = {
        label: class_correct[label] / class_totals[label] if class_totals[label] else 0.0
        for label in LABELS
    }
    return metrics
