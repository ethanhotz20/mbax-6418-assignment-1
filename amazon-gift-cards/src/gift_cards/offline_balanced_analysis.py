"""API-free metrics and NRC enrichment for balanced sentiment results."""
import csv
from collections import Counter

from gift_cards.emotion import EMOTIONS
from gift_cards.sentiment_three_class import LABELS
from gift_cards.three_class_sample import rating_to_sentiment

OUTPUT_COLUMNS = [
    "sample_id", "source_line", "rating", "title", "text",
    "actual_sentiment", "predicted_sentiment", "correct", "nrc_emotion",
]


def _rating_key(value):
    rating = float(value)
    if rating not in (1.0, 2.0, 3.0, 4.0, 5.0):
        raise ValueError(f"Invalid rating: {value!r}")
    return str(int(rating))


def analyze_and_enrich(rows, nrc_predict, output_path):
    """Validate rating-derived truth, compute metrics, and save NRC emotions."""
    confusion = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    actual_counts = Counter({label: 0 for label in LABELS})
    predicted_counts = Counter({label: 0 for label in LABELS})
    rating_counts = Counter({str(rating): 0 for rating in range(1, 6)})
    nrc_counts = Counter()
    correct = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            rating_key = _rating_key(row["rating"])
            expected = rating_to_sentiment(int(rating_key))
            actual = row["actual_sentiment"]
            predicted = row["predicted_sentiment"]
            if actual != expected:
                raise ValueError(
                    f"Invalid ground truth for sample {row['sample_id']}: "
                    f"rating {row['rating']} maps to {expected}, not {actual}"
                )
            if predicted not in LABELS:
                raise ValueError(f"Invalid predicted sentiment: {predicted!r}")
            is_correct = actual == predicted
            if str(row["correct"]).lower() != str(is_correct).lower():
                raise ValueError(f"Incorrect correctness flag for sample {row['sample_id']}")
            emotion = nrc_predict({"title": row["title"], "text": row["text"]})
            if emotion not in EMOTIONS:
                raise ValueError(f"Invalid NRC emotion: {emotion!r}")
            writer.writerow({**row, "nrc_emotion": emotion})

            correct += is_correct
            actual_counts[actual] += 1
            predicted_counts[predicted] += 1
            confusion[actual][predicted] += 1
            rating_counts[rating_key] += 1
            nrc_counts[emotion] += 1

    total = len(rows)
    return {
        "ground_truth_mapping": {"4-5": "POSITIVE", "3": "NEUTRAL", "1-2": "NEGATIVE"},
        "total_reviews": total,
        "correct": correct,
        "incorrect": total - correct,
        "overall_accuracy": correct / total if total else 0.0,
        "star_rating_distribution": dict(rating_counts),
        "actual_sentiment_distribution": dict(actual_counts),
        "predicted_sentiment_distribution": dict(predicted_counts),
        "confusion_matrix": confusion,
        "accuracy_by_class": {
            label: confusion[label][label] / actual_counts[label] if actual_counts[label] else 0.0
            for label in LABELS
        },
        "nrc_emotion_distribution": dict(sorted(nrc_counts.items())),
        "nrc_completed_reviews": total,
        "llm_nrc_agreement_rate": None,
        "llm_nrc_agreement_status": "not_calculated_incomplete_llm_emotions",
    }
