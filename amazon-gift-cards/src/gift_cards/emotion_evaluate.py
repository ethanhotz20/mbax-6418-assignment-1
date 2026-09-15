"""Evaluate one LLM sentiment/emotion result against rating and NRC emotion."""
import csv

from gift_cards.emotion import EMOTIONS
from gift_cards.evaluate import rating_to_sentiment

COLUMNS = [
    "rating",
    "title",
    "text",
    "actual_sentiment",
    "predicted_sentiment",
    "correct",
    "llm_emotion",
    "nrc_emotion",
    "emotion_match",
]


class EmotionEvaluationAPIError(Exception):
    def __init__(self, review_number, source_line, status_code):
        super().__init__(
            f"API stopped at review {review_number}, source line {source_line}, HTTP {status_code}"
        )
        self.review_number = review_number
        self.source_line = source_line
        self.status_code = status_code


def evaluate_emotions(records, predict, nrc_predict, output_path):
    """Call the rating-blind LLM, score NRC independently, and append each row."""
    totals = {
        "total_reviews": 0,
        "correct": 0,
        "incorrect": 0,
        "accuracy": 0.0,
        "actual_positive": 0,
        "actual_negative": 0,
        "emotion_matches": 0,
        "emotion_mismatches": 0,
        "emotion_agreement_rate": 0.0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for review_number, record in enumerate(records, 1):
            review = {"title": record["title"], "text": record["text"]}
            try:
                predicted = predict(review)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429:
                    source_line = record.get("_source_line", review_number)
                    raise EmotionEvaluationAPIError(
                        review_number, source_line, 429
                    ) from None
                raise
            if (
                not isinstance(predicted, dict)
                or predicted.get("sentiment") not in ("POSITIVE", "NEGATIVE")
                or predicted.get("emotion") not in EMOTIONS
            ):
                raise ValueError(f"Invalid LLM prediction: {predicted!r}")
            nrc_emotion = nrc_predict(review)
            if nrc_emotion not in EMOTIONS:
                raise ValueError(f"Invalid NRC emotion: {nrc_emotion!r}")

            rating = float(record["rating"])
            if rating not in (1, 2, 3, 4, 5):
                raise ValueError(f"Invalid rating: {record['rating']!r}")
            actual = rating_to_sentiment(rating)
            correct = predicted["sentiment"] == actual
            emotion_match = predicted["emotion"] == nrc_emotion
            writer.writerow(
                {
                    "rating": record["rating"],
                    "title": record["title"],
                    "text": record["text"],
                    "actual_sentiment": actual,
                    "predicted_sentiment": predicted["sentiment"],
                    "correct": correct,
                    "llm_emotion": predicted["emotion"],
                    "nrc_emotion": nrc_emotion,
                    "emotion_match": emotion_match,
                }
            )
            stream.flush()
            totals["total_reviews"] += 1
            totals["correct" if correct else "incorrect"] += 1
            totals[
                "actual_positive" if actual == "POSITIVE" else "actual_negative"
            ] += 1
            totals["emotion_matches" if emotion_match else "emotion_mismatches"] += 1

    count = totals["total_reviews"]
    if count:
        totals["accuracy"] = totals["correct"] / count
        totals["emotion_agreement_rate"] = totals["emotion_matches"] / count
    return totals
