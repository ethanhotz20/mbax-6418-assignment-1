"""Fixed-seed stratified reservoir sampling for three rating-derived classes."""
import gzip
import json
import random

from gift_cards.sentiment import prepare_review
from gift_cards.sentiment_three_class import LABELS


def rating_to_sentiment(rating):
    if rating in (4, 5):
        return "POSITIVE"
    if rating == 3:
        return "NEUTRAL"
    if rating in (1, 2):
        return "NEGATIVE"
    raise ValueError(f"Invalid rating: {rating!r}")


def balanced_sample(path, per_class=50, seed=6418):
    """Uniformly reservoir-sample each class while scanning the entire gzip file."""
    if per_class < 1:
        raise ValueError("per_class must be positive")
    rng = random.Random(seed)
    reservoirs = {label: [] for label in LABELS}
    eligible = {label: 0 for label in LABELS}
    scanned = 0
    skipped = 0

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for source_line, line in enumerate(stream, 1):
            scanned += 1
            record = json.loads(line)
            try:
                label = rating_to_sentiment(record.get("rating"))
                prepare_review(
                    {"title": record.get("title"), "text": record.get("text")}
                )
            except ValueError:
                skipped += 1
                continue

            eligible[label] += 1
            candidate = {
                "source_line": source_line,
                "rating": record["rating"],
                "title": record["title"],
                "text": record["text"],
                "actual_sentiment": label,
            }
            bucket = reservoirs[label]
            if len(bucket) < per_class:
                bucket.append(candidate)
            else:
                replacement = rng.randrange(eligible[label])
                if replacement < per_class:
                    bucket[replacement] = candidate

    short = {label: len(rows) for label, rows in reservoirs.items() if len(rows) < per_class}
    if short:
        raise ValueError(f"Insufficient eligible reviews: {short}")

    sample = [row for label in LABELS for row in reservoirs[label]]
    rng.shuffle(sample)
    for sample_id, row in enumerate(sample, 1):
        row["sample_id"] = sample_id
    stats = {
        "seed": seed,
        "per_class": per_class,
        "source_records_scanned": scanned,
        "eligible_by_class": eligible,
        "skipped_unusable": skipped,
    }
    return sample, stats
