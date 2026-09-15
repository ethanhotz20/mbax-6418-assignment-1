"""Add LLM and NRC emotion columns to saved balanced sentiment results."""
import csv
from collections import Counter

from gift_cards.emotion import EMOTIONS

EMOTION_COLUMNS = ["llm_emotion", "nrc_emotion", "emotion_match"]


class EnrichmentAPIError(Exception):
    def __init__(self, sample_id, source_line, status_code):
        super().__init__(
            f"API stopped at sample {sample_id}, source line {source_line}, HTTP {status_code}"
        )
        self.sample_id = sample_id
        self.source_line = source_line
        self.status_code = status_code


def enrich(rows, predict, nrc_predict, output_path):
    """Append one LLM and one NRC primary emotion to each saved result row."""
    if not rows:
        raise ValueError("No rows to enrich")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) + EMOTION_COLUMNS
    matches = 0
    llm_counts = Counter()
    nrc_counts = Counter()

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            review = {"title": row["title"], "text": row["text"]}
            try:
                llm_emotion = predict(review)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429:
                    raise EnrichmentAPIError(
                        int(row["sample_id"]), int(row["source_line"]), 429
                    ) from None
                raise
            nrc_emotion = nrc_predict(review)
            if llm_emotion not in EMOTIONS or nrc_emotion not in EMOTIONS:
                raise ValueError("Invalid emotion label")
            emotion_match = llm_emotion == nrc_emotion
            writer.writerow(
                {
                    **row,
                    "llm_emotion": llm_emotion,
                    "nrc_emotion": nrc_emotion,
                    "emotion_match": emotion_match,
                }
            )
            stream.flush()
            matches += emotion_match
            llm_counts[llm_emotion] += 1
            nrc_counts[nrc_emotion] += 1

    total = len(rows)
    return {
        "total_reviews": total,
        "emotion_matches": matches,
        "emotion_mismatches": total - matches,
        "emotion_agreement_rate": matches / total,
        "llm_emotion_distribution": dict(sorted(llm_counts.items())),
        "nrc_emotion_distribution": dict(sorted(nrc_counts.items())),
    }


def resume_enrich(rows, predict, nrc_predict, output_path):
    """Resume a prefix-complete enrichment without repeating completed calls."""
    if not rows or not output_path.exists():
        raise ValueError("A non-empty base and existing partial output are required")
    base_fields = list(rows[0])
    fieldnames = base_fields + EMOTION_COLUMNS
    with output_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fieldnames:
            raise ValueError("Partial output columns do not match the base results")
        completed = list(reader)
    if len(completed) > len(rows):
        raise ValueError("Partial output has more rows than the base results")
    for index, saved in enumerate(completed):
        if any(saved[key] != str(rows[index][key]) for key in base_fields):
            raise ValueError("Partial output is not an exact prefix of the base results")

    with output_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        for row in rows[len(completed):]:
            review = {"title": row["title"], "text": row["text"]}
            try:
                llm_emotion = predict(review)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429:
                    raise EnrichmentAPIError(
                        int(row["sample_id"]), int(row["source_line"]), 429
                    ) from None
                raise
            nrc_emotion = nrc_predict(review)
            if llm_emotion not in EMOTIONS or nrc_emotion not in EMOTIONS:
                raise ValueError("Invalid emotion label")
            writer.writerow({
                **row,
                "llm_emotion": llm_emotion,
                "nrc_emotion": nrc_emotion,
                "emotion_match": llm_emotion == nrc_emotion,
            })
            stream.flush()

    with output_path.open(encoding="utf-8", newline="") as stream:
        saved_rows = list(csv.DictReader(stream))
    llm_counts = Counter(row["llm_emotion"] for row in saved_rows)
    nrc_counts = Counter(row["nrc_emotion"] for row in saved_rows)
    matches = sum(row["emotion_match"] == "True" for row in saved_rows)
    total = len(saved_rows)
    return {
        "total_reviews": total,
        "emotion_matches": matches,
        "emotion_mismatches": total - matches,
        "emotion_agreement_rate": matches / total,
        "llm_emotion_distribution": dict(sorted(llm_counts.items())),
        "nrc_emotion_distribution": dict(sorted(nrc_counts.items())),
    }
