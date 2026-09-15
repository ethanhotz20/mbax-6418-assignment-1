"""Compute balanced sentiment metrics and NRC emotions without API calls."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gift_cards.emotion import nrc_primary_emotion
from gift_cards.offline_balanced_analysis import analyze_and_enrich

RESULT_DIR = ROOT / "data/processed/balanced_150_sentiment_3class_gpt-5.6-sol"
INPUT = RESULT_DIR / "results.csv"
OUTPUT = RESULT_DIR / "results_with_nrc_all_150.csv"
SUMMARY = RESULT_DIR / "offline_sentiment_nrc_summary.json"
EXPECTED_COLUMNS = [
    "sample_id", "source_line", "rating", "title", "text",
    "actual_sentiment", "predicted_sentiment", "correct",
]


def main():
    if OUTPUT.exists() or SUMMARY.exists():
        raise RuntimeError("Refusing to overwrite existing offline NRC outputs")
    with INPUT.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Expected columns: {EXPECTED_COLUMNS}")
        rows = list(reader)
    if len(rows) != 150:
        raise ValueError(f"Expected 150 balanced reviews, found {len(rows)}")

    metrics = analyze_and_enrich(rows, nrc_primary_emotion, OUTPUT)
    summary = {
        "status": "completed_without_api_calls",
        **metrics,
        "input_file": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "output_file": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "nrclex_version": importlib.metadata.version("nrclex"),
        "nrc_method": (
            "Lowercase ASCII word tokens from title and review text; count NRC "
            "associations; fixed emotion-order tie break; TRUST when no term matches."
        ),
        "llm_api_calls_made": 0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Total reviews: {metrics['total_reviews']}")
    print(f"Sentiment accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"NRC emotions completed: {metrics['nrc_completed_reviews']}")
    print(f"Results: {OUTPUT}")
    print(f"Summary: {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
