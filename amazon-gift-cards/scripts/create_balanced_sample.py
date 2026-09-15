"""Create the fixed-seed balanced three-class review sample."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gift_cards.three_class_sample import balanced_sample

DATASET = ROOT / "data/raw/Gift_Cards.jsonl.gz"
OUTPUT_DIR = ROOT / "data/processed/balanced_150_sentiment_3class_gpt-5.6-sol"
SAMPLE = OUTPUT_DIR / "sample.csv"
MANIFEST = OUTPUT_DIR / "sample_manifest.json"
COLUMNS = ["sample_id", "source_line", "rating", "title", "text", "actual_sentiment"]


def main():
    settings = json.loads(
        (ROOT / "config/sentiment_three_class.json").read_text(encoding="utf-8")
    )
    if SAMPLE.exists() or MANIFEST.exists():
        raise RuntimeError(f"Sample outputs already exist in {OUTPUT_DIR}")
    rows, stats = balanced_sample(
        DATASET, per_class=50, seed=settings["sample_seed"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in COLUMNS} for row in rows)
    manifest = {
        **stats,
        "sample_size": len(rows),
        "sample_counts": {
            label: sum(row["actual_sentiment"] == label for row in rows)
            for label in ("POSITIVE", "NEUTRAL", "NEGATIVE")
        },
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "sample_sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
        "eligibility_rule": (
            "Valid rating and non-empty title/text with records containing possible "
            "rating references or numeric content excluded before sampling."
        ),
        "sampling_method": "One-pass stratified reservoir sample across the full gzip JSONL file.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {SAMPLE}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
