"""Run three-class sentiment inference on the saved balanced sample."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gift_cards.sentiment_three_class import LABELS, classify
from gift_cards.three_class_evaluate import EvaluationAPIError, evaluate

OUTPUT_DIR = ROOT / "data/processed/balanced_150_sentiment_3class_gpt-5.6-sol"
SAMPLE = OUTPUT_DIR / "sample.csv"
RESULTS = OUTPUT_DIR / "results.csv"
SUMMARY = OUTPUT_DIR / "summary.json"
INCORRECT = OUTPUT_DIR / "incorrect_reviews.csv"
RESPONSES = OUTPUT_DIR / "api_responses.jsonl"
SAMPLE_COLUMNS = [
    "sample_id", "source_line", "rating", "title", "text", "actual_sentiment"
]


def read_sample():
    with SAMPLE.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != SAMPLE_COLUMNS:
            raise ValueError(f"Expected sample columns: {SAMPLE_COLUMNS}")
        rows = list(reader)
    counts = {label: sum(row["actual_sentiment"] == label for row in rows) for label in LABELS}
    if len(rows) != 150 or any(count != 50 for count in counts.values()):
        raise ValueError(f"Expected 150 balanced rows; found {counts}")
    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        row["source_line"] = int(row["source_line"])
        row["rating"] = float(row["rating"])
    return rows


def main():
    settings_path = ROOT / "config/sentiment_three_class.json"
    prompt_path = ROOT / "prompts/sentiment_three_class_v1.txt"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    prompt = prompt_path.read_text(encoding="utf-8")
    rows = read_sample()
    if any(path.exists() for path in (RESULTS, SUMMARY, INCORRECT, RESPONSES)):
        raise RuntimeError(f"Classification outputs already exist in {OUTPUT_DIR}")

    import agent.auxiliary_client as auxiliary
    from hermes_cli.config import load_config

    saved = load_config().get("model", {})
    expected = {
        "provider": settings["provider"],
        "default": settings["model"],
        "base_url": settings["base_url"],
    }
    if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            "Hermes provider/model/endpoint differs from config/sentiment_three_class.json"
        )
    client, model = auxiliary.resolve_provider_client(
        settings["provider"], model=settings["model"], raw_codex=True
    )
    if client is None or model != settings["model"]:
        raise RuntimeError("Hermes authentication or exact model unavailable")
    if str(client.base_url).rstrip("/") != settings["base_url"].rstrip("/"):
        client.close()
        raise RuntimeError("Resolved endpoint differs from three-class config")
    client = client.with_options(max_retries=0)
    call_number = 0

    def predict(review):
        nonlocal call_number
        call_number += 1
        result = classify(review, client, settings, prompt)
        with RESPONSES.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "sample_id": call_number,
                        "sentiment": result["sentiment"],
                        "response_id": result["response_id"],
                        "returned_model": result["returned_model"],
                        "usage": result["usage"],
                    }
                )
                + "\n"
            )
        return result["sentiment"]

    try:
        metrics = evaluate(rows, predict, RESULTS)
    except EvaluationAPIError as exc:
        completed = exc.sample_id - 1
        stopped = {
            "status": "stopped",
            "http_status": exc.status_code,
            "sample_id": exc.sample_id,
            "source_line": exc.source_line,
            "completed_reviews": completed,
            "message": (
                f"HTTP 429 stopped calls immediately at sample {exc.sample_id} "
                f"(dataset source line {exc.source_line})."
            ),
        }
        SUMMARY.write_text(json.dumps(stopped, indent=2) + "\n", encoding="utf-8")
        print(stopped["message"])
        return 1
    finally:
        client.close()

    with RESULTS.open(encoding="utf-8", newline="") as stream:
        result_rows = list(csv.DictReader(stream))
    with INCORRECT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(row for row in result_rows if row["correct"] == "False")

    summary = {
        "status": "completed",
        **metrics,
        "model": settings["model"],
        "provider": settings["provider"],
        "sample_seed": settings["sample_seed"],
        "openai_version": importlib.metadata.version("openai"),
        "sample_sha256": hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Total reviews: {metrics['total_reviews']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    for label in LABELS:
        print(f"{label} accuracy: {metrics['accuracy_by_class'][label]:.2%}")
    print("Neutral reviews predicted as:", metrics["neutral_predictions"])
    print(f"Results CSV: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
