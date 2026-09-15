"""Add LLM and NRC emotion columns to the saved balanced three-class results."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gift_cards.balanced_emotions import EnrichmentAPIError, resume_enrich
from gift_cards.emotion import nrc_primary_emotion
from gift_cards.emotion_only import classify

OUTPUT_DIR = ROOT / "data/processed/balanced_150_sentiment_3class_gpt-5.6-sol"
INPUT = OUTPUT_DIR / "results.csv"
OUTPUT = OUTPUT_DIR / "results_with_emotions.csv"
SUMMARY = OUTPUT_DIR / "emotion_summary.json"
RESPONSES = OUTPUT_DIR / "emotion_api_responses.jsonl"
RESULT_COLUMNS = [
    "sample_id", "source_line", "rating", "title", "text",
    "actual_sentiment", "predicted_sentiment", "correct",
]


def main():
    settings_path = ROOT / "config/emotion_balanced.json"
    prompt_path = ROOT / "prompts/emotion_only_v1.txt"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    prompt = prompt_path.read_text(encoding="utf-8")
    with INPUT.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(f"Expected result columns: {RESULT_COLUMNS}")
        rows = list(reader)
    if len(rows) != 150:
        raise ValueError(f"Expected 150 balanced results, found {len(rows)}")
    if not OUTPUT.exists():
        raise RuntimeError(f"Partial emotion output is required for resume: {OUTPUT}")
    with OUTPUT.open(encoding="utf-8", newline="") as stream:
        completed_before = list(csv.DictReader(stream))
    if len(completed_before) >= len(rows):
        print(f"All {len(rows)} reviews already have LLM emotions.")
        return 0

    import agent.auxiliary_client as auxiliary
    from hermes_cli.config import load_config

    saved = load_config().get("model", {})
    expected = {
        "provider": settings["provider"],
        "default": settings["model"],
        "base_url": settings["base_url"],
    }
    if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Hermes provider/model/endpoint differs from emotion config")
    client, model = auxiliary.resolve_provider_client(
        settings["provider"], model=settings["model"], raw_codex=True
    )
    if client is None or model != settings["model"]:
        raise RuntimeError("Hermes authentication or exact model unavailable")
    if str(client.base_url).rstrip("/") != settings["base_url"].rstrip("/"):
        client.close()
        raise RuntimeError("Resolved endpoint differs from emotion config")
    client = client.with_options(max_retries=0)
    call_number = len(completed_before)

    def predict(review):
        nonlocal call_number
        if call_number > len(completed_before):
            time.sleep(settings["request_delay_seconds"])
        call_number += 1
        result = classify(review, client, settings, prompt)
        with RESPONSES.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "sample_id": call_number,
                "emotion": result["emotion"],
                "response_id": result["response_id"],
                "returned_model": result["returned_model"],
                "usage": result["usage"],
            }) + "\n")
        return result["emotion"]

    try:
        metrics = resume_enrich(rows, predict, nrc_primary_emotion, OUTPUT)
    except EnrichmentAPIError as exc:
        with OUTPUT.open(encoding="utf-8", newline="") as stream:
            completed_reviews = sum(1 for _ in csv.DictReader(stream))
        stopped = {
            "status": "stopped",
            "http_status": exc.status_code,
            "sample_id": exc.sample_id,
            "source_line": exc.source_line,
            "completed_reviews": completed_reviews,
            "next_sample_id": exc.sample_id,
            "message": (
                f"HTTP 429 stopped calls immediately. {completed_reviews} of 150 "
                f"reviews are complete; process sample {exc.sample_id} next "
                f"(dataset source line {exc.source_line})."
            ),
        }
        SUMMARY.write_text(json.dumps(stopped, indent=2) + "\n", encoding="utf-8")
        print(stopped["message"])
        return 1
    finally:
        client.close()

    summary = {
        "status": "completed",
        **metrics,
        "model": settings["model"],
        "provider": settings["provider"],
        "openai_version": importlib.metadata.version("openai"),
        "nrclex_version": importlib.metadata.version("nrclex"),
        "input_results_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "output_results_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
        "nrc_method": (
            "Lowercase ASCII word tokens from title and text; count NRC associations; "
            "fixed emotion-order tie break; TRUST when no emotion term matches."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Emotion agreement: {metrics['emotion_agreement_rate']:.2%}")
    print(f"Results: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
