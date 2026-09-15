"""Classify sentiment/emotion and NRC emotion for the first 100 usable reviews."""
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gift_cards.emotion import classify_sentiment_emotion, nrc_primary_emotion
from gift_cards.emotion_evaluate import EmotionEvaluationAPIError, evaluate_emotions
from gift_cards.evaluate import select_first_usable

DATASET = ROOT / "data/raw/Gift_Cards.jsonl.gz"
OUTPUT_DIR = ROOT / "data/processed/first_100_sentiment_emotion_gpt-5.6-sol"
RESULTS = OUTPUT_DIR / "results.csv"
SUMMARY = OUTPUT_DIR / "summary.json"


def main():
    settings = json.loads((ROOT / "config/emotion.json").read_text(encoding="utf-8"))
    prompt = (ROOT / "prompts/sentiment_emotion_v1.txt").read_text(encoding="utf-8")
    reviews, skipped = select_first_usable(DATASET, limit=100)
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import agent.auxiliary_client as auxiliary
    from hermes_cli.config import load_config

    saved = load_config().get("model", {})
    expected = {
        "provider": settings["provider"],
        "default": settings["model"],
        "base_url": settings["base_url"],
    }
    if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Hermes provider/model/endpoint differs from config/emotion.json")
    client, model = auxiliary.resolve_provider_client(
        settings["provider"], model=settings["model"], raw_codex=True
    )
    if client is None or model != settings["model"]:
        raise RuntimeError("Hermes authentication or exact model unavailable")
    if str(client.base_url).rstrip("/") != settings["base_url"].rstrip("/"):
        client.close()
        raise RuntimeError("Resolved endpoint differs from config/emotion.json")
    client = client.with_options(max_retries=0)

    def predict(review):
        result = classify_sentiment_emotion(review, client, settings, prompt)
        return {"sentiment": result["sentiment"], "emotion": result["emotion"]}

    try:
        metrics = evaluate_emotions(reviews, predict, nrc_primary_emotion, RESULTS)
    except EmotionEvaluationAPIError as exc:
        completed = exc.review_number - 1
        status = {
            "status": "stopped",
            "http_status": exc.status_code,
            "sample_review_number": exc.review_number,
            "source_line": exc.source_line,
            "completed_reviews": completed,
            "message": (
                "HTTP 429 stopped model calls immediately at sample review "
                f"{exc.review_number} (dataset source line {exc.source_line})."
            ),
        }
        SUMMARY.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(status["message"])
        print(f"Partial CSV contains {completed} completed reviews: {RESULTS}")
        return 1
    finally:
        client.close()

    summary = {
        "status": "completed",
        **metrics,
        "usable_reviews": 100,
        "skipped_before_selection_completed": skipped,
        "last_selected_source_line": reviews[-1]["_source_line"],
        "model": settings["model"],
        "nrclex_version": importlib.metadata.version("nrclex"),
        "nrc_method": (
            "Lowercase ASCII word tokens from title and text; count NRC associations; "
            "choose the first maximum in the documented emotion order; use TRUST when no emotion term matches."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Total reviews: {metrics['total_reviews']}")
    print(f"Sentiment accuracy: {metrics['accuracy']:.2%}")
    print(f"Emotion matches: {metrics['emotion_matches']}")
    print(f"Emotion mismatches: {metrics['emotion_mismatches']}")
    print(f"Emotion agreement: {metrics['emotion_agreement_rate']:.2%}")
    print(f"Results CSV: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
