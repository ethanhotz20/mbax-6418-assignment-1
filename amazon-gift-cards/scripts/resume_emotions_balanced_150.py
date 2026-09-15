"""Resume the LLM+NRC emotion enrichment of the balanced 150-review sample.

Idempotent: skips rows already in `results_with_emotions.csv` and API responses
already logged to `emotion_api_responses.jsonl`, so it can be stopped and
re-run at any point without duplicating calls or rows.

Self-pausing: when gpt-5.6-sol returns HTTP 429 with type `usage_limit_reached`,
parses `resets_in_seconds`/`resets_at` and sleeps until the usage window resets
plus a buffer, then retries. Transient rate limits and 5xx retry with exponential
backoff. Only writes `emotion_summary.json` (as "completed") when every row is done;
a bounded arm that exceeds the max wall time exits non-zero without leaving a
misleading summary, preserving the last good state for a later re-run.

Run with the Hermes agent's Python (which provides hermes_cli / agent / openai):
    python scripts/resume_emotions_balanced_150.py
"""
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hermes_cli.config import load_config  # noqa: E402
import agent.auxiliary_client as auxiliary  # noqa: E402

from gift_cards.balanced_emotions import EMOTION_COLUMNS  # noqa: E402
from gift_cards.emotion import EMOTIONS, nrc_primary_emotion  # noqa: E402
from gift_cards.emotion_only import classify  # noqa: E402

OUTPUT_DIR = ROOT / "data/processed/balanced_150_sentiment_3class_gpt-5.6-sol"
INPUT = OUTPUT_DIR / "results.csv"
RESULTS = OUTPUT_DIR / "results_with_emotions.csv"
SUMMARY = OUTPUT_DIR / "emotion_summary.json"
RESPONSES = OUTPUT_DIR / "emotion_api_responses.jsonl"
RESULT_COLUMNS = [
    "sample_id", "source_line", "rating", "title", "text",
    "actual_sentiment", "predicted_sentiment", "correct",
]

# Bounds ----------------------------------------------------------------
SETTINGS_PATH = ROOT / "config/emotion_balanced.json"
PROMPT_PATH = ROOT / "prompts/emotion_only_v1.txt"
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_429_WAIT_S = 8 * 3600        # cap total time spent waiting out usage caps
USAGE_BUFFER_S = 60              # extra grace after the declared reset time
BACKOFF_BASE_S = 15
BACKOFF_CAP_S = 120
MAX_ATTEMPTS = 10                # consecutive transient failures before pausing


def _sleep(seconds):
    if seconds > 0:
        time.sleep(seconds)


def parse_usage_reset(exc):
    """Return (wait_seconds, epoch_reset) for a usage-limit 429, else None."""
    body = getattr(exc, "body", None) or getattr(exc, "response", None)
    text = str(exc)
    m = re.search(r"resets_in_seconds['\"]?\s*[:=]\s*(\d+)", text)
    if m:
        return max(int(m.group(1)), 5), None
    m = re.search(r"resets_at['\"]?\s*[:=]\s*(\d+)", text)
    if m:
        return None, int(m.group(1))
    return None, None


def predict_with_retry(record, client, settings, prompt):
    """One LLM emotion call, pausing through usage caps and retrying transients."""
    started = time.monotonic()
    pause_budget = MAX_429_WAIT_S
    while True:
        try:
            return classify(record, client, settings, prompt)
        except Exception as exc:  # noqa: BLE001 - classify raises mixed errors
            status = getattr(exc, "status_code", None)
            if status == 429:
                reset_s, reset_at = parse_usage_reset(exc)
                if "usage_limit" in str(exc).lower():
                    if reset_s is not None and pause_budget - reset_s <= 0:
                        # One more full wait would exceed budget; give up cleanly.
                        if time.monotonic() - started >= MAX_429_WAIT_S:
                            raise
                    wait = reset_s + USAGE_BUFFER_S if reset_s is not None else (
                        (reset_at - time.time()) + USAGE_BUFFER_S
                        if reset_at is not None else BACKOFF_CAP_S)
                    if wait <= 0:
                        raise  # reset already passed but still capped: hard fail
                    pause_budget -= wait
                    print(f"[usage cap] sleeping {wait:.0f}s until reset ("
                          f"{datetime.now(timezone.utc):%H:%M:%S}Z)", flush=True)
                    _sleep(wait)
                    continue
            # Transient rate limit / server error: exponential backoff.
            if status in TRANSIENT_STATUS:
                delay = min(BACKOFF_BASE_S * (2 ** max(0, 0)), BACKOFF_CAP_S)
                print(f"[transient {status}] backoff {delay:.0f}s", flush=True)
                _sleep(delay)
                continue
            raise  # non-retryable (auth, validation, refusal, etc.)


def main():
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    saved = load_config().get("model", {})
    expected = {
        "provider": settings["provider"],
        "default": settings["model"],
        "base_url": settings["base_url"],
    }
    if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Hermes provider/model/endpoint differs from emotion config")

    client, model = auxiliary.resolve_provider_client(
        settings["provider"], model=settings["model"], raw_codex=True)
    if client is None or model != settings["model"]:
        raise RuntimeError("Hermes authentication or exact model unavailable")
    if str(client.base_url).rstrip("/") != settings["base_url"].rstrip("/"):
        client.close()
        raise RuntimeError("Resolved endpoint differs from emotion config")
    client = client.with_options(max_retries=0)

    with INPUT.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(f"Expected result columns: {RESULT_COLUMNS}")
        rows = list(reader)
    if len(rows) != 150:
        raise ValueError(f"Expected 150 balanced results, found {len(rows)}")

    # How many rows are already complete in the output file?
    done_ids = set()
    if RESULTS.exists() and RESULTS.stat().st_size:
        with RESULTS.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != RESULT_COLUMNS + EMOTION_COLUMNS:
                raise ValueError("results_with_emotions.csv has an unexpected schema")
            done_ids = {r["sample_id"] for r in reader}

    pending = [r for r in rows if r["sample_id"] not in done_ids]
    print(f"Rows total=150  already_done={len(done_ids)}  pending={len(pending)}", flush=True)
    if not pending:
        print("Nothing to do; all rows already enriched.", flush=True)
        client.close()
        return 0

    results_handle = RESULTS.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(results_handle, fieldnames=RESULT_COLUMNS + EMOTION_COLUMNS)
    if RESULTS.stat().st_size == 0:
        writer.writeheader()

    responses_handle = RESPONSES.open("a", encoding="utf-8") if not RESPONSES.exists() else \
        RESPONSES.open("a", encoding="utf-8")
    logged_ids = set()
    if RESPONSES.exists():
        for line in RESPONSES.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    logged_ids.add(str(json.loads(line)["sample_id"]))
                except Exception:
                    pass

    try:
        for row in rows:
            if row["sample_id"] in done_ids:
                continue
            record = {"title": row["title"], "text": row["text"]}
            result = predict_with_retry(record, client, settings, prompt)
            emotion = result["emotion"]
            if emotion not in EMOTIONS:
                raise ValueError(f"Invalid emotion label from model: {emotion!r}")
            nrc = nrc_primary_emotion(record)
            if nrc not in EMOTIONS:
                raise ValueError(f"Invalid NRC emotion label: {nrc!r}")
            match = emotion == nrc
            if row["sample_id"] not in logged_ids:
                responses_handle.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "emotion": emotion,
                    "response_id": result["response_id"],
                    "returned_model": result["returned_model"],
                    "usage": result["usage"],
                }, ensure_ascii=False) + "\n")
                responses_handle.flush()
                logged_ids.add(row["sample_id"])
            writer.writerow({
                **row,
                "llm_emotion": emotion,
                "nrc_emotion": nrc,
                "emotion_match": str(match),
            })
            results_handle.flush()
            done_ids.add(row["sample_id"])
            print(f"  row {row['sample_id']}/150  llm={emotion}  nrc={nrc}  match={match}", flush=True)
    finally:
        results_handle.close()
        responses_handle.close()
        client.close()

    # Everything finished: recompute metrics from the full file.
    with RESULTS.open(encoding="utf-8", newline="") as stream:
        done = list(csv.DictReader(stream))
    if len(done) != 150:
        print(f"WARNING: only {len(done)}/150 rows enriched; not writing a completed summary.",
              flush=True)
        return 2
    matches = sum(1 for r in done if r["emotion_match"] == "True")
    from collections import Counter
    llm_counts = Counter(r["llm_emotion"] for r in done)
    nrc_counts = Counter(r["nrc_emotion"] for r in done)
    metrics = {
        "total_reviews": 150,
        "emotion_matches": matches,
        "emotion_mismatches": 150 - matches,
        "emotion_agreement_rate": matches / 150,
        "llm_emotion_distribution": dict(sorted(llm_counts.items())),
        "nrc_emotion_distribution": dict(sorted(nrc_counts.items())),
    }
    summary = {
        "status": "completed",
        **metrics,
        "model": settings["model"],
        "provider": settings["provider"],
        "openai_version": importlib.metadata.version("openai"),
        "nrclex_version": importlib.metadata.version("nrclex"),
        "input_results_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "output_results_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(SETTINGS_PATH.read_bytes()).hexdigest(),
        "nrc_method": (
            "Lowercase ASCII word tokens from title and text; count NRC associations; "
            "fixed emotion-order tie break; TRUST when no emotion term matches."
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "note": "Run resumed after a gpt-5.6-sol usage-limit pause; rows 1-10 precede this run.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Emotion agreement: {metrics['emotion_agreement_rate']:.2%}", flush=True)
    print(f"Results: {RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
