"""Synthetic sentiment/emotion checks; --live uses Hermes-managed OAuth."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gift_cards.emotion import classify_sentiment_emotion
from gift_cards.sentiment import prepare_review, safe_api_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/emotion_spot_check")
    args = parser.parse_args()

    settings_path = ROOT / "config/emotion.json"
    prompt_path = ROOT / "prompts/sentiment_emotion_v1.txt"
    fixtures_path = ROOT / "tests/fixtures/emotion_cases.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    prompt = prompt_path.read_text(encoding="utf-8")
    cases = json.loads(fixtures_path.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; use a new directory")
    args.output.mkdir(parents=True, exist_ok=True)

    prepared = []
    for case in cases:
        low = prepare_review({**case, "rating": 1, "user_id": "PRIVATE_SENTINEL"})
        high = prepare_review({**case, "rating": 5, "user_id": "PRIVATE_SENTINEL"})
        if low != high or set(json.loads(low)) != {"title", "text"}:
            raise ValueError("Input isolation check failed")
        prepared.append({"id": case["id"], "model_input": json.loads(low)})
    (args.output / "prepared_inputs.json").write_text(
        json.dumps(prepared, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "settings": settings,
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                prompt_path,
                settings_path,
                fixtures_path,
                ROOT / "src/gift_cards/emotion.py",
                Path(__file__),
            )
        },
        "raw_dataset_read": False,
        "conversation_history_sent": False,
    }
    summary = {
        "mode": "live" if args.live else "dry_run",
        "planned": len(cases),
        "api_calls": 0,
        "parsed_valid": 0,
        "sentiment_matches": 0,
        "emotion_matches": 0,
        "results": [],
    }
    client = None
    try:
        if args.live:
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
                raise RuntimeError("Resolved endpoint differs from config/emotion.json")
            client = client.with_options(max_retries=0)
            manifest["openai_version"] = importlib.metadata.version("openai")
            hermes_root = Path(auxiliary.__file__).resolve().parents[1]
            revision = subprocess.run(
                ["git", "-C", str(hermes_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            manifest["hermes_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
            manifest["hermes_adapter_sha256"] = hashlib.sha256(
                Path(auxiliary.__file__).read_bytes()
            ).hexdigest()

            for case in cases:
                summary["api_calls"] += 1
                result = classify_sentiment_emotion(case, client, settings, prompt)
                sentiment_match = result["sentiment"] == case["expected_sentiment"]
                emotion_match = result["emotion"] == case["expected_emotion"]
                saved_result = {
                    "id": case["id"],
                    "sentiment": result["sentiment"],
                    "expected_sentiment": case["expected_sentiment"],
                    "sentiment_match": sentiment_match,
                    "emotion": result["emotion"],
                    "expected_emotion": case["expected_emotion"],
                    "emotion_match": emotion_match,
                    "response_id": result["response_id"],
                    "returned_model": result["returned_model"],
                    "usage": result["usage"],
                }
                with (args.output / "responses.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(saved_result) + "\n")
                summary["parsed_valid"] += 1
                summary["sentiment_matches"] += sentiment_match
                summary["emotion_matches"] += emotion_match
                summary["results"].append(saved_result)
                print(
                    f"{case['id']}: {result['sentiment']} / {result['emotion']}",
                    flush=True,
                )
    except Exception as exc:
        summary["error"] = safe_api_error(exc)
        print(summary["error"]["message"], file=sys.stderr)
    finally:
        if client is not None:
            client.close()
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))
    return 1 if "error" in summary else 0


if __name__ == "__main__":
    raise SystemExit(main())
