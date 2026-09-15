"""Strict three-class sentiment output handling."""
import json

from gift_cards.sentiment import prepare_review

LABELS = ("POSITIVE", "NEUTRAL", "NEGATIVE")


def parse_sentiment(raw):
    """Accept exactly one sentiment key with one allowed label."""
    pairs = json.loads(raw, object_pairs_hook=list)
    if (
        not isinstance(pairs, list)
        or len(pairs) != 1
        or not isinstance(pairs[0], tuple)
        or pairs[0][0] != "sentiment"
        or pairs[0][1] not in LABELS
    ):
        raise ValueError("Expected exactly one sentiment key with a three-class label")
    return pairs[0][1]


def classify(record, client, settings, prompt):
    """Make one rating-blind Responses API call without retries or history."""
    request = {
        "model": settings["model"],
        "instructions": prompt,
        "input": [{"role": "user", "content": prepare_review(record)}],
        "store": False,
        "reasoning": {"effort": settings["reasoning_effort"]},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "review_sentiment_three_class",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": list(LABELS)}
                    },
                    "required": ["sentiment"],
                    "additionalProperties": False,
                },
            }
        },
        "stream": True,
    }
    response = None
    text_parts = []
    with client.responses.create(
        **request, timeout=settings["timeout_seconds"]
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.done":
                text_parts.append(event.text)
            elif event.type == "response.completed":
                response = event.response
            elif event.type in (
                "response.failed",
                "response.incomplete",
                "error",
                "response.refusal.done",
            ):
                raise ValueError("Model refused or stream failed")
    if response is None or response.status != "completed":
        raise ValueError("Model response did not complete")
    raw = "".join(text_parts)
    return {
        "request": request,
        "raw_response": raw,
        "sentiment": parse_sentiment(raw),
        "response_id": response.id,
        "returned_model": response.model,
        "usage": response.usage.model_dump() if response.usage is not None else None,
    }
