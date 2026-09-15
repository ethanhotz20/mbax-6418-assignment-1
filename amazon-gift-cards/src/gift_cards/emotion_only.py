"""Strict rating-blind emotion-only classification."""
import json

from gift_cards.emotion import EMOTIONS
from gift_cards.sentiment import prepare_review


def parse_emotion(raw):
    pairs = json.loads(raw, object_pairs_hook=list)
    if (
        not isinstance(pairs, list)
        or len(pairs) != 1
        or not isinstance(pairs[0], tuple)
        or pairs[0][0] != "emotion"
        or pairs[0][1] not in EMOTIONS
    ):
        raise ValueError("Expected exactly one emotion key with an allowed label")
    return pairs[0][1]


def classify(record, client, settings, prompt):
    """Make one isolated emotion call with title/text only."""
    request = {
        "model": settings["model"],
        "instructions": prompt,
        "input": [{"role": "user", "content": prepare_review(record)}],
        "store": False,
        "reasoning": {"effort": settings["reasoning_effort"]},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "review_primary_emotion",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "emotion": {"type": "string", "enum": list(EMOTIONS)}
                    },
                    "required": ["emotion"],
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
        "emotion": parse_emotion(raw),
        "response_id": response.id,
        "returned_model": response.model,
        "usage": response.usage.model_dump() if response.usage is not None else None,
    }
