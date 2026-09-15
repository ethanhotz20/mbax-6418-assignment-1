"""Rating-blind LLM emotion classification and NRC lexicon scoring."""
import json
import re

from gift_cards.sentiment import prepare_review

EMOTIONS = (
    "ANGER",
    "ANTICIPATION",
    "DISGUST",
    "FEAR",
    "JOY",
    "SADNESS",
    "SURPRISE",
    "TRUST",
)
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def parse_sentiment_emotion(raw):
    """Parse exactly one binary sentiment and one allowed primary emotion."""
    pairs = json.loads(raw, object_pairs_hook=list)
    if (
        not isinstance(pairs, list)
        or len(pairs) != 2
        or any(not isinstance(pair, tuple) for pair in pairs)
        or [key for key, _ in pairs] != ["sentiment", "emotion"]
        or pairs[0][1] not in ("POSITIVE", "NEGATIVE")
        or pairs[1][1] not in EMOTIONS
    ):
        raise ValueError("Expected exactly sentiment and emotion with allowed labels")
    return dict(pairs)


def classify_sentiment_emotion(record, client, settings, prompt):
    """Make one isolated Responses API call with title and text only."""
    review_json = prepare_review(record)
    request = {
        "model": settings["model"],
        "instructions": prompt,
        "input": [{"role": "user", "content": review_json}],
        "store": False,
        "reasoning": {"effort": settings["reasoning_effort"]},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "review_sentiment_emotion",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "enum": ["POSITIVE", "NEGATIVE"],
                        },
                        "emotion": {"type": "string", "enum": list(EMOTIONS)},
                    },
                    "required": ["sentiment", "emotion"],
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
    parsed = parse_sentiment_emotion(raw)
    return {
        "request": request,
        "raw_response": raw,
        **parsed,
        "response_id": response.id,
        "returned_model": response.model,
        "usage": response.usage.model_dump() if response.usage is not None else None,
    }


def primary_emotion_from_scores(scores):
    """Select the largest NRC count; use label order for ties and TRUST for no hits."""
    counts = {}
    for emotion in EMOTIONS:
        value = scores.get(emotion.lower(), 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError("NRC emotion scores must be non-negative integers")
        counts[emotion] = value
    largest = max(counts.values())
    if largest == 0:
        return "TRUST"
    return next(emotion for emotion in EMOTIONS if counts[emotion] == largest)


def nrc_primary_emotion(record, analyzer_factory=None):
    """Assign one NRC emotion from lower-cased title/text word tokens."""
    safe = json.loads(prepare_review(record))
    tokens = TOKEN_PATTERN.findall(f"{safe['title']} {safe['text']}".lower())
    if analyzer_factory is None:
        from nrclex import NRCLex

        analyzer_factory = NRCLex
    analyzer = analyzer_factory()
    analyzer.load_token_list(tokens)
    return primary_emotion_from_scores(analyzer.raw_emotion_scores)
