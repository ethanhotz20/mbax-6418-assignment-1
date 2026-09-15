"""Rating-blind input preparation and strict output validation. No API dependencies."""
import html
import json
import re
import unicodedata

RATING_WORDS = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|stars?|ratings?|rated|rates?|scores?|scored)\b', re.IGNORECASE)


def safe_api_error(exc):
    """Map provider failures to non-secret diagnostics without echoing exception text."""
    status = getattr(exc, 'status_code', None)
    message_lower = str(exc).lower()
    if status == 429 and ('usage limit' in message_lower or 'quota' in message_lower
                          or 'credit' in message_lower):
        code = 'API_USAGE_LIMIT_REACHED'
        message = ('The request could not run because the API usage limit was reached. '
                   'Resolve the provider account quota, credits, or subscription usage cap before retrying.')
    elif status == 429:
        code = 'API_RATE_LIMITED'
        message = 'The request could not run because the provider rate limit was reached. Retry later.'
    else:
        code = 'API_REQUEST_FAILED'
        message = 'The model request failed. Review the safe error type and HTTP status; credentials are not shown.'
    return {'type': type(exc).__name__, 'http_status': status, 'code': code, 'message': message}


def prepare_review(record):
    """Project an input record onto a strict title/text allowlist."""
    review = {}
    for key in ('title', 'text'):
        value = record.get(key)
        if not isinstance(value, str):
            raise ValueError('Title and text must be strings')
        normalized = unicodedata.normalize('NFKC', html.unescape(value))
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Cf')
        if (any(c.isnumeric() or c in '★☆⭐✩✪✫✬✭✮✯' for c in normalized)
                or RATING_WORDS.search(normalized)):
            raise ValueError('Review held: possible rating reference or numeric content')
        review[key] = value
    if not any(value.strip() for value in review.values()):
        raise ValueError('Review held: no title or text content')
    return json.dumps(review, ensure_ascii=False)


def parse_sentiment(raw):
    """Reject invalid JSON, duplicate/extra keys, and any non-enum value."""
    pairs = json.loads(raw, object_pairs_hook=list)
    if (not isinstance(pairs, list) or len(pairs) != 1
            or not isinstance(pairs[0], tuple) or pairs[0][0] != 'sentiment'
            or pairs[0][1] not in ('POSITIVE', 'NEGATIVE')):
        raise ValueError('Expected exactly one sentiment key with POSITIVE or NEGATIVE')
    return pairs[0][1]


def classify(record, client, settings, prompt):
    """Make one isolated Responses API call; no chat history, tools, or retries."""
    review_json = prepare_review(record)
    request = {
        'model': settings['model'],
        'instructions': prompt,
        'input': [{'role': 'user', 'content': review_json}],
        'store': False,
        'reasoning': {'effort': settings['reasoning_effort']},
        'text': {'format': {
            'type': 'json_schema', 'name': 'review_sentiment', 'strict': True,
            'schema': {'type': 'object', 'properties': {
                'sentiment': {'type': 'string', 'enum': ['POSITIVE', 'NEGATIVE']}},
                'required': ['sentiment'], 'additionalProperties': False}}},
    }
    request['stream'] = True
    response = None
    text_parts = []
    # Codex may leave response.completed.output empty: consume completed text events.
    with client.responses.create(**request, timeout=settings['timeout_seconds']) as stream:
        for event in stream:
            if event.type == 'response.output_text.done':
                text_parts.append(event.text)
            elif event.type == 'response.completed':
                response = event.response
            elif event.type in ('response.failed', 'response.incomplete', 'error',
                                'response.refusal.done'):
                raise ValueError('Model refused or stream failed')
    if response is None or response.status != 'completed':
        raise ValueError('Model response did not complete')
    raw = ''.join(text_parts)
    sentiment = parse_sentiment(raw)
    return {'request': request, 'raw_response': raw, 'sentiment': sentiment,
            'response_id': response.id, 'returned_model': response.model,
            'usage': response.usage.model_dump() if response.usage is not None else None}
