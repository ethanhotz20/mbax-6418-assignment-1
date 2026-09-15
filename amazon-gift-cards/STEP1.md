# Assignment Step 1: binary sentiment prompt and synthetic spot checks

## Deliverables

- `prompts/sentiment_v1.txt`: exact reusable classifier instructions.
- `src/gift_cards/sentiment.py`: title/text input allowlist, rating-reference hold policy, strict JSON parser, isolated Responses API call.
- `config/sentiment.json`: fixed endpoint/model/settings for this stage.
- `scripts/spot_check_sentiment.py`: offline preparation by default; explicit `--live` for synthetic fixture calls only.
- `tests/fixtures/sentiment_cases.json`: predeclared expected labels, separate from model input.
- `tests/test_sentiment.py`, `tests/test_spot_check.py`: offline regression tests.
- `reports/sentiment_spot_check/`: saved requests, raw JSON answers, parsed labels, request IDs, usage, settings, hashes and summary.
- `requirements-inference.txt`: exercised OpenAI SDK version, 2.24.0.

## Prompt and binary labeling policy

Use the prompt file as `instructions` and a JSON object with only `title` and `text` as the isolated user input. No examples' expected labels, record identifiers, ratings, conversation history, dataset aggregates, memory, or tools are provided. Reviews are untrusted data, not task instructions.

Output is exactly `{"sentiment":"POSITIVE"}` or `{"sentiment":"NEGATIVE"}`. The request supplies a strict JSON Schema with an enum, required key, and `additionalProperties: false`. Python also validates the complete response independently. Duplicate keys, extra keys, Markdown fences, invalid labels, incomplete streams, and refusals are errors, never silently coerced to a valid label.

Mixed reviews are judged by significance and explicit final verdict, not word counts. Delivery/customer-service experiences are valid targets. Sarcasm follows the described outcome. For the unavoidable binary-only neutral case, the prompt uses **NEGATIVE as a fixed tie-break**. This is an explicit assignment convention, not evidence that a factual review expresses dissatisfaction. Empty title AND body are held without a model call. This policy can bias class proportions and must be revisited when designing substantive evaluation.

## Rating isolation and limits

`prepare_review` accesses only `title` and `text`, never `rating`. Local tests replace a synthetic rating with opposing values and require identical model input. Metadata and expected labels never cross the input boundary.

To prevent obvious rating disclosure embedded in the allowed text, preprocessing fails closed on numeric characters, star symbols, common English rating terms, and number words zero through ten. Detection checks HTML-unescaped, NFKC-normalized text with zero-width format characters removed. It does not alter the original accepted text and does not consult the rating field. Examples: `Five Stars`, `5/5`, fullwidth digits, HTML-encoded digits, and star glyphs are held before any call.

This deliberately conservative guard also holds harmless dates, prices, and words such as 'one'. It is **not** a production-ready multilingual detector and cannot prove that all obfuscated rating references or covert instructions are impossible. No prompt can prove the model's internal reasoning. What is directly verified here is the input allowlist, rating invariance, tested hold cases, and the saved synthetic requests. Manual review or a validated broader leakage policy is required before applying this to unrestricted real reviews. Do not quietly relax the guard or remove true opinion words to improve accuracy.

## Endpoint and settings

Saved Hermes configuration and resolved client both selected:
- Provider: `openai-codex`
- Model: `gpt-5.6-sol`
- Base URL: `https://chatgpt.com/backend-api/codex`
- Protocol: OpenAI SDK Responses API, streaming
- Reasoning effort: `low`
- `store: false`; no prior-response ID; no tool definitions
- SDK retries: zero; timeout: 120 seconds (SDK network timeout, not a hard total wall-clock deadline)
- Temperature, seed, and max output tokens: omitted because the Codex endpoint does not support the usual controls.

Thus inputs/settings are fixed and auditable, but deterministic hosted-model output is not guaranteed. The model identifier is fixed, not guaranteed to be an immutable provider snapshot. Requests do not claim that project sampling seed 6418 controls generation.

Authentication remains inside Hermes's existing credential resolver. No credentials are copied into this project, printed, or included in result files. The script refuses a mismatch between its fixed settings, saved Hermes model configuration, and resolved endpoint. No fallback provider is used.

## Reproduce

From `amazon-gift-cards/`, offline (no Hermes imports or network):

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/spot_check_sentiment.py --output reports/new-dry-run
```

For live inference, use the Python interpreter from the authenticated Hermes installation, not the isolated standard-library-only project environment. On the environment used for this run:

```powershell
& "$env:LOCALAPPDATA/hermes/hermes-agent/venv/Scripts/python.exe" scripts/spot_check_sentiment.py --live --output reports/new-live-run
```

`--live` consumes provider quota. Output directories must be empty to prevent overwriting evidence. `--limit 1` runs only the first synthetic fixture. The runner has no raw dataset loading path. Installing `requirements-inference.txt` alone does not install Hermes or authenticate an account; reproduce the Hermes revision recorded in the manifest and authenticate through its supported workflow. Never put OAuth tokens in the repository.

## Integration issue resolved

The first high-level SDK stream attempt failed parsing because `get_final_response().output_text` was empty. A diagnostic rerun confirmed the symptom. A lower-level event probe then showed valid JSON in `response.output_text.done` while `response.completed.response.output` was an empty array. The implementation now consumes completed text events with `responses.create(stream=True)` and requires a successful completion event before parsing. This matches the transport workaround used by Hermes itself. Strict JSON Schema and the prompt were retained; there is no regex extraction or fabricated fallback answer.

The initial failed capability summary remains in `reports/sentiment_capability_check/`. The original successful GPT-6 spot-check report remains in `reports/sentiment_spot_check/`; the GPT-5.6 validation is saved separately in `reports/sentiment_spot_check_gpt-5.6-sol/`. Diagnostic probes are not part of the scored fixture set.

## Result interpretation

The successful live run returned valid labels matching all 14 predeclared fixture expectations, including six obvious positive/negative cases. These are engineering spot checks, not a representative evaluation or a claim of perfect real-world accuracy. The limited-context case tests the explicit tie-break policy rather than an objectively negative opinion. No Amazon dataset reviews have been classified.
