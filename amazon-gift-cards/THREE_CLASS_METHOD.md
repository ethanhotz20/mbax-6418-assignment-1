# Three-Class Sentiment Method

## Labels

- Ratings 4–5: `POSITIVE`
- Rating 3: `NEUTRAL`
- Ratings 1–2: `NEGATIVE`

Ratings are used only after inference to create evaluation labels. The LLM request contains only the review title and review text.

## Balanced sample

The sample uses seed `6418` and one-pass stratified reservoir sampling. The complete `Gift_Cards.jsonl.gz` file is scanned before finalizing 50 reviews from each class, so each eligible review within a class has the same selection probability.

The existing rating-leakage safeguard excludes empty reviews and reviews whose title or text may disclose a rating or contain numeric content. This produced the following eligible population before sampling:

- Positive: 90,290
- Neutral: 1,865
- Negative: 6,897
- Excluded: 53,358

This safeguard protects the rating-blind evaluation but makes the sample representative of eligible text-only reviews rather than every review in the category.

## Model settings

- Provider: `openai-codex`
- Model: `gpt-5.6-sol`
- Reasoning effort: `low`
- Structured output: strict JSON Schema
- API retries: zero
- Response storage: disabled

The endpoint does not support temperature or a decoding seed in this configuration. The sample is exactly reproducible, but a new hosted-model run is not guaranteed to return identical predictions. Saved predictions should therefore be treated as the experiment record.

## Saved run results

- Overall: 101/150 correct (67.33%)
- Positive: 46/50 correct (92.00%)
- Neutral: 6/50 correct (12.00%)
- Negative: 49/50 correct (98.00%)

Confusion matrix (rows are actual labels; columns are predicted labels):

| Actual \\ Predicted | POSITIVE | NEUTRAL | NEGATIVE |
|---|---:|---:|---:|
| POSITIVE | 46 | 3 | 1 |
| NEUTRAL | 10 | 6 | 34 |
| NEGATIVE | 0 | 1 | 49 |

Among the 50 rating-derived neutral reviews, the model predicted 10 as positive (20%), 6 as neutral (12%), and 34 as negative (68%). In this run, the LLM's text-based labels therefore disagreed with the rating-derived neutral label for 44 of 50 reviews; this does not establish which method was correct.

## Reproduce

```bash
python scripts/create_balanced_sample.py
python scripts/classify_balanced_150.py
python -m unittest discover -s tests -v
```

The scripts refuse to overwrite existing sample or classification outputs. Move or archive the existing output directory before intentionally running a new experiment.
