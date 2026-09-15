# Emotion Classification Method

## Scope

This stage uses the same first 100 usable Gift Cards reviews as the binary sentiment evaluation. No balanced three-class sample is created.

The LLM receives only `title` and `text`. It returns strict JSON with one binary `sentiment` and one `emotion` from `ANGER`, `ANTICIPATION`, `DISGUST`, `FEAR`, `JOY`, `SADNESS`, `SURPRISE`, or `TRUST`. The request excludes the numeric rating, identifiers, expected labels, prior conversation, and tools. The model is fixed to `gpt-5.6-sol`, reasoning effort is `low`, response storage is disabled, and API retries are disabled. The endpoint does not support a decoding seed or temperature in this configuration, so repeated hosted-model runs are not guaranteed to be identical.

## NRC method

The NRC Word-Emotion Association Lexicon associates English words with eight emotions using binary association flags; version 0.92 was manually annotated through crowdsourcing and is available for non-commercial research and educational use.[1] This project uses the bundled English dictionary in `NRCLex==4.1.0` through its token-list interface.[2]

For each review:

1. Concatenate the title and review text.
2. Lowercase and extract ASCII word tokens with `[A-Za-z]+(?:'[A-Za-z]+)?`.
3. Count NRC word associations for the eight allowed emotions. Positive and negative lexicon categories are ignored.
4. Choose the emotion with the largest count.
5. Resolve equal maximum counts using this fixed order: `ANGER`, `ANTICIPATION`, `DISGUST`, `FEAR`, `JOY`, `SADNESS`, `SURPRISE`, `TRUST`.
6. If no token has an NRC emotion association, assign `TRUST` as the documented deterministic fallback required to produce one of the eight labels.

`emotion_match` is `True` when `llm_emotion == nrc_emotion`. Overall agreement is the number of matching rows divided by the total number of rows.

## Limitations

The NRC result is a lexical baseline rather than emotion ground truth. It does not model context, negation, sarcasm, or word order. Short reviews create many score ties or no lexicon hit; in this sample, 57 reviews had a tied positive maximum and 16 had no emotion-word match. The fixed tie rule and fallback make the result reproducible but can materially affect the agreement rate. Agreement therefore measures consistency between two methods, not classification accuracy.

## Reproduce

From the project root:

```bash
python scripts/spot_check_emotion.py --live --output reports/emotion_spot_check_gpt-5.6-sol
python scripts/classify_emotions_first_100.py
python scripts/build_dashboard.py \
  --input data/processed/first_100_sentiment_emotion_gpt-5.6-sol/results.csv \
  --output reports/sentiment_dashboard.html
python -m unittest discover -s tests -v
```

## Sources

[1] https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm — NRC Word-Emotion Association Lexicon
[2] https://pypi.org/project/NRCLex — NRCLex 4.1.0
