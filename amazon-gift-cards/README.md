# Amazon Gift Cards: Sentiment and Emotion Analytics

MBAX 6418 graduate analytics project. **Phase 1: ingestion and descriptive profiling only.** No predictions, API calls, or model-derived labels have been produced.

## Sources

- Reviews: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz
- Documentation: https://amazon-reviews-2023.github.io
- Citation: Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952.

Download the original gzip, not a web-extracted Markdown representation: extraction can decode HTML entities or otherwise alter the review text. Review source terms before redistribution; this repository excludes raw data and reviewer-level example exports.

## Structure

```text
amazon-gift-cards/
  .gitignore                 # Secrets, environments, and bulk data excluded
  .python-version            # Python version pin
  requirements.txt           # Phase 1: no third-party dependencies
  config/analysis.json        # Source, seed, disabled classification policy
  src/gift_cards/ingest.py    # Streaming gzip JSONL profile function
  scripts/prepare_data.py     # Download/local-file CLI and report persistence
  tests/                     # Small synthetic fixtures, no external API calls
  data/raw/                  # Original gzip + download provenance (ignored)
  data/interim/              # Future cleaned/sampled data (ignored)
  data/processed/            # Future predictions/scoring outputs (ignored)
  notebooks/                 # Optional presentation/exploration, not core pipeline
  reports/
    sources.json             # Source ledger
    manifest.json            # Dataset SHA-256, bytes, runtime, download metadata
    profile.json             # Full schema, row counts, quality checks
    rating_distribution.csv  # Counts and percent of all reviews
    examples.json            # First five complete records (ignored)
    verification.json        # Repeated-run and count checks
    figures/                 # Future plots
```

## Reproduce (Windows, from this project folder)

Install Python 3.11.16 and uv, then run:

```powershell
uv venv --python 3.11.16 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe scripts/prepare_data.py
```

On POSIX, use `.venv/bin/python` instead. No activation is necessary when invoking the environment's Python explicitly. Do not recreate an environment while another process is using it.

The script reuses an existing raw file; it only downloads when absent. New downloads use a temporary `.part` file and must successfully decompress and parse before being promoted. Invalid JSON raises an error with its line number; malformed data is not silently discarded. A full-file scan computes counts without loading a DataFrame into memory. Reports are overwritten on each run, so archive them before a deliberate source or code change.

To use a separately obtained gzip:

```powershell
.venv/Scripts/python.exe scripts/prepare_data.py --input path/to/reviews.jsonl.gz --output reports/local-check
```

Local input has no claimed download URL in its manifest. Compare SHA-256 with the original manifest to confirm source identity. The saved SHA-256 fingerprints the bytes used; it is not an independently publisher-signed checksum.

## Dependencies

Phase 1 uses only Python's standard library: `gzip`, `json`, `collections`, `urllib.request`, `csv`, `hashlib`, `pathlib`, `datetime`, `argparse`, and `unittest`. No third-party package is needed; the requirements file intentionally contains documentation rather than unnecessary packages.

For later phases, add and pin tested versions of:
- `pandas`: analysis tables, joins, sampling, and exports.
- `openai`: OpenAI-compatible API client, only after verifying compatibility with the Hermes-configured endpoint and authentication flow.
- `scikit-learn`: evaluation metrics and split utilities.
- `matplotlib` (optionally `seaborn`): publication-ready figures.
- `pyarrow`: optional Parquet persistence if justified.

Do not install heavyweight NLP frameworks or a model-serving stack for API-based classification unless a later requirement demands them.

## Reproducibility and safeguards

- Seed: **6418**, reserved in `config/analysis.json`. Phase 1 is deterministic and uses all records, so no random sampling occurs.
- Examples are the **first five source records**, not a representative or random sample.
- Raw data remains unchanged. Schema checks measure missing/null fields, types, invalid ratings, and blank text; they are not a complete duplicate/language/content audit.
- Ratings are currently used only for descriptive analysis. Future prediction payloads must exclude `rating` and all unrelated metadata through a tested input allowlist.
- Titles and text may themselves contain explicit ratings (e.g., 'Five Stars'). Define and test a rating-reference redaction/exclusion policy before any inference; merely dropping the rating column is insufficient.
- Keep evaluation ratings separate from model inputs. Join predictions back by an internal record identifier only after prediction.
- Classification is disabled. No model or generation settings are selected yet. Before enabling it, freeze the model identifier, prompts, label schema, generation limits, supported seed/temperature settings, and endpoint identity; record request/response metadata and cache outputs. Fixed settings reduce variation but do not guarantee bitwise deterministic hosted-model outputs.
- Use the OpenAI-compatible endpoint configured for Hermes, not an assumed default endpoint. Runtime chat provider metadata alone does not establish a usable standalone API authentication method.
- Credentials must stay in an approved runtime credential mechanism, never in source, notebooks, outputs, or Git. This phase does not inspect Hermes credential files.
- `.gitignore` is defense-in-depth, not a credential scanner. Before eventual GitHub publication, inspect staged changes and file sizes. No Git repository was initialized and nothing was committed or pushed in this phase.

## Interpretation and next stage

Review text contains HTML artifacts; timestamps are represented in milliseconds in this file. Use an explicit UTC millisecond conversion in later cleaning. Blank bodies need an explicit title-only/exclusion policy. Star-rating imbalance means raw accuracy alone will be misleading; ratings are imperfect sentiment proxies and are not emotion ground truth.

Next: define sentiment/emotion label rubrics and evaluation design, audit duplicates and language, specify cleaning and rating-leakage controls, then create a seeded pilot sample with a separate evaluation table. Do not classify until that design and the endpoint/model settings are agreed.
