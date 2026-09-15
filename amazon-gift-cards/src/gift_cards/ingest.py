"""Memory-bounded, strict UTF-8 gzip JSONL profiling. No classification."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path


def profile(path):
    columns = set()
    types = defaultdict(Counter)
    present = Counter()
    nulls = Counter()
    ratings = Counter({str(n): 0 for n in range(1, 6)})
    examples = []
    count = invalid = blank = blank_lines = 0
    timestamps = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                blank_lines += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {line_number}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'Expected object on line {line_number}')
            count += 1
            columns.update(record)
            for key, value in record.items():
                present[key] += 1
                types[key][type(value).__name__] += 1
                nulls[key] += value is None
            rating = record.get('rating')
            if not isinstance(rating, bool) and isinstance(rating, (int, float)) and rating in (1, 2, 3, 4, 5):
                ratings[str(int(rating))] += 1
            else:
                invalid += 1
            blank += not str(record.get('text') or '').strip()
            timestamp = record.get('timestamp')
            if isinstance(timestamp, int):
                if not timestamps:
                    timestamps = [timestamp, timestamp]
                timestamps = [min(timestamps[0], timestamp), max(timestamps[1], timestamp)]
            if len(examples) < 5:
                examples.append(record)
    return {
        'observations': count, 'columns': sorted(columns),
        'schema': {k: {'types': dict(sorted(types[k].items())), 'missing': count-present[k], 'null': nulls[k]} for k in sorted(columns)},
        'rating_counts': dict(sorted(ratings.items())),
        'invalid_rating_count': invalid, 'blank_text_count': blank,
        'blank_lines': blank_lines, 'timestamp_range_raw': timestamps,
        'example_selection': 'First five records in source order; not a representative sample',
        'examples': examples,
    }
