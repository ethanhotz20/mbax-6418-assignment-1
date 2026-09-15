"""Download source bytes and save reproducible full-dataset descriptive reports."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gift_cards.ingest import profile

if __name__ == '__main__':
    config = json.loads((ROOT / 'config' / 'analysis.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Use a local gzip instead of downloading')
    parser.add_argument('--output', type=Path, default=ROOT/'reports')
    args = parser.parse_args()
    source = args.input or ROOT/'data'/'raw'/'Gift_Cards.jsonl.gz'
    provenance_path = source.with_suffix(source.suffix+'.download.json')
    if args.input is None and not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        partial = source.with_suffix(source.suffix+'.part')
        request = urllib.request.Request(config['dataset_url'], headers={'User-Agent': 'MBAX6418-research/1.0', 'Accept-Encoding': 'identity'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as dest:
                headers = dict(response.headers)
                final_url = response.url
                shutil.copyfileobj(response, dest)
            # Fully decompress and parse before promoting the temporary download.
            profile(partial)
            partial.replace(source)
            provenance_path.write_text(json.dumps({'url':config['dataset_url'], 'resolved_url':final_url, 'downloaded_at_utc':datetime.now(timezone.utc).isoformat(), 'headers':headers}, indent=2), encoding='utf-8')
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    result = profile(source)
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    examples = result.pop('examples')
    (args.output/'profile.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    (args.output/'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with (args.output/'rating_distribution.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rating','count','percent_of_all_reviews'])
        for rating, count in result['rating_counts'].items():
            writer.writerow([rating, count, round(100*count/result['observations'], 6) if result['observations'] else 0])
    manifest = {'source_url': config['dataset_url'] if args.input is None else None,
                'source_file':source.name, 'sha256':digest, 'compressed_bytes':source.stat().st_size,
                'python':platform.python_version(), 'platform':platform.platform(),
                'random_seed':config['random_seed'], 'sampling':'None; all records profiled',
                'classification_performed':False}
    if args.input is None and provenance_path.exists():
        manifest['download'] = json.loads(provenance_path.read_text(encoding='utf-8'))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'profile':result, 'manifest':manifest, 'example_count':len(examples)}, indent=2))
