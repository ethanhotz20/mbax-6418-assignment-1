"""Synthetic-only sentiment checks. Default offline; --live uses Hermes-managed OAuth."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gift_cards.sentiment import classify, prepare_review, safe_api_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Make paid/quota-consuming provider calls')
    parser.add_argument('--limit', type=int, help='Run only the first N synthetic fixtures')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/sentiment_spot_check')
    args = parser.parse_args()
    settings_path = ROOT / 'config/sentiment.json'
    prompt_path = ROOT / 'prompts/sentiment_v1.txt'
    fixtures_path = ROOT / 'tests/fixtures/sentiment_cases.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8'))
    prompt = prompt_path.read_text(encoding='utf-8')
    cases = json.loads(fixtures_path.read_text(encoding='utf-8'))
    if args.limit is not None:
        if not 1 <= args.limit <= len(cases):
            parser.error('--limit must be within the fixture count')
        cases = cases[:args.limit]
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory to preserve prior evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for case in cases:
        # Counterfactual metadata test is local only. Neither expected labels nor IDs are inputs.
        low = prepare_review({**case, 'rating': 1, 'user_id': 'PRIVATE_SENTINEL'})
        high = prepare_review({**case, 'rating': 5, 'user_id': 'PRIVATE_SENTINEL'})
        if low != high or set(json.loads(low)) != {'title', 'text'} or 'PRIVATE_SENTINEL' in low:
            raise ValueError('Input isolation check failed')
        prepared.append({'id': case['id'], 'model_input': json.loads(low)})
    (args.output / 'prepared_inputs.json').write_text(json.dumps(prepared, indent=2)+'\n', encoding='utf-8')
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'settings': settings,
        'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (prompt_path, settings_path, fixtures_path,
                                ROOT / 'src/gift_cards/sentiment.py', Path(__file__))},
        'raw_dataset_read': False, 'conversation_history_sent': False,
    }
    summary = {'mode': 'live' if args.live else 'dry_run', 'planned': len(cases),
               'api_calls': 0, 'parsed_valid': 0, 'matched_expected': 0,
               'rating_invariance_passed': True, 'results': []}
    client = None
    try:
        if args.live:
            # Import installed Hermes only for live inference. Never open or print credential files.
            import agent.auxiliary_client as auxiliary
            from hermes_cli.config import load_config
            saved = load_config().get('model', {})
            if not isinstance(saved, dict) or any(saved.get(k) != settings[v] for k, v in
                    (('provider', 'provider'), ('default', 'model'), ('base_url', 'base_url'))):
                raise RuntimeError('Hermes provider/model/endpoint changed; review config/sentiment.json first')
            client, model = auxiliary.resolve_provider_client(
                settings['provider'], model=settings['model'], raw_codex=True)
            if client is None or model != settings['model']:
                raise RuntimeError('Hermes authentication or exact model resolution unavailable')
            if str(client.base_url).rstrip('/') != settings['base_url'].rstrip('/'):
                raise RuntimeError('Resolved endpoint differs from the approved Hermes endpoint')
            client = client.with_options(max_retries=settings['max_retries'])
            manifest['openai_version'] = importlib.metadata.version('openai')
            hermes_root = Path(auxiliary.__file__).resolve().parents[1]
            revision = subprocess.run(['git', '-C', str(hermes_root), 'rev-parse', 'HEAD'],
                                      capture_output=True, text=True)
            manifest['hermes_revision'] = revision.stdout.strip() if revision.returncode == 0 else None
            manifest['hermes_adapter_sha256'] = hashlib.sha256(Path(auxiliary.__file__).read_bytes()).hexdigest()
            for case in cases:
                summary['api_calls'] += 1
                # This is the only inference call site. No raw dataset loader exists in this script.
                result = classify(case, client, settings, prompt)
                result.update(id=case['id'], kind=case['kind'], expected=case['expected'],
                              matched_expected=result['sentiment'] == case['expected'])
                with (args.output / 'responses.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                summary['parsed_valid'] += 1
                summary['matched_expected'] += result['matched_expected']
                summary['results'].append({k: result[k] for k in
                                          ('id', 'kind', 'sentiment', 'expected', 'matched_expected')})
                print(f"{case['id']}: {result['sentiment']} (expected {case['expected']})", flush=True)
    except Exception as exc:
        # Error strings, headers and client reprs can include secrets: persist only mapped diagnostics.
        summary['error'] = safe_api_error(exc)
        print(summary['error']['message'], file=sys.stderr)
        print('Safe error metadata saved in summary.json; no response was invented.', file=sys.stderr)
    finally:
        if client is not None:
            client.close()
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    if 'error' in summary or (args.live and summary['matched_expected'] != len(cases)):
        sys.exit(1)
