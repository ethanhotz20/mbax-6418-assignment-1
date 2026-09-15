import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'src' / 'gift_cards' / 'ingest.py'

class IngestionTests(unittest.TestCase):
    def test_gzip_profile_preserves_fields_counts_ratings_and_first_five(self):
        self.assertTrue(MODULE.exists(), 'Ingestion module must exist')
        spec = importlib.util.spec_from_file_location('ingest', MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = [dict(rating=float(r), title='Example', text='Review', images=[],
                        asin='A', parent_asin='P', user_id='U', timestamp=1000,
                        helpful_vote=0, verified_purchase=True) for r in [5, 1, 5, 3, 4, 2]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reviews.jsonl.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                for record in records:
                    stream.write(json.dumps(record) + '\n')
            result = module.profile(path)
        self.assertEqual(result['observations'], 6)
        self.assertEqual(result['rating_counts'], {'1': 1, '2': 1, '3': 1, '4': 1, '5': 2})
        self.assertEqual(result['examples'], records[:5])
        self.assertEqual(set(result['columns']), set(records[0]))
        self.assertEqual(result['invalid_rating_count'], 0)
        self.assertEqual(result['blank_text_count'], 0)

if __name__ == '__main__':
    unittest.main()
