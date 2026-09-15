import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_auditable_reports_from_local_gzip(self):
        script = ROOT / 'scripts' / 'prepare_data.py'
        self.assertTrue(script.exists(), 'Pipeline entry point must exist')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'fixture.gz'
            with gzip.open(source, 'wt', encoding='utf-8') as stream:
                stream.write(json.dumps({'rating': 5.0, 'title': 'Hi', 'text': 'Good'})+'\n')
            output = root / 'reports'
            run = subprocess.run([sys.executable, str(script), '--input', str(source), '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((output/'profile.json').read_text())
            self.assertEqual(report['observations'], 1)
            self.assertEqual(len(json.loads((output/'examples.json').read_text())), 1)
            self.assertTrue((output/'rating_distribution.csv').exists())
            self.assertEqual(len(json.loads((output/'manifest.json').read_text())['sha256']), 64)

if __name__ == '__main__':
    unittest.main()
