"""A dry run must work without Hermes, credentials, network, or the raw dataset."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SpotCheckTests(unittest.TestCase):
    def test_offline_spot_check_prepares_all_cases_without_network(self):
        script = ROOT / 'scripts/spot_check_sentiment.py'
        self.assertTrue(script.exists(), 'Spot check script must exist')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'check'
            run = subprocess.run([sys.executable, str(script), '--output', str(output)],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((output / 'summary.json').read_text())
            cases = json.loads((ROOT / 'tests/fixtures/sentiment_cases.json').read_text())
            self.assertEqual(report['planned'], len(cases))
            self.assertEqual(report['api_calls'], 0)
            self.assertTrue(report['rating_invariance_passed'])
            self.assertEqual(report['mode'], 'dry_run')


if __name__ == '__main__':
    unittest.main()
