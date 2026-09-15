"""Verify the dashboard is generated entirely from a CSV fixture."""
import csv
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['rating', 'title', 'text', 'actual_sentiment',
           'predicted_sentiment', 'correct', 'llm_emotion',
           'nrc_emotion', 'emotion_match']


class DashboardTests(unittest.TestCase):
    def test_dashboard_numbers_and_rows_come_from_csv(self):
        script = ROOT / 'scripts/build_dashboard.py'
        self.assertTrue(script.exists(), 'Dashboard generator must exist')
        rows = [
            ['5.0', 'Great <gift>', 'Loved & used it', 'POSITIVE', 'POSITIVE', 'True',
             'JOY', 'JOY', 'True'],
            ['1.0', 'Bad', 'Failed', 'NEGATIVE', 'POSITIVE', 'False',
             'ANGER', 'SADNESS', 'False'],
            ['2.0', 'Awful', 'Never worked', 'NEGATIVE', 'NEGATIVE', 'True',
             'SADNESS', 'SADNESS', 'True'],
        ]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'results.csv'
            output = Path(folder) / 'dashboard.html'
            with source.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(COLUMNS)
                writer.writerows(rows)
            run = subprocess.run([sys.executable, str(script), '--input', str(source),
                                  '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            page = output.read_text(encoding='utf-8')
        self.assertIn('data-total="3"', page)
        self.assertIn('data-correct="2"', page)
        self.assertIn('data-incorrect="1"', page)
        self.assertIn('data-actual-positive="1"', page)
        self.assertIn('data-actual-negative="2"', page)
        self.assertIn('data-predicted-positive="2"', page)
        self.assertIn('data-predicted-negative="1"', page)
        self.assertIn('data-tp="1"', page)
        self.assertIn('data-fp="1"', page)
        self.assertIn('data-tn="1"', page)
        self.assertIn('data-fn="0"', page)
        self.assertIn('66.67%', page)
        self.assertIn('data-emotion-agreement="0.666667"', page)
        self.assertIn('<title>Amazon Gift Cards Sentiment and Emotion Results</title>', page)
        self.assertIn('Emotion agreement', page)
        self.assertIn('<th scope="col">LLM emotion</th>', page)
        self.assertIn('<th scope="col">NRC emotion</th>', page)
        self.assertIn('<td>ANGER</td>', page)
        self.assertIn('<td>SADNESS</td>', page)
        self.assertIn('Great &lt;gift&gt;', page)
        self.assertIn('Loved &amp; used it', page)
        self.assertEqual(page.count('<tr class="review-row'), 3)
        self.assertIn('data-filter="all"', page)
        self.assertIn('data-filter="correct"', page)
        self.assertIn('data-filter="incorrect"', page)
        self.assertIn('id="visible-count"', page)
        self.assertIn("row.classList.toggle('filtered-out'", page)


if __name__ == '__main__':
    unittest.main()
