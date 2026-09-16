"""Verify the balanced dashboard derives all values from its CSV input."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NRC_COLUMNS = [
    "sample_id", "source_line", "rating", "title", "text",
    "actual_sentiment", "predicted_sentiment", "correct",
    "nrc_emotion",
]
LLM_COLUMNS = NRC_COLUMNS[:-1] + ["llm_emotion", "nrc_emotion", "emotion_match"]


class BalancedDashboardTests(unittest.TestCase):
    def test_dashboard_metrics_distributions_matrix_and_rows_come_from_csv(self):
        script = ROOT / "scripts/build_balanced_dashboard.py"
        self.assertTrue(script.exists(), "Balanced dashboard generator must exist")
        rows = [
            [1, 11, 5.0, "Great <gift>", "Loved & used it", "POSITIVE", "POSITIVE", "True", "JOY"],
            [2, 12, 4.0, "Fine", "Works", "POSITIVE", "NEUTRAL", "False", "JOY"],
            [3, 13, 3.0, "Mixed", "Average", "NEUTRAL", "NEUTRAL", "True", "SURPRISE"],
            [4, 14, 3.0, "Poor", "Problems", "NEUTRAL", "NEGATIVE", "False", "ANGER"],
            [5, 15, 2.0, "Bad", "Failed", "NEGATIVE", "NEGATIVE", "True", "ANGER"],
            [6, 16, 1.0, "Awful", "Broken", "NEGATIVE", "POSITIVE", "False", "SADNESS"],
        ]
        llm_emotions = ["JOY", "TRUST", "SURPRISE", "ANGER", "ANGER", "DISGUST"]
        llm_rows = [
            row[:-1] + [llm, row[-1], str(llm == row[-1])]
            for row, llm in zip(rows, llm_emotions)
        ]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "results.csv"
            llm_partial = Path(folder) / "partial.csv"
            output = Path(folder) / "dashboard.html"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(NRC_COLUMNS)
                writer.writerows(rows)
            with llm_partial.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(LLM_COLUMNS)
                writer.writerows(llm_rows)
            run = subprocess.run(
                [sys.executable, str(script), "--input", str(source),
                 "--llm-partial", str(llm_partial), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            page = output.read_text(encoding="utf-8")

        for marker in (
            'data-total="6"', 'data-accuracy="0.500000"',
            'data-llm-emotion-completed="6"',
            'data-correct="3"', 'data-incorrect="3"',
            'data-rating-1="1"', 'data-rating-2="1"', 'data-rating-3="2"',
            'data-rating-4="1"', 'data-rating-5="1"',
            'data-actual-positive="2"', 'data-actual-neutral="2"',
            'data-actual-negative="2"', 'data-predicted-positive="2"',
            'data-predicted-neutral="2"', 'data-predicted-negative="2"',
            'data-cm-positive-positive="1"', 'data-cm-positive-neutral="1"',
            'data-cm-neutral-neutral="1"', 'data-cm-neutral-negative="1"',
            'data-cm-negative-positive="1"', 'data-cm-negative-negative="1"',
            'data-neutral-predicted-positive="0"',
            'data-neutral-predicted-neutral="1"',
            'data-neutral-predicted-negative="1"',
            'data-class-positive="0.500000"', 'data-class-neutral="0.500000"',
            'data-class-negative="0.500000"',
            'data-nrc-anger="2"', 'data-nrc-joy="2"',
        ):
            self.assertIn(marker, page)
        self.assertIn("50.00%", page)
        self.assertIn("Great &lt;gift&gt;", page)
        self.assertIn("Loved &amp; used it", page)
        self.assertEqual(page.count('<tr class="review-row'), 6)
        self.assertIn('<th scope="col">LLM emotion</th>', page)
        self.assertIn('<th scope="col">NRC emotion</th>', page)
        self.assertNotIn("LLM emotion analysis is incomplete", page)
        self.assertNotIn("Not available", page)
        self.assertIn("LLM emotion distribution", page)
        self.assertIn("LLM vs NRC emotion agreement", page)
        self.assertIn('data-llm-anger="2"', page)
        self.assertIn('data-llm-joy="1"', page)
        self.assertIn('data-emotion-agreements="4"', page)
        self.assertIn('data-emotion-disagreements="2"', page)
        self.assertIn('data-emotion-agreement-rate="0.666667"', page)
        self.assertIn("NRC emotion distribution", page)
        self.assertIn("Actual NEUTRAL review outcomes", page)
        self.assertIn('class="largest-problem"', page)
        self.assertIn('id="filter-all"', page)
        self.assertIn('id="filter-correct"', page)
        self.assertIn('id="filter-incorrect"', page)
        self.assertIn('class="count-all">6</span>', page)
        self.assertIn('class="count-correct">3</span>', page)
        self.assertIn('class="count-incorrect">3</span>', page)
        self.assertIn('#filter-correct:checked ~ .table-wrap .review-row.incorrect', page)
        self.assertIn('#filter-incorrect:checked ~ .table-wrap .review-row.correct', page)


if __name__ == "__main__":
    unittest.main()
