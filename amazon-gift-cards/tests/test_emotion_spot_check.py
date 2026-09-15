"""Offline contract for the sentiment-plus-emotion spot check."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EmotionSpotCheckTests(unittest.TestCase):
    def test_offline_spot_check_prepares_only_title_and_text(self):
        script = ROOT / "scripts/spot_check_emotion.py"
        self.assertTrue(script.exists(), "Emotion spot-check script must exist")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "spot"
            result = subprocess.run(
                ["python", str(script), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            prepared = json.loads(
                (output / "prepared_inputs.json").read_text(encoding="utf-8")
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["mode"], "dry_run")
        self.assertEqual(summary["api_calls"], 0)
        self.assertEqual(summary["planned"], 8)
        self.assertEqual(len(prepared), 8)
        self.assertTrue(all(set(item["model_input"]) == {"title", "text"} for item in prepared))
        self.assertFalse(manifest["raw_dataset_read"])
        self.assertFalse(manifest["conversation_history_sent"])


if __name__ == "__main__":
    unittest.main()
