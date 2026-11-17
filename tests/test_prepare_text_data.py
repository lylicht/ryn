import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_text_data import iter_plain_text  # noqa: E402


class PreparationTests(unittest.TestCase):
    def test_large_plain_text_streams_as_bounded_records(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "large.txt"
            content = "A long plain-text source split into bounded streaming chunks.\n" * 8
            source.write_text(content, encoding="utf-8")
            chunks = list(iter_plain_text(source, 16))
            self.assertEqual("".join(chunks), content)
            self.assertTrue(all(len(chunk.encode("utf-8")) <= 16 for chunk in chunks))

    def test_dedup_privacy_filter_provenance_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [
                {"text": f"Clean synthetic example {index} explains multilingual corpus quality and provenance in detail.", "language": "en"}
                for index in range(24)
            ]
            rows.append(rows[0])
            rows.append({"text": "Contact someone@example.com for this synthetic training record.", "language": "en"})
            rows.append({"text": "A separate provenance example with its own source license and source metadata.", "language": "en", "source": "corpus/part-1", "license": "MIT synthetic"})
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "prepared.jsonl"
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "prepare_text_data.py"),
                 "--input", str(source), "--output", str(output), "--license", "CC0 synthetic",
                 "--min-chars", "5", "--validation-fraction", "0.5"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
            train = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            validation = [json.loads(line) for line in Path(summary["validation_output"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["duplicates"], 1)
            self.assertEqual(summary["filtered_sensitive_records"], 1)
            self.assertTrue(train and validation)
            self.assertTrue(all(row["license"] in {"CC0 synthetic", "MIT synthetic"} for row in train + validation))
            provenance_row = next(row for row in train + validation if row["license"] == "MIT synthetic")
            self.assertEqual(provenance_row["source"], "corpus/part-1")
            self.assertEqual(manifest["outputs"]["train"]["sha256"], __import__("hashlib").sha256(output.read_bytes()).hexdigest())
            self.assertEqual(manifest["tokenizer_tokens_by_language"], None)


if __name__ == "__main__":
    unittest.main()
