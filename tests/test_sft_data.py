import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from hf_dataset_runtime import is_remote_uri  # noqa: E402
from prepare_oasst_sft import conversation_for  # noqa: E402


class SftDataTests(unittest.TestCase):
    def test_remote_uri_detection_does_not_treat_windows_paths_as_urls(self):
        self.assertFalse(is_remote_uri(r"C:\data\corpus.parquet"))
        self.assertFalse(is_remote_uri("file:///C:/data/corpus.parquet"))
        self.assertTrue(is_remote_uri("hf://datasets/OpenAssistant/oasst1/data/train.parquet"))

    def test_reconstructs_multiturn_user_assistant_path(self):
        messages = {
            "u1": {"message_id": "u1", "parent_id": None, "role": "prompter", "lang": "en", "text": "first question"},
            "a1": {"message_id": "a1", "parent_id": "u1", "role": "assistant", "lang": "en", "text": "first answer"},
            "u2": {"message_id": "u2", "parent_id": "a1", "role": "prompter", "lang": "en", "text": "follow-up"},
            "a2": {"message_id": "a2", "parent_id": "u2", "role": "assistant", "lang": "en", "text": "follow-up answer"},
        }
        self.assertEqual(conversation_for("a2", messages), [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "follow-up answer"},
        ])

    def test_rejects_missing_parent_and_cross_language_context(self):
        missing = {"a": {"message_id": "a", "parent_id": "missing", "role": "assistant", "lang": "en", "text": "answer"}}
        self.assertIsNone(conversation_for("a", missing))
        mixed = {
            "u": {"message_id": "u", "parent_id": None, "role": "prompter", "lang": "es", "text": "pregunta"},
            "a": {"message_id": "a", "parent_id": "u", "role": "assistant", "lang": "en", "text": "answer"},
        }
        self.assertIsNone(conversation_for("a", mixed))


if __name__ == "__main__":
    unittest.main()
