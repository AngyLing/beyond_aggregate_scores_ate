"""Resume identity checks use only synthetic batches and standard library."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ate_paper.experiments.llm_run_paths import pin_batch_selection, pin_run_identity


class LLMRunIdentityTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.batches = self.root / "batches"
        self.batches.mkdir()
        self.prompt = self.batches / "batch_0001_prompt.txt"
        self.meta = self.batches / "batch_0001_meta.json"
        self.prompt.write_text("Synthetic prompt.", encoding="utf-8")
        self.write_meta("first")
        (self.batches / "batch_0001_input.json").write_text(
            json.dumps({"sentences": [{"doc_id": "toy", "sent_id": 1, "text": "Alpha."}]}),
            encoding="utf-8")
        self.args = SimpleNamespace(raw_output=str(self.root / "raw"), batches=str(self.batches),
                                    provider="deepseek", model="toy", task="genia_span_class",
                                    prompt_version="v1.1", temperature=0.0, top_p=1.0,
                                    seed=42, max_output_tokens=32768, base_url=None, endpoint=None, overwrite=False)

    def write_meta(self, created_at):
        self.meta.write_text(json.dumps({"prompt_hash": hashlib.sha256(self.prompt.read_bytes()).hexdigest(),
                                         "created_at": created_at,
                                         "input_rows": [{"doc_id": "toy", "sent_id": 1, "text": "Alpha."}]}),
                             encoding="utf-8")

    def pin(self):
        return pin_run_identity(self.args, "sync", "toy")

    def test_resume_ignores_preparation_timestamp(self):
        first = self.pin()
        self.write_meta("later")
        self.assertEqual(self.pin(), first)

    def test_changed_prompt_is_rejected(self):
        self.pin()
        self.prompt.write_text("Changed synthetic prompt.", encoding="utf-8")
        self.write_meta("later")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.pin()

    def test_changed_decoding_is_rejected(self):
        for field, changed in (("temperature", 0.5), ("top_p", 0.5), ("seed", 7),
                               ("max_output_tokens", 100), ("base_url", "https://example.invalid/v1"),
                               ("endpoint", "/v1/changed-endpoint")):
            with self.subTest(field=field):
                self.pin()
                old = getattr(self.args, field)
                setattr(self.args, field, changed)
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.pin()
                setattr(self.args, field, old)

    def test_provider_endpoint_change_is_hashed_and_rejected(self):
        url = "https://user:secret@example.invalid/v1"
        with patch.dict("os.environ", {"DEEPSEEK_BASE_URL": url}):
            self.pin()
            receipt = (self.root / "raw/_run_identity.json").read_text()
            self.assertNotIn(url, receipt)
            self.assertNotIn("secret", receipt)
        with patch.dict("os.environ", {"DEEPSEEK_BASE_URL": "https://other.invalid/v1"}):
            with self.assertRaisesRegex(ValueError, "changed"):
                self.pin()

    def test_unbound_raw_is_rejected(self):
        raw = self.root / "raw"
        raw.mkdir()
        (raw / "batch_0001_raw.json").write_text('{"status":"ok"}')
        with self.assertRaisesRegex(ValueError, "no run identity"):
            self.pin()

    def test_async_resume_requires_same_saved_job_and_selection(self):
        self.pin()
        pin_batch_selection(self.args, ["batch_0001"])
        job = self.root / "raw/_batch_job.json"
        job.write_text('{"id":"own-job"}')
        pin_batch_selection(self.args, ["batch_0001"], "own-job")
        with self.assertRaisesRegex(ValueError, "saved job"):
            pin_batch_selection(self.args, ["batch_0001"], "foreign-job")
        with self.assertRaisesRegex(ValueError, "selection changed"):
            pin_batch_selection(self.args, ["batch_0002"])
        (self.root / "raw/_batch_selection.json").unlink()
        with self.assertRaisesRegex(ValueError, "no batch selection"):
            pin_batch_selection(self.args, ["batch_0001"])


if __name__ == "__main__":
    unittest.main()
