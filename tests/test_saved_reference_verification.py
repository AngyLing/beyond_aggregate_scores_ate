"""Read-only aggregate checks and corrupted numeric-reference regression."""
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ate_paper
ROOT = Path(ate_paper.__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("saved_numeric_verifier", ROOT / "scripts/verify_saved_results.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class SavedReferenceVerification(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.results = self.root / "results"
        shutil.copytree(ROOT / "results", self.results)
        self.enterContext(patch.object(verifier, "ROOT", self.root))
        self.enterContext(patch.object(verifier, "RESULTS", self.results))

    def fingerprints(self):
        return {p.relative_to(self.results).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.results.rglob("*") if p.is_file()}

    def test_default_verification_preserves_all_numeric_references(self):
        before = self.fingerprints()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            verifier.main()
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["sources"]["files"], 49)
        self.assertEqual(self.fingerprints(), before)
        self.assertFalse((self.root / "build").exists())

    def test_corrupted_headline_reference_is_rejected(self):
        path = self.results / "saved_results_verification.json"
        reference = json.loads(path.read_text())
        reference["headlines"]["ACL RD-TEC 2.0"]["main_table_medians"][0] += 0.01
        path.write_text(json.dumps(reference))
        with self.assertRaisesRegex(ValueError, "Headline median"):
            verifier.check_headlines()


if __name__ == "__main__":
    unittest.main()
