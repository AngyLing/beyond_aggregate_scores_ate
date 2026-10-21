"""Resume/recipe contracts using toy files and a blocked subprocess boundary."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ate_paper.experiments import launcher


class ExperimentResumeTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = (Path(self.folder.name) / "repo").resolve()
        for name in ["src/ate_paper/experiments/example.py", "scripts/run_experiments.py",
                     "src/ate_paper/prepare/__init__.py", "prompts/shared_system_base.txt"]:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# toy fixture\n")
        self.out = self.root / "build/experiments/check"
        self.root_patch = patch.object(launcher, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def item(self, name="one", llm=False):
        run = self.out / "acl" / ("llm" if llm else "supervised") / name
        item = {"dataset": "acl", "group": "llm" if llm else "supervised",
                "run": name, "run_dir": str(run), "prediction": str(run / "predictions.jsonl"),
                "commands": [["must-not-run"]], "requires": []}
        if llm:
            item["prompt_files"] = ["prompts/shared_system_base.txt"]
        return item

    def save_plan(self, item):
        folder = Path(item["run_dir"])
        folder.mkdir(parents=True)
        value = {"registry_sha256": "fixture-registry", "source_sha256": launcher.execution_sources([item]), **item}
        (folder / "plan.json").write_text(json.dumps(value))

    def reject_without_work(self, items, expression):
        with patch.object(launcher, "prepare_inputs") as prepare, patch.object(launcher.subprocess, "run") as child:
            with self.assertRaisesRegex(ValueError, expression):
                launcher.execute(items, self.root / "processed", self.out, "fixture-registry", resume=True)
            prepare.assert_not_called()
            child.assert_not_called()

    def test_existing_run_without_plan_is_not_adopted(self):
        item = self.item()
        Path(item["run_dir"]).mkdir(parents=True)
        self.reject_without_work([item], "no saved plan")

    def test_later_invalid_run_is_checked_before_first_run(self):
        first, second = self.item("first"), self.item("second")
        self.save_plan(second)
        (Path(second["run_dir"]) / "plan.json").write_text("{}")
        self.reject_without_work([first, second], "differ from saved plan")
        self.assertFalse(Path(first["run_dir"]).exists())

    def test_changed_program_prevents_resume(self):
        item = self.item()
        self.save_plan(item)
        (self.root / "src/ate_paper/experiments/example.py").write_text("# changed fixture\n")
        self.reject_without_work([item], "code or prompts differ")

    def test_changed_prompt_prevents_resume(self):
        item = self.item(llm=True)
        self.save_plan(item)
        (self.root / "prompts/shared_system_base.txt").write_text("changed fixture prompt")
        self.reject_without_work([item], "code or prompts differ")

    def test_extra_selected_run_does_not_change_completed_run_identity(self):
        first, second = self.item(), self.item("extra", llm=True)
        self.save_plan(first)
        prediction = Path(first["prediction"])
        prediction.write_text("fixture prediction")
        (prediction.parent / "complete.json").write_text(json.dumps({"prediction_sha256": launcher.sha256(prediction)}))
        self.save_plan(second)
        other = Path(second["prediction"])
        other.write_text("fixture prediction")
        (other.parent / "complete.json").write_text(json.dumps({"prediction_sha256": launcher.sha256(other)}))
        with patch.object(launcher, "prepare_inputs"), patch.object(launcher.subprocess, "run") as child:
            launcher.execute([first, second], self.root / "processed", self.out, "fixture-registry", resume=True)
            child.assert_not_called()

    def test_modified_article_prompt_is_rejected_during_planning(self):
        item = {"dataset_key": "acl", "dataset": "ACL RD-TEC 2.0", "group": "llm",
                "run": "toy", "release_path": "unused-reference",
                "accepted_settings": {"prompt_files": [{"path": "prompts/shared_system_base.txt",
                                                          "sha256": hashlib.sha256(b"original fixture").hexdigest()}]}}
        with self.assertRaisesRegex(ValueError, "Prompt differs from the article recipe"):
            launcher.plan_record(item, self.root / "processed", self.out)


if __name__ == "__main__":
    unittest.main()
