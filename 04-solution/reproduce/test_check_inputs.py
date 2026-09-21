"""Input gate regressions: missing test data must not block validation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check_inputs import check_inputs, sha256


class InputGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.data = self.root / "repo", self.root / "data"
        self.repo.mkdir()
        (self.data / "images").mkdir(parents=True)
        self.split = self.repo / "val.csv"
        self.split.write_text("image_id,x,y,w,h\nval,0,0,1,1\n")
        self.val = self.data / "images/val.jpg"
        self.val.write_bytes(b"validation fixture")
        def entry(p, base):
            return {"path": str(p.relative_to(base)), "bytes": p.stat().st_size, "sha256": sha256(p)}
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "repo_files": [entry(self.split, self.repo)],
            "data": {"val": [entry(self.val, self.data)],
                     "test": [{"path": "test_query.csv", "bytes": 10, "sha256": "missing"},
                              {"path": "images/test.jpg", "bytes": 10, "sha256": "missing"}]}}))

    def check(self, mode):
        return check_inputs(self.repo, self.data, mode, manifest_path=self.manifest)

    def test_validation_without_test_files(self):
        self.assertTrue(self.check("val")["ok"])
        self.assertEqual(len(self.check("test")["missing"]), 2)
        self.assertEqual(len(self.check("all")["missing"]), 2)

    def test_missing_files_collected_together(self):
        self.split.unlink()
        self.val.unlink()
        result = self.check("all")
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["missing"]), 4)

    def test_same_size_corruption_rejected(self):
        self.val.write_bytes(b"Validation fixture")
        result = self.check("val")
        self.assertFalse(result["ok"])
        self.assertEqual(result["mismatched"][0]["path"], str(self.val))
        self.assertIn("actual_sha256", result["mismatched"][0])

    def test_test_does_not_require_validation(self):
        self.split.unlink()
        result = self.check("test")
        self.assertNotIn(str(self.split), result["missing"])

    def test_runner_fails_before_ml_imports(self):
        # -S removes site-packages. The missing-input diagnostic must still work.
        run = subprocess.run([sys.executable, "-S", str(Path(__file__).with_name("run.py")),
                              "--mode", "val", "--data-dir", str(self.root / "absent"),
                              "--out-dir", str(self.root / "out")], text=True, capture_output=True)
        self.assertEqual(run.returncode, 2, run.stderr)
        self.assertIn("Инференс не запущен", run.stderr)
        self.assertNotIn("Traceback", run.stderr)
        report = json.loads((self.root / "out/inputs-val.json").read_text())
        self.assertEqual(len(report["missing"]), 1860)


if __name__ == "__main__":
    unittest.main()
