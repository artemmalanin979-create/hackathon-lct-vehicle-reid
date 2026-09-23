"""CLI input gates must work without ML packages and accept supported inputs."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "04-solution/service"))
from app.input_checks import require_dataset, require_files


class CliInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_supported_extensions_and_named_file(self):
        images = self.root / "images"
        images.mkdir()
        ids = ["jpg", "jpeg", "png", "literal.png"]
        for name in ("jpg.jpg", "jpeg.jpeg", "png.png", "literal.png"):
            (images / name).touch()
        csv = self.root / "custom.csv"
        csv.write_text("image_id\n" + "\n".join(ids) + "\n")
        require_dataset([csv], images)

    def test_limit_does_not_require_unused_images(self):
        csv = self.root / "custom.csv"
        csv.write_text("image_id\npresent\nabsent\n")
        (self.root / "present.jpg").touch()
        require_dataset([csv], self.root, suffixes=(".jpg",), limit=1)
        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(io.StringIO()):
            require_dataset([csv], self.root, suffixes=(".jpg",))
        self.assertEqual(exc.exception.code, 2)

    def test_missing_files_listed_together(self):
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(stderr):
            require_files([self.root / "a.npy", self.root / "b.ids"], hint="Run extraction.")
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("a.npy", stderr.getvalue())
        self.assertIn("b.ids", stderr.getvalue())
        self.assertIn("Run extraction.", stderr.getvalue())

    def test_documented_commands_fail_before_site_packages(self):
        # A fresh tree has code but no external CSV, images or derived vectors.
        for directory in ("baseline/scripts", "postproc/scripts", "service/app"):
            src = REPO / "04-solution" / directory
            dst = self.root / "04-solution" / directory
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        solution = self.root / "04-solution"
        commands = [
            ("baseline", ["scripts/extract_embeddings.py", "--csv", "missing.csv",
                          "--images-dir", "images", "--model", "missing.onnx", "--out", "out.npy"]),
            ("baseline", ["scripts/run_eval.py"]),
            ("baseline", ["scripts/make_submission.py", "--threshold", "0.34921352213815304"]),
            ("baseline", ["scripts/check_embeddings_order.py"]),
            ("postproc", ["scripts/s01_verify_baseline.py"]),
            ("postproc", ["scripts/s03_extract.py", "--sets", "val", "--variants", "208,208f,256"]),
            ("postproc", ["scripts/s05b_make_tta_vectors.py", "--sets", "val"]),
            ("postproc", ["scripts/s06_final_val.py"]),
            ("service", ["-m", "app.load_gallery", "--gallery", "missing.csv", "--images-dir", "images"]),
        ]
        batch = ["-m", "app.batch", "--query", "q.csv", "--gallery", "g.csv",
                 "--images-dir", "images", "--out-dir", "out"]
        commands += [("service", batch), ("service", batch + ["--no-rerank"])]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("REID_", "PYTHON"))}
        for directory, args in commands:
            with self.subTest(command=args):
                result = subprocess.run([sys.executable, "-B", "-S", *args],
                                        cwd=solution / directory, env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Входы не готовы", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse((solution / directory / "out").exists())

    def test_missing_image_before_ml_import(self):
        service = REPO / "04-solution/service"
        csv = self.root / "custom.csv"
        csv.write_text("image_id,x,y,w,h\nmissing,0,0,10,10\n")
        for module, args in (("app.batch", ["--query", str(csv), "--out-dir", str(self.root / "out")]),
                             ("app.load_gallery", [])):
            with self.subTest(module=module):
                p = subprocess.run([sys.executable, "-B", "-S", "-m", module,
                                    "--gallery", str(csv), "--images-dir", str(self.root), *args],
                                   cwd=service, capture_output=True, text=True)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertIn("image_id=missing", p.stderr)
                self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
