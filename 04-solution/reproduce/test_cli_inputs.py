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

    def test_directory_is_not_an_input_file(self):
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(stderr):
            require_files([self.root], hint="Expected a file.")
        self.assertEqual(exc.exception.code, 2)
        self.assertIn(str(self.root), stderr.getvalue())

    def test_missing_second_csv_is_reported(self):
        query = self.root / "query.csv"
        query.write_text("image_id\npresent\n")
        (self.root / "present.jpg").touch()
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(stderr):
            require_dataset([query, self.root / "gallery.csv"], self.root)
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("gallery.csv", stderr.getvalue())

    @unittest.skipUnless(os.name == "posix", "POSIX pipes and FIFOs")
    def test_stream_preflight_does_not_consume_or_open_csv(self):
        # A subprocess timeout also detects a second FIFO open after its writer
        # exits. No model or inference substitute is needed to check consumption.
        code = '''
import os, sys, threading
from pathlib import Path
from app.input_checks import require_dataset
root, kind = Path(sys.argv[1]), sys.argv[2]
payload = "image_id,x,y,w,h\\npresent,0,0,1,1\\n"
if kind == "pipe":
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload.encode())
    os.close(write_fd)
    path = Path(f"/dev/fd/{read_fd}")
else:
    path = root / "input.fifo"
    os.mkfifo(path)
    # Preflight must even work before the writer exists.
    require_dataset([path], root)
    def write():
        path.write_text(payload)
    threading.Thread(target=write, daemon=True).start()
require_dataset([path], root)
assert path.read_text() == payload
print("stream preserved")
'''
        for kind in ("pipe", "fifo"):
            with self.subTest(kind=kind):
                p = subprocess.run([sys.executable, "-B", "-S", "-c", code,
                                    str(self.root), kind],
                                   cwd=REPO / "04-solution/service", timeout=10,
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn("stream preserved", p.stdout)

    def assert_early_failure(self, args, cwd, missing, env=None):
        p = subprocess.run([sys.executable, "-B", "-S", *args], cwd=cwd,
                           env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertIn("Входы не готовы", p.stderr)
        self.assertIn(str(missing), p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_each_service_weight_is_required_before_ml_import(self):
        csv = self.root / "input.csv"
        csv.write_text("image_id,x,y,w,h\npresent,0,0,1,1\n")
        (self.root / "present.jpg").touch()
        names = ("MODEL_PATH", "MODEL2_PATH", "WHITENING_PATH")
        for name in names:
            (self.root / name).touch()
        for module in ("app.batch", "app.load_gallery"):
            for missing in names:
                with self.subTest(module=module, missing=missing):
                    paths = {name: str(self.root / name) for name in names}
                    paths[missing] = str(self.root / (missing + ".absent"))
                    args = ["-m", module, "--gallery", str(csv),
                            "--images-dir", str(self.root)]
                    if module == "app.batch":
                        args += ["--query", str(csv), "--out-dir", str(self.root / "out")]
                    self.assert_early_failure(args, REPO / "04-solution/service",
                                              paths[missing], {**os.environ, **paths})
        self.assertFalse((self.root / "out").exists())

    def test_each_tta2_part_is_required_before_ml_import(self):
        scripts = Path("04-solution/postproc/scripts")
        shutil.copytree(REPO / scripts, self.root / scripts,
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "04-solution/service/app", self.root / "04-solution/service/app",
                        ignore=shutil.ignore_patterns("__pycache__"))
        split = self.root / "04-solution/split/files"
        out = self.root / "04-solution/postproc/out"
        split.mkdir(parents=True)
        out.mkdir()
        for part in ("query", "gallery"):
            (split / f"val_{part}.csv").touch()
            for tag in ("208", "TTA", "TTA2"):
                (out / f"val_{part}_{tag}.npy").touch()
        for part in ("query", "gallery"):
            missing = out / f"val_{part}_TTA2.npy"
            missing.unlink()
            self.assert_early_failure(["scripts/s06_final_val.py"],
                                      out.parent, missing)
            missing.touch()

    def test_module_commands_reach_input_diagnostics(self):
        for args in (
            ["-m", "04-solution.baseline.scripts.extract_embeddings", "--csv", "missing.csv",
             "--images-dir", str(self.root), "--model", "missing.onnx", "--out", "out.npy"],
            ["-m", "04-solution.postproc.scripts.s03_extract", "--sets", "val",
             "--variants", "208f,256"],
        ):
            with self.subTest(args=args):
                self.assert_early_failure(args, REPO, "Входы не готовы",
                                          {**os.environ, "REID_DATA_DIR": str(self.root),
                                           "REID_MODEL_PATH": str(self.root / "missing.onnx")})

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
