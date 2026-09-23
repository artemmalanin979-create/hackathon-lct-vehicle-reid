"""Real ONNX regressions. Run explicitly with the service runtime and weights:

    python -m unittest -v check_cli_runtime

MODEL_PATH may point to the shipped OSNet outside the checkout. All inference
uses the real model; temporary images/cache arrays are small test fixtures.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("MODEL_PATH", REPO / "04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx"))


class RuntimeCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("baseline/scripts", "postproc/scripts", "service/app", "eval"):
            shutil.copytree(REPO / "04-solution" / folder, self.root / "04-solution" / folder,
                            ignore=shutil.ignore_patterns("__pycache__"))
        self.images = self.root / "images"
        self.images.mkdir()
        Image.new("RGB", (32, 32), (45, 130, 170)).save(self.images / "present.jpg")
        self.csv = self.root / "partial.csv"
        self.csv.write_text("image_id,x,y,w,h\npresent,0,0,32,32\nunused,0,0,32,32\n")
        self.env = {**os.environ, "REID_DATA_DIR": str(self.root / "no-data"),
                    "REID_MODEL_PATH": str(MODEL), "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1"}
        for key, name in (("MODEL_PATH", "osnet_ain_x1_0_vehicle_reid.onnx"),
                          ("MODEL2_PATH", "osnet_ain_combined_v1.onnx"),
                          ("WHITENING_PATH", "lw_ens_j48_rho0.5.npz")):
            self.env[key] = os.environ.get(key, str(REPO / "04-solution/service/model" / name))

    def run_cli(self, args, cwd=None):
        p = subprocess.run([sys.executable, "-B", *args], cwd=cwd or self.root,
                           env=self.env, capture_output=True, text=True, timeout=90)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def test_baseline_module_and_direct_match_with_limit(self):
        outputs = []
        for index, (entry, cwd, limit) in enumerate((
            (["-m", "04-solution.baseline.scripts.extract_embeddings"], self.root, "1"),
            ([str(self.root / "04-solution/baseline/scripts/extract_embeddings.py")], self.images, "-1"),
        )):
            out = self.root / f"baseline-{index}.npy"
            self.run_cli([*entry, "--csv", str(self.csv), "--images-dir", str(self.images),
                          "--model", str(MODEL), "--out", str(out), "--limit", limit,
                          "--threads", "1"], cwd)
            self.assertEqual(np.load(out).shape, (1, 512))
            outputs.append(out.read_bytes())
        self.assertEqual(*outputs)

    def test_cached_postproc_module_and_direct_without_images(self):
        split = self.root / "04-solution/split/files"
        out = self.root / "04-solution/postproc/out"
        split.mkdir(parents=True)
        out.mkdir()
        for part in ("query", "gallery"):
            (split / f"val_{part}.csv").write_bytes(self.csv.read_bytes())
            for tag in ("208f", "256"):
                np.save(out / f"val_{part}_{tag}.npy", np.eye(2, 512, dtype=np.float32))
        before = {p.name: p.read_bytes() for p in out.glob("*.npy")}
        for entry, cwd in (
            (["-m", "04-solution.postproc.scripts.s03_extract"], self.root),
            ([str(self.root / "04-solution/postproc/scripts/s03_extract.py")], self.images),
        ):
            with self.subTest(entry=entry):
                p = self.run_cli([*entry, "--sets", "val", "--variants", "208f,256"], cwd)
                self.assertIn("частичный прогон", p.stdout)
                self.assertEqual(before, {p.name: p.read_bytes() for p in out.glob("*.npy")})
        self.assertFalse((self.root / "no-data/images").exists())
        self.assertEqual((out / "val_query.ids").read_text(), "present\nunused\n")

    @unittest.skipUnless(os.name == "posix", "POSIX process substitution and FIFO")
    def test_batch_pipe_and_fifo_match_regular_csv(self):
        self.csv.write_text("image_id,x,y,w,h\npresent,0,0,32,32\n")
        service = self.root / "04-solution/service"
        base = [sys.executable, "-B", "-m", "app.batch", "--images-dir", str(self.images),
                "--gallery", str(self.csv), "--threads", "1", "--no-rerank"]
        # All three shipped artifacts are provided through the runtime env.
        expected = None
        for kind in ("regular", "pipe", "fifo"):
            with self.subTest(kind=kind):
                out = self.root / kind
                args = [*base, "--out-dir", str(out)]
                if kind == "pipe":
                    args = ["bash", "-c", 'exec "$@" --query <(cat "$CSV")', "bash", *args]
                else:
                    query = self.csv
                    if kind == "fifo":
                        query = self.root / "input.fifo"
                        os.mkfifo(query)
                        writer = threading.Thread(target=lambda: query.write_bytes(self.csv.read_bytes()),
                                                  daemon=True)
                        writer.start()
                    args += ["--query", str(query)]
                p = subprocess.run(args, cwd=service, env={**self.env, "CSV": str(self.csv)},
                                   capture_output=True, text=True, timeout=90)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                actual = {name: (out / name).read_bytes() for name in
                          ("embeddings.npy", "submission.csv", "candidates.csv")}
                if expected is None:
                    expected = actual
                self.assertEqual(actual, expected)
                if kind == "fifo":
                    writer.join(timeout=5)
                    self.assertFalse(writer.is_alive())


if __name__ == "__main__":
    unittest.main()
