"""The speed probe must time decoded frames and never label a CPU run as GPU."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


SERVICE = Path(__file__).resolve().parents[1]
BENCHMARK = SERVICE / "tools" / "benchmark_gpu.py"


class GpuBenchmarkTests(unittest.TestCase):
    def test_cpu_smoke_requires_opt_in_and_measures_full_extract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (32, 32), (70, 100, 140)).save(images / "frame.jpg")
            csv_path = root / "query.csv"
            csv_path.write_text("image_id,x,y,w,h\nframe,0,0,32,32\n")
            output = root / "benchmark.json"
            command = [sys.executable, str(BENCHMARK),
                       "--images-dir", str(images), "--csv", str(csv_path),
                       "--out", str(output), "--warmup", "1", "--latency-runs", "3",
                       "--throughput-seconds", "0.02"]
            env = {**os.environ, "LCT_DEVICE": "cpu"}

            rejected = subprocess.run(command, cwd=SERVICE, env=env,
                                      capture_output=True, text=True, timeout=60)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("--allow-cpu", rejected.stderr)
            self.assertFalse(output.exists())

            accepted = subprocess.run([*command, "--allow-cpu"], cwd=SERVICE, env=env,
                                      capture_output=True, text=True, timeout=60)
            self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["inference_backend"]["active_device"], "cpu")
            self.assertFalse(result["official_counts_and_gpu"])
            self.assertEqual(result["scope"], "disk_jpeg_bbox_preprocess_two_onnx_whitening_l2")
            self.assertEqual(len(result["latency_b1"]["samples_ms"]), 3)
            self.assertGreater(result["latency_b1"]["median_ms"], 0)
            self.assertEqual(set(result["throughput_fps_by_batch"]), {"1", "8", "16", "32"})
            self.assertTrue(all(value > 0 for value in result["throughput_fps_by_batch"].values()))


if __name__ == "__main__":
    unittest.main()
