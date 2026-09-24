"""N1: invalid numbers must fail before I/O, inference or publication."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from app import batch, load_gallery
from app.core import model
from app.core.preprocess import BBoxRow
from app.core.submission import save_embeddings

REPO = Path(__file__).resolve().parents[3]
SERVICE = REPO / "04-solution/service"


class NumericCliTests(unittest.TestCase):
    def invalid_cli(self, module, extra, message, env=None):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            out.mkdir()
            previous = {}
            for name in ("embeddings.npy", "submission.csv", "candidates.csv", "run_info.json"):
                previous[name] = b"previous complete result\n"
                (out / name).write_bytes(previous[name])
            args = ["--images-dir", str(root / "missing-images")]
            if module == "app.batch":
                command = ["-m", module]
                args += ["--query", str(root / "query.csv"), "--gallery", str(root / "gallery.csv"),
                         "--out-dir", str(out)]
            elif module == "app.load_gallery":
                command = ["-m", module]
                args += ["--gallery", str(root / "gallery.csv")]
            else:
                command = [str(REPO / module)]
                args += ["--csv", str(root / "input.csv"), "--out", str(out / "embeddings.npy")]
                if "baseline" in module:
                    args += ["--model", str(root / "missing.onnx")]
                else:
                    args = args[2:]  # fastreid reads images from REID_DATA_DIR
            # -S proves no runtime packages, model or network are needed to reject.
            clean = {k: v for k, v in os.environ.items() if not k.startswith("REID_")}
            proc = subprocess.run([sys.executable, "-B", "-S", *command, *args, *extra],
                                  cwd=SERVICE, env={**clean, **(env or {})},
                                  capture_output=True, text=True, timeout=15)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn(message, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertNotIn("ModuleNotFoundError", proc.stderr)
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, previous)

    def test_batch_zero_is_rejected_before_work(self):
        for mode in ("--rerank", "--no-rerank"):
            with self.subTest(mode=mode):
                self.invalid_cli("app.batch", [mode, "--batch", "0"], "--batch")

    def test_batch_negative_is_rejected_before_work(self):
        for mode in ("--rerank", "--no-rerank"):
            with self.subTest(mode=mode):
                self.invalid_cli("app.batch", [mode, "--batch", "-1"], "--batch")

    def test_loader_batch_is_rejected_before_database_or_model(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                self.invalid_cli("app.load_gallery", ["--batch", value], "--batch")

    def test_negative_threads_are_rejected(self):
        self.invalid_cli("app.batch", ["--threads", "-1"], "--threads")

    def test_loader_wait_must_be_finite_and_nonnegative(self):
        for value in ("-1", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.invalid_cli("app.load_gallery", [f"--wait={value}"], "--wait")

    def test_threshold_overrides_are_checked_in_both_modes(self):
        for mode, variable in (("--rerank", "REID_THRESHOLD_RERANK"),
                               ("--no-rerank", "REID_THRESHOLD")):
            for value in ("nan", "inf", "-inf"):
                with self.subTest(mode=mode, value=value):
                    self.invalid_cli("app.batch", [mode], "threshold", {variable: value})

    def test_rerank_parameters_are_checked_before_inference(self):
        for variable, values in (("REID_RERANK_K1", ("0", "-1")),
                                 ("REID_RERANK_K2", ("0", "-1")),
                                 ("REID_RERANK_LAMBDA", ("nan", "inf", "-inf", "-0.1", "1.1"))):
            for value in values:
                with self.subTest(variable=variable, value=value):
                    self.invalid_cli("app.batch", ["--rerank"], "REID_RERANK", {variable: value})

    def test_research_extractors_reject_invalid_numeric_options(self):
        for script in ("04-solution/baseline/scripts/extract_embeddings.py",
                       "04-solution/postproc/scripts/s07_fastreid_extract.py"):
            options = [("--batch", "0"), ("--batch", "-1"), ("--threads", "-1"), ("--limit", "-1")]
            if "baseline" in script:
                options += [("--mask-bottom", v) for v in ("-0.1", "1.1", "nan", "inf", "-inf")]
            for flag, value in options:
                with self.subTest(script=script, flag=flag, value=value):
                    self.invalid_cli(script, [f"{flag}={value}"], flag)

    def test_valid_sentinels_and_explicit_thresholds_reach_input_checks(self):
        class InputReached(Exception):
            pass
        for threshold in ("-1", "0", "1.1"):
            args = ["batch", "--images-dir", ".", "--query", "q.csv", "--gallery", "g.csv",
                    "--out-dir", "out", "--batch", "32", "--threads", "0",
                    "--no-rerank", "--threshold", threshold]
            with self.subTest(threshold=threshold), patch("sys.argv", args), \
                 patch.object(batch.config, "RERANK_K1", 0), \
                 patch.object(batch, "require_dataset", side_effect=InputReached):
                with self.assertRaises(InputReached):
                    batch.main()
        args = ["loader", "--images-dir", ".", "--gallery", "g.csv", "--batch", "32", "--wait", "0"]
        with patch("sys.argv", args), patch.object(load_gallery, "require_dataset", side_effect=InputReached):
            with self.assertRaises(InputReached):
                load_gallery.main()


class EmbeddingContractTests(unittest.TestCase):
    def test_empty_inputs_cannot_be_published_as_an_inference_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv = root / "empty.csv"
            csv.write_text("image_id,x,y,w,h\n")
            store = Mock()
            args = ["loader", "--images-dir", str(root), "--gallery", str(csv)]
            with patch("sys.argv", args), \
                 patch.dict(sys.modules, {"app.api.store": SimpleNamespace(GalleryStore=store)}), \
                 patch.object(model, "Embedder") as embedder:
                with self.assertRaisesRegex(SystemExit, "пустой gallery"):
                    load_gallery.main()
                embedder.assert_not_called()
                store.assert_not_called()
            fake_model = root / "unusable.onnx"
            fake_model.write_bytes(b"not an ONNX model")
            output = root / "output.npy"
            p = subprocess.run([sys.executable, "-B", str(REPO / "04-solution/baseline/scripts/extract_embeddings.py"),
                                "--images-dir", str(root), "--csv", str(csv), "--model", str(fake_model),
                                "--out", str(output)], capture_output=True, text=True, timeout=15)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("пустой CSV", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(output.exists())

    def test_loader_invalid_vectors_cannot_replace_existing_collection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new("RGB", (4, 4)).save(root / "good.png")
            csv = root / "gallery.csv"
            csv.write_text("image_id,x,y,w,h\ngood,0,0,4,4\n")
            store = Mock()
            store.reachable.return_value = True
            embedder = Mock()
            embedder.embed_rows.return_value = np.zeros((1, 512), dtype=np.float32)
            args = ["loader", "--images-dir", str(root), "--gallery", str(csv), "--wait", "0"]
            with patch("sys.argv", args), \
                 patch.dict(sys.modules, {"app.api.store": SimpleNamespace(GalleryStore=Mock(return_value=store))}), \
                 patch.object(model, "Embedder", return_value=embedder):
                with self.assertRaisesRegex(ValueError, "L2-норму"):
                    load_gallery.main()
            store.recreate.assert_not_called()
            store.upsert_rows.assert_not_called()

    def test_direct_embed_rows_rejects_nonpositive_batch_without_loading_crops(self):
        embedder = model.Embedder.__new__(model.Embedder)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), patch.object(model, "load_crop") as crop, \
                 patch.object(embedder, "embed_tensors") as infer:
                with self.assertRaisesRegex(ValueError, "batch_size"):
                    embedder.embed_rows(Path("missing"), [BBoxRow("x", 0, 0, 1, 1)], size)
                crop.assert_not_called()
                infer.assert_not_called()

    def test_direct_embedder_rejects_negative_threads_before_weights(self):
        with patch.object(model, "model_file_sha256") as digest:
            with self.assertRaisesRegex(ValueError, "threads"):
                model.Embedder(threads=-1)
            digest.assert_not_called()

    def test_batch_32_calls_model_for_every_row_and_preserves_order(self):
        embedder = model.Embedder.__new__(model.Embedder)
        rows = [BBoxRow(str(i), 0, 0, 1, 1) for i in range(33)]
        expected = np.eye(512, dtype=np.float32)[:33]
        def infer(tensors):
            return expected[tensors[:, 0].astype(int)]
        with patch.object(model, "load_crop", side_effect=lambda _, r: np.array([int(r.image_id)])), \
             patch.object(embedder, "embed_tensors", side_effect=infer) as inference:
            actual = embedder.embed_rows(Path("."), rows, 32)
        self.assertEqual([len(c.args[0]) for c in inference.call_args_list], [32, 1])
        np.testing.assert_array_equal(actual, expected)

    def test_skipped_inference_cannot_return_uninitialized_memory(self):
        embedder = model.Embedder.__new__(model.Embedder)
        with patch.object(model, "range", return_value=[], create=True), \
             patch.object(embedder, "embed_tensors") as inference:
            with self.assertRaisesRegex(ValueError, "инференс"):
                embedder.embed_rows(Path("."), [BBoxRow("x", 0, 0, 1, 1)], 32)
            inference.assert_not_called()

    def test_model_output_cannot_broadcast_one_vector_across_rows(self):
        embedder = model.Embedder.__new__(model.Embedder)
        with patch.object(model, "load_crop", return_value=np.zeros((3, 2, 2))), \
             patch.object(embedder, "embed_tensors", return_value=np.eye(512, dtype=np.float32)[:1]):
            with self.assertRaisesRegex(ValueError, "форма"):
                embedder.embed_rows(Path("."), [BBoxRow("x", 0, 0, 1, 1)] * 2, 32)

    def test_writer_rejects_invalid_vectors_without_overwriting(self):
        good = np.eye(512, dtype=np.float32)[:1]
        bad_vectors = [np.full((1, 512), value, dtype=np.float32)
                       for value in (0, 1, 1e35, np.nan, np.inf, -np.inf)]
        bad_vectors += [np.ones((1, 511)), np.ones(512)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "embeddings.npy"
            path.write_bytes(b"previous embeddings")
            for bad in bad_vectors:
                for bad_query in (True, False):
                    with self.subTest(shape=bad.shape, bad_query=bad_query):
                        with self.assertRaises(ValueError):
                            save_embeddings(path, bad if bad_query else good, good if bad_query else bad)
                        self.assertEqual(path.read_bytes(), b"previous embeddings")

    def test_valid_float32_embeddings_are_saved_unchanged(self):
        values = np.random.default_rng(97).normal(size=(4, 512))
        values = (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "embeddings.npy"
            save_embeddings(path, values[:1], values[1:])
            np.testing.assert_array_equal(np.load(path), values)

    def test_batch_invalid_or_missing_vectors_never_publish_any_file(self):
        good = np.eye(512, dtype=np.float32)[:1]
        for bad in (np.zeros((1, 512), dtype=np.float32), good[:0], np.repeat(good, 2, axis=0)):
            for role in ("query", "gallery"):
                for reuse in (False, True):
                    with self.subTest(shape=bad.shape, role=role, reuse=reuse), tempfile.TemporaryDirectory() as td:
                        root = Path(td)
                        Image.new("RGB", (4, 4)).save(root / "good.png")
                        csv = root / "input.csv"
                        csv.write_text("image_id,x,y,w,h\ngood,0,0,4,4\n")
                        out = root / "out"
                        previous = {}
                        if reuse:
                            out.mkdir()
                            for name in ("embeddings.npy", "submission.csv", "candidates.csv", "run_info.json"):
                                previous[name] = b"previous complete output\n"
                                (out / name).write_bytes(previous[name])
                        instance = Mock()
                        instance.embed_rows.side_effect = [bad, good] if role == "query" else [good, bad]
                        args = ["batch", "--images-dir", str(root), "--query", str(csv), "--gallery", str(csv),
                                "--out-dir", str(out)]
                        with patch("sys.argv", args), patch.object(model, "Embedder", return_value=instance):
                            with self.assertRaises(ValueError):
                                batch.main()
                        self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()} if out.exists() else {}, previous)


if __name__ == "__main__":
    unittest.main()
