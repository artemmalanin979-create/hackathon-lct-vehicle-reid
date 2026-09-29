"""The organizer reads a headerless ranking and a separate refusal decision."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import batch
from app.core import model, rerank


class OfficialBatchContractTests(unittest.TestCase):
    def test_ranking_and_candidates_use_their_respective_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = Image.new("RGB", (16, 16), (80, 90, 100))
            ids = ["q0", *[f"g{i}" for i in range(10)]]
            for name in ids:
                image.save(root / f"{name}.jpg")
            query, gallery, out = root / "query.csv", root / "gallery.csv", root / "out"
            query.write_text("image_id,x,y,w,h\nq0,0,0,16,16\n")
            gallery.write_text("image_id,x,y,w,h\n" +
                               "".join(f"g{i},0,0,16,16\n" for i in range(10)))

            vectors = {}
            for index, name in enumerate(ids):
                vector = np.zeros(512, dtype=np.float32)
                vector[0 if name == "q0" else index - 1] = 1.0
                vectors[name] = vector

            class FakeEmbedder:
                inference_backend = {"requested_device": "cpu", "active_device": "cpu",
                                     "providers": {"osnet": ["CPUExecutionProvider"],
                                                   "combined_v1": ["CPUExecutionProvider"]}}

                def __init__(self, threads):
                    pass

                def embed_rows(self, images_dir, rows, batch_size):
                    return np.stack([vectors[row.image_id] for row in rows])

            def fake_rerank(query_vectors, gallery_vectors, k1, k2, lam):
                scores = np.arange(len(gallery_vectors), dtype=np.float64)
                return np.tile(scores, (len(query_vectors), 1))

            argv = ["batch", "--images-dir", str(root), "--query", str(query),
                    "--gallery", str(gallery), "--out-dir", str(out),
                    "--threshold", "0.5"]
            with patch.object(sys, "argv", argv), \
                 patch.object(batch, "require_files"), \
                 patch.object(model, "Embedder", FakeEmbedder), \
                 patch.object(rerank, "rerank_scores_independent", fake_rerank):
                batch.main()

            with (out / "submission.csv").open(newline="") as f:
                ranking = list(csv.reader(f))
            self.assertEqual(len(ranking), 1)
            self.assertEqual(ranking[0][:2], ["q0", "g9"])
            self.assertEqual(len(ranking[0]), 11)

            with (out / "candidates.csv").open(newline="") as f:
                candidates = list(csv.DictReader(f))
            self.assertEqual(candidates, [{"query_id": "q0", "gallery_id": "g0",
                                           "confidence": "1.000000"}])
            info = json.loads((out / "run_info.json").read_text())
            self.assertEqual(info["score_scale"], "cosine")
            self.assertEqual(info["rerank_top_k"], 50)
            self.assertEqual(info["inference_backend"], FakeEmbedder.inference_backend)


if __name__ == "__main__":
    unittest.main()
