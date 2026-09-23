"""F5: invalid crops must not be encoded as an ordinary black vehicle."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.core.preprocess import BBoxRow, CropInputError, crop_problems, crop_to_input, load_crop
from app.core.config import INPUT_SIZE


class CropTests(unittest.TestCase):
    def test_zero_or_negative_area_is_rejected(self):
        image = Image.new("RGB", (20, 30), (20, 60, 90))
        for box in ((0, 0, 0, 5), (0, 0, 5, 0), (0, 0, 0, 0),
                    (0, 0, -1, 4), (0, 0, 4, -1)):
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, "bbox"):
                crop_to_input(image, *box)

    def test_entirely_outside_on_each_side_is_rejected(self):
        image = Image.new("RGB", (20, 30))
        for box in ((20, 0, 2, 3), (0, 30, 2, 3), (-2, 0, 2, 3), (0, -3, 2, 3)):
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, "bbox"):
                crop_to_input(image, *box)

    def test_valid_edges_partial_overlap_and_extreme_aspects_keep_pil_pixels(self):
        image = Image.fromarray(np.random.default_rng(93).integers(0, 256, (30, 20, 3), dtype=np.uint8))
        for box in ((0, 0, 20, 30), (19, 29, 1, 1), (-3, -4, 5, 6),
                    (18, 28, 5, 6), (0, 0, 20, 1), (0, 0, 1, 30)):
            x, y, w, h = box
            expected = np.asarray(image.convert("RGB").crop((x, y, x+w, y+h))
                                  .resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR),
                                  dtype=np.float32).transpose(2, 0, 1)
            with self.subTest(box=box):
                np.testing.assert_array_equal(crop_to_input(image, *box), expected)


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        pixels = np.random.default_rng(93).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(self.root / "good.jpg")
        (self.root / "broken.jpg").write_bytes(b"not a JPEG")
        (self.root / "truncated.jpg").write_bytes((self.root / "good.jpg").read_bytes()[:-32])
        self.good = BBoxRow("good", 0, 0, 64, 64)

    def test_corrupt_and_truncated_images_are_reported_with_row_and_path(self):
        # Pillow accepts the truncated header; failure happens only on decode.
        with Image.open(self.root / "truncated.jpg") as image:
            self.assertEqual(image.size, (64, 64))
        rows = [self.good, BBoxRow("broken", 0, 0, 64, 64), self.good,
                BBoxRow("truncated", 0, 0, 64, 64)]
        problems = crop_problems(self.root, rows, Path("query.csv"))
        self.assertEqual(len(problems), 2)
        for problem, name, line in zip(problems, ("broken", "truncated"), (3, 5)):
            self.assertIn("query.csv", problem)
            self.assertIn(f"строка {line}", problem)
            self.assertIn(f"image_id={name}", problem)
            self.assertIn(str(self.root / f"{name}.jpg"), problem)

    def test_valid_black_image_is_allowed(self):
        Image.new("RGB", (64, 64)).save(self.root / "black.png")
        row = BBoxRow("black", 0, 0, 64, 64)
        self.assertEqual(crop_problems(self.root, [row], Path("query.csv")), [])
        self.assertFalse(load_crop(self.root, row).any())

    def test_repeated_valid_rows_and_named_extensions_are_not_filtered(self):
        for name in ("explicit.png", "alternate.jpeg"):
            Image.new("RGB", (64, 64), (30, 60, 90)).save(self.root / name)
        rows = [self.good, self.good, BBoxRow("explicit.png", 0, 0, 1, 1),
                BBoxRow("alternate", -1, -1, 5, 5)]
        self.assertEqual(crop_problems(self.root, rows, Path("query.csv")), [])
        for row in rows:
            self.assertEqual(load_crop(self.root, row).shape, (3, INPUT_SIZE, INPUT_SIZE))

    def test_load_crop_preserves_context_for_late_file_damage(self):
        for name in ("broken", "truncated", "missing"):
            with self.subTest(name=name), self.assertRaisesRegex(CropInputError, f"image_id={name}"):
                load_crop(self.root, BBoxRow(name, 0, 0, 64, 64))

    def test_invalid_query_or_gallery_fails_before_model_and_preserves_old_output(self):
        from app import batch
        from app.core import model
        query, gallery, out = self.root / "query.csv", self.root / "gallery.csv", self.root / "out"
        header = "image_id,x,y,w,h\n"
        good = "good,0,0,64,64\n"
        out.mkdir()
        old = out / "old-result.txt"
        old.write_text("previous result\n")
        for role in (query, gallery):
            for bad in ("broken,0,0,64,64\n", "truncated,0,0,64,64\n",
                        "good,0,0,0,64\n", "good,64,0,1,1\n"):
                query.write_text(header + good)
                gallery.write_text(header + good)
                role.write_text(header + good + bad + good)
                args = ["batch", "--images-dir", str(self.root), "--query", str(query),
                        "--gallery", str(gallery), "--out-dir", str(out)]
                stderr = io.StringIO()
                with self.subTest(role=role.name, bad=bad), patch("sys.argv", args), \
                     patch.object(model, "Embedder") as embedder, contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as exc:
                        batch.main()
                    self.assertEqual(exc.exception.code, 2)
                    embedder.assert_not_called()
                self.assertIn(str(role), stderr.getvalue())
                self.assertIn("строка 3", stderr.getvalue())
                self.assertIn("Вычисления не запущены", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertEqual(list(out.iterdir()), [old])
                self.assertEqual(old.read_text(), "previous result\n")

    def test_errors_in_both_csvs_are_collected_in_one_attempt(self):
        from app import batch
        from app.core import model
        query, gallery = self.root / "query.csv", self.root / "gallery.csv"
        query.write_text("image_id,x,y,w,h\nbroken,0,0,1,1\n")
        gallery.write_text("image_id,x,y,w,h\ngood,0,0,0,64\n")
        args = ["batch", "--images-dir", str(self.root), "--query", str(query),
                "--gallery", str(gallery), "--out-dir", str(self.root / "out")]
        stderr = io.StringIO()
        with patch("sys.argv", args), patch.object(model, "Embedder") as embedder, \
             contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
            batch.main()
        self.assertEqual(exc.exception.code, 2)
        embedder.assert_not_called()
        self.assertIn("2 проблем", stderr.getvalue())
        self.assertIn(str(query), stderr.getvalue())
        self.assertIn(str(gallery), stderr.getvalue())
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
