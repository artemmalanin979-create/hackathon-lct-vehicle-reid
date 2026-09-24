"""N2: preflight and decoding must agree on literal filenames."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from app.core.preprocess import BBoxRow, CropInputError, load_crop, resolve_image_path
from app.input_checks import require_dataset


class ImageNameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_explicit_filename_is_not_replaced_by_appended_extension(self):
        for name in ("named.png", "named.jpeg", "named.jpg", "named.PNG", "frame.v1.png"):
            with self.subTest(name=name):
                Image.new("RGB", (8, 8), (10, 20, 30)).save(self.root / name)
                for suffix in (".jpg", ".jpeg", ".png"):
                    Image.new("RGB", (8, 8), (240, 200, 160)).save(self.root / (name + suffix))
                self.assertEqual(resolve_image_path(self.root, name), self.root / name)
                decoded = load_crop(self.root, BBoxRow(name, 0, 0, 8, 8))
                np.testing.assert_array_equal(decoded[:, 0, 0], [10, 20, 30])

    def test_missing_explicit_filename_never_falls_back_to_shadow_file(self):
        Image.new("RGB", (8, 8)).save(self.root / "named.png.jpg")
        with self.assertRaises(FileNotFoundError):
            resolve_image_path(self.root, "named.png")
        with self.assertRaises(CropInputError):
            load_crop(self.root, BBoxRow("named.png", 0, 0, 8, 8))
        csv = self.root / "query.csv"
        csv.write_text("image_id,x,y,w,h\nnamed.png,0,0,8,8\n")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
            require_dataset([csv], self.root)
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("named.png", stderr.getvalue())

    def test_corrupt_explicit_file_does_not_fall_back_to_valid_shadow(self):
        (self.root / "named.png").write_bytes(b"broken")
        Image.new("RGB", (8, 8)).save(self.root / "named.png.jpg")
        with self.assertRaises(CropInputError):
            load_crop(self.root, BBoxRow("named.png", 0, 0, 8, 8))

    def test_extensionless_ids_keep_existing_priority_and_fallbacks(self):
        for n, suffix in enumerate((".jpg", ".jpeg", ".png", "")):
            name = f"case{n}"
            for available in (".jpg", ".jpeg", ".png", "")[n:]:
                Image.new("RGB", (8, 8)).save(self.root / (name + available), format="PNG")
            with self.subTest(suffix=suffix):
                self.assertEqual(resolve_image_path(self.root, name), self.root / (name + suffix))
                csv = self.root / f"query{n}.csv"
                csv.write_text(f"image_id,x,y,w,h\n{name},0,0,8,8\n")
                require_dataset([csv], self.root)

    def test_jpeg_only_research_preflight_keeps_its_explicit_suffix_contract(self):
        Image.new("RGB", (8, 8)).save(self.root / "stem.v1.jpg")
        csv = self.root / "query.csv"
        csv.write_text("image_id,x,y,w,h\nstem.v1,0,0,8,8\n")
        require_dataset([csv], self.root, suffixes=(".jpg",))


if __name__ == "__main__":
    unittest.main()
