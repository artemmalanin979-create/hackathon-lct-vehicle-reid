"""The demo exposes approved query frames and only installed public materials."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app
from app.core import config


class DemoMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.materials = self.root / "materials"
        self.materials.mkdir()
        self.csv = self.root / "query.csv"
        self.image_id = "a4f2a13bd2b54360a921c8ef7366e535"
        self.csv.write_text(f"image_id,x,y,w,h\n{self.image_id},1,2,6,5\n")
        self.frame = self.images / (self.image_id + ".jpg")
        Image.new("RGB", (12, 10), (42, 100, 150)).save(self.frame)
        self.env = patch.dict(os.environ, {"DEMO_CSV_PATH": str(self.csv),
                                         "MATERIALS_DIR": str(self.materials)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.images_patch = patch.object(config, "IMAGES_DIR", self.images)
        self.images_patch.start()
        self.addCleanup(self.images_patch.stop)
        # These routes do not need the inference lifespan or a live database.
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_demo_catalog_returns_real_frame_and_exact_csv_bbox(self):
        response = self.client.get("/api/demo")
        self.assertEqual(response.status_code, 200)
        examples = response.json()["examples"]
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["bbox"], {"x": 1, "y": 2, "w": 6, "h": 5})
        frame = self.client.get(examples[0]["image_url"])
        self.assertEqual(frame.status_code, 200)
        self.assertEqual(frame.content, self.frame.read_bytes())
        self.assertTrue(frame.headers["content-type"].startswith("image/"))

    def test_unlisted_query_and_unknown_image_cannot_be_downloaded(self):
        self.csv.write_text("image_id,x,y,w,h\nunlisted,0,0,1,1\n")
        Image.new("RGB", (4, 4)).save(self.images / "unlisted.jpg")
        response = self.client.get("/api/demo")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["examples"], [])
        self.assertEqual(self.client.get("/api/demo/unlisted/image").status_code, 404)

    def test_missing_data_is_empty_and_symlink_outside_images_is_not_served(self):
        outside = self.root / "outside.jpg"
        self.frame.rename(outside)
        self.frame.symlink_to(outside)
        response = self.client.get("/api/demo")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["examples"], [])
        self.csv.unlink()
        self.assertEqual(self.client.get("/api/demo").json()["examples"], [])

    def test_materials_only_link_installed_allowlisted_files(self):
        (self.materials / "solution.pdf").write_bytes(b"local public PDF")
        (self.materials / "team-data.md").write_text("not public")
        response = self.client.get("/api/materials")
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertEqual([item["id"] for item in items], ["solution"])
        payload = self.client.get(items[0]["url"])
        self.assertEqual(payload.content, b"local public PDF")
        self.assertEqual(items[0]["sha256"], hashlib.sha256(payload.content).hexdigest())
        self.assertEqual(self.client.get("/materials/team-data.md").status_code, 404)
        self.assertEqual(self.client.get("/materials/presentation").status_code, 404)

    def test_material_symlink_cannot_expose_a_file_outside_public_directory(self):
        secret = self.root / "private.pdf"
        secret.write_bytes(b"private")
        (self.materials / "solution.pdf").symlink_to(secret)
        response = self.client.get("/api/materials")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])
        self.assertEqual(self.client.get("/materials/solution").status_code, 404)


if __name__ == "__main__":
    unittest.main()
