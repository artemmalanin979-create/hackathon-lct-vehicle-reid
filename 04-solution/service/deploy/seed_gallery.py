#!/usr/bin/env python3
"""Install the verified released gallery without overwriting any collection.

This deployment path consumes the existing canonical run, not a new evaluator.
For arbitrary organizer inputs use app.load_gallery and app.batch as documented.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from qdrant_client import models as qm

from app.api.store import GalleryStore
from app.core import config
from app.core.preprocess import read_rows
from app.core.validation import validate_embeddings


def checked_file(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def ensure_schema(store):
    """A release collection must preserve cosine score meaning and 512-D rows."""
    vectors = store.client.get_collection(store.collection).config.params.vectors
    if not isinstance(vectors, qm.VectorParams) or vectors.size != config.EMBEDDING_DIM or vectors.distance != qm.Distance.COSINE:
        raise ValueError("Existing gallery schema differs; preserved for inspection")


def main():
    artifacts = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
    data = Path("/data")
    manifest_path = artifacts / "manifest.json"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    expected_collection = "lct_" + digest[:16]
    if config.QDRANT_COLLECTION != expected_collection:
        raise ValueError(f"Use QDRANT_COLLECTION={expected_collection} for this release")
    for name, spec in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact filename")
        checked_file(artifacts / name, spec["sha256"])
    for spec in manifest["input_csvs"]:
        if Path(spec["path"]).name != spec["path"]:
            raise ValueError("Invalid input filename")
        checked_file(data / spec["path"], spec["sha256"])
    for key, actual in (("model_sha256", config.MODEL_SHA256),
                        ("model2_sha256", config.MODEL2_SHA256),
                        ("whitening_sha256", config.WHITENING_SHA256)):
        if manifest["run_info"][key] != actual:
            raise ValueError(f"Model contract mismatch: {key}")
    queries = read_rows(data / "test_query.csv")
    rows = read_rows(data / "test_gallery.csv")
    vectors = np.load(artifacts / "embeddings.npy", allow_pickle=False)
    validate_embeddings(vectors, rows=len(queries) + len(rows))
    if list(vectors.shape) != manifest["run_info"]["embeddings_shape"]:
        raise ValueError("Embedding shape does not match manifest")
    gallery = vectors[len(queries):]
    store = GalleryStore()
    if not store.reachable():
        raise RuntimeError("Qdrant not ready")
    created = not store.client.collection_exists(store.collection)
    if created:
        # Atomic create; deliberately no delete/recreate path, even on a race.
        store.client.create_collection(store.collection, vectors_config=qm.VectorParams(
            size=config.EMBEDDING_DIM, distance=qm.Distance.COSINE))
        store.upsert_rows(gallery, rows)
    ensure_schema(store)
    if store.count() != len(rows):
        raise ValueError("Existing gallery count differs; preserved for inspection")
    for start in range(0, len(rows), 128):
        points = store.client.retrieve(store.collection,
            ids=list(range(start, min(start + 128, len(rows)))),
            with_payload=True, with_vectors=True)
        by_id = {point.id: point for point in points}
        for i in range(start, min(start + 128, len(rows))):
            row = rows[i]
            payload = {"image_id": row.image_id, "x": row.x, "y": row.y, "w": row.w, "h": row.h}
            point = by_id.get(i)
            if point is None or point.payload != payload or not np.allclose(
                    point.vector, gallery[i], rtol=1e-5, atol=1e-6):
                raise ValueError(f"Existing gallery row {i} differs; preserved for inspection")
    print(json.dumps({"status": "PASS", "created": created, "points": len(rows),
                      "collection": store.collection, "manifest_sha256": digest}))


if __name__ == "__main__":
    main()
