#!/usr/bin/env python3
"""Загрузка галереи в хранилище (Qdrant) — отдельный воспроизводимый шаг.

Считает векторы галереи тем же конвейером, что и пакетный прогон, и кладёт их
в коллекцию вместе с метаданными (image_id, bbox). Коллекция пересоздаётся —
повторный запуск даёт то же состояние.

    python -m app.load_gallery --images-dir /data/images --gallery /data/test_gallery.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api.store import GalleryStore
from .core import config
from .core.model import Embedder
from .core.preprocess import read_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--gallery", type=Path, required=True,
                    help="CSV галереи: image_id,x,y,w,h")
    ap.add_argument("--url", default=config.QDRANT_URL, help="адрес Qdrant")
    ap.add_argument("--collection", default=config.QDRANT_COLLECTION)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rows = read_rows(args.gallery)
    embedder = Embedder()
    vectors = embedder.embed_rows(args.images_dir, rows, args.batch)

    store = GalleryStore(url=args.url, collection=args.collection)
    if not store.reachable():
        raise SystemExit(f"Qdrant недоступен по {args.url} — поднимите хранилище "
                         "(docker compose up) или передайте --url")
    store.recreate()
    store.upsert_rows(vectors, rows)

    info = {"collection": args.collection, "points": store.count(), "rows": len(rows)}
    assert info["points"] == len(rows), "число точек в коллекции != числу строк CSV"
    print(json.dumps(info))


if __name__ == "__main__":
    main()
