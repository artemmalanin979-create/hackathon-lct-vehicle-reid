#!/usr/bin/env python3
"""Загрузка галереи в хранилище (Qdrant) — отдельный воспроизводимый шаг.

Считает векторы галереи тем же конвейером, что и пакетный прогон, и кладёт их
в коллекцию вместе с метаданными (image_id, bbox). Коллекция пересоздаётся —
повторный запуск даёт то же состояние.

    python -m app.load_gallery --images-dir /data/images --gallery /data/test_gallery.csv

Хранилище может подниматься дольше загрузчика (в compose loader стартует сразу
после контейнера Qdrant), поэтому готовность ожидается до начала извлечения:
--wait секунд опроса, 0 — не ждать.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .core import config
from .input_checks import require_dataset, require_files
from .numeric_inputs import nonnegative_float, positive_int


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--gallery", type=Path, required=True,
                    help="CSV галереи: image_id,x,y,w,h")
    ap.add_argument("--url", default=config.QDRANT_URL, help="адрес Qdrant")
    ap.add_argument("--collection", default=config.QDRANT_COLLECTION)
    ap.add_argument("--batch", type=positive_int, default=32)
    ap.add_argument("--wait", type=nonnegative_float, default=120.0,
                    help="сколько секунд ждать готовности Qdrant (0 — не ждать)")
    args = ap.parse_args()

    require_dataset([args.gallery], args.images_dir)
    require_files([config.MODEL_PATH, config.MODEL2_PATH, config.WHITENING_PATH],
                  hint="Восстановите веса из Git; см. service/model/fetch_model.sh.")

    from .api.store import GalleryStore
    from .core.model import Embedder
    from .core.preprocess import read_rows
    from .core.validation import validate_embeddings

    rows = read_rows(args.gallery)
    if not rows:
        raise SystemExit("пустой gallery CSV — прогон не имеет смысла")

    # Ждать хранилище ДО извлечения векторов: иначе минута работы модели
    # пропадает из-за ещё не поднявшейся БД.
    store = GalleryStore(url=args.url, collection=args.collection)
    deadline = time.monotonic() + args.wait
    while not store.reachable():
        if time.monotonic() >= deadline:
            raise SystemExit(f"Qdrant недоступен по {args.url} за {args.wait:g} с — "
                             "поднимите хранилище (docker compose up), передайте "
                             "--url или увеличьте --wait")
        time.sleep(1.0)

    embedder = Embedder()
    vectors = embedder.embed_rows(args.images_dir, rows, args.batch)
    validate_embeddings(vectors, rows=len(rows))

    store.recreate()
    store.upsert_rows(vectors, rows)

    info = {"collection": args.collection, "points": store.count(), "rows": len(rows)}
    assert info["points"] == len(rows), "число точек в коллекции != числу строк CSV"
    print(json.dumps(info))


if __name__ == "__main__":
    main()
