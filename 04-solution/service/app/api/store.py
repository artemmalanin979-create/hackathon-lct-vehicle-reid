"""Хранилище галереи: Qdrant (свободная векторная СУБД, отдельный контейнер).

Хранит L2-нормированные векторы 512 и метаданные (image_id, bbox). Поиск —
точный (exact=True): галерея мала, а ответы обязаны совпадать с пакетным
прогоном; ANN-режим для галерей ~10^6 — отдельная задача масштабирования.
"""
from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from ..core import config
from ..core.preprocess import BBoxRow


class GalleryStore:
    def __init__(self, url: str = config.QDRANT_URL,
                 collection: str = config.QDRANT_COLLECTION):
        self.collection = collection
        # check_compatibility=False: клиент и сервер пиновятся в compose,
        # предупреждение о минорной разнице версий не должно ронять запуск.
        self.client = QdrantClient(url=url, timeout=30, check_compatibility=False)

    def reachable(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def count(self) -> int | None:
        """Число точек в коллекции; None, если коллекции нет или сервер недоступен."""
        try:
            return self.client.count(self.collection, exact=True).count
        except Exception:
            return None

    def recreate(self) -> None:
        """Пересоздать коллекцию под 512-мерные векторы с косинусной метрикой."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=config.EMBEDDING_DIM,
                                           distance=qm.Distance.COSINE),
        )

    def upsert_rows(self, vectors: np.ndarray, rows: list[BBoxRow],
                    batch_size: int = 256) -> None:
        """Загрузить векторы галереи; id точки = номер строки CSV галереи."""
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            self.client.upsert(
                self.collection,
                points=[
                    qm.PointStruct(
                        id=start + k,
                        vector=vectors[start + k].tolist(),
                        payload={"image_id": r.image_id,
                                 "x": r.x, "y": r.y, "w": r.w, "h": r.h},
                    )
                    for k, r in enumerate(chunk)
                ],
                wait=True,
            )

    def search(self, vector: np.ndarray, top_k: int) -> list[dict]:
        """top_k ближайших по косинусу, по убыванию, без порога.

        Порог отказа применяется вызывающим кодом (единая точка истины в API),
        поэтому здесь возвращаются и кандидаты ниже порога — API показывает
        лучшую близость даже при отказе.
        """
        hits = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=top_k,
            with_payload=True,
            search_params=qm.SearchParams(exact=True),
        ).points
        return [
            {"gallery_id": h.payload["image_id"], "confidence": float(h.score)}
            for h in hits
        ]
