"""Чтение CSV с bbox и подготовка кропа для модели.

Конвейер повторяет проверенный бейзлайн бит-в-бит:
кроп по bbox -> PIL RGB -> resize 208x208 (bilinear) -> float32 0..255 -> NCHW.
Нормализация не нужна: первый узел ONNX-графа — InstanceNormalization.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from .config import INPUT_SIZE


class BBoxRow(NamedTuple):
    """Одна строка тестового CSV: изображение и рамка ТС в пикселях кадра."""

    image_id: str
    x: int
    y: int
    w: int
    h: int


def read_rows(csv_path: Path) -> list[BBoxRow]:
    """Строки CSV в порядке файла. Лишние колонки игнорируются.

    Порядок строк сохраняется во всех выходных файлах — этого требует формат
    embeddings.npy (сначала все query по порядку файла, затем вся галерея).
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "x", "y", "w", "h"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{csv_path}: нет колонок {sorted(missing)}")
        return [
            BBoxRow(r["image_id"], int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            for r in reader
        ]


def resolve_image_path(images_dir: Path, image_id: str) -> Path:
    """Файл кадра по image_id.

    ТЗ допускает JPEG и PNG; выданный набор — .jpg. Перебираем расширения
    детерминированно, чтобы закрытый тест с .png не уронил прогон.
    """
    for suffix in (".jpg", ".jpeg", ".png", ""):
        p = images_dir / f"{image_id}{suffix}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"нет файла изображения для image_id={image_id} в {images_dir}")


def crop_to_input(image: Image.Image, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Кроп PIL-кадра по bbox -> тензор C,H,W float32 (RGB, 0..255)."""
    crop = image.convert("RGB").crop((x, y, x + w, y + h))
    crop = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)


def load_crop(images_dir: Path, row: BBoxRow) -> np.ndarray:
    """Тензор кропа для строки CSV (чтение кадра с диска)."""
    with Image.open(resolve_image_path(images_dir, row.image_id)) as im:
        return crop_to_input(im, row.x, row.y, row.w, row.h)
