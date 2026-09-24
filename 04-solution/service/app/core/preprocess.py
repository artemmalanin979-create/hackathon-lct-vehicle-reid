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


class CropInputError(ValueError):
    """Ошибка конкретного изображения/bbox, а не ошибка инференса."""


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

    Имя с расширением означает только этот файл, без подстановки. Для ID без
    расширения сохраняется порядок .jpg, .jpeg, .png, затем точное имя.
    """
    from ..input_checks import image_candidates

    for p in image_candidates(images_dir, image_id):
        if p.is_file():
            return p
    raise FileNotFoundError(f"нет файла изображения для image_id={image_id} в {images_dir}")


def validate_bbox(image: Image.Image, x: int, y: int, w: int, h: int) -> None:
    """Рамка должна иметь положительную площадь и пересекаться с кадром.

    Частичный выход сохраняет прежнее дополнение PIL чёрным цветом.
    """
    if w <= 0 or h <= 0:
        raise CropInputError(f"bbox ({x}, {y}, {w}, {h}): ширина и высота должны быть > 0")
    if x >= image.width or y >= image.height or x + w <= 0 or y + h <= 0:
        raise CropInputError(f"bbox ({x}, {y}, {w}, {h}) не пересекает кадр "
                             f"{image.width}x{image.height}")


def crop_problems(images_dir: Path, rows: list[BBoxRow], csv_path: Path) -> list[str]:
    """Декодировать все кадры до инференса; собрать ошибки с местом во входе.

    Используются уже прочитанные строки, поэтому потоковый CSV не открывается
    повторно. В памяти живёт один декодированный кадр, тензоры не накапливаются.
    """
    problems = []
    for line, row in enumerate(rows, start=2):
        path = None
        try:
            path = resolve_image_path(images_dir, row.image_id)
            with Image.open(path) as image:
                image.load()  # open()/verify() недостаточно для усечённого JPEG
                validate_bbox(image, row.x, row.y, row.w, row.h)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            problems.append(f"{csv_path}: строка {line}, image_id={row.image_id}, "
                            f"файл {path or images_dir}: {exc}")
    return problems


def crop_to_input(image: Image.Image, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Кроп PIL-кадра по bbox -> тензор C,H,W float32 (RGB, 0..255)."""
    validate_bbox(image, x, y, w, h)
    crop = image.convert("RGB").crop((x, y, x + w, y + h))
    crop = crop.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)


def load_crop(images_dir: Path, row: BBoxRow) -> np.ndarray:
    """Тензор кропа для строки CSV (чтение кадра с диска)."""
    path = None
    try:
        path = resolve_image_path(images_dir, row.image_id)
        with Image.open(path) as im:
            return crop_to_input(im, row.x, row.y, row.w, row.h)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise CropInputError(f"image_id={row.image_id}, файл {path or images_dir}: {exc}") from exc
