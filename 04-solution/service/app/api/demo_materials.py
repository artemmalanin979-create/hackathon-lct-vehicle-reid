"""Local demonstration inputs and an explicit catalogue of public materials.

Examples are query frames, not gallery self-matches. The client sends their
original bytes and CSV bbox through the ordinary search API. This module never
stores search results or supplies canned candidates.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..core import config
from ..core.preprocess import resolve_image_path

router = APIRouter()

# The released test query CSV contains these anonymous frames. Labels describe
# the frame, not a known identity or a ground-truth match in the unlabelled test.
EXAMPLES = (
    ("street", "Автомобиль на дороге", "a4f2a13bd2b54360a921c8ef7366e535"),
    ("second-view", "Другой кадр", "5cfbbd42352245fb9ab4e93f0e17452a"),
    ("edge", "Автомобиль у края кадра", "c15e316f43844831a8809cd85924e3b1"),
)
MATERIALS = {
    "solution": ("Документация решения", "solution.pdf", "application/pdf"),
    "presentation": ("Технические слайды", "presentation.pdf", "application/pdf"),
    "manifest": ("Состав и хеши сдаваемого прогона", "manifest.json", "application/json"),
    "submission": ("Пакетное ранжирование", "submission.csv", "text/csv"),
    "candidates": ("Пакетные кандидаты и отказ", "candidates.csv", "text/csv"),
    "embeddings": ("Признаки тестовой выборки", "embeddings.npy", "application/octet-stream"),
}


class DemoBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DemoExample(BaseModel):
    id: str
    label: str
    bbox: DemoBox
    image_url: str


class DemoCatalog(BaseModel):
    examples: list[DemoExample]


class MaterialItem(BaseModel):
    id: str
    label: str
    url: str
    bytes: int
    sha256: str


class MaterialCatalog(BaseModel):
    items: list[MaterialItem]
    deployment_status: str


def _contained_file(path: Path, directory: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
        if resolved.is_relative_to(directory.resolve()) and resolved.is_file():
            return resolved
    except (OSError, RuntimeError):
        pass
    return None


def _examples() -> dict:
    csv_path = Path(os.environ.get("DEMO_CSV_PATH", "/data/test_query.csv"))
    allowed = {image_id for _, _, image_id in EXAMPLES}
    rows: dict[str, list] = {}
    try:
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("image_id") in allowed:
                    rows.setdefault(row["image_id"], []).append(row)
    except (OSError, UnicodeError, csv.Error):
        return {}
    result = {}
    for key, label, image_id in EXAMPLES:
        matches = rows.get(image_id, [])
        if len(matches) != 1:
            continue
        try:
            box = {field: int(matches[0][field]) for field in ("x", "y", "w", "h")}
            if box["x"] < 0 or box["y"] < 0 or box["w"] <= 0 or box["h"] <= 0:
                continue
            path = _contained_file(resolve_image_path(config.IMAGES_DIR, image_id), config.IMAGES_DIR)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if path is not None:
            result[key] = (DemoExample(id=key, label=label, bbox=DemoBox(**box),
                                       image_url=f"/api/demo/{key}/image"), path)
    return result


@router.get("/api/demo", response_model=DemoCatalog, tags=["client"])
def demo_catalog() -> DemoCatalog:
    """Available approved examples, empty when the local dataset is absent."""
    return DemoCatalog(examples=[entry[0] for entry in _examples().values()])


@router.get("/api/demo/{example_id}/image", response_class=FileResponse, tags=["client"])
def demo_image(example_id: str) -> FileResponse:
    entry = _examples().get(example_id)
    if entry is None:
        raise HTTPException(404, "Демонстрационный кадр недоступен")
    return FileResponse(entry[1], headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})


def _material_file(material_id: str) -> Path | None:
    definition = MATERIALS.get(material_id)
    if definition is None:
        return None
    root = Path(os.environ.get("MATERIALS_DIR", "/materials"))
    return _contained_file(root / definition[1], root)


@router.get("/api/materials", response_model=MaterialCatalog, tags=["client"])
def material_catalog() -> MaterialCatalog:
    """List only files actually installed by the reviewed delivery bundle."""
    items = []
    for key, (label, _, _) in MATERIALS.items():
        path = _material_file(key)
        if path is not None:
            items.append(MaterialItem(id=key, label=label, url=f"/materials/{key}",
                                      bytes=path.stat().st_size,
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return MaterialCatalog(items=items, deployment_status=os.environ.get("DEPLOYMENT_STATUS", "DEPLOY PENDING"))


@router.get("/materials/{material_id}", response_class=FileResponse, tags=["client"])
def material_file(material_id: str) -> FileResponse:
    path = _material_file(material_id)
    if path is None:
        raise HTTPException(404, "Материал пока не опубликован")
    _, filename, media_type = MATERIALS[material_id]
    return FileResponse(path, media_type=media_type, filename=filename,
                        content_disposition_type="inline" if media_type == "application/pdf" else "attachment",
                        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"})
