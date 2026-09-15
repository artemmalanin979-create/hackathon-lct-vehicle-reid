"""HTTP-API сервиса инференса (FastAPI).

Методы — минимум, нужный пользователю и пакетному сценарию:
  GET  /api/health         — состояние сервиса и хранилища;
  GET  /api/version        — версия модели/сервиса и параметры;
  POST /api/embed          — изображение + bbox -> вектор признака;
  POST /api/search         — изображение + bbox -> кандидаты или честный отказ;
  POST /api/search/vector  — то же по готовому вектору.

Спецификация OpenAPI: /openapi.json, интерактивная документация: /docs.
Swagger UI отдаётся из локальных файлов (app/static/vendor) — документация
открывается и без доступа в интернет.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from ..core import config
from ..core.model import Embedder, model_file_sha256
from ..core.preprocess import crop_to_input
from .store import GalleryStore

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Модель загружается один раз на старте; sha256 весов проверяется здесь же,
    # чтобы повреждённый файл валил запуск, а не первый запрос.
    state["embedder"] = Embedder()
    state["model_sha256"] = model_file_sha256()
    state["store"] = GalleryStore()
    yield
    state.clear()


app = FastAPI(
    title="Vehicle ReID service",
    version=config.SERVICE_VERSION,
    description="Формирование цифрового признака ТС и поиск по галерее "
                "(без использования государственного номера).",
    lifespan=lifespan,
    docs_url=None,  # /docs собирается вручную из локальных файлов Swagger UI
    redoc_url=None,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- схемы ответов (видны в OpenAPI) ----------

class Candidate(BaseModel):
    gallery_id: str = Field(description="image_id объекта галереи")
    confidence: float = Field(description="косинусная близость, [-1, 1]")


class SearchResponse(BaseModel):
    refusal: bool = Field(description="true — уверенного совпадения нет, список пуст")
    threshold: float = Field(description="применённый порог отказа")
    best_confidence: float | None = Field(
        description="близость лучшего кандидата до порога (видна и при отказе)")
    candidates: list[Candidate]


class EmbedResponse(BaseModel):
    dim: int
    embedding: list[float] = Field(description="L2-нормированный вектор признака")


class VectorQuery(BaseModel):
    vector: list[float] = Field(description="вектор признака размерности 512")
    top_k: int = Field(default=10, ge=1, le=100)
    threshold: float | None = Field(
        default=None, description="порог отказа; по умолчанию — порог сервиса")


# ---------- вспомогательные ----------

def _crop_from_upload(data: bytes, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Байты изображения + bbox -> тензор кропа; базовая валидация входа."""
    if w <= 0 or h <= 0:
        raise HTTPException(422, "ширина и высота bbox должны быть положительными")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if x < 0 or y < 0 or x + w > im.width or y + h > im.height:
                raise HTTPException(422, f"bbox выходит за пределы кадра "
                                         f"{im.width}x{im.height}")
            return crop_to_input(im, x, y, w, h)
    except UnidentifiedImageError:
        raise HTTPException(422, "файл не является изображением (JPEG/PNG)")


def _search_by_vector(vec: np.ndarray, top_k: int, threshold: float) -> SearchResponse:
    """Онлайновый поиск — на шкале КОСИНУСА, с косинусным порогом.

    Переранжирование сюда не заводится сознательно: k-reciprocal определено на
    множестве «все запросы прогона + вся галерея» и для одиночного запроса
    посчиталось бы по другой окрестности, чем в пакетном прогоне. Смешивать
    шкалы в одном сервисе нельзя, поэтому API остаётся косинусным, а его порог —
    config.DEFAULT_THRESHOLD (см. README, раздел про две шкалы).
    """
    store: GalleryStore = state["store"]
    if not store.reachable():
        raise HTTPException(503, "хранилище галереи (Qdrant) недоступно")
    if not store.count():
        raise HTTPException(409, "галерея не загружена — выполните app.load_gallery")
    hits = store.search(vec, top_k)
    best = hits[0]["confidence"] if hits else None
    accepted = [Candidate(gallery_id=h["gallery_id"],
                          confidence=round(h["confidence"], 6))
                for h in hits if h["confidence"] >= threshold]
    # Честный отказ: ни один кандидат не прошёл порог -> пустой список, а не топ-1.
    return SearchResponse(refusal=not accepted, threshold=threshold,
                          best_confidence=round(best, 6) if best is not None else None,
                          candidates=accepted)


# ---------- методы API ----------

@app.get("/api/health", tags=["service"])
def health() -> dict:
    """Состояние сервиса: модель в памяти, доступность и наполнение хранилища."""
    store: GalleryStore = state["store"]
    reachable = store.reachable()
    return {
        "status": "ok",
        "model_loaded": "embedder" in state,
        "storage": {
            "url": config.QDRANT_URL,
            "reachable": reachable,
            "gallery_points": store.count() if reachable else None,
        },
    }


@app.get("/api/version", tags=["service"])
def version() -> dict:
    """Версия сервиса, модель и численные параметры конвейера."""
    return {
        "service": config.SERVICE_VERSION,
        "model": config.MODEL_NAME,
        "model_sha256": state["model_sha256"],
        "embedding_dim": config.EMBEDDING_DIM,
        "input_size": config.INPUT_SIZE,
        "score_scale": "cosine",
        "default_threshold": config.DEFAULT_THRESHOLD,
        "batch_rerank": {
            "enabled_by_default": config.RERANK_DEFAULT,
            "params": [config.RERANK_K1, config.RERANK_K2, config.RERANK_LAMBDA],
            "score_scale": "rerank_confidence_1_minus_distance",
            "default_threshold": config.DEFAULT_THRESHOLD_RERANK,
        },
        "onnxruntime": __import__("onnxruntime").__version__,
        "numpy": np.__version__,
    }


@app.post("/api/embed", response_model=EmbedResponse, tags=["inference"])
async def embed(file: UploadFile = File(description="кадр JPEG/PNG"),
                x: int = Form(), y: int = Form(),
                w: int = Form(), h: int = Form()) -> EmbedResponse:
    """Извлечь вектор признака ТС из изображения по bbox (тот же конвейер,
    что и в пакетном прогоне)."""
    crop = _crop_from_upload(await file.read(), x, y, w, h)
    vec = state["embedder"].embed_one(crop)
    return EmbedResponse(dim=len(vec), embedding=[float(v) for v in vec])


@app.post("/api/search", response_model=SearchResponse, tags=["inference"])
async def search(file: UploadFile = File(description="кадр JPEG/PNG"),
                 x: int = Form(), y: int = Form(),
                 w: int = Form(), h: int = Form(),
                 top_k: int = Form(default=10, ge=1, le=100),
                 threshold: float | None = Form(default=None)) -> SearchResponse:
    """Найти в галерее кандидатов на то же ТС; при отсутствии уверенного
    совпадения — пустой ответ (режим отказа)."""
    crop = _crop_from_upload(await file.read(), x, y, w, h)
    vec = state["embedder"].embed_one(crop)
    return _search_by_vector(vec, top_k,
                             config.DEFAULT_THRESHOLD if threshold is None else threshold)


@app.post("/api/search/vector", response_model=SearchResponse, tags=["inference"])
def search_by_vector(q: VectorQuery) -> SearchResponse:
    """Поиск кандидатов по готовому вектору (например, из /api/embed)."""
    if len(q.vector) != config.EMBEDDING_DIM:
        raise HTTPException(422, f"ожидается вектор размерности {config.EMBEDDING_DIM}")
    vec = np.asarray(q.vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise HTTPException(422, "нулевой вектор не допускается")
    return _search_by_vector(vec / norm, q.top_k,
                             config.DEFAULT_THRESHOLD if q.threshold is None else q.threshold)


# ---------- тонкий клиент и офлайн-документация ----------

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/docs", include_in_schema=False)
def docs() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} — документация",
        swagger_js_url="/static/vendor/swagger-ui-bundle.js",
        swagger_css_url="/static/vendor/swagger-ui.css",
        swagger_favicon_url="/static/vendor/favicon.png",
    )
