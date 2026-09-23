"""HTTP-API сервиса инференса (FastAPI).

Методы — минимум, нужный пользователю и пакетному сценарию:
  GET  /api/health         — состояние сервиса и хранилища;
  GET  /api/version        — версия модели/сервиса и параметры;
  POST /api/embed          — изображение + bbox -> вектор признака;
  POST /api/search         — изображение + bbox -> кандидаты или честный отказ;
  POST /api/search/vector  — то же по готовому вектору.

Методы тонкого клиента (добавлены к прежним, прежние ответы не менялись):
  GET  /api/ui/state              — что клиенту показывать: хранилище, порог,
                                    доступность кадров и объяснений;
  GET  /api/gallery/{id}/crop     — кроп кандидата или кадр целиком с рамкой;
  POST /api/explain               — карты вклада областей в оценку близости.

Спецификация OpenAPI: /openapi.json, интерактивная документация: /docs.
Swagger UI отдаётся из локальных файлов (app/static/vendor) — документация
открывается и без доступа в интернет.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field

from ..core import config
from ..core import explain as explain_mod
from ..core.model import Embedder, model_file_sha256
from ..core.preprocess import crop_to_input, resolve_image_path
from ..core.ranking import validate_scores
from .store import GalleryStore

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Модель загружается один раз на старте; sha256 всех трёх файлов весов
    # (две ONNX-модели + матрица whitening) проверяется здесь же, чтобы
    # повреждённый файл валил запуск, а не первый запрос.
    state["embedder"] = Embedder()
    state["model_sha256"] = model_file_sha256()
    state["model2_sha256"] = model_file_sha256(config.MODEL2_PATH)
    state["whitening_sha256"] = model_file_sha256(config.WHITENING_PATH)
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
class _Static(StaticFiles):
    """Статика тонкого клиента с обязательной перепроверкой.

    Без Cache-Control браузер кэширует css/js по эвристике, и после обновления
    образа оператор может увидеть старый интерфейс поверх нового API. Ответы
    маленькие, перепроверка стоит один 304.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", _Static(directory=STATIC_DIR), name="static")


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


class ExplainResponse(BaseModel):
    """Точное разложение косинуса по позициям карты признаков (13x13)."""

    gallery_id: str
    cos: float = Field(description="близость пары запрос-кандидат")
    grid: list[int] = Field(description="размер карты признаков [H, W]")
    query: list[list[float]] = Field(
        description="вклад позиций карты ЗАПРОСА в cos, в единицах близости")
    gallery: list[list[float]] = Field(description="то же со стороны кандидата")
    bias: dict[str, float] = Field(description="свободный член разложения")
    residual: dict[str, float] = Field(
        description="невязка «сумма вкладов + свободный член − cos»; "
                    "разложение точное, ожидается уровень округления float32")
    ms: float = Field(description="время расчёта пары на сервере")


class UIState(BaseModel):
    """Состояние для тонкого клиента одним запросом."""

    service: str
    model: str
    score_scale: str
    default_threshold: float
    storage_reachable: bool
    gallery_points: int | None
    images_available: bool = Field(
        description="каталог кадров смонтирован — можно показывать кропы")
    explain_available: bool = Field(
        description="разложение близости по областям доступно")


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
    try:
        validate_scores([], threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store: GalleryStore = state["store"]
    if not store.reachable():
        raise HTTPException(503, "хранилище галереи (Qdrant) недоступно")
    if not store.count():
        raise HTTPException(409, "галерея не загружена — выполните app.load_gallery")
    hits = store.search(vec, top_k)
    try:
        validate_scores([h["confidence"] for h in hits])
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc
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
    """Версия сервиса, состав конвейера и численные параметры."""
    return {
        "service": config.SERVICE_VERSION,
        "model": config.MODEL_NAME,
        "model_sha256": state["model_sha256"],
        "model2": config.MODEL2_NAME,
        "model2_sha256": state["model2_sha256"],
        "whitening_sha256": state["whitening_sha256"],
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


# ---------- методы тонкого клиента ----------
#
# Добавлены поверх прежних; ни один существующий ответ здесь не меняется.
# Всё, что нужно интерфейсу сверх поиска, живёт в отдельных методах.

def _gallery_frame(gallery_id: str) -> tuple[Image.Image, dict]:
    """Кадр галереи с диска и его рамка из хранилища."""
    store: GalleryStore = state["store"]
    if not store.reachable():
        raise HTTPException(503, "хранилище галереи (Qdrant) недоступно")
    box = store.bbox_of(gallery_id)
    if box is None:
        raise HTTPException(404, f"в галерее нет объекта {gallery_id}")
    if not config.IMAGES_DIR.is_dir():
        raise HTTPException(503, f"каталог кадров {config.IMAGES_DIR} недоступен — "
                                 "смонтируйте его (в compose это DATA_DIR)")
    try:
        path = resolve_image_path(config.IMAGES_DIR, gallery_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return Image.open(path), box


@app.get("/api/ui/state", response_model=UIState, tags=["client"])
def ui_state() -> UIState:
    """Всё, что тонкому клиенту нужно знать до первого запроса.

    Один вызов вместо трёх: клиент должен уметь объяснить пользователю, почему
    поиск сейчас не выполнить (хранилище не поднято, галерея не загружена),
    и знать, доступны ли картинки и объяснения.
    """
    store: GalleryStore = state["store"]
    reachable = store.reachable()
    return UIState(
        service=config.SERVICE_VERSION,
        model=config.MODEL_NAME,
        score_scale="cosine",
        default_threshold=config.DEFAULT_THRESHOLD,
        storage_reachable=reachable,
        gallery_points=store.count() if reachable else None,
        images_available=config.IMAGES_DIR.is_dir(),
        explain_available=config.IMAGES_DIR.is_dir(),
    )


@app.get("/api/gallery/{gallery_id}/crop", tags=["client"],
         response_class=Response,
         responses={200: {"content": {"image/jpeg": {}},
                          "description": "кроп объекта галереи или кадр с рамкой"}})
def gallery_crop(gallery_id: str,
                 view: str = Query(default="crop", pattern="^(crop|frame)$",
                                   description="crop — объект, frame — весь кадр"),
                 size: int = Query(default=320, ge=64, le=1600,
                                   description="длинная сторона результата")) -> Response:
    """Изображение объекта галереи: сам объект или кадр целиком с рамкой.

    Нужен клиенту, чтобы показывать кандидатов картинками, а не строками
    идентификаторов. Рамка берётся из того же payload, по которому шёл поиск.
    """
    frame, box = _gallery_frame(gallery_id)
    with frame as im:
        im = im.convert("RGB")
        if view == "crop":
            out = im.crop((box["x"], box["y"], box["x"] + box["w"],
                           box["y"] + box["h"]))
        else:
            out = im.copy()
            width = max(2, round(min(out.size) / 240))
            ImageDraw.Draw(out).rectangle(
                [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]],
                outline=(227, 73, 72), width=width)
    out.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    # Кроп детерминирован для пары (image_id, рамка), поэтому кэшируется:
    # повторный показ тех же кандидатов не тревожит диск.
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/explain", response_model=ExplainResponse, tags=["client"])
async def explain(file: UploadFile = File(description="кадр запроса JPEG/PNG"),
                  x: int = Form(), y: int = Form(),
                  w: int = Form(), h: int = Form(),
                  gallery_id: str = Form(description="кандидат из /api/search")
                  ) -> ExplainResponse:
    """На что модель смотрела, сопоставляя запрос с этим кандидатом.

    Не приближение и не Grad-CAM: голова сети после глобального пулинга
    аффинна, поэтому косинус раскладывается по позициям карты признаков ТОЧНО,
    за один прогон сети (вывод и проверки — 04-solution/explainability/).
    Возвращаются обе карты и невязка разложения — оператор видит и картинку,
    и меру её достоверности.

    Работает на ПЕРВОЙ модели конвейера (OSNet-AIN, config.MODEL_PATH) — это
    зафиксировано осознанно: разложение опирается на аффинную голову именно
    её графа, а возвращаемый cos — на шкале признаков OSNet, а не на шкале
    поиска (d1_j48: whitening-ансамбль). На выдачу поиска не влияет: отдельный
    метод, отдельная сессия, вызывается только по явному действию оператора.
    """
    q_crop = _crop_from_upload(await file.read(), x, y, w, h)
    frame, box = _gallery_frame(gallery_id)
    with frame as im:
        g_crop = crop_to_input(im, box["x"], box["y"], box["w"], box["h"])
    try:
        res = explain_mod.get().pair(q_crop, g_crop)
    except Exception as e:  # повреждённый или иной файл весов
        raise HTTPException(503, f"разложение недоступно: {e}")
    return ExplainResponse(
        gallery_id=gallery_id,
        cos=round(res["cos"], 6),
        grid=res["grid"],
        query=np.round(res["maps"]["query"], 9).tolist(),
        gallery=np.round(res["maps"]["gallery"], 9).tolist(),
        bias={k: round(v, 9) for k, v in res["bias"].items()},
        residual={k: float(f"{v:.3g}") for k, v in res["residual"].items()},
        ms=res["ms"],
    )


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
