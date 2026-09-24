"""ONNX-инференс: батчи кропов -> признак d1_j48 (512, L2-нормированный).

Конвейер сдаваемой конфигурации d1_j48 (численно повторяет измерительный контур
04-solution/training/combined/): два сырых вектора от двух моделей с общим
препроцессингом -> L2-нормировка каждого -> среднее -> whitening
y = l2n((x - m) @ P.T) в float32 -> финальная L2-нормировка с накоплением
в float64. Сырые выходы сетей копятся в float32, как в прежней одномодельной
версии. Дельта метрик от float32-хранения P, m — ~1e-6 (SOLUTION.md).

Три файла весов проверяются по SHA-256 при создании Embedder: две ONNX-модели
и матрица whitening.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import (EMBEDDING_DIM, MODEL2_PATH, MODEL2_SHA256, MODEL_PATH,
                     MODEL_SHA256, WHITENING_PATH, WHITENING_SHA256)
from .preprocess import BBoxRow, load_crop
from .validation import validate_embeddings


def model_file_sha256(path: Path = MODEL_PATH) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def l2norm(m: np.ndarray) -> np.ndarray:
    """L2-нормировка строк с накоплением в float64, результат float32."""
    m = m.astype(np.float64)
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def _open_session(model_path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # молчать про неиспользуемые инициализаторы
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def _load_whitening(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """P (512,512) и m (512,) whitening; файл поставляется в float32."""
    with np.load(path) as z:
        P, m = z["P"], z["m"]
    if P.shape != (EMBEDDING_DIM, EMBEDDING_DIM) or m.shape != (EMBEDDING_DIM,):
        raise SystemExit(f" whitening {path}: неверные формы {P.shape} / {m.shape}")
    return np.ascontiguousarray(P), np.ascontiguousarray(m)


class Embedder:
    """Обёртка над двумя ONNX-сессиями и whitening; создаётся один раз,
    потокобезопасна для run()."""

    def __init__(self, model_path: Path = MODEL_PATH, model2_path: Path = MODEL2_PATH,
                 whitening_path: Path = WHITENING_PATH, threads: int = 0,
                 verify_sha256: bool = True):
        if isinstance(threads, bool) or not isinstance(threads, Integral) or threads < 0:
            raise ValueError("threads должен быть целым числом >= 0")
        if verify_sha256:
            # Проверяем ВСЕ три файла: любая подмена/потеря весов или матрицы
            # валит запуск здесь, а не первым запросом.
            for path, expected, kind in (
                (model_path, MODEL_SHA256, "модели 1 (OSNet)"),
                (model2_path, MODEL2_SHA256, "модели 2 (combined_v1)"),
                (whitening_path, WHITENING_SHA256, "матрицы whitening"),
            ):
                digest = model_file_sha256(path)
                if digest != expected:
                    raise SystemExit(
                        f"sha256 {kind} не совпал: {digest} != {expected} ({path}); "
                        "файл повреждён или подменён"
                    )
        self.session = _open_session(model_path, threads)       # первая модель (OSNet)
        self.session2 = _open_session(model2_path, threads)     # вторая модель (combined_v1)
        self.input_name = self.session.get_inputs()[0].name
        self.input_name2 = self.session2.get_inputs()[0].name
        self.P, self.m = _load_whitening(whitening_path)

    def embed_tensors(self, batch: np.ndarray) -> np.ndarray:
        """Батч N,C,H,W float32 -> признак d1_j48 (N x 512, L2-нормированный).

        Две модели на одном препроцессированном батче -> L2-нормировка каждого
        сырого вектора -> среднее -> whitening -> L2-нормировка.
        """
        (raw1,) = self.session.run(None, {self.input_name: batch})
        (raw2,) = self.session2.run(None, {self.input_name2: batch})
        # Как в контуре: l2n каждого сырого вектора -> среднее -> l2n -> whitening.
        # Порядок важен: m обучена на нормированных входах, поэтому нормировка
        # среднего ДО (x - m) @ P.T обязательна, иначе сдвиг по масштабу ~x0.55.
        x = l2norm(l2norm(raw1) + l2norm(raw2))
        w = (x - self.m) @ self.P.T
        return l2norm(w)

    def embed_rows(self, images_dir: Path, rows: list[BBoxRow], batch_size: int = 32) -> np.ndarray:
        """Векторы для строк CSV в порядке строк; выход L2-нормирован, float32.

        В памяти живёт один батч кропов и выходная матрица N x 512 — тестовая
        выборка любого разумного размера не приводит к OOM.
        """
        if (isinstance(batch_size, bool) or not isinstance(batch_size, Integral)
                or batch_size <= 0):
            raise ValueError("batch_size должен быть целым числом > 0")
        # A skipped/unfilled batch must never look like valid model output.
        out = np.full((len(rows), EMBEDDING_DIM), np.nan, dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            tensors = np.stack([load_crop(images_dir, r) for r in chunk])
            vectors = self.embed_tensors(tensors)
            validate_embeddings(vectors, rows=len(chunk))
            out[start : start + len(chunk)] = vectors
        validate_embeddings(out, rows=len(rows))
        return out

    def embed_one(self, tensor: np.ndarray) -> np.ndarray:
        """Один кроп C,H,W -> L2-нормированный вектор (512,) float32."""
        return self.embed_tensors(tensor[None])[0]
