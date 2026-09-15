"""ONNX-инференс: батчи кропов -> L2-нормированные векторы 512.

Численно повторяет бейзлайн: CPUExecutionProvider, сырые выходы копятся в
float32, финальная L2-нормировка — с накоплением в float64.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import EMBEDDING_DIM, MODEL_PATH, MODEL_SHA256
from .preprocess import BBoxRow, load_crop


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


class Embedder:
    """Обёртка над ONNX-сессией; создаётся один раз, потокобезопасна для run()."""

    def __init__(self, model_path: Path = MODEL_PATH, threads: int = 0,
                 verify_sha256: bool = True):
        if verify_sha256:
            digest = model_file_sha256(model_path)
            if digest != MODEL_SHA256:
                raise SystemExit(
                    f"sha256 весов не совпал: {digest} != {MODEL_SHA256} ({model_path}); "
                    "файл повреждён или подменён"
                )
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # молчать про неиспользуемые инициализаторы
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def embed_tensors(self, batch: np.ndarray) -> np.ndarray:
        """Батч N,C,H,W float32 -> сырые (ненормированные) векторы N x 512."""
        (emb,) = self.session.run(None, {self.input_name: batch})
        return emb

    def embed_rows(self, images_dir: Path, rows: list[BBoxRow], batch_size: int = 32) -> np.ndarray:
        """Векторы для строк CSV в порядке строк; выход L2-нормирован, float32.

        В памяти живёт один батч кропов и выходная матрица N x 512 — тестовая
        выборка любого разумного размера не приводит к OOM.
        """
        out = np.empty((len(rows), EMBEDDING_DIM), dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            tensors = np.stack([load_crop(images_dir, r) for r in chunk])
            out[start : start + len(chunk)] = self.embed_tensors(tensors)
        return l2norm(out)

    def embed_one(self, tensor: np.ndarray) -> np.ndarray:
        """Один кроп C,H,W -> L2-нормированный вектор (512,) float32."""
        return l2norm(self.embed_tensors(tensor[None]))[0]
