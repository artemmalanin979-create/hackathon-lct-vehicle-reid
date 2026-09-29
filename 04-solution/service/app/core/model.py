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
import ctypes
import os
import sys
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


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


def _cuda_driver_status() -> tuple[bool, str]:
    """Check the injected NVIDIA driver before ORT attempts CUDA initialization.

    A CUDA-enabled ORT wheel can advertise its provider without a usable host
    driver. In that state session construction can crash in native code rather
    than raise a Python exception, so get_available_providers() is insufficient.
    """
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        status = driver.cuInit(0)
        if status != 0:
            return False, f"cuInit вернул код {status}"
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        count = ctypes.c_int()
        status = driver.cuDeviceGetCount(ctypes.byref(count))
        if status != 0:
            return False, f"cuDeviceGetCount вернул код {status}"
        if count.value <= 0:
            return False, "доступных GPU нет"
        return True, f"устройств {count.value}"
    except (OSError, AttributeError) as exc:
        return False, str(exc)


def _open_session(model_path: Path, threads: int,
                  providers: list[str]) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # молчать про неиспользуемые инициализаторы
    if threads:
        opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), sess_options=opts,
                                   providers=providers)
    # ORT otherwise retries a failed CUDA run on CPU without telling the caller.
    session.disable_fallback()
    return session


def _cuda_active(session: ort.InferenceSession) -> bool:
    providers = session.get_providers()
    return bool(providers) and providers[0] == CUDA_PROVIDER


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
        requested_device = os.environ.get("LCT_DEVICE", "auto").strip().lower()
        if requested_device not in {"auto", "cuda", "cpu"}:
            raise ValueError("LCT_DEVICE должен быть auto, cuda или cpu")
        available = set(ort.get_available_providers())
        if requested_device == "cuda" and CUDA_PROVIDER not in available:
            raise RuntimeError("CUDAExecutionProvider недоступен: проверьте GPU-образ, "
                               "NVIDIA Container Toolkit и драйвер; для CPU задайте LCT_DEVICE=cpu")

        use_cuda = requested_device != "cpu" and CUDA_PROVIDER in available
        if use_cuda:
            driver_ok, driver_status = _cuda_driver_status()
            if not driver_ok:
                if requested_device == "cuda":
                    raise RuntimeError("CUDA-драйвер или GPU недоступны контейнеру "
                                       f"({driver_status}); проверьте --gpus all")
                print("LCT_DEVICE=auto: CUDA-драйвер или GPU недоступны контейнеру "
                      f"({driver_status}); обе модели запущены на CPU", file=sys.stderr)
                use_cuda = False
        if use_cuda:
            try:
                session = _open_session(model_path, threads, [CUDA_PROVIDER, CPU_PROVIDER])
                session2 = _open_session(model2_path, threads, [CUDA_PROVIDER, CPU_PROVIDER])
                if not (_cuda_active(session) and _cuda_active(session2)):
                    raise RuntimeError("CUDAExecutionProvider не стал первым активным провайдером")
            except Exception as exc:
                if requested_device == "cuda":
                    raise RuntimeError("CUDA-инференс не запустился для обеих моделей; "
                                       "проверьте CUDA/cuDNN и доступ контейнера к GPU") from exc
                print(f"LCT_DEVICE=auto: CUDA недоступна при запуске ONNX; "
                      f"обе модели переключены на CPU ({exc})", file=sys.stderr)
                use_cuda = False
        if not use_cuda:
            session = _open_session(model_path, threads, [CPU_PROVIDER])
            session2 = _open_session(model2_path, threads, [CPU_PROVIDER])
            if (CPU_PROVIDER not in session.get_providers()
                    or CPU_PROVIDER not in session2.get_providers()):
                raise RuntimeError("CPUExecutionProvider не стал активным для обеих моделей")

        self.session = session       # первая модель (OSNet)
        self.session2 = session2     # вторая модель (combined_v1)
        self.inference_backend = {
            "requested_device": requested_device,
            "active_device": "cuda" if use_cuda else "cpu",
            "providers": {
                "osnet": list(session.get_providers()),
                "combined_v1": list(session2.get_providers()),
            },
        }
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
