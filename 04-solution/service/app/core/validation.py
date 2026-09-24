"""Check the embedding contract without changing or renormalizing vectors."""
import numpy as np


def validate_embeddings(vectors, *, rows=None, dimensions=512, name="embeddings"):
    vectors = np.asarray(vectors)
    if (vectors.ndim != 2 or vectors.shape[1] != dimensions
            or (rows is not None and len(vectors) != rows)):
        raise ValueError(f"{name}: неверная форма {vectors.shape}; "
                         f"ожидается ({rows if rows is not None else 'N'}, {dimensions})")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name}: эмбеддинги содержат NaN или бесконечность; "
                         "инференс не заполнил выход или вернул некорректный результат")
    # float64 accumulation matches the documented check in reproduce/run.py.
    # The tolerance admits float32 rounding, not uninitialized or zero vectors.
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.all(np.abs(norms - 1) < 1e-6):
        raise ValueError(f"{name}: эмбеддинги должны иметь L2-норму 1 (допуск 1e-6)")
