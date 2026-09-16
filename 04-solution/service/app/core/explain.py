"""Объяснимость: точное разложение оценки близости по позициям карты признаков.

Метод и его проверки — `04-solution/explainability/`. Здесь он переписан под
сервис: тот же тождественный расчёт, но БЕЗ пакета `onnx` (его нет в семи
зависимостях сервиса и в офлайновом наборе wheels). Всё, что нужно от файла
весов, читается минимальным разбором protobuf — стандартной библиотекой.

Суть. После глобального пулинга голова сети чисто аффинная (два Gemm + BN в
режиме вывода + Concat), а пулинг линеен, поэтому

    output(Q) = A · gap(Q) + c,        gap = (1/HW) Σ_p z_p
    cos(f_Q, f_G) = Σ_p C_p + b,       C_p = (f_G · A) z_p / (HW · ‖output(Q)‖)

Вклад позиции p считается ТОЧНО, за один прогон сети: это тождество, а не
приближение. Проверка сходимости (`Σ C_p + b − cos`) возвращается вместе с
картами — сервис показывает оператору не только картинку, но и невязку.

Матрицы A и c извлекаются из тех же весов, чья sha256 проверяется при запуске;
отдельных производных артефактов решение не хранит.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import MODEL_PATH

# Имена тензоров графа osnet_ain_x1_0_vehicle_reid (OMZ 2022.1).
FEAT = "1027"           # выход Relu после conv5 — вход GlobalAveragePool
BN_OUTPUTS = ("1038", "1040")   # выходы BatchNormalization двух ветвей головы
BRANCHES = (("fc.0.0", "fc.0.1", "1038"), ("fc.1.0", "fc.1.1", "1040"))

# Номера полей protobuf-схемы onnx.proto (onnx/onnx.proto, версия 3).
_MODEL_GRAPH = 7
_GRAPH_NODE, _GRAPH_INIT, _GRAPH_OUTPUT = 1, 5, 12
_NODE_OUTPUT, _NODE_OPTYPE, _NODE_ATTR = 2, 4, 5
_ATTR_NAME, _ATTR_F = 1, 2
_TENSOR_DIMS, _TENSOR_DTYPE, _TENSOR_NAME, _TENSOR_RAW = 1, 2, 8, 9
_DTYPE_FLOAT = 1


# ---------- минимальный разбор protobuf ----------

def _varint(buf: bytes, i: int) -> tuple[int, int]:
    val = shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return val, i


def _fields(buf: bytes, start: int, end: int):
    """(номер поля, начало значения, конец значения, значение varint | None).

    Поддержаны только те типы, что встречаются в onnx.proto: varint, 64-бит,
    длина-значение, 32-бит. Группы (устаревшие) не поддержаны намеренно.
    """
    i = start
    while i < end:
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            val, nxt = _varint(buf, i)
            yield num, i, nxt, val
            i = nxt
        elif wire == 2:
            ln, i = _varint(buf, i)
            yield num, i, i + ln, None
            i += ln
        elif wire == 5:
            yield num, i, i + 4, None
            i += 4
        elif wire == 1:
            yield num, i, i + 8, None
            i += 8
        else:
            raise ValueError(f"неподдерживаемый wire type {wire} в ONNX-файле")


def _graph_span(buf: bytes) -> tuple[int, int]:
    for num, s, e, _ in _fields(buf, 0, len(buf)):
        if num == _MODEL_GRAPH:
            return s, e
    raise ValueError("в файле весов нет GraphProto (поле 7 ModelProto)")


def _initializers(buf: bytes, gs: int, ge: int, want: set[str]) -> dict[str, np.ndarray]:
    """Инициализаторы по именам; читается только float32 с raw_data."""
    out: dict[str, np.ndarray] = {}
    for num, s, e, _ in _fields(buf, gs, ge):
        if num != _GRAPH_INIT:
            continue
        dims: list[int] = []
        dtype = name = None
        raw: tuple[int, int] | None = None
        for f, s2, e2, v in _fields(buf, s, e):
            if f == _TENSOR_DIMS and v is not None:
                dims.append(v)
            elif f == _TENSOR_DTYPE and v is not None:
                dtype = v
            elif f == _TENSOR_NAME:
                name = buf[s2:e2].decode()
            elif f == _TENSOR_RAW:
                raw = (s2, e2)
        if name not in want:
            continue
        if dtype != _DTYPE_FLOAT or raw is None:
            raise ValueError(f"инициализатор {name}: ожидался float32 в raw_data")
        arr = np.frombuffer(buf, dtype="<f4", count=int(np.prod(dims)),
                            offset=raw[0]).reshape(dims)
        out[name] = arr
    missing = want - set(out)
    if missing:
        raise ValueError(f"в файле весов нет инициализаторов {sorted(missing)}")
    return out


def _bn_epsilon(buf: bytes, gs: int, ge: int) -> dict[str, float]:
    """epsilon узлов BatchNormalization по имени их выхода (умолчание — 1e-5)."""
    eps = {o: 1e-5 for o in BN_OUTPUTS}
    for num, s, e, _ in _fields(buf, gs, ge):
        if num != _GRAPH_NODE:
            continue
        outs, optype, attrs = [], None, []
        for f, s2, e2, _v in _fields(buf, s, e):
            if f == _NODE_OUTPUT:
                outs.append(buf[s2:e2].decode())
            elif f == _NODE_OPTYPE:
                optype = buf[s2:e2].decode()
            elif f == _NODE_ATTR:
                attrs.append((s2, e2))
        if optype != "BatchNormalization":
            continue
        for o in outs:
            if o not in eps:
                continue
            for a_s, a_e in attrs:
                a_name, a_val = None, None
                for f, s3, e3, _v in _fields(buf, a_s, a_e):
                    if f == _ATTR_NAME:
                        a_name = buf[s3:e3].decode()
                    elif f == _ATTR_F:
                        a_val = float(np.frombuffer(buf[s3:e3], dtype="<f4")[0])
                if a_name == "epsilon" and a_val is not None:
                    eps[o] = a_val
    return eps


def _with_feature_output(buf: bytes, name: str) -> bytes:
    """Копия графа с дополнительным выходом `name` (файл на диске не трогаем).

    Добавляется ValueInfoProto{name, type.tensor_type.elem_type=FLOAT} без
    указания формы — ровно то, что делает onnx.helper.make_tensor_value_info
    с shape=None; onnxruntime выводит форму сам.
    """
    def _tag(field: int) -> bytes:
        return bytes([field << 3 | 2])

    def _len(n: int) -> bytes:
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)

    def _msg(field: int, payload: bytes) -> bytes:
        return _tag(field) + _len(len(payload)) + payload

    tensor_type = b"\x08" + bytes([_DTYPE_FLOAT])           # elem_type = FLOAT
    type_proto = _msg(1, tensor_type)                       # TypeProto.tensor_type
    value_info = _msg(1, name.encode()) + _msg(2, type_proto)
    extra = _msg(_GRAPH_OUTPUT, value_info)

    gs, ge = _graph_span(buf)
    head = buf[: gs - len(_len(ge - gs))]                    # всё до длины graph
    return head + _len(ge - gs + len(extra)) + buf[gs:ge] + extra + buf[ge:]


# ---------- расчёт ----------

class Explainer:
    """Сессия с дополнительным выходом карты признаков + аффинная голова.

    Создаётся лениво, при первом обращении к объяснению: обычный поиск не должен
    платить ни памятью, ни временем старта за необязательную возможность.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        buf = model_path.read_bytes()
        gs, ge = _graph_span(buf)
        want = {f"{gemm}.{s}" for gemm, _, _ in BRANCHES for s in ("weight", "bias")}
        want |= {f"{bn}.{s}" for _, bn, _ in BRANCHES
                 for s in ("weight", "bias", "running_mean", "running_var")}
        init = _initializers(buf, gs, ge, want)
        eps = _bn_epsilon(buf, gs, ge)

        # A (512 x C) и c (512,): BN в режиме вывода — аффинное преобразование,
        # поэтому обе ветви складываются в одну матрицу и один свободный член.
        parts_a, parts_c = [], []
        for gemm, bn, bn_out in BRANCHES:
            w = init[f"{gemm}.weight"].astype(np.float64)
            b = init[f"{gemm}.bias"].astype(np.float64)
            gamma = init[f"{bn}.weight"].astype(np.float64)
            beta = init[f"{bn}.bias"].astype(np.float64)
            mean = init[f"{bn}.running_mean"].astype(np.float64)
            var = init[f"{bn}.running_var"].astype(np.float64)
            s = gamma / np.sqrt(var + eps[bn_out])
            parts_a.append(s[:, None] * w)
            parts_c.append(s * b + beta - s * mean)
        self.a = np.vstack(parts_a)
        self.c = np.concatenate(parts_c)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            _with_feature_output(buf, FEAT), sess_options=opts,
            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def pair(self, q: np.ndarray, g: np.ndarray) -> dict:
        """Два кропа C,H,W float32 -> карты вкладов обеих сторон и сверки.

        cq[p] — вклад позиции p карты признаков ЗАПРОСА в cos(Q, G);
        cg[p] — то же со стороны КАНДИДАТА. Карты разные: каждая отвечает на
        вопрос «чем это изображение объясняет совпадение с тем».
        """
        t0 = time.perf_counter()
        res = self.session.run(None, {self.input_name: np.stack([q, g])})
        got = dict(zip(self.output_names, res))
        out = got["output"].astype(np.float64)
        feat = got[FEAT]
        nrm = np.linalg.norm(out, axis=1)
        f = out / nrm[:, None]
        cos = float(f[0] @ f[1])

        maps, biases, checks = {}, {}, {}
        for i, key in ((0, "query"), (1, "gallery")):
            z = feat[i].astype(np.float64)
            ch, hh, ww = z.shape
            # Порядок умножения важен: (f·A)·z — 0,35 МFLOP, (A·z)·f — 44 МFLOP.
            contrib = ((f[1 - i] @ self.a) @ z.reshape(ch, -1)) / (hh * ww * nrm[i])
            maps[key] = contrib.reshape(hh, ww)
            biases[key] = float((self.c @ f[1 - i]) / nrm[i])
            checks[key] = float(maps[key].sum() + biases[key] - cos)
        return {"cos": cos, "grid": list(feat.shape[2:]), "maps": maps,
                "bias": biases, "residual": checks,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}


_lock = threading.Lock()
_instance: Explainer | None = None


def get() -> Explainer:
    """Единственный экземпляр на процесс; безопасно из пула потоков FastAPI."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = Explainer()
        return _instance
