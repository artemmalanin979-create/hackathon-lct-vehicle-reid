"""Точное разложение оценки близости по позициям карты признаков.

Голова сети после глобального пулинга чисто аффинная (два Gemm + BN + Concat),
а пулинг линеен, поэтому

    output(Q) = A · gap(Q) + c,      gap = (1/HW) Σ_p z_p
    cos(f_Q, f_G) = [ (1/HW) Σ_p (A z_p)·f_G + c·f_G ] / ‖output(Q)‖

Вклад позиции p карты признаков: C_p = (A z_p)·f_G / (HW · ‖output(Q)‖).
Сумма всех C_p плюс член свободного слагаемого равна косинусу ТОЧНО.

Ни одного настраиваемого параметра: ни окна, ни шага, ни заливки. Один прогон сети.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

import occl

FEAT = "1027"      # выход Relu после conv5 — вход GlobalAveragePool
GAP = "1036"       # Reshape(GAP) -> (N, C)
WORK = Path(__file__).resolve().parent.parent / "out"
PATCHED = WORK / "model_with_feats.onnx"

_S = None
_AC = None


def _affine_head(g):
    """A (512 x C) и c (512,) из двух ветвей Gemm+BN. BN в выводе — аффинное."""
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    eps = {}
    for n in g.node:
        if n.op_type == "BatchNormalization":
            e = [a.f for a in n.attribute if a.name == "epsilon"]
            eps[n.output[0]] = e[0] if e else 1e-5
    parts_A, parts_c = [], []
    for br, bn_out in (("fc.0.0", "1038"), ("fc.1.0", "1040")):
        W = init[f"{br}.weight"].astype(np.float64)      # (256, C)
        b = init[f"{br}.bias"].astype(np.float64)
        p = br.rsplit(".", 1)[0] + ".1"                   # fc.0.1 / fc.1.1
        gm = init[f"{p}.weight"].astype(np.float64)
        bt = init[f"{p}.bias"].astype(np.float64)
        mu = init[f"{p}.running_mean"].astype(np.float64)
        var = init[f"{p}.running_var"].astype(np.float64)
        s = gm / np.sqrt(var + eps[bn_out])
        parts_A.append(s[:, None] * W)
        parts_c.append(s * b + bt - s * mu)
    return np.vstack(parts_A), np.concatenate(parts_c)


def _prepare():
    """Копия графа с дополнительными выходами (исходный файл не трогаем)."""
    global _S, _AC
    if _S is not None:
        return
    m = onnx.load(str(occl.MODEL))
    if not PATCHED.exists():
        have = {o.name for o in m.graph.output}
        for name in (FEAT, GAP):
            if name not in have:
                m.graph.output.append(onnx.helper.make_tensor_value_info(
                    name, onnx.TensorProto.FLOAT, None))
        PATCHED.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(m, str(PATCHED))
    o = ort.SessionOptions()
    o.log_severity_level = 3
    _S = ort.InferenceSession(str(PATCHED), sess_options=o,
                              providers=["CPUExecutionProvider"])
    _AC = _affine_head(onnx.load(str(occl.MODEL)).graph)


def forward(imgs):
    """-> (output N x 512 сырой, feat N x C x H x W, gap N x C)."""
    _prepare()
    names = [o.name for o in _S.get_outputs()]
    inp = _S.get_inputs()[0].name
    b = np.stack([a.astype(np.float32).transpose(2, 0, 1) for a in imgs])
    res = _S.run(None, {inp: b})
    d = dict(zip(names, res))
    return d["output"], d[FEAT], d[GAP]


def check_head(imgs, verbose=True):
    """Проверка: A·gap + c должно совпасть с выходом ONNX."""
    out, feat, gap = forward(imgs)
    A, c = _AC
    rec = gap.astype(np.float64) @ A.T + c
    err = np.abs(rec - out.astype(np.float64))
    rel = err.max() / max(np.abs(out).max(), 1e-12)
    # и линейность пулинга: gap == среднее по позициям feat
    gerr = np.abs(feat.astype(np.float64).mean(axis=(2, 3)) - gap.astype(np.float64)).max()
    if verbose:
        print(f"голова: max|A·gap+c − output| = {err.max():.3e} "
              f"(относительно {np.abs(out).max():.3f} → {rel:.3e})")
        print(f"пулинг: max|mean(feat) − gap| = {gerr:.3e}")
    return {"max_abs_err": float(err.max()), "rel_err": float(rel),
            "pool_err": float(gerr), "feat_shape": list(feat.shape)}


def pair_contrib(q_img, g_img):
    """Карты точного вклада для обеих сторон пары.

    cq[p] — вклад позиции p карты признаков ЗАПРОСА в cos(Q,G);
    cg[p] — то же для КАНДИДАТА. Суммы + свободные члены равны cos.
    """
    _prepare()
    A, c = _AC
    out, feat, _ = forward([q_img, g_img])
    out = out.astype(np.float64)
    nrm = np.linalg.norm(out, axis=1)
    f = out / nrm[:, None]
    s0 = float(f[0] @ f[1])
    res = {"cos": s0, "grid": list(feat.shape[2:])}
    for i, key, bias_key in ((0, "cq", "bq"), (1, "cg", "bg")):
        z = feat[i].astype(np.float64)                 # C x H x W
        C, H, W = z.shape
        # порядок умножения важен: (f·A)·z — это 0,35 МFLOP, а (A·z)·f — 44 МFLOP
        contrib = ((f[1 - i] @ A) @ z.reshape(C, -1)) / (H * W * nrm[i])  # HW
        res[key] = contrib.reshape(H, W)
        res[bias_key] = float((c @ f[1 - i]) / nrm[i])
        res[key + "_sum_check"] = float(res[key].sum() + res[bias_key] - s0)
    return res
