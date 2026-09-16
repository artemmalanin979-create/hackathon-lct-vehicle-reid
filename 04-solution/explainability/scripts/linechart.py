"""Минимальный линейный график на PIL: три ряда, прямые подписи, лёгкая сетка.

Палитра — первые три слота (они проходят проверку по всем парам в обоих режимах):
синий #2a78d6, оранжевый #eb6834, бирюзовый #1baf7a. Одна ось. Сетка приглушена.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from render import font, SURFACE, INK, INK2, MUTED, RULE

SLOTS = [(0x2A, 0x78, 0xD6), (0xEB, 0x68, 0x34), (0x1B, 0xAF, 0x7A)]


def chart(series, xlabels, title, subtitle, ylabel, note="",
          W=780, H=446, pad=(64, 212, 74, 46)):
    """series: список (имя, значения). pad: слева, справа, сверху, снизу."""
    L, Rr, T, Bm = pad
    im = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    d.text((12, 8), title, font=font(16, True), fill=INK)
    d.text((12, 30), subtitle, font=font(12), fill=INK2)
    x0, x1 = L, W - Rr
    y0, y1 = T, H - Bm
    vals = np.concatenate([np.asarray(v, float) for _, v in series])
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-9)
    lo, hi = lo - 0.12 * span, hi + 0.12 * span

    def px(i, n):
        return x0 + (x1 - x0) * i / max(n - 1, 1)

    def py(v):
        return y1 - (y1 - y0) * (v - lo) / (hi - lo)

    # сетка и ось значений
    ticks = np.linspace(lo, hi, 5)
    for t in ticks:
        d.line([x0, py(t), x1, py(t)], fill=RULE, width=1)
        d.text((x0 - 8, py(t)), f"{t:+.3f}", font=font(11), fill=MUTED, anchor="rm")
    if lo < 0 < hi:                       # нулевая линия заметнее прочих
        d.line([x0, py(0), x1, py(0)], fill=(195, 194, 183), width=2)
    d.text((12, T - 21), ylabel, font=font(11), fill=INK2)
    for i, lab in enumerate(xlabels):
        d.text((px(i, len(xlabels)), y1 + 8), lab, font=font(10), fill=MUTED, anchor="ma")

    for s, (name, v) in enumerate(series):
        col = SLOTS[s % len(SLOTS)]
        v = np.asarray(v, float)
        pts = [(px(i, len(v)), py(x)) for i, x in enumerate(v)]
        d.line(pts, fill=col, width=2, joint="curve")
        for p in pts:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=col,
                      outline=SURFACE, width=2)
        d.text((x1 + 12, pts[-1][1]), name, font=font(12, True), fill=col, anchor="lm")
    # легенда (идентичность не только цветом)
    ly = H - 24
    lx = 12
    for s, (name, _) in enumerate(series):
        col = SLOTS[s % len(SLOTS)]
        d.rectangle([lx, ly + 3, lx + 10, ly + 11], fill=col)
        d.text((lx + 15, ly), name, font=font(11), fill=INK2)
        lx += 16 + int(d.textlength(name, font=font(11))) + 20
    if note:
        d.text((W - 12, H - 24), note, font=font(11), fill=MUTED, anchor="ra")
    return im
