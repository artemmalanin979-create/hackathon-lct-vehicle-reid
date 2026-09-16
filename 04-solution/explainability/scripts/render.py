"""Отрисовка карт влияния поверх кропов (PIL, matplotlib в .venv нет).

Палитра расходящаяся: два тона + нейтральная середина, равные шаги по плечам.
  Δ > 0 (закрытие СНИЖАЕТ близость, область держит совпадение) — красный #e34948
  Δ < 0 (закрытие ПОВЫШАЕТ близость, область мешает)           — синий  #2a78d6
  Δ = 0 — прозрачно, видно исходный пиксель (нейтральная середина).
Никаких радуг: величину несёт прозрачность одного тона, знак — тон.
Шкала всегда подписана числами в единицах Δcos — цвет один магнитуду не несёт.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

F = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
FB = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"

POS_HEX = (0xE3, 0x49, 0x48)   # красный, слот 8 палитры
NEG_HEX = (0x2A, 0x78, 0xD6)   # синий,   слот 1 палитры
A_MAX = 0.76                   # верхняя прозрачность плеча

SURFACE = (252, 252, 251)
INK = (11, 11, 11)
INK2 = (82, 81, 78)
MUTED = (137, 135, 129)
RULE = (225, 224, 217)


def font(size, bold=False):
    return ImageFont.truetype(FB if bold else F, size)


def overlay(img_u8: np.ndarray, m: np.ndarray, scale: float) -> Image.Image:
    """img: HxWx3 uint8, m: HxW значения Δcos, scale: |Δ| на полное плечо."""
    a = np.clip(np.abs(m) / max(scale, 1e-9), 0, 1)[..., None] * A_MAX
    col = np.where((m > 0)[..., None], np.array(POS_HEX), np.array(NEG_HEX)).astype(np.float64)
    out = img_u8.astype(np.float64) * (1 - a) + col * a
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))


def colorbar(w: int, h: int, scale: float, label="Δcos при закрытии участка") -> Image.Image:
    """Горизонтальная шкала −scale .. +scale с подписями чисел."""
    pad_l, pad_r = 4, 4
    bar_h = 14
    im = Image.new("RGB", (w, h), SURFACE)
    d = ImageDraw.Draw(im)
    bw = w - pad_l - pad_r
    x = np.linspace(-scale, scale, bw)
    strip = np.zeros((bar_h, bw, 3), np.float64)
    a = np.clip(np.abs(x) / max(scale, 1e-9), 0, 1) * A_MAX
    col = np.where((x > 0)[:, None], np.array(POS_HEX), np.array(NEG_HEX)).astype(np.float64)
    strip[:] = (np.array(SURFACE) * (1 - a[:, None]) + col * a[:, None])[None, :, :]
    im.paste(Image.fromarray(strip.round().clip(0, 255).astype(np.uint8)), (pad_l, 16))
    d.rectangle([pad_l - 1, 15, pad_l + bw, 16 + bar_h], outline=RULE, width=1)
    d.text((pad_l, 1), label, font=font(11), fill=INK2)
    f = font(11)
    for frac, txt, anchor in ((0.0, f"−{scale:.3f}", "la"),
                              (0.5, "0", "ma"),
                              (1.0, f"+{scale:.3f}", "ra")):
        d.text((pad_l + frac * bw, 16 + bar_h + 3), txt, font=f, fill=MUTED, anchor=anchor)
    y = 16 + bar_h + 19
    for col, txt in ((POS_HEX, "красный — закрытие СНИЖАЕТ близость: участок держит совпадение"),
                     (NEG_HEX, "синий — закрытие ПОВЫШАЕТ близость: участок мешает")):
        d.rectangle([pad_l, y + 2, pad_l + 8, y + 10], fill=col)
        d.text((pad_l + 13, y), txt, font=font(10), fill=INK2)
        y += 14
    return im


def tile(img_u8, m=None, scale=None, box=None, size=260, boxcol=None):
    """Одна ячейка: вход сети (208x208) с картой или без, увеличенный до size.

    Фото увеличивается LANCZOS, карта — BILINEAR (она и так гладкая), смешиваются
    уже в размере показа: так фото остаётся резким, а тепло не звенит.
    """
    if m is None:
        im = Image.fromarray(img_u8).resize((size, size), Image.LANCZOS)
    else:
        big = np.asarray(Image.fromarray(img_u8).resize((size, size), Image.LANCZOS))
        mb = np.asarray(Image.fromarray(m.astype(np.float32), mode="F")
                        .resize((size, size), Image.BILINEAR)).astype(np.float64)
        im = overlay(big, mb, scale)
    if box is not None:
        k = size / img_u8.shape[0]
        x, y, w, h = box
        ImageDraw.Draw(im).rectangle(
            [x * k, y * k, (x + w) * k, (y + h) * k],
            outline=boxcol or (255, 255, 255), width=2)
    return im


def sheet(title: str, subtitle: str, cells, scale: float, footer: str,
          size=260, cb_note: str = "", show_cb: bool = True) -> Image.Image:
    """cells: список (PIL.Image, подпись, подподпись); в подподписи допустим \n."""
    n = len(cells)
    PAD, TOP = 10, 52
    nsub = max(len(str(sub).split("\n")) for _, _, sub in cells)
    CAP = 24 + 15 * nsub
    W = PAD + n * (size + PAD)
    H = TOP + size + CAP + (88 if show_cb else 34)
    im = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(im)
    d.text((PAD, 8), title, font=font(16, True), fill=INK)
    d.text((PAD, 30), subtitle, font=font(12), fill=INK2)
    for i, (cell, cap, sub) in enumerate(cells):
        x = PAD + i * (size + PAD)
        im.paste(cell, (x, TOP))
        d.rectangle([x - 1, TOP - 1, x + size, TOP + size], outline=RULE, width=1)
        d.text((x, TOP + size + 5), cap, font=font(12, True), fill=INK)
        for j, line in enumerate(str(sub).split("\n")):
            d.text((x, TOP + size + 21 + 14 * j), line, font=font(11), fill=INK2)
    fx, fy = PAD, TOP + size + CAP + 2
    if show_cb:
        cbw = min(400, W - 2 * PAD)
        im.paste(colorbar(cbw, 80, scale), (PAD, fy))
        fx = PAD + cbw + 18
    if footer:
        d.text((fx, fy + 2), footer, font=font(12), fill=INK)
    if cb_note:
        d.text((fx, fy + 24), cb_note, font=font(11), fill=INK2)
    return im
