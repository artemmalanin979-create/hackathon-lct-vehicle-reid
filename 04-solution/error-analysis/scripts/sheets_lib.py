"""Сборка контрольных листов: строки из кропов с подписями (PIL)."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
IMG = P / "data/images"
F = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
FB = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
CW, CH = 230, 170          # ячейка под кроп
PAD, CAP, HEAD = 8, 34, 26  # отступ, подпись под ячейкой, шапка листа


def crop_img(meta, i, ctx=0.0):
    r = meta[i]
    x, y, w, h = r["x"], r["y"], r["w"], r["h"]
    with Image.open(IMG / f"{r['image_id']}.jpg") as im:
        W, H = im.size
        if ctx:
            dx, dy = int(w * ctx), int(h * ctx)
            x, y, w, h = max(0, x - dx), max(0, y - dy), w + 2 * dx, h + 2 * dy
            w, h = min(w, W - x), min(h, H - y)
        return im.convert("RGB").crop((x, y, x + w, y + h))


def fit(im, w=None, h=None, bg=(30, 30, 34)):
    w, h = w or CW, h or CH
    s = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def build_sheet(rows, col_titles, title, out_path, border_by=None, cw=None, ch=None):
    """rows: список списков (PIL.Image, подпись). border_by: цвет рамки по колонке."""
    cw, ch = cw or CW, ch or CH
    ncol = len(col_titles)
    W = PAD + ncol * (cw + PAD)
    H = HEAD + 18 + len(rows) * (ch + CAP + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (245, 245, 247))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 5), title, font=ImageFont.truetype(FB, 15), fill=(10, 10, 10))
    ft = ImageFont.truetype(FB, 12)
    fc = ImageFont.truetype(F, 11)
    for c, t in enumerate(col_titles):
        d.text((PAD + c * (cw + PAD), HEAD), t, font=ft, fill=(60, 60, 70))
    y0 = HEAD + 18
    for r, row in enumerate(rows):
        y = y0 + r * (ch + CAP + PAD)
        for c, cell in enumerate(row):
            x = PAD + c * (cw + PAD)
            if cell is None:
                continue
            im, cap = cell
            sheet.paste(fit(im, cw, ch), (x, y))
            col = (border_by or [(120, 120, 130)] * ncol)[c]
            d.rectangle([x - 1, y - 1, x + cw, y + ch], outline=col, width=2)
            for k, line in enumerate(cap.split("\n")[:2]):
                d.text((x + 2, y + ch + 3 + k * 14), line, font=fc, fill=(20, 20, 25))
    sheet.save(out_path, quality=92)
    return out_path
