#!/usr/bin/env python3
"""s06c: одноразовый потоковый рескан всех изображений в миниатюры 96x54 (grayscale)
для проверки, ограничена ли полнота суррогата камеры разрешением 32x32.
Память: потоково, по одному файлу; draft-декодирование JPEG на 1/8.
Выход: thumbs96.npy (uint8, N x 5184), порядок как в scan_files.json."""
import json, os
import numpy as np
from PIL import Image

DATA = "/home/artem/projects/hackathon-lct-vehicle-reid/data"
HERE = os.path.dirname(os.path.abspath(__file__))
files = json.load(open(os.path.join(HERE, "scan_files.json")))
out = np.zeros((len(files), 54*96), np.uint8)
for k, fn in enumerate(files):
    with Image.open(os.path.join(DATA, "images", fn)) as im:
        im.draft("L", (240, 135))
        g = im.convert("L").resize((96, 54), Image.BILINEAR)
        out[k] = np.asarray(g, np.uint8).ravel()
    if k % 2000 == 0:
        print(k, flush=True)
np.save(os.path.join(HERE, "thumbs96.npy"), out)
print("done", out.shape)
