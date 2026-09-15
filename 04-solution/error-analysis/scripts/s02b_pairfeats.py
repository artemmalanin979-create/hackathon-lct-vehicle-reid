#!/usr/bin/env python3
"""Парные признаки «запрос ↔ его верные кандидаты» и прокси ливреи.

Всё считается из признаков кадров (s02_features.py) и аннотации, без участия модели.
Для каждого признака берётся минимум по всем кросс-камерным парам запроса, то есть
«самая лёгкая доступная пара» — величина не зависит от того, что выдала модель.

Пишет: out/tmp_geo.npz (dasp, dscale, dlab_mate, twins),
       out/tmp_light.npz (dfL, dcL, dsat, night, mixed),
       out/tmp_livery.npy.
"""
import csv
from pathlib import Path
import numpy as np
from PIL import Image

JOB = Path(__file__).resolve().parent.parent
P = Path("/home/artem/projects/hackathon-lct-vehicle-reid")
IMG = P/"data/images"
NIGHT = 70.0          # порог «ночного» кадра по средней яркости
TWIN_DE, TWIN_ASP = 10.0, 0.2   # «цветовой двойник»: ΔE и допуск по aspect

fq = np.load(JOB/'out/feats_query.npz', allow_pickle=True)
fg = np.load(JOB/'out/feats_gallery.npz', allow_pickle=True)
pb = np.load(JOB/'out/perquery_base.npz', allow_pickle=True)
idx = np.flatnonzero(pb['status'] == 'known')
qv, gv = fq['vehicle_id'], fg['vehicle_id']
qc, gc = fq['camera_id'], fg['camera_id']
qlab = np.stack([fq['labL'], fq['laba'], fq['labb']], 1)
glab = np.stack([fg['labL'], fg['laba'], fg['labb']], 1)
qa, ga, qar, gar = fq['aspect'], fg['aspect'], fq['area_frac'], fg['area_frac']
n = len(qv)
nan = lambda: np.full(n, np.nan)
dasp, dscale, dlab, dfL, dcL, dsat = (nan() for _ in range(6))
twins = np.zeros(n)
mixed = np.zeros(n, bool)

for i in idx:
    m = np.flatnonzero((gv == qv[i]) & (gc != qc[i]))
    dasp[i] = np.abs(np.log(qa[i]/ga[m])).min()
    dscale[i] = np.abs(np.log(qar[i]/gar[m])).min()
    dlab[i] = np.linalg.norm(qlab[i]-glab[m], axis=1).min()
    dfL[i] = np.abs(fq['frameL'][i]-fg['frameL'][m]).min()
    dcL[i] = np.abs(fq['cropL'][i]-fg['cropL'][m]).min()
    dsat[i] = np.abs(fq['sat'][i]-fg['sat'][m]).min()
    mixed[i] = ((fq['frameL'][i] < NIGHT) != (fg['frameL'][m] < NIGHT)).all()
    d = np.linalg.norm(glab-qlab[i], axis=1)
    twins[i] = ((gv != qv[i]) & (d < TWIN_DE) & (np.abs(np.log(ga/qa[i])) < TWIN_ASP)).sum()

np.savez(JOB/'out/tmp_geo.npz', dasp=dasp, dscale=dscale, dlab_mate=dlab, twins=twins)
np.savez(JOB/'out/tmp_light.npz', dfL=dfL, dcL=dcL, dsat=dsat,
         night=fq['frameL'] < NIGHT, mixed=mixed)

rows = list(csv.DictReader(open(P/"04-solution/split/files/val_query.csv", newline="")))
liv = np.zeros(n, bool)
for i in idx:
    r = rows[i]
    x, y, w, h = (int(r[k]) for k in "xywh")
    with Image.open(IMG/f"{r['image_id']}.jpg") as im:
        c = np.asarray(im.convert("RGB").crop((x, y, x+w, y+h)).resize((96, 96), Image.BILINEAR),
                       dtype=np.float32)
    core = c[24:72, 24:72]
    mx, mn = core.max(-1), core.min(-1)
    sat = np.where(mx > 0, (mx-mn)/np.maximum(mx, 1e-6), 0)
    bright = mx/255.
    # ливрея = крупный насыщенный цветной блок И крупное белое поле в одном кузове
    liv[i] = ((sat > 0.35) & (bright > 0.35)).mean() > 0.15 and \
             ((sat < 0.15) & (bright > 0.6)).mean() > 0.15
np.save(JOB/'out/tmp_livery.npy', liv)
print("готово:", int(liv[idx].sum()), "ливрейных запросов из", len(idx))
