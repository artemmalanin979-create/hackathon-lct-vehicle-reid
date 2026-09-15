#!/usr/bin/env python3
"""Группы ошибок по проверяемым признакам: n, Rank-1, mAP, отклонение от среднего."""
import json
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
fq = np.load(JOB/'out/feats_query.npz', allow_pickle=True)
fg = np.load(JOB/'out/feats_gallery.npz', allow_pickle=True)
pb = np.load(JOB/'out/perquery_base.npz', allow_pickle=True)
pr = np.load(JOB/'out/perquery_rr.npz', allow_pickle=True)
geo = np.load(JOB/'out/tmp_geo.npz'); lig = np.load(JOB/'out/tmp_light.npz')

known = pb['status'] == 'known'; idx = np.flatnonzero(known)
r1, ap, fr, top = pb['rank1'], pb['ap'], pb['first_rank'], pb['top_gidx']
r1r, apr = pr['rank1'], pr['ap']
qv, gv = fq['vehicle_id'], fg['vehicle_id']; qc, gc = fq['camera_id'], fg['camera_id']
N = len(idx); R1, MAP = np.nanmean(r1[idx]), np.nanmean(ap[idx])
ERR = int((r1[idx] == 0).sum())

# срез краем у ближайшей верной пары
mate_edge = np.zeros(len(qv)); mate_dark = np.zeros(len(qv), bool)
for i in idx:
    m = np.flatnonzero((gv == qv[i]) & (gc != qc[i]))
    mate_edge[i] = fg['edges'][m].min()
    mate_dark[i] = (fg['frameL'][m] < 70).any()

q75_asp = np.nanquantile(geo['dasp'][idx], .75)
q75_lab = np.nanquantile(geo['dlab_mate'][idx], .75)
q75_dfL = np.nanquantile(lig['dfL'][idx], .75)
q75_tw = np.nanquantile(geo['twins'][idx], .75)
q25_sat = np.quantile(fq['sat'][idx], .25)
p05_area = np.quantile(fq['area_frac'][idx], .05)

same_cam_top1 = np.zeros(len(qv), bool)
same_cam_top1[idx] = gc[top[idx]] == qc[idx]

groups = {
  "Смена ракурса: Δaspect до ближайшей верной пары в верхнем квартиле (>%.2f)" % q75_asp:
      geo['dasp'] > q75_asp,
  "Фотометрический разрыв: ΔLab кузова до ближайшей пары в верхнем квартиле (>%.1f)" % q75_lab:
      geo['dlab_mate'] > q75_lab,
  "Смена освещения: Δяркости кадра до ближайшей пары в верхнем квартиле (>%.0f)" % q75_dfL:
      lig['dfL'] > q75_dfL,
  "Все верные пары сняты в другое время суток (ночь↔день, порог frameL=70)": lig['mixed'],
  "Запрос срезан краем кадра (bbox касается границы)": fq['edges'] >= 1,
  "Верная пара срезана краем кадра (все доступные пары)": mate_edge >= 1,
  "В галерее ровно одна кросс-камерная пара": pb['num_relevant'] == 1,
  "Ночной запрос (яркость кадра < 70)": fq['frameL'] < 70,
  "Мелкий объект (площадь bbox < %.1f%% кадра, нижние 5%%)" % (p05_area*100):
      fq['area_frac'] < p05_area,
  "Ахроматичный кузов (насыщенность в нижнем квартиле, белый/серый/чёрный)":
      fq['sat'] < q25_sat,
  "Много «цветовых двойников» в галерее (верхний квартиль, ΔE<10 и aspect ±20%)":
      geo['twins'] > q75_tw,
  "Размытый/нерезкий кроп (нижний квартиль по градиенту)":
      fq['sharp'] < np.quantile(fq['sharp'][idx], .25),
}

rows = []
for name, mask in groups.items():
    m = mask[idx].astype(bool)
    n = int(m.sum())
    if n == 0:
        continue
    g_r1, g_ap = float(np.nanmean(r1[idx][m])), float(np.nanmean(ap[idx][m]))
    err_in = int((r1[idx][m] == 0).sum())
    rows.append(dict(group=name, n=n, share_q=n/N, R1=g_r1, mAP=g_ap,
                     dR1=g_r1-R1, dmAP=g_ap-MAP, errors=err_in, share_err=err_in/ERR,
                     lift=(err_in/ERR)/(n/N),
                     R1_rr=float(np.nanmean(r1r[idx][m])), mAP_rr=float(np.nanmean(apr[idx][m]))))
rows.sort(key=lambda r: r['dR1'])

out = dict(overall=dict(n=N, R1=float(R1), mAP=float(MAP), errors=ERR,
                        R1_rr=float(np.nanmean(r1r[idx])), mAP_rr=float(np.nanmean(apr[idx]))),
           groups=rows,
           same_camera_top1=dict(errors_with=int(same_cam_top1[idx].sum()),
                                 share_of_errors=float(same_cam_top1[idx].sum()/ERR)))
(JOB/'out/groups.json').write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")

print(f"общее: n={N} R1={R1:.3f} mAP={MAP:.3f} ошибок={ERR}\n")
print(f"{'группа':78s} {'n':>4s} {'R1':>6s} {'ΔR1':>7s} {'mAP':>6s} {'ΔmAP':>7s} {'%ош':>5s} {'lift':>5s}")
for r in rows:
    print(f"{r['group'][:78]:78s} {r['n']:4d} {r['R1']:6.3f} {r['dR1']:+7.3f} {r['mAP']:6.3f} "
          f"{r['dmAP']:+7.3f} {100*r['share_err']:5.1f} {r['lift']:5.2f}")
