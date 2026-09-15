#!/usr/bin/env python3
"""Листы: однотипные машины (самые уверенные ошибки) и успешные трудные случаи."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sheets_lib import build_sheet, crop_img, P

JOB = Path(__file__).resolve().parent.parent
SPLIT = P / "04-solution/split/files"


def read_meta(p):
    rows = list(csv.DictReader(open(p, newline="")))
    for r in rows:
        for k in ("x", "y", "w", "h", "vehicle_id", "camera_id"):
            r[k] = int(r[k])
    return rows


qm, gm = read_meta(SPLIT/"val_query.csv"), read_meta(SPLIT/"val_gallery.csv")
pb = np.load(JOB/'out/perquery_base.npz', allow_pickle=True)
pr = np.load(JOB/'out/perquery_rr.npz', allow_pickle=True)
fq = np.load(JOB/'out/feats_query.npz', allow_pickle=True)
fg = np.load(JOB/'out/feats_gallery.npz', allow_pickle=True)
geo = np.load(JOB/'out/tmp_geo.npz'); lig = np.load(JOB/'out/tmp_light.npz')
S = pb['score_matrix']; known = pb['status'] == 'known'; idx = np.flatnonzero(known)
r1, fr, top = pb['rank1'], pb['first_rank'], pb['top_gidx']
qv, gv = fq['vehicle_id'], fg['vehicle_id']; qc, gc = fq['camera_id'], fg['camera_id']
err = idx[r1[idx] == 0]; ok = idx[r1[idx] == 1]
BLUE, GREEN, RED = (40, 90, 200), (20, 140, 60), (200, 40, 40)
manifest = []


def uniq_by_vid(order, k=8):
    seen, out = set(), []
    for i in order:
        if int(qv[i]) in seen:
            continue
        seen.add(int(qv[i])); out.append(int(i))
        if len(out) == k:
            break
    return np.array(out)


def mate_of(i):
    m = np.flatnonzero((gv == qv[i]) & (gc != qc[i]))
    return int(m[np.argmax(S[i, m])])


# --- 1. самые уверенные ошибки: кандидат-двойник ---
cw = np.array([S[i, top[i]] for i in err])
order = err[np.argsort(-cw)][:16]
rows = []
for n, i in enumerate(order, 1):
    j, k = mate_of(i), int(top[i])
    rows.append([(crop_img(qm, i), f"#{n} запрос v{qv[i]} cam{qc[i]}"),
                 (crop_img(gm, k), f"top-1 ЧУЖАЯ: cos {S[i,k]:.3f}\nv{gv[k]} cam{gc[k]}"),
                 (crop_img(gm, j), f"своя, ранг {int(fr[i])}: cos {S[i,j]:.3f}\ncam{gc[j]}")])
for s in (0, 8):
    p = JOB/f"sheets/twins_{s//8+1:02d}.jpg"
    build_sheet(rows[s:s+8], ["ЗАПРОС", "ВЫДАНО ПЕРВЫМ (другая машина)", "ВЕРНЫЙ ОТВЕТ И ЕГО РАНГ"],
                f"Самые уверенные ошибки: cos к чужой выше, чем к своей (лист {s//8+1}/2)",
                p, border_by=[BLUE, RED, GREEN])
    manifest.append(dict(file=p.name, what="16 ошибок с наибольшим cos у ошибочного top-1",
                         cases=[int(x) for x in order[s:s+8]]))

# --- 2. крупный план пары «неразличимы без примет» ---
big = order[:4]
rows = []
for i in big:
    k = int(top[i])
    rows.append([(crop_img(qm, i, ctx=0.05), f"запрос v{qv[i]} cam{qc[i]}"),
                 (crop_img(gm, k, ctx=0.05), f"top-1: v{gv[k]} cam{gc[k]} cos {S[i,k]:.3f}")])
p = JOB/"sheets/twins_zoom_01.jpg"
build_sheet(rows, ["ЗАПРОС (крупно)", "ЧУЖАЯ МАШИНА, ВЫДАННАЯ ПЕРВОЙ (крупно)"],
            "Крупный план четырёх самых уверенных ошибок: чем эти машины различаются", p,
            border_by=[BLUE, RED], cw=470, ch=340)
manifest.append(dict(file=p.name, what="крупный план 4 самых уверенных ошибок", cases=[int(x) for x in big]))

# --- 3. успехи при сильной смене ракурса ---
def top2(i):
    rk = pb['top50'][i]
    for g in rk:
        if g >= 0 and gv[g] != qv[i]:
            return int(g)
    return -1


cand = uniq_by_vid(ok[np.argsort(-geo['dasp'][ok])])
rows = []
for i in cand:
    j, k = mate_of(i), top2(i)
    rows.append([(crop_img(qm, i), f"запрос v{qv[i]} cam{qc[i]}\nΔaspect={geo['dasp'][i]:.2f}"),
                 (crop_img(gm, j), f"НАЙДЕН 1-м: cos {S[i,j]:.3f}\ncam{gc[j]}"),
                 (crop_img(gm, k), f"ближайшая чужая: cos {S[i,k]:.3f}\nv{gv[k]}")])
p = JOB/"sheets/success_viewpoint_01.jpg"
build_sheet(rows, ["ЗАПРОС", "ВЕРНО НАЙДЕНО (ранг 1)", "БЛИЖАЙШАЯ ЧУЖАЯ"],
            "Успехи при сильной смене ракурса (8 разных машин, верх по Δaspect)", p,
            border_by=[BLUE, GREEN, RED])
manifest.append(dict(file=p.name, what="8 успехов с наибольшей сменой ракурса", cases=[int(x) for x in cand]))

# --- 4. успехи при смене освещения (ночь<->день) ---
mix = ok[lig['mixed'][ok]]
cand = uniq_by_vid(mix[np.argsort(-lig['dfL'][mix])])
rows = []
for i in cand:
    j, k = mate_of(i), top2(i)
    rows.append([(crop_img(qm, i), f"запрос v{qv[i]} cam{qc[i]}\nяркость кадра {fq['frameL'][i]:.0f}"),
                 (crop_img(gm, j), f"НАЙДЕН 1-м: cos {S[i,j]:.3f}\nяркость {fg['frameL'][j]:.0f}"),
                 (crop_img(gm, k), f"ближайшая чужая: cos {S[i,k]:.3f}\nv{gv[k]}")])
p = JOB/"sheets/success_daynight_01.jpg"
build_sheet(rows, ["ЗАПРОС", "ВЕРНО НАЙДЕНО (ранг 1)", "БЛИЖАЙШАЯ ЧУЖАЯ"],
            "Успехи на паре ночь↔день (8 разных машин, верх по перепаду яркости)", p,
            border_by=[BLUE, GREEN, RED])
manifest.append(dict(file=p.name, what="8 успехов на парах ночь-день", cases=[int(x) for x in cand]))

# --- 5. что чинит переранжирование ---
fixed = idx[(r1[idx] == 0) & (pr['rank1'][idx] == 1)]
cand = uniq_by_vid(fixed[np.argsort(-fr[fixed])])
rows = []
for i in cand:
    j, k = mate_of(i), int(top[i])
    rows.append([(crop_img(qm, i), f"запрос v{qv[i]} cam{qc[i]}"),
                 (crop_img(gm, k), f"было 1-м (чужая): cos {S[i,k]:.3f}\nv{gv[k]}"),
                 (crop_img(gm, j), f"стало 1-м после rerank\nбыло ранг {int(fr[i])}, cos {S[i,j]:.3f}")])
p = JOB/"sheets/rerank_fixed_01.jpg"
build_sheet(rows, ["ЗАПРОС", "БЫЛО ПЕРВЫМ (ошибка)", "СТАЛО ПЕРВЫМ ПОСЛЕ ПЕРЕРАНЖИРОВАНИЯ"],
            "Что чинит rerank (6,3,0.3): 8 машин с наибольшим исходным рангом", p,
            border_by=[BLUE, RED, GREEN])
manifest.append(dict(file=p.name, what="8 случаев, починенных переранжированием", cases=[int(x) for x in cand]))

# --- 6. глубокие провалы: верный ответ дальше 20-го места ---
hard = idx[(r1[idx] == 0) & (fr[idx] > 20)]
cand = uniq_by_vid(hard[np.argsort(-fr[hard])])
rows = []
for i in cand:
    j, k = mate_of(i), int(top[i])
    rows.append([(crop_img(qm, i), f"запрос v{qv[i]} cam{qc[i]}"),
                 (crop_img(gm, j), f"верный: ранг {int(fr[i])}, cos {S[i,j]:.3f}\ncam{gc[j]}"),
                 (crop_img(gm, k), f"top-1: cos {S[i,k]:.3f}\nv{gv[k]} cam{gc[k]}")])
p = JOB/"sheets/hard_failures_01.jpg"
build_sheet(rows, ["ЗАПРОС", "ВЕРНЫЙ ОТВЕТ (глубоко в списке)", "ЧТО ВЫДАНО ПЕРВЫМ"],
            "Глубокие провалы: 8 разных машин с наибольшим рангом верного ответа", p,
            border_by=[BLUE, GREEN, RED])
manifest.append(dict(file=p.name, what="8 самых глубоких провалов", cases=[int(x) for x in cand]))

(JOB/'out/sheets_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
print("\n".join(m['file'] for m in manifest))
