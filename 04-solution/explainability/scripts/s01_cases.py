#!/usr/bin/env python3
"""Отбор случаев по правилам, объявленным в journal.md ДО запуска.

Рейтинг и статус запроса берём из измерительного контура 04-solution/eval
(evaluate(..., include_rankings=True), политика камер 'market'), чтобы не
изобретать свой протокол. Отбор — численный, без просмотра картинок.

Выход: out/cases.json — список случаев с полями
  kind: twin | hardok | badcrop | plate
  qi, gi (индексы строк val_query.csv / val_gallery.csv), пояснение.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "scripts"))
import occl  # noqa: E402

REPO = occl.REPO
EA = REPO / "04-solution/error-analysis/out"
sys.path.insert(0, str(REPO / "04-solution/eval"))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402

qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
qe = np.load(REPO / "04-solution/baseline/out/val_query.npy")
ge = np.load(REPO / "04-solution/baseline/out/val_gallery.npy")
qids = (REPO / "04-solution/baseline/out/val_query.ids").read_text().split()
gids = (REPO / "04-solution/baseline/out/val_gallery.ids").read_text().split()
assert qids == [r["image_id"] for r in qm] and gids == [r["image_id"] for r in gm]

cos = scores_from_embeddings(qe, ge, metric="cosine")
res = evaluate(
    cos,
    query_ids=[r["vehicle_id"] for r in qm],
    gallery_ids=[r["vehicle_id"] for r in gm],
    query_cameras=[r["camera_id"] for r in qm],
    gallery_cameras=[r["camera_id"] for r in gm],
    known_absent=np.array([r["has_mate"] == "0" for r in qm]),
    threshold=0.0, camera_policy="market", refusal_mode="presence",
    include_rankings=True,
)
pq = res["per_query"]
full = res["ranking_full_gallery"]
print(f"контур: mAP={full['mAP']:.4f} Rank-1={full['Rank-1']:.4f} "
      f"valid={full['num_valid_queries']}")

gv = np.array([int(r["vehicle_id"]) for r in gm])
gc = np.array([int(r["camera_id"]) for r in gm])
qv = np.array([int(r["vehicle_id"]) for r in qm])
qc = np.array([int(r["camera_id"]) for r in qm])


def top1(i):
    return pq[i]["ranking"][0] if pq[i].get("ranking") else None


def mate_of(i):
    """Лучшая по cos кросс-камерная верная пара запроса i."""
    m = np.flatnonzero((gv == qv[i]) & (gc != qc[i]))
    return int(m[np.argmax(cos[i, m])]) if m.size else None


valid = [i for i, r in enumerate(pq) if r["status"] == "known" and r["rank1"] is not None]
print("запросов со статусом known:", len(valid))

cases = []

# ---- 1) ДВОЙНИКИ: взаимные пары машин из twin_stats.json ----------------------
tw = json.load(open(EA / "twin_stats.json"))
used_vid = set()
for a, b, n in tw["top_pairs"]:
    if (a, b) in used_vid or (b, a) in used_vid:
        continue
    # запрос машины a, у которого top-1 — машина b
    qcand = [i for i in valid if qv[i] == a and top1(i) is not None and gv[top1(i)] == b]
    if not qcand:
        continue
    i = max(qcand, key=lambda i: cos[i, top1(i)])   # самый уверенный промах
    w, m = top1(i), mate_of(i)
    if m is None:
        continue
    used_vid.add((a, b))
    cases.append({"kind": "twin", "qi": int(i), "gi": int(w), "gi_alt": int(m),
                  "vid_q": int(a), "vid_g": int(b), "n_err": int(n),
                  "cos_wrong": float(cos[i, w]), "cos_mate": float(cos[i, m]),
                  "note": f"взаимные двойники v{a}<->v{b}, {n} ошибок; "
                          f"показан чужой top-1, альтернатива — верная пара"})
    if len(cases) >= 6:
        break

# ---- 2) УСПЕШНЫЕ ТРУДНЫЕ: rank1=1 при самом низком cos к верной паре ---------
ok = [i for i in valid if pq[i]["rank1"] == 1]
ok_cos = [(i, cos[i, mate_of(i)]) for i in ok if mate_of(i) is not None]
ok_cos.sort(key=lambda t: t[1])
seen_v = set()
hard = []
for i, c in ok_cos:                                    # нижний хвост по cos
    if qv[i] in seen_v:
        continue
    seen_v.add(qv[i])
    m = mate_of(i)
    hard.append({"kind": "hardok", "qi": int(i), "gi": int(m), "gi_alt": None,
                 "vid_q": int(qv[i]), "vid_g": int(gv[m]), "n_err": 0,
                 "cos_wrong": None, "cos_mate": float(c),
                 "note": f"верно найдено при cos={c:.3f} "
                         f"(кам. {qc[i]}->{gc[m]}), нижний хвост успешных"})
    if len(hard) >= 6:
        break
cases += hard

# ---- 3) ИСПОРЧЕННЫЙ КРОП: вердикты ручной сверки lowcos ---------------------
lc = json.load(open(EA / "lowcos_cases.json"))
lv = json.load(open(EA / "lowcos_verdicts.json"))["cases"]
bad = []
for rec in lc:
    key = [k for k in lv if k.startswith(f"{rec['n']}_")]
    if not key:
        continue
    verdict = lv[key[0]]
    if verdict not in ("разметка", "рамка_на_несколько_машин"):
        continue
    bad.append({"kind": "badcrop", "qi": int(rec["qi"]), "gi": int(rec["mate"]),
                "gi_alt": None, "vid_q": int(rec["q_vid"]),
                "vid_g": int(rec["q_vid"]), "n_err": 0,
                "cos_wrong": None, "cos_mate": float(rec["cos"]),
                "note": f"вердикт ручной сверки: «{verdict}»; "
                        f"кам. {rec['q_cam']}->{rec['mate_cam']}, ранг верного {rec['rank']}"})
    if len(bad) >= 6:
        break
cases += bad

# ---- 4) кроме того: категория C ручной разметки 64 ошибок -------------------
ml = json.load(open(EA / "manual_labels.json"))["labels"]
meta = {m["case"]: m for m in json.load(open(EA / "error_sample_meta.json"))}
cc = [int(k) for k, v in ml.items() if v[0] == "C"]
for c in cc[:3]:
    m = meta[c]
    cases.append({"kind": "badcrop_err", "qi": int(m["qi"]), "gi": int(m["wrong"]),
                  "gi_alt": int(m["mate"]), "vid_q": int(m["q_vid"]),
                  "vid_g": int(m["wrong_vid"]), "n_err": 0,
                  "cos_wrong": float(m["cos_wrong"]), "cos_mate": float(m["cos_mate"]),
                  "note": f"ручная разметка ошибки #{c}: «кроп испорчен»; "
                          f"ранг верного {m['rank']}"})

out = {"n": len(cases), "cases": cases,
       "eval": {"mAP": full["mAP"], "Rank-1": full["Rank-1"]}}
(JOB / "out/cases.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
for c in cases:
    print(f"{c['kind']:12s} q{c['qi']:5d} g{c['gi']:4d}  {c['note']}")
print("всего:", len(cases))

# ---- 5) ОПОРНАЯ ГРУППА: случайные ОБЫЧНЫЕ верные совпадения (rank1=1) --------
# Нужна как фон для агрегатов (фон против центра, зона пластины против карты):
# без неё непонятно, что считать обычным значением.
rng = np.random.default_rng(20260916)
pool = [i for i in valid if pq[i]["rank1"] == 1 and mate_of(i) is not None]
pick = rng.permutation(len(pool))[:40]
ref = []
for j in pick:
    i = pool[int(j)]
    m = mate_of(i)
    ref.append({"kind": "ref", "qi": int(i), "gi": int(m), "gi_alt": None,
                "vid_q": int(qv[i]), "vid_g": int(gv[m]), "n_err": 0,
                "cos_wrong": None, "cos_mate": float(cos[i, m]),
                "note": f"обычное верное совпадение, cos={cos[i, m]:.3f}, "
                        f"кам. {qc[i]}->{gc[m]}"})
(JOB / "out/ref_cases.json").write_text(
    json.dumps({"n": len(ref), "seed": 20260916, "cases": ref},
               ensure_ascii=False, indent=1))
print("опорная группа:", len(ref), "случаев, медиана cos =",
      round(float(np.median([r["cos_mate"] for r in ref])), 3))
