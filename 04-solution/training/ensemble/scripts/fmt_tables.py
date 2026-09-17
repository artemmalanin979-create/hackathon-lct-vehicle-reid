#!/usr/bin/env python3
"""Форматирование таблиц отчёта из out/*.json (только текст)."""
import json
from pathlib import Path
OUT = Path.home() / "lct-reid/jobs/job_43/out"
T = json.loads((OUT / "s04_table.json").read_text())
B = json.loads((OUT / "s05_boot.json").read_text())
R = json.loads((OUT / "s06_refusal.json").read_text())
names = {"osnet": "базовая OSNet-AIN", "ainv1": "ain_v1 (прогон 1)", "ainv2": "ain_v2 (прогон 2)",
         "cat_w0.25": "concat [база, 0,25·ain_v2]", "cat_w0.5": "concat [база, 0,5·ain_v2]",
         "cat_w1.0": "concat [база, 1,0·ain_v2]", "avg": "среднее (база + ain_v2)",
         "avg_w0.25": "среднее (база + 0,25·ain_v2)*", "avg_w0.5": "среднее (база + 0,5·ain_v2)*"}
def f(x): return f"{x:.4f}"
print("### Таблица A. Полная галерея, камерная политика market (832 запроса с кросс-камерной парой)\n")
print("| Конфигурация | шкала | mAP | Rank-1 | Rank-5 | mINP |")
print("|---|---|---:|---:|---:|---:|")
for k, nm in names.items():
    for sc, lab in (("cos", "косинус"), ("rr", "переранж. (6,3,0,3)")):
        m = T[f"{k}|{sc}|market"]
        print(f"| {nm} | {lab} | {f(m['mAP'])} | {f(m['Rank-1'])} | {f(m['Rank-5'])} | {f(m['mINP'])} |")
print("\n\\* строки со звёздочкой — дополнительные (в брифе не запрошены).\n")
print("### Таблица B. Разрыв «без исключения камеры − market» (mAP / Rank-1)\n")
print("| Конфигурация | косинус: без искл. | разрыв mAP | разрыв Rank-1 | переранж.: без искл. | разрыв mAP | разрыв Rank-1 |")
print("|---|---:|---:|---:|---:|---:|---:|")
for k, nm in names.items():
    c, cg = T[f"{k}|cos|nocam"], T[f"{k}|cos|gap"]; r, rg = T[f"{k}|rr|nocam"], T[f"{k}|rr|gap"]
    print(f"| {nm} | {f(c['mAP'])} / {f(c['Rank-1'])} | {cg['mAP']:+.4f} | {cg['Rank-1']:+.4f} | {f(r['mAP'])} / {f(r['Rank-1'])} | {rg['mAP']:+.4f} | {rg['Rank-1']:+.4f} |")
print("\n### Таблица C. Парный бутстрэп против базовой + переранжирование (mAP 0,6937), 4000 ресэмплов, seed 20260916\n")
print("| Конфигурация | шкала | mAP | Δ mAP | 95 % ДИ | p | Δ Rank-1 | 95 % ДИ | p | Δ > σ = 0,0134? |")
print("|---|---|---:|---:|---|---:|---:|---|---:|---|")
for k, nm in names.items():
    for sc, lab in (("cos", "косинус"), ("rr", "переранж.")):
        key = f"{k}|{sc}"
        if key == "osnet|rr":
            print(f"| {nm} | {lab} | {f(T['osnet|rr|market']['mAP'])} | опора | — | — | опора | — | — | — |"); continue
        m, r = B["mAP"][key], B["Rank-1"][key]
        flag = "да" if m["delta"] > 0.0134 and m["ci95"][0] > 0 else ("нет" if m["delta"] <= 0.0134 else "ДИ включает 0")
        print(f"| {nm} | {lab} | {f(m['mean_a'])} | {m['delta']:+.4f} | [{m['ci95'][0]:+.4f}; {m['ci95'][1]:+.4f}] | {m['p_two_sided']:.3f} | "
              f"{r['delta']:+.4f} | [{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}] | {r['p_two_sided']:.3f} | {flag} |")
print("\n### Таблица D. Режим отказа: порог по правилу max min(TNR, F1@{0,10; 0,25; 0,40}), market, presence\n")
print("| Конфигурация | шкала | порог | F1 | precision | recall | TNR | AUC-PR | F1 при TNR 0,7 | F1 при TNR 0,8 | F1@доля 0,10 / 0,40 |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for k in ("osnet", "ainv2", "avg", "cat_w1.0"):
    for sc, lab in (("cos", "косинус"), ("rr", "переранж.")):
        r = R[k][sc]; rb = r["robust"]; pr = r["robust_f1_at_priors"]
        print(f"| {names[k]} | {lab} | {rb['threshold']:.4f} | {f(rb['f1'])} | {f(rb['precision'])} | {f(rb['recall'])} | {f(rb['tnr'])} | {f(r['auc_pr_trapezoid'])} | "
              f"{f(r['tnr_0.7']['f1'])} | {f(r['tnr_0.8']['f1'])} | {f(pr['0.1'])} / {f(pr['0.4'])} |")
print("\n### Таблица E. Что даёт старый порог сервиса без перекалибровки (0,5496 косинус / 0,4994 переранж.)\n")
print("| Конфигурация | шкала | F1 | TNR | принято с парой (из 832) | ложных принятий (из 278) |")
print("|---|---|---:|---:|---:|---:|")
for k in ("osnet", "ainv2", "avg", "cat_w1.0"):
    for sc, lab in (("cos", "косинус"), ("rr", "переранж.")):
        o = R[k][sc]["old_base_threshold_applied"]
        print(f"| {names[k]} | {lab} | {f(o['f1'])} | {f(o['tnr'])} | {o['tp']} | {o['fp_unknown']} |")
