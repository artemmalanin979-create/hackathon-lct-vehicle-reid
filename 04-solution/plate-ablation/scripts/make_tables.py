"""Сборка markdown-таблиц отчёта из out/ablation.json (числа не набираются руками)."""
import json
from pathlib import Path
a = json.load(open("out/ablation.json"))
m = a["metrics"]
base = m["base"]["mAP"]
NAMES = {
 "base": "ничего не закрыто",
 "plate_ring": "**зона пластины** (бокс детектора)",
 "shift_ring": "контроль: тот же бокс, сдвинут по вертикали на 1.6 высоты",
 "mirror_ring": "контроль: тот же бокс, зеркально относительно центра кропа",
 "side_ring": "контроль: тот же бокс, сдвинут вбок на 1.3 ширины",
 "rand1_ring": "контроль: тот же бокс, случайное место (seed 1)",
 "rand2_ring": "контроль: тот же бокс, случайное место (seed 2)",
 "platepad_ring": "**зона пластины, бокс расширен на 12 %**",
 "shiftpad_ring": "контроль к расширенному боксу (сдвиг)",
 "plate_gray127": "**зона пластины**, заливка серым 127",
 "shift_gray127": "контроль (сдвиг), заливка серым 127",
 "plate_desat": "**обесцвечена только зона пластины**",
 "shift_desat": "контроль (сдвиг), обесцвечен",
}
lines = ["| Что сделано с кропом | mAP | Δ к «ничего» | Rank-1 | mINP |",
         "|---|---:|---:|---:|---:|"]
for k, nm in NAMES.items():
    r = m[k]
    d = "—" if k == "base" else f"{r['mAP']-base:+.4f}"
    lines.append(f"| {nm} | {r['mAP']:.4f} | {d} | {r['Rank-1']:.4f} | {r['mINP']:.4f} |")
tbl1 = "\n".join(lines)

lines = ["| Контраст (контроль − пластина) | mAP пластина | mAP контроль | Δ | 95 % ДИ | p |",
         "|---|---:|---:|---:|---:|---:|"]
for c in a["contrasts"]:
    lines.append(f"| {c['пара']} | {c['mAP_a']:.4f} | {c['mAP_b']:.4f} | "
                 f"{c['delta_b_minus_a']:+.4f} | [{c['ci95'][0]:+.4f}; {c['ci95'][1]:+.4f}] | {c['p_two_sided']:.3f} |")
tbl2 = "\n".join(lines)
Path("out/tables.md").write_text(tbl1 + "\n\n" + tbl2 + "\n")
print(tbl1); print(); print(tbl2)
