#!/usr/bin/env python3
"""s03: целостность. CSV<->файлы в обе стороны; bbox внутри кадра (реальный размер
из scan_meta.jsonl); нулевые/отрицательные w,h; побайтовые дубликаты файлов и их
принадлежность выборкам. Требует прогона s02. Выход: out_s03.json."""
from pathlib import Path
import csv, json, os
from collections import defaultdict

REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = str(Path(os.environ.get("REID_DATA_DIR", REPO / "data")))
HERE = os.path.dirname(os.path.abspath(__file__))

def rows_of(name):
    with open(os.path.join(DATA, name), newline="") as f:
        rd = csv.DictReader(f)
        return list(rd)

meta = {}
with open(os.path.join(HERE, "scan_meta.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        meta[r["f"]] = r

out = {}
splits = {n: rows_of(n) for n in ["train.csv", "test_query.csv", "test_gallery.csv"]}
csv_ids = {n: {r["image_id"] for r in rows} for n, rows in splits.items()}
all_csv = set().union(*csv_ids.values())
files = set(meta)
file_stems = {os.path.splitext(f)[0]: f for f in files}

out["csv_ids_total"] = len(all_csv)
out["files_total"] = len(files)
missing_files = sorted(i for i in all_csv if i not in file_stems)
orphan_files = sorted(f for s, f in file_stems.items() if s not in all_csv)
out["csv_ids_without_file"] = {"n": len(missing_files), "sample": missing_files[:5]}
out["files_not_in_any_csv"] = {"n": len(orphan_files), "sample": orphan_files[:5]}

# bbox против реального размера кадра
bad = defaultdict(list)
stats = {}
for name, rows in splits.items():
    n_zero = n_neg = n_out = 0
    for r in rows:
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        f = file_stems.get(r["image_id"])
        W, H = meta[f]["size"] if f else (None, None)
        if w == 0 or h == 0: n_zero += 1; bad["zero"].append(r["image_id"])
        if w < 0 or h < 0 or x < 0 or y < 0: n_neg += 1; bad["neg"].append(r["image_id"])
        if W and (x + w > W or y + h > H):
            n_out += 1
            bad["out"].append({"id": r["image_id"], "x": x, "y": y, "w": w, "h": h,
                               "overx": x + w - W, "overy": y + h - H})
    stats[name] = {"rows": len(rows), "zero_wh": n_zero, "negative": n_neg,
                   "bbox_out_of_frame": n_out}
out["bbox_checks"] = stats
out["bbox_out_examples"] = bad["out"][:10]
out["bbox_out_total"] = len(bad["out"])
if bad["out"]:
    mx = max(max(b["overx"], b["overy"]) for b in bad["out"])
    out["bbox_out_max_overhang_px"] = mx

# побайтовые дубликаты: группы и принадлежность
by_md5 = defaultdict(list)
for f, r in meta.items():
    by_md5[r["md5"]].append(os.path.splitext(f)[0])
which = {}
for n, ids in csv_ids.items():
    for i in ids: which[i] = n.replace(".csv", "")
rowmap = {r["image_id"]: r for rows in splits.values() for r in rows}
dupg = []
for h, ids in sorted(by_md5.items()):
    if len(ids) > 1:
        g = []
        for i in sorted(ids):
            r = rowmap.get(i, {})
            g.append({"image_id": i, "split": which.get(i, "?"),
                      "bbox": [r.get("x"), r.get("y"), r.get("w"), r.get("h")],
                      "vehicle_id": r.get("vehicle_id"), "camera_id": r.get("camera_id")})
        dupg.append(g)
out["byte_duplicate_groups"] = dupg
out["byte_dup_split_pattern"] = {}
from collections import Counter
pat = Counter(tuple(sorted(m["split"] for m in g)) for g in dupg)
out["byte_dup_split_pattern"] = {" + ".join(k): v for k, v in pat.items()}

with open(os.path.join(HERE, "out_s03.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:5500])
