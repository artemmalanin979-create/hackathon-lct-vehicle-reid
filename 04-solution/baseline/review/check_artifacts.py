#!/usr/bin/env python3
"""Дословная сверка сдаваемых файлов со схемой dataset-readme.md + пересборка из embeddings.npy."""
import os
import csv, json, sys
from pathlib import Path
import numpy as np
JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
ART = JOB/"subject"/"artifacts"
sys.path.insert(0, str(REPO / "04-solution/eval"))
from reid_metrics import scores_from_embeddings
def ids(p):
    with open(p, newline="") as f: return [r["image_id"] for r in csv.DictReader(f)]
qid, gid = ids(DATA/"test_query.csv"), ids(DATA/"test_gallery.csv")
E = np.load(ART/"embeddings.npy")
print("embeddings.npy: shape", E.shape, "dtype", E.dtype, "finite", bool(np.isfinite(E).all()),
      "max|1-||v|||", float(np.abs(np.linalg.norm(E,axis=1)-1).max()))
print("  == concat(out/test_query.npy,out/test_gallery.npy):",
      np.array_equal(E, np.concatenate([np.load(JOB/"subject/out/test_query.npy"),
                                        np.load(JOB/"subject/out/test_gallery.npy")])))
Q, G = E[:len(qid)].astype(np.float64), E[len(qid):].astype(np.float64)
S = scores_from_embeddings(Q, G, metric="cosine")

# ---- submission.csv ----
raw = (ART/"submission.csv").read_bytes().decode()
hdr = raw.split("\n")[0]
exp_hdr = "query_id," + ",".join(f"gallery_id_{k}" for k in range(1,11))
print("\nsubmission заголовок:", repr(hdr), "| ожидаемый:", "СОВПАДАЕТ" if hdr==exp_hdr else "НЕ СОВПАДАЕТ")
rows = list(csv.reader((ART/"submission.csv").open(newline="")))[1:]
print("  строк:", len(rows), "(ожидалось", len(qid), ")",
      "| порядок query_id == test_query.csv:", [r[0] for r in rows]==qid,
      "| ширина всех строк 11:", all(len(r)==11 for r in rows))
allg=set(gid); bad_ids=sum(1 for r in rows if not set(r[1:])<=allg)
dupes=sum(1 for r in rows if len(set(r[1:]))!=10)
print("  gallery_id вне test_gallery:", bad_ids, "| строк с повторами:", dupes)
mis=0; mono=0
for i,r in enumerate(rows):
    top = np.argsort(-S[i], kind="stable")[:10]
    if [gid[j] for j in top] != r[1:]: mis+=1
    sc=[S[i, gid.index(x)] for x in r[1:]]
    if any(sc[k]<sc[k+1]-1e-12 for k in range(9)): mono+=1
print("  строк, не совпавших с пересборкой из embeddings.npy:", mis, "| нарушен порядок убывания:", mono)

# ---- candidates.csv ----
raw = (ART/"candidates.csv").read_bytes().decode()
print("\ncandidates заголовок:", repr(raw.split("\n")[0]))
crows = list(csv.reader((ART/"candidates.csv").open(newline="")))[1:]
print("  строк:", len(crows))
T = json.load(open(ART/"threshold.json"))["threshold"]
from collections import OrderedDict
byq = OrderedDict()
for r in crows: byq.setdefault(r[0], []).append(r)
print("  уникальных query_id:", len(byq), "| отказов (нет строк):", len(qid)-len(byq),
      "| threshold.json queries_refused:", json.load(open(ART/"threshold.json"))["queries_refused"])
print("  порядок query_id в файле == порядок в test_query.csv (среди присутствующих):",
      list(byq.keys()) == [q for q in qid if q in byq])
badconf=0; badsort=0; badthr=0; mism=0
for q,rs in byq.items():
    i = qid.index(q)
    exp = [j for j in np.argsort(-S[i], kind="stable") if S[i,j] >= T]
    if [gid[j] for j in exp] != [r[1] for r in rs]: mism+=1
    cs=[float(r[2]) for r in rs]
    if any(cs[k]<cs[k+1]-1e-9 for k in range(len(cs)-1)): badsort+=1
    if any(c < T-1e-6 for c in cs): badthr+=1
    for r in rs:
        if abs(float(r[2]) - S[i, gid.index(r[1])]) > 1e-6: badconf+=1
print("  запросов с расхождением списка:", mism, "| нарушен порядок:", badsort,
      "| confidence ниже порога:", badthr, "| confidence != cos:", badconf)
lens=[len(v) for v in byq.values()]
print("  кандидатов на запрос: min",min(lens),"медиана",int(np.median(lens)),"max",max(lens),"сумма",sum(lens))
# доля запросов, где принято >10 (submission обрезает до 10, candidates — нет)
print("  запросов с >10 кандидатами:", sum(1 for l in lens if l>10))
