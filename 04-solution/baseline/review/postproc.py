#!/usr/bin/env python3
"""Пост-обработка ТЕМИ ЖЕ векторами: k-reciprocal re-ranking и alpha-QE/DBA.
Обучения нет, веса те же, входные векторы — их собственные subject/out/val_*.npy.
Метрики считает ИХ контур (evaluate), чтобы сравнение было в тех же терминах."""
import csv, json, sys
from pathlib import Path
import numpy as np

JOB = Path(__file__).resolve().parent.parent
SPLIT = Path("/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/files")
sys.path.insert(0, "/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval")
from reid_metrics import evaluate, scores_from_embeddings

def rd(p):
    with open(p, newline="") as f: return list(csv.DictReader(f))
qm, gm = rd(SPLIT/"val_query.csv"), rd(SPLIT/"val_gallery.csv")
QV=[r["vehicle_id"] for r in qm]; GV=[r["vehicle_id"] for r in gm]
QC=[r["camera_id"] for r in qm]; GC=[r["camera_id"] for r in gm]
ABS=np.array([r["has_mate"]=="0" for r in qm])

def report(name, scores):
    res = evaluate(scores, query_ids=QV, gallery_ids=GV, query_cameras=QC, gallery_cameras=GC,
                   known_absent=ABS, threshold=0.0, camera_policy="market", refusal_mode="presence")
    f = res["ranking_full_gallery"]; t = res["ranking_top_k"]
    print(f"{name:34s} mAP={f['mAP']:.4f} R1={f['Rank-1']:.4f} R5={f['Rank-5']:.4f} mAP@10={t['mAP']:.4f}", flush=True)
    return dict(name=name, mAP=f["mAP"], R1=f["Rank-1"], R5=f["Rank-5"], mAP10=t["mAP"])

def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)

def k_reciprocal(q, g, k1=20, k2=6, lam=0.3):
    feat = np.concatenate([q, g]).astype(np.float32)
    n, nq = len(feat), len(q)
    d = 2.0 - 2.0*(feat @ feat.T)            # квадрат евклида для L2-норм. векторов
    np.maximum(d, 0, out=d)
    d = np.transpose(d / np.max(d, axis=0))  # как в оригинальной реализации
    V = np.zeros((n, n), np.float32)
    rank = np.argsort(d, axis=1, kind="stable").astype(np.int32)
    half = int(round(k1/2.))+1
    for i in range(n):
        fw = rank[i, :k1+1]; bw = rank[fw, :k1+1]
        kr = fw[np.where(bw == i)[0]]
        exp = kr
        for c in kr:
            cf = rank[c, :half]; cb = rank[cf, :half]
            ckr = cf[np.where(cb == c)[0]]
            if len(np.intersect1d(ckr, kr)) > 2./3*len(ckr):
                exp = np.append(exp, ckr)
        exp = np.unique(exp)
        w = np.exp(-d[i, exp]); V[i, exp] = w/np.sum(w)
    d = d[:nq]
    if k2 != 1:
        Vq = np.zeros_like(V)
        for i in range(n): Vq[i] = np.mean(V[rank[i, :k2]], axis=0)
        V = Vq
    inv = [np.where(V[:, i] != 0)[0] for i in range(n)]
    jac = np.zeros_like(d)
    for i in range(nq):
        tmp = np.zeros(n, np.float32)
        nz = np.where(V[i] != 0)[0]
        for j in nz:
            im = inv[j]
            tmp[im] += np.minimum(V[i, j], V[im, j])
        jac[i] = 1 - tmp/(2.-tmp)
    return (jac*(1-lam) + d*lam)[:, nq:]  # расстояние Q x G, меньше = лучше

def aqe(q, g, k=3, alpha=3.0, rounds=1):
    q, g = q.copy(), g.copy()
    for _ in range(rounds):
        s = q @ g.T
        idx = np.argsort(-s, axis=1)[:, :k]
        w = np.take_along_axis(s, idx, 1).clip(0)**alpha
        q = l2(q + np.einsum("qk,qkd->qd", w, g[idx]))
    return q, g

def dba(g, k=2, alpha=3.0):
    s = g @ g.T
    idx = np.argsort(-s, axis=1)[:, :k+1]
    w = np.take_along_axis(s, idx, 1).clip(0)**alpha
    return l2(np.einsum("qk,qkd->qd", w, g[idx]))

q = np.load(JOB/"subject/out/val_query.npy").astype(np.float64)
g = np.load(JOB/"subject/out/val_gallery.npy").astype(np.float64)
q, g = l2(q), l2(g)
rows = [report("base (их вектора)", scores_from_embeddings(q, g))]
for k1,k2,lam in [(20,6,0.3),(30,6,0.3),(20,6,0.0),(50,10,0.3),(10,3,0.3),(30,10,0.5)]:
    fd = k_reciprocal(q, g, k1, k2, lam)
    rows.append(report(f"k-reciprocal k1={k1} k2={k2} l={lam}", -fd.astype(np.float64)))
for k,a in [(1,3.0),(2,3.0),(3,3.0),(5,3.0),(3,1.0)]:
    qa, ga = aqe(q, g, k, a)
    rows.append(report(f"alphaQE k={k} a={a}", scores_from_embeddings(qa, ga)))
gd = dba(g)
rows.append(report("DBA(gallery) k=2", scores_from_embeddings(q, gd)))
json.dump(rows, open(JOB/"work/postproc.json","w"), indent=1)
