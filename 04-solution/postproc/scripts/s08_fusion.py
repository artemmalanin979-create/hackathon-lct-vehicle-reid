#!/usr/bin/env python3
"""Шаг 8: ансамбль OSNet + fast-reid R50-IBN (VeRi).

Слияние на уровне косинусных близостей: S = (1-w)*S_osnet + w*S_fastreid
(вектора разной размерности — 512 и 2048, конкатенация без веса некорректна).
Подбор: w на протоколе подбора; отдельно rerank поверх слитой дистанции
(быстрая сетка, w фиксирован). Затем — те же конфигурации на вал.
Выход: out/fusion.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (JOB, SPLIT, read_meta, eval_scores, l2norm,
                    rerank_prepare, rerank_finalize)
from s04_grid import K1S_FAST, K2S_FAST, LAMS_FAST

WS = [round(0.1 * i, 1) for i in range(11)]


def cos_all(q, g):
    feats = l2norm(np.concatenate([q, g]).astype(np.float64))
    return np.clip(feats @ feats.T, -1.0, 1.0)


def grid_on_dist(dist_all, nq, qm, gm, k1s, k2s, lams):
    points = []
    for k1 in k1s:
        od, ir, V = rerank_prepare(None, None, k1, dist_all=dist_all)
        for k2 in k2s:
            if k2 > k1:
                continue
            for lam, d in rerank_finalize(od, ir, V, nq, k2, lams).items():
                m = eval_scores(-d, qm, gm)
                points.append({"k1": k1, "k2": k2, "lam": lam, **m})
    return points


def main():
    out = {}
    meta = {
        "tune": (read_meta(JOB / "tune/tune_query.csv"), read_meta(JOB / "tune/tune_gallery.csv")),
        "val": (read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")),
    }
    sims = {}
    for s in ("tune", "val"):
        for tag in ("208", "TTA", "FR"):
            q = np.load(JOB / f"out/{s}_query_{tag}.npy")
            g = np.load(JOB / f"out/{s}_gallery_{tag}.npy")
            sims[(s, tag)] = (cos_all(q, g), len(q))

    # 1) fast-reid сам по себе
    for s in ("tune", "val"):
        qm, gm = meta[s]
        sim, nq = sims[(s, "FR")]
        out[f"fr_alone_{s}"] = eval_scores(sim[:nq, nq:], qm, gm)
        print(f"fr_alone_{s}:", json.dumps(out[f"fr_alone_{s}"]), flush=True)

    # 2) подбор w на tune: OSNet(208) + FR и OSNet(TTA) + FR
    qm, gm = meta["tune"]
    best = None
    for base_tag in ("208", "TTA"):
        so, nq = sims[("tune", base_tag)]
        sf, _ = sims[("tune", "FR")]
        for w in WS:
            fused = (1 - w) * so + w * sf
            m = eval_scores(fused[:nq, nq:], qm, gm)
            rec = {"base": base_tag, "w": w, **m}
            out.setdefault("fusion_w_scan_tune", []).append(rec)
            if best is None or m["mAP"] > best["mAP"]:
                best = rec
    out["fusion_best_tune"] = best
    print("выбор на подборе:", json.dumps(best), flush=True)

    # 3) rerank поверх слитой дистанции (быстрая сетка на tune, w и база из п.2)
    so, nq = sims[("tune", best["base"])]
    sf, _ = sims[("tune", "FR")]
    dist_all = (1 - best["w"]) * (1 - so) + best["w"] * (1 - sf)
    pts = grid_on_dist(dist_all, nq, qm, gm, K1S_FAST, K2S_FAST, LAMS_FAST)
    bp = max(pts, key=lambda p: p["mAP"])
    out["fusion_rerank_grid_tune_best"] = bp
    print("fusion+rerank на подборе:", json.dumps(bp), flush=True)

    # 4) объявление на вал: fusion(w*), fusion(w*)+rerank(bp)
    qm, gm = meta["val"]
    so, nq = sims[("val", best["base"])]
    sf, _ = sims[("val", "FR")]
    fused_sim = (1 - best["w"]) * so + best["w"] * sf
    out["fusion_val"] = eval_scores(fused_sim[:nq, nq:], qm, gm)
    dist_all = (1 - best["w"]) * (1 - so) + best["w"] * (1 - sf)
    od, ir, V = rerank_prepare(None, None, bp["k1"], dist_all=dist_all)
    d = rerank_finalize(od, ir, V, nq, bp["k2"], [bp["lam"]])[bp["lam"]]
    out["fusion_rerank_val"] = eval_scores(-d, qm, gm)
    print("fusion_val:", json.dumps(out["fusion_val"]), flush=True)
    print("fusion_rerank_val:", json.dumps(out["fusion_rerank_val"]), flush=True)

    (JOB / "out/fusion.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
