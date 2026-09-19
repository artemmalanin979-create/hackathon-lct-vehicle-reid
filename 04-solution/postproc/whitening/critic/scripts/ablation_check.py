#!/usr/bin/env python3
"""job_49: проверка абляции номера для d1 (задание 4).

1) Входы абляции — именно d1? Берём СЫРЫЕ векторы вариантов (osnet+ainv2 из
   mirror/), применяем СВОЙ d1-преобразователь (свой whitening с train_fit),
   считаем cosine и KR(6,3,0,3) контуром и сравниваем с:
   - ablation_d1_w1.0.json (cosine-метрики 13 вариантов)
   - s03_ablation.json [d1_w1.0][kr_supplement] (KR-метрики + контрасты)
   Если бы входы были НЕ d1, base дал бы 0,6566 (osnet) / 0,6905 (ainv2) /
   0,6962 (ансамбль без whitening) / ~0,7598 (d2) вместо 0,7198 (cos) и 0,7621 (KR).
2) Маска и контроль одного размера: прогон controls() из mask_ops на всех боксах,
   проверка (w,h) попиксельно.
3) Контроль «случайная область»: rand1/rand2 против plate — тот же эффект или нет.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recalc_d1 import (l2n, read_meta, learn_whitening, apply_whitening,
                       kreciprocal, contour, RHO, K1, K2, LAM,
                       S, SPLIT, A2, J42, J45, OUT)

MIRROR = Path.home() / "lct-reid/jobs/job_45b/mirror"
EMB = {m: MIRROR / m / "04-solution/plate-ablation/work/emb" for m in ("osnet", "ainv2")}
BOXES = MIRROR / "osnet/04-solution/plate-ablation/work/boxes.json"
VARIANTS = ["base", "plate_ring", "shift_ring", "mirror_ring", "side_ring",
            "rand1_ring", "rand2_ring", "platepad_ring", "shiftpad_ring",
            "plate_gray127", "shift_gray127", "plate_desat", "shift_desat"]
RING7 = ["base", "plate_ring", "shift_ring", "mirror_ring", "side_ring",
         "rand1_ring", "rand2_ring"]


def my_d1(o, a, lw):
    return apply_whitening(l2n(o + a), lw)


def main():
    t0 = time.perf_counter()
    qm, gm = read_meta(SPLIT / "val_query.csv"), read_meta(SPLIT / "val_gallery.csv")
    tf = read_meta(SPLIT / "train_fit.csv")
    Xt = {"osnet": l2n(np.load(J42 / "train_fit_osnet.npy")),
          "ainv2": l2n(np.load(J45 / "train_fit_ainv2.npy"))}
    vid_t = np.array([r["vehicle_id"] for r in tf]); cam_t = np.array([r["camera_id"] for r in tf])
    lw = learn_whitening(l2n(Xt["osnet"] + Xt["ainv2"]), vid_t, cam_t, RHO)

    raw = {m: {f"{sp}_{v}": np.load(EMB[m] / f"{sp}_{v}.npy").astype(np.float64)
               for sp in ("val_query", "val_gallery") for v in VARIANTS}
           for m in ("osnet", "ainv2")}
    # размеры и порядок должны совпадать со сплитом
    for m in ("osnet", "ainv2"):
        assert raw[m]["val_query_base"].shape == (len(qm), 512)
        assert raw[m]["val_gallery_base"].shape == (len(gm), 512)

    # ---------- (1) cosine-метрики 13 вариантов ----------
    ref_cos = json.load(open(J45 / "ablation_d1_w1.0.json"))["metrics"]
    print("[cosine: мой d1-преобразователь по сырым векторам вариантов]")
    max_d = 0.0
    my_cos = {}
    for v in VARIANTS:
        q = my_d1(raw["osnet"][f"val_query_{v}"], raw["ainv2"][f"val_query_{v}"], lw)
        g = my_d1(raw["osnet"][f"val_gallery_{v}"], raw["ainv2"][f"val_gallery_{v}"], lw)
        from reid_metrics import scores_from_embeddings
        r = contour(scores_from_embeddings(q, g, metric="cosine"), qm, gm)
        my_cos[v] = r
        d = max(abs(r["mAP"] - ref_cos[v]["mAP"]), abs(r["Rank-1"] - ref_cos[v]["Rank-1"]))
        max_d = max(max_d, d)
        print(f"  {v:16s} mAP={r['mAP']:.6f} (ref {ref_cos[v]['mAP']:.6f}) "
              f"R1={r['Rank-1']:.6f} (ref {ref_cos[v]['Rank-1']:.6f})", flush=True)
    print(f"  MAX |diff| по 13 вариантам = {max_d:.3e}")
    print(f"  эталоны НЕ-d1: osnet 0.656569 / ainv2 0.690505 / c(без lw) 0.696159 / "
          f"base здесь = {my_cos['base']['mAP']:.6f} -> это d1")

    # ---------- (2) KR-метрики и контрасты ----------
    ref = json.load(open(J45 / "s03_ablation.json"))["d1_w1.0"]["kr_supplement"]
    print("[KR(6,3,0.3): свои прогоны по вариантам]")
    ap_kr, met_kr = {}, {}
    for v in VARIANTS:
        q = my_d1(raw["osnet"][f"val_query_{v}"], raw["ainv2"][f"val_query_{v}"], lw)
        g = my_d1(raw["osnet"][f"val_gallery_{v}"], raw["ainv2"][f"val_gallery_{v}"], lw)
        r = contour(-kreciprocal(q, g, K1, K2, LAM), qm, gm)
        ap_kr[v], met_kr[v] = r["ap"], r
        print(f"  {v:16s} mAP={r['mAP']:.6f} (ref {ref['metrics'][v]['mAP']:.6f})", flush=True)
    max_dk = max(abs(met_kr[v]["mAP"] - ref["metrics"][v]["mAP"]) for v in VARIANTS)
    print(f"  MAX |diff KR mAP| = {max_dk:.3e}")

    n = len(ap_kr["base"])
    rng = np.random.default_rng(20260915)
    idx = rng.integers(0, n, size=(4000, n))

    def contrast(a, b):
        dd = ap_kr[b][idx].mean(1) - ap_kr[a][idx].mean(1)
        return {"delta": float(met_kr[b]["mAP"] - met_kr[a]["mAP"]),
                "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                "p": float(2 * min((dd <= 0).mean(), (dd >= 0).mean()))}

    print("[KR-контрасты: мои vs s03]")
    my_ct = {}
    for c in ("shift_ring", "mirror_ring", "side_ring", "rand1_ring", "rand2_ring"):
        my_ct[c] = contrast("plate_ring", c)
        rc = ref["contrasts"][c]
        print(f"  plate vs {c:12s} Δ={my_ct[c]['delta']:+.6f} CI={my_ct[c]['ci95']} "
              f"(ref Δ={rc['delta_b_minus_a']:+.6f} CI={[round(x,6) for x in rc['ci95']]})")

    # ---------- (3) размеры маски и контролей ----------
    sys.path.insert(0, str(MIRROR / "osnet/04-solution/plate-ablation/scripts"))
    from mask_ops import controls  # их код генерации масок — проверяемый объект

    def pad_box(b, shape, f=0.12):  # как в extract_variants.py
        H, W = shape; x, y, w, h = b
        dx, dy = int(round(w * f)), int(round(h * f))
        x0, y0 = max(0, x - dx), max(0, y - dy)
        x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
        return (x0, y0, x1 - x0, y1 - y0)
    boxes = json.load(open(BOXES))
    n_bad = 0; n_tot = 0
    for split in ("val_query", "val_gallery"):
        for b in boxes[split]:
            if not b:
                continue
            box = tuple(b[:4])
            # размер кропа нужен для controls; берём из csv (x,y,w,h кропа = поле снимка)
            n_tot += 1
        # размеры кропов из csv
    crops = {}
    for split in ("val_query", "val_gallery"):
        rows = read_meta(SPLIT / f"{split}.csv")
        crops[split] = [(int(r["w"]), int(r["h"])) for r in rows]
    n_box = 0
    for split in ("val_query", "val_gallery"):
        for j, b in enumerate(boxes[split]):
            if not b:
                continue
            n_box += 1
            W, H = crops[split][j]
            box = tuple(b[:4])
            w0, h0 = box[2], box[3]
            ctl = controls(box, (H, W), seed=123)
            for name, c in ctl.items():
                if c[2] != w0 or c[3] != h0:
                    n_bad += 1
            pb = pad_box(box, (H, W))
            pctl = controls(pb, (H, W), seed=123)
            for name, c in pctl.items():
                if c[2] != pb[2] or c[3] != pb[3]:
                    n_bad += 1
    print(f"[размеры масок] боксов с детекцией: {n_box}, контролей несовпадающего размера: {n_bad}")

    json.dump({"cos_max_diff": max_d, "kr_max_diff": max_dk,
               "my_cos_map": {v: my_cos[v]["mAP"] for v in VARIANTS},
               "my_kr_map": {v: met_kr[v]["mAP"] for v in VARIANTS},
               "my_kr_contrasts": my_ct,
               "mask_size_mismatches": n_bad, "n_boxes": n_box},
              open(OUT / "ablation_check.json", "w"), indent=1)
    print(f"total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
