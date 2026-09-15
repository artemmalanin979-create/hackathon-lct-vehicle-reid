"""Метрики по всем вариантам — ТОЛЬКО через контур 04-solution/eval/reid_metrics.py.

Свой расчёт метрик не пишется: evaluate() импортируется и вызывается как в
04-solution/baseline/scripts/run_eval.py (camera_policy=market, refusal_mode=presence).
Парный бутстрэп считается по per-query AP, которые вернул тот же контур.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
JOB = Path(__file__).resolve().parent.parent
REPO = Path(__file__).resolve().parents[3]  # корень репозитория
EVAL = REPO / "04-solution/eval"
SPLIT = REPO / "04-solution/split/files"
sys.path.insert(0, str(EVAL))
from reid_metrics import evaluate, scores_from_embeddings  # noqa: E402
sys.path.insert(0, str(JOB / "scripts"))  # mask_ops лежит здесь, а не в work/
from extract_variants import VARIANTS  # noqa: E402

THRESHOLD = 0.349214  # порог бейзлайна; на mAP не влияет, нужен только для refusal-ветки


def rd(p):
    return list(csv.DictReader(open(p, newline="")))


def main():
    missing = [v for v in VARIANTS
               if not (JOB / "work" / "emb" / f"val_query_{v}.npy").is_file()]
    if missing:
        raise SystemExit(
            f"нет векторов вариантов в {JOB / 'work' / 'emb'} ({len(missing)} из "
            f"{len(VARIANTS)}, например {missing[0]}). Их создаёт extract_variants.py "
            "по кэшу детекций work/boxes.json (cache_boxes.py); см. REPORT.md, разд. 7")
    qm, gm = rd(SPLIT / "val_query.csv"), rd(SPLIT / "val_gallery.csv")
    common = dict(query_ids=[r["vehicle_id"] for r in qm],
                  gallery_ids=[r["vehicle_id"] for r in gm],
                  query_cameras=[r["camera_id"] for r in qm],
                  gallery_cameras=[r["camera_id"] for r in gm],
                  known_absent=np.array([r["has_mate"] == "0" for r in qm]),
                  threshold=THRESHOLD, camera_policy="market", refusal_mode="presence")
    res, aps = {}, {}
    for v in VARIANTS:
        q = np.load(JOB / "work" / "emb" / f"val_query_{v}.npy")
        g = np.load(JOB / "work" / "emb" / f"val_gallery_{v}.npy")
        r = evaluate(scores_from_embeddings(q, g, metric="cosine"), **common)
        f = r["ranking_full_gallery"]
        res[v] = {"mAP": f["mAP"], "Rank-1": f["Rank-1"], "Rank-5": f["Rank-5"],
                  "mINP": f["mINP"], "valid_queries": f["num_valid_queries"]}
        aps[v] = np.array([row["ap"] for row in r["per_query"] if row["status"] == "known"])
        print(f"{v:16s} mAP={f['mAP']:.4f} R1={f['Rank-1']:.4f} "
              f"(n={f['num_valid_queries']})", flush=True)
    n = len(aps["base"])
    assert all(len(a) == n for a in aps.values()), "разное число валидных запросов"
    for v in VARIANTS:  # сверка: mAP контура == среднее его же per-query AP
        assert abs(aps[v].mean() - res[v]["mAP"]) < 1e-9, v

    rng = np.random.default_rng(20260915)
    idx = rng.integers(0, n, size=(4000, n))

    def contrast(a, b, label):
        """b - a по mAP (a=пластина, b=контроль): сколько метрики даёт зона пластины
        СВЕРХ потери такой же площади в другом месте. Парный бутстрэп по запросам."""
        d = aps[b][idx].mean(axis=1) - aps[a][idx].mean(axis=1)
        return {"пара": label, "mAP_a": res[a]["mAP"], "mAP_b": res[b]["mAP"],
                "delta_b_minus_a": res[b]["mAP"] - res[a]["mAP"],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

    ctl = ["shift_ring", "mirror_ring", "side_ring", "rand1_ring", "rand2_ring"]
    contrasts = [contrast("plate_ring", c, f"пластина vs контроль {c}") for c in ctl]
    med = float(np.median([res[c]["mAP"] for c in ctl]))
    extra = [contrast("platepad_ring", "shiftpad_ring", "пластина+12% vs контроль+12%"),
             contrast("plate_gray127", "shift_gray127", "серый 127: пластина vs контроль"),
             contrast("plate_desat", "shift_desat", "обесцвечивание: пластина vs контроль")]
    # насколько само вмешательство сдвигает вектор (без всякой метрики ранжирования)
    cos = {}
    for v in VARIANTS[1:]:
        acc = []
        for split in ("val_query", "val_gallery"):
            b = np.load(JOB / "work" / "emb" / f"{split}_base.npy").astype(np.float64)
            x = np.load(JOB / "work" / "emb" / f"{split}_{v}.npy").astype(np.float64)
            acc.append((b * x).sum(axis=1))
        c = np.concatenate(acc)
        cos[v] = {"mean": round(float(c.mean()), 4), "median": round(float(np.median(c)), 4)}

    summary = {"metrics": res, "cos_to_base": cos,
               "median_control_mAP_ring": med,
               "plate_minus_median_control": med - res["plate_ring"]["mAP"],
               "contrasts": contrasts + extra,
               "n_valid_queries": int(n), "bootstrap": 4000, "threshold": THRESHOLD}
    (JOB / "out").mkdir(exist_ok=True)
    json.dump(summary, open(JOB / "out" / "ablation.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
