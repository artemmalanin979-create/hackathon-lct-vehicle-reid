#!/usr/bin/env python3
"""Сравнение структуры нашей вал-части с выданным тестом — числами.
Всё, что в тесте измеримо без меток (размеры, серии, почти-дубли, bbox), меряется
ОДНИМ и тем же кодом там и там; что требует меток — даётся для нас точно, для теста
как «неизвестно» или внешняя оценка EDA."""
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

import common as C


def bbox_median_share(rows):
    a = np.array([int(r["w"]) * int(r["h"]) for r in rows], dtype=np.float64)
    return round(float(np.median(a)) / (1920 * 1080) * 100, 1)


def cross_share_ge(Vq, Vg, thr=C.SERIES_THRESHOLD, block=256):
    """Доля запросов, у которых есть кадр галереи с полнокадровым cos>=thr."""
    hit = np.zeros(len(Vq), dtype=bool)
    for s in range(0, len(Vq), block):
        S = Vq[s:s + block] @ Vg.T
        hit[s:s + block] = (S >= thr).any(axis=1)
    return int(hit.sum())


def main():
    split_dir = os.path.join(C.JOB, "split")
    with open(os.path.join(split_dir, "val_query.csv"), newline="") as f:
        vq = list(csv.DictReader(f))
    with open(os.path.join(split_dir, "val_gallery.csv"), newline="") as f:
        vg = list(csv.DictReader(f))
    tq = C.read_csv("test_query.csv")
    tg = C.read_csv("test_gallery.csv")

    Vq = C.normed(C.thumbs_for([r["image_id"] for r in vq], "thumbs_val_query"))
    Vg = C.normed(C.thumbs_for([r["image_id"] for r in vg], "thumbs_val_gallery"))
    Tq = C.normed(C.thumbs_for([r["image_id"] for r in tq], "thumbs_test_query"))
    Tg = C.normed(C.thumbs_for([r["image_id"] for r in tg], "thumbs_test_gallery"))

    ours = {"query_frames": len(vq), "gallery_frames": len(vg),
            "ratio_q_to_g": round(len(vq) / len(vg), 3),
            "query_series": C.series_stats(Vq),
            "gallery_series": C.series_stats(Vg),
            "queries_with_same_moment_in_gallery_cos090": cross_share_ge(Vq, Vg),
            "bbox_median_share_query_pct": bbox_median_share(vq),
            "bbox_median_share_gallery_pct": bbox_median_share(vg)}
    test = {"query_frames": len(tq), "gallery_frames": len(tg),
            "ratio_q_to_g": round(len(tq) / len(tg), 3),
            "query_series": C.series_stats(Tq),
            "gallery_series": C.series_stats(Tg),
            "queries_with_same_moment_in_gallery_cos090": cross_share_ge(Tq, Tg),
            "bbox_median_share_query_pct": bbox_median_share(tq),
            "bbox_median_share_gallery_pct": bbox_median_share(tg)}

    # --- только для нашего сплита (в тесте меток нет) ---
    per_vid = Counter()
    for r in vq + vg:
        per_vid[int(r["vehicle_id"])] += 1
    frames_per_id = np.array(sorted(per_vid.values()))
    gal_cams = defaultdict(set)
    for r in vg:
        gal_cams[int(r["vehicle_id"])].add(int(r["camera_id"]))
    mates = []
    same_cam_mate = 0
    for r in vq:
        v, c = int(r["vehicle_id"]), int(r["camera_id"])
        if v in gal_cams:
            mates.append(len(gal_cams[v] - {c}))
            same_cam_mate += int(c in gal_cams[v])
    cams_val = Counter(int(r["camera_id"]) for r in vq + vg)
    train = C.load_train()
    cams_train = Counter(r["camera_id"] for r in train)
    top5_val = [(c, n, round(n / sum(cams_val.values()) * 100, 1))
                for c, n in cams_val.most_common(5)]
    top5_train = [(c, n, round(n / 9556 * 100, 1)) for c, n in cams_train.most_common(5)]

    labels_only_ours = {
        "protocol_identities": len(per_vid),
        "frames_per_identity": {"min": int(frames_per_id.min()),
                                "median": float(np.median(frames_per_id)),
                                "max": int(frames_per_id.max())},
        "known_queries": len(mates),
        "refusal_queries": len(vq) - len(mates),
        "refusal_share": round((len(vq) - len(mates)) / len(vq), 4),
        "cross_camera_mates_per_known_query": dict(Counter(mates)),
        "known_queries_with_same_camera_mate": same_cam_mate,
        "distinct_cameras_in_val": len(cams_val),
        "top5_cameras_val_frames_pct": top5_val,
        "top5_cameras_train_frames_pct": top5_train,
    }
    out = {"ours": ours, "test": test, "ours_labels_only": labels_only_ours,
           "test_external_estimates": {
               "queries_with_same_point_double_lower_bound": "635/1110 (EDA s05b)",
               "identities_scenario": "250–350 (EDA §6, сценарная оценка)",
               "refusal_share": "неизвестна (гарантировано лишь >0)"}}
    with open(os.path.join(C.OUT, "structure_comparison.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
