#!/usr/bin/env python3
"""Детерминированный сплит train.csv на train_fit / val_query / val_gallery (+val_unused).

Конструкция (обоснование — journal.md и REPORT.md):
- единица разбиения — кластер идентичностей (union-find по md5-парам скопированных
  кадров), чтобы копии кадра не разошлись по сторонам;
- вал-идентичности: paired (query+gallery), refusal (только query, материал отказа),
  gallery-only (только gallery, дистракторы);
- галерея: 1 кадр с каждой камеры paired-идентичности → каждый paired-запрос имеет
  кросс-камерное верное совпадение; дедупликация "1 кадр = 1 наблюдение";
- запросы: сериями (компоненты cos32>=0.90 внутри (vid,cam)-групп), 1 кадр на серию
  + добор вторых кадров до целевого числа кадров — воспроизводит структуру выданного
  query (1110 кадров / 953 серии);
- параметры не подбирались по метрикам; целевые числа — размеры и структура
  выданного теста, доля отказных — отдельный обоснованный параметр.

Запуск: python3 make_split.py [--out DIR] [--seed 20260915] [--p-refusal 0.25]
Повторный запуск с теми же параметрами даёт бит-в-бит те же файлы.
"""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

import common as C


def build_clusters(vids, id2vid):
    parent = {v: v for v in vids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in C.MD5_PAIRS_TRAIN:
        ra, rb = find(id2vid[a]), find(id2vid[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for v in vids:
        groups[find(v)].append(v)
    return [sorted(g) for _, g in sorted(groups.items())]


def group_series(frame_idxs, V):
    """Серии внутри набора кадров одной (vid,cam)-группы: компоненты cos>=0.90."""
    if len(frame_idxs) == 1:
        return [list(frame_idxs)]
    sub = V[frame_idxs]
    labels = C.components(sub)
    out = defaultdict(list)
    for local, lab in enumerate(labels):
        out[int(lab)].append(frame_idxs[local])
    return [out[k] for k in sorted(out)]


def pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def select_series(rng, pool, n_series_target, n_frames_target, V):
    """Из пула серий: n_series_target серий по 1 кадру + добор вторых кадров
    до n_frames_target. Возвращает (кадры, фактическое число серий, добрано_вторых)."""
    order = rng.permutation(len(pool))
    chosen, frames = [], []
    for k in order[:n_series_target]:
        s = pool[int(k)]
        f = pick(rng, s)
        chosen.append((s, f))
        frames.append(f)
    need = n_frames_target - len(frames)
    multi = [i for i, (s, f) in enumerate(chosen) if len(s) >= 2]
    take2 = rng.permutation(len(multi))[:need]
    added = 0
    for t in take2:
        s, f = chosen[multi[int(t)]]
        rest = [x for x in s if x != f]
        frames.append(pick(rng, rest))
        added += 1
    ptr = n_series_target
    while len(frames) < n_frames_target and ptr < len(order):
        s = pool[int(order[ptr])]
        frames.append(pick(rng, s))
        chosen.append((s, None))
        ptr += 1
    if len(frames) != n_frames_target:
        raise SystemExit(f"пул серий исчерпан: {len(frames)} < {n_frames_target}")
    return frames, len(chosen), added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.SPLIT_DIR)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--n-query", type=int, default=1110)
    ap.add_argument("--n-gallery", type=int, default=750)
    ap.add_argument("--target-query-series", type=int, default=953)
    ap.add_argument("--p-refusal", type=float, default=0.25,
                    help="доля кадров val_query без пары в галерее")
    args = ap.parse_args()

    rows = C.load_train()
    V = C.normed(C.thumbs_for([r["image_id"] for r in rows], "thumbs_train"))
    id2vid = {r["image_id"]: r["vehicle_id"] for r in rows}
    by_vid_cam = defaultdict(list)  # (vid, cam) -> [row_idx] в порядке train.csv
    for i, r in enumerate(rows):
        by_vid_cam[(r["vehicle_id"], r["camera_id"])].append(i)
    vid_cams = defaultdict(list)
    for (v, c) in sorted(by_vid_cam):
        vid_cams[v].append(c)

    clusters = build_clusters(sorted(vid_cams), id2vid)
    rng = np.random.default_rng(args.seed)
    cluster_order = rng.permutation(len(clusters))

    n_ref = round(args.p_refusal * args.n_query)
    n_known = args.n_query - n_ref
    s_ref = round(args.target_query_series * n_ref / args.n_query)
    s_known = args.target_query_series - s_ref

    gallery_frames = []            # row_idx
    known_pool, refusal_pool = [], []   # списки серий (списков row_idx)
    paired, refusal, gallery_only = [], [], []
    gal_left = args.n_gallery
    ref_series_goal = int(np.ceil(s_ref * 1.5)) + 10

    val_vids = set()
    for ci in cluster_order:
        if gal_left == 0 and len(refusal_pool) >= ref_series_goal:
            break
        cluster = clusters[int(ci)]
        for vid in cluster:
            cams = vid_cams[vid]
            if gal_left >= 2:
                g_cams = cams if len(cams) <= gal_left else \
                    sorted(rng.choice(cams, size=gal_left, replace=False).tolist())
                for cam in g_cams:
                    gallery_frames.append(pick(rng, by_vid_cam[(vid, cam)]))
                gal_left -= len(g_cams)
                taken = set(gallery_frames[-len(g_cams):])
                for cam in cams:
                    rest = [i for i in by_vid_cam[(vid, cam)] if i not in taken]
                    if rest:
                        known_pool.extend(group_series(rest, V))
                paired.append(vid)
            elif gal_left == 1:
                cam = pick(rng, cams)
                gallery_frames.append(pick(rng, by_vid_cam[(vid, cam)]))
                gal_left = 0
                gallery_only.append(vid)
            elif len(refusal_pool) < ref_series_goal:
                for cam in cams:
                    refusal_pool.extend(group_series(by_vid_cam[(vid, cam)], V))
                refusal.append(vid)
            else:
                # квоты закрыты внутри кластера — остаток кластера в gallery-only
                # нельзя (галерея полна); отдаём кадры в val_unused как есть
                gallery_only.append(vid)  # помечаем val-стороной без ролей
            val_vids.add(vid)

    kq, kq_series, kq_second = select_series(rng, known_pool, s_known, n_known, V)
    rq, rq_series, rq_second = select_series(rng, refusal_pool, s_ref, n_ref, V)

    role = {}
    for i, r in enumerate(rows):
        role[i] = "train_fit" if r["vehicle_id"] not in val_vids else "val_unused"
    for i in gallery_frames:
        role[i] = "val_gallery"
    for i in kq + rq:
        assert role[i] == "val_unused", "кадр запроса пересёкся с галереей"
        role[i] = "val_query"

    # --- внутренние проверки до записи (генератор не доверяет сам себе) ---
    gal_vids = {rows[i]["vehicle_id"] for i in gallery_frames}
    gal_vid_cams = defaultdict(set)
    for i in gallery_frames:
        gal_vid_cams[rows[i]["vehicle_id"]].add(rows[i]["camera_id"])
    for i in kq:
        v, c = rows[i]["vehicle_id"], rows[i]["camera_id"]
        assert v in gal_vids and gal_vid_cams[v] - {c}, "нет кросс-камерной пары"
    for i in rq:
        assert rows[i]["vehicle_id"] not in gal_vids, "отказной запрос имеет пару"
    fit_vids = {r["vehicle_id"] for i, r in enumerate(rows) if role[i] == "train_fit"}
    assert not (fit_vids & val_vids), "утечка идентичностей"
    assert len(gallery_frames) == args.n_gallery and len(kq) + len(rq) == args.n_query

    os.makedirs(args.out, exist_ok=True)
    header = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]

    def write_rows(name, idxs, extra=None):
        path = os.path.join(args.out, name)
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header + ([extra[0]] if extra else []))
            for i in sorted(idxs):
                r = rows[i]
                vals = [r[c] for c in header]
                if extra:
                    vals.append(extra[1](i))
                w.writerow(vals)
        return path

    query_set = set(kq)
    files = [
        write_rows("train_fit.csv", [i for i in role if role[i] == "train_fit"]),
        write_rows("val_query.csv", kq + rq,
                   ("has_mate", lambda i: 1 if i in query_set else 0)),
        write_rows("val_gallery.csv", gallery_frames),
    ]
    apath = os.path.join(args.out, "split_assignment.csv")
    with open(apath, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_id", "role"])
        for i in range(len(rows)):
            w.writerow([rows[i]["image_id"], role[i]])
    files.append(apath)

    same_cam_mate = sum(
        1 for i in kq
        if rows[i]["camera_id"] in gal_vid_cams[rows[i]["vehicle_id"]])
    manifest = {
        "params": {k: getattr(args, k.replace("-", "_")) for k in
                   ["seed", "n_query", "n_gallery", "target_query_series", "p_refusal"]},
        "counts": {
            "train_fit": sum(1 for i in role if role[i] == "train_fit"),
            "val_query": len(kq) + len(rq),
            "val_query_with_mate": len(kq),
            "val_query_refusal": len(rq),
            "val_gallery": len(gallery_frames),
            "val_unused": sum(1 for i in role if role[i] == "val_unused"),
            "identities": {
                "train_fit": len(fit_vids), "val_total": len(val_vids),
                "paired": len(paired), "refusal": len(refusal),
                "gallery_only_or_unused": len(gallery_only)},
            "query_series_selected": {"known": kq_series, "refusal": rq_series},
            "second_frames_added": {"known": kq_second, "refusal": rq_second},
            "queries_with_same_camera_mate_in_gallery": same_cam_mate,
        },
        "sha256": {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()
                   for p in files},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False, sort_keys=True)
    print(json.dumps(manifest["counts"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
