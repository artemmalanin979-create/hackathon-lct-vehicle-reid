#!/usr/bin/env python3
"""Намеренная порча сплита четырьмя способами — доказательство, что проверяльщик
ловит нарушения. Каждый вариант меняет МИНИМУМ, чтобы упало ровно целевое свойство.

  break1_identity_leak   кадр вал-идентичности уходит в train_fit → P1
  break2_no_cross_mate   у идентичности с 2 камерами в галерее удаляется кадр
                         второй камеры → у её запроса остаётся только
                         одноместное совпадение → P2
  break3_split_md5_pair  одна идентичность md5-пары целиком уезжает в val_unused,
                         вторая остаётся в train_fit → пара по разные стороны → P4
                         (утечки P1 при этом НЕТ — идентичность уехала целиком)
  break4_fake_refusal    кадр «отказной» идентичности добавляется в галерею с
                         камеры, отличной от камер её запросов → has_mate=0 лжёт
                         → P3 (P2 при этом выполняется — совпадение кросс-камерное)

Выход: out/corrupted/<name>/ (без manifest.json) + corruption_log.json.
"""
import csv
import json
import os
from collections import defaultdict

import common as C

SRC = C.SPLIT_DIR
DST = os.path.join(C.OUT, "corrupted")
HEADER = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]


def load_split():
    with open(os.path.join(SRC, "split_assignment.csv"), newline="") as f:
        amap = {r["image_id"]: r["role"] for r in csv.DictReader(f)}
    with open(os.path.join(SRC, "val_query.csv"), newline="") as f:
        has_mate = {r["image_id"]: r["has_mate"] for r in csv.DictReader(f)}
    return amap, has_mate


def write_variant(name, amap, has_mate, note):
    rows = C.load_train()
    d = os.path.join(DST, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "split_assignment.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_id", "role"])
        for r in rows:
            w.writerow([r["image_id"], amap[r["image_id"]]])
    for fname, role in [("train_fit.csv", "train_fit"),
                        ("val_query.csv", "val_query"),
                        ("val_gallery.csv", "val_gallery")]:
        with open(os.path.join(d, fname), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            extra = ["has_mate"] if role == "val_query" else []
            w.writerow(HEADER + extra)
            for r in rows:
                if amap[r["image_id"]] != role:
                    continue
                vals = [r[c] for c in HEADER]
                if extra:
                    vals.append(has_mate.get(r["image_id"], "0"))
                w.writerow(vals)
    return {"variant": name, "note": note}


def main():
    rows = C.load_train()
    by_id = {r["image_id"]: r for r in rows}
    vid_frames = defaultdict(list)
    for r in rows:
        vid_frames[r["vehicle_id"]].append(r["image_id"])
    md5_vids = {by_id[i]["vehicle_id"] for p in C.MD5_PAIRS_TRAIN for i in p}
    log = []

    # --- break1: утечка идентичности ---
    amap, hm = load_split()
    victim = None
    for r in rows:  # первый val_unused кадр идентичности вне md5-кластеров
        if amap[r["image_id"]] == "val_unused" and r["vehicle_id"] not in md5_vids:
            victim = r
            break
    amap[victim["image_id"]] = "train_fit"
    log.append(write_variant(
        "break1_identity_leak", amap, hm,
        f"кадр {victim['image_id']} (vehicle_id={victim['vehicle_id']}) переложен "
        f"из val_unused в train_fit — идентичность теперь по обе стороны"))

    # --- break2: запрос без кросс-камерного совпадения ---
    amap, hm = load_split()
    gal_cams = defaultdict(set)
    for i, ro in amap.items():
        if ro == "val_gallery":
            gal_cams[by_id[i]["vehicle_id"]].add(by_id[i]["camera_id"])
    q_cams = defaultdict(set)
    for i, ro in amap.items():
        if ro == "val_query":
            q_cams[by_id[i]["vehicle_id"]].add(by_id[i]["camera_id"])
    victim = None
    for i, ro in sorted(amap.items()):
        if ro != "val_gallery":
            continue
        v, c = by_id[i]["vehicle_id"], by_id[i]["camera_id"]
        if len(gal_cams[v]) == 2 and q_cams.get(v) and (gal_cams[v] - {c}) <= q_cams[v]:
            victim = (i, v, c)  # удаляем кадр камеры c; остаётся камера, с которой
            break               # есть запрос → у того запроса только одноместная пара
    i, v, c = victim
    amap[i] = "val_unused"
    log.append(write_variant(
        "break2_no_cross_mate", amap, hm,
        f"из галереи убран кадр {i} (vehicle_id={v}, camera_id={c}); у идентичности "
        f"остаётся одна камера галереи, совпадающая с камерой её запроса"))

    # --- break3: разрыв md5-пары ---
    amap, hm = load_split()
    victim = None
    for a, b in C.MD5_PAIRS_TRAIN:
        va, vb = by_id[a]["vehicle_id"], by_id[b]["vehicle_id"]
        if amap[a] == "train_fit" and amap[b] == "train_fit" and va != vb:
            # va уезжает целиком (без утечки P1); вторая половина пары остаётся
            if all(amap[x] == "train_fit" for x in vid_frames[va]):
                victim = (a, b, va)
                break
    a, b, va = victim
    for x in vid_frames[va]:
        amap[x] = "val_unused"
    log.append(write_variant(
        "break3_split_md5_pair", amap, hm,
        f"идентичность {va} (все {len(vid_frames[va])} кадров, включая копию {a}) "
        f"переведена в val_unused; копия {b} осталась в train_fit — физический кадр "
        f"по обе стороны"))

    # --- break4: «отказные» на самом деле с парой ---
    amap, hm = load_split()
    refusal_vids = {by_id[i]["vehicle_id"] for i, m in hm.items() if m == "0"}
    victim = None
    for i, ro in sorted(amap.items()):
        if ro != "val_unused":
            continue
        v, c = by_id[i]["vehicle_id"], by_id[i]["camera_id"]
        if v in refusal_vids and c not in q_cams[v]:
            victim = (i, v, c)  # камера отлична от камер запросов → P2 не заденет
            break
    i, v, c = victim
    amap[i] = "val_gallery"
    log.append(write_variant(
        "break4_fake_refusal", amap, hm,
        f"кадр {i} (vehicle_id={v}, camera_id={c}) добавлен в галерею; запросы этой "
        f"идентичности объявлены отказными (has_mate=0), но пара теперь существует"))

    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, "corruption_log.json"), "w") as f:
        json.dump(log, f, indent=1, ensure_ascii=False)
    print(json.dumps(log, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
