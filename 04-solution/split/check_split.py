#!/usr/bin/env python3
"""Проверяльщик сплита. Задача — УПАСТЬ (exit 1), если нарушено любое свойство.

Свойства (нумерация = BRIEF):
  S0  структура: split_assignment покрывает train.csv ровно один раз; файлы ролей
      согласованы с assignment; содержимое строк совпадает с train.csv.
  P1  идентичности train_fit и вал-части (query+gallery+unused) не пересекаются.
  P2  у каждого запроса, чья идентичность есть в галерее, есть верное совпадение
      с ДРУГОЙ камеры (по настоящему camera_id).
  P3  отказные запросы существуют; колонка has_mate строго равна факту
      "идентичность есть в галерее" (в обе стороны).
  P4  пары скопированных кадров (побайтово одинаковые файлы) не расходятся по
      разные стороны train_fit / val. Пары либо перепроверяются побайтово по
      известному списку, либо (--full-scan) пересчитываются md5 всех кадров train.
  P5  sha256 файлов совпадают с manifest.json (если манифест есть).

Запуск: python3 check_split.py [SPLIT_DIR] [--full-scan]
Всё считается заново из data/train.csv и файлов сплита; генератору не доверяем.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FAILS = []


def fail(prop, msg):
    FAILS.append((prop, msg))
    print(f"FAIL[{prop}] {msg}")


def ok(prop, msg):
    print(f"  ok [{prop}] {msg}")


def read(path, name):
    if not os.path.exists(path):
        fail("S0", f"нет файла {name}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", nargs="?", default=C.SPLIT_DIR)
    ap.add_argument("--full-scan", action="store_true",
                    help="пересчитать md5 всех кадров train (медленно, независимо)")
    args = ap.parse_args()
    sd = args.split_dir

    train = C.load_train()
    by_id = {r["image_id"]: r for r in train}

    assign = read(os.path.join(sd, "split_assignment.csv"), "split_assignment.csv")
    fit = read(os.path.join(sd, "train_fit.csv"), "train_fit.csv")
    query = read(os.path.join(sd, "val_query.csv"), "val_query.csv")
    gallery = read(os.path.join(sd, "val_gallery.csv"), "val_gallery.csv")
    if FAILS:
        finish()

    # ---- S0: структура ----
    roles = {"train_fit", "val_query", "val_gallery", "val_unused"}
    amap = {}
    for r in assign:
        if r["image_id"] in amap:
            fail("S0", f"image_id дважды в assignment: {r['image_id']}")
        if r["role"] not in roles:
            fail("S0", f"неизвестная роль {r['role']!r} у {r['image_id']}")
        amap[r["image_id"]] = r["role"]
    if set(amap) != set(by_id):
        d1 = set(by_id) - set(amap)
        d2 = set(amap) - set(by_id)
        fail("S0", f"assignment не совпадает с train.csv: нет {len(d1)}, лишних {len(d2)}")
    for name, rows_, role in [("train_fit.csv", fit, "train_fit"),
                              ("val_query.csv", query, "val_query"),
                              ("val_gallery.csv", gallery, "val_gallery")]:
        ids = [r["image_id"] for r in rows_]
        if len(ids) != len(set(ids)):
            fail("S0", f"{name}: повторы image_id")
        want = {i for i, ro in amap.items() if ro == role}
        if set(ids) != want:
            fail("S0", f"{name}: множество image_id не равно роли {role} "
                       f"в assignment ({len(set(ids))} vs {len(want)})")
        for r in rows_:
            src = by_id.get(r["image_id"])
            if src and any(r[c] != src[str(c)] if c in ("x", "y", "w", "h")
                           else int(r[c]) != src[c]
                           for c in ["x", "y", "w", "h", "vehicle_id", "camera_id"]):
                fail("S0", f"{name}: строка {r['image_id']} расходится с train.csv")
    if query and "has_mate" not in query[0]:
        fail("S0", "val_query.csv: нет колонки has_mate")
    if not FAILS:
        ok("S0", f"структура согласована: {len(fit)} fit / {len(query)} query / "
                 f"{len(gallery)} gallery / "
                 f"{sum(1 for r in amap.values() if r == 'val_unused')} unused")

    # ---- P1: непересечение идентичностей ----
    fit_vids = {int(by_id[i]["vehicle_id"]) for i, ro in amap.items() if ro == "train_fit"}
    val_vids = {int(by_id[i]["vehicle_id"]) for i, ro in amap.items() if ro != "train_fit"}
    inter = fit_vids & val_vids
    if inter:
        fail("P1", f"идентичности в обеих частях: {sorted(inter)[:10]}"
                   f"{' …' if len(inter) > 10 else ''} (всего {len(inter)})")
    else:
        ok("P1", f"{len(fit_vids)} идентичностей train_fit и {len(val_vids)} вал — "
                 f"пересечение пусто")

    # ---- P2: кросс-камерное совпадение ----
    gal_cams = defaultdict(set)
    for r in gallery:
        gal_cams[int(r["vehicle_id"])].add(int(r["camera_id"]))
    bad = []
    for r in query:
        v, c = int(r["vehicle_id"]), int(r["camera_id"])
        if v in gal_cams and not (gal_cams[v] - {c}):
            bad.append((r["image_id"], v, c))
    if bad:
        for iid, v, c in bad[:5]:
            fail("P2", f"запрос {iid} (vehicle_id={v}, camera_id={c}): совпадения в "
                       f"галерее только с той же камеры {sorted(gal_cams[v])}")
        if len(bad) > 5:
            fail("P2", f"… и ещё {len(bad) - 5} таких запросов")
    else:
        n_with = sum(1 for r in query if int(r["vehicle_id"]) in gal_cams)
        ok("P2", f"у всех {n_with} запросов с парой есть совпадение с другой камеры")

    # ---- P3: отказные запросы ----
    n_refusal = 0
    for r in query:
        v = int(r["vehicle_id"])
        declared = r.get("has_mate") == "1"
        actual = v in gal_cams
        if declared != actual:
            fail("P3", f"запрос {r['image_id']} (vehicle_id={v}): has_mate="
                       f"{int(declared)}, но идентичность "
                       f"{'ЕСТЬ' if actual else 'отсутствует'} в галерее")
        n_refusal += int(not actual)
    if n_refusal == 0:
        fail("P3", "нет ни одного запроса без пары в галерее — режим отказа "
                   "проверить не на чем")
    if not any(p == "P3" for p, _ in FAILS):
        ok("P3", f"{n_refusal} отказных запросов; колонка has_mate совпадает с фактом")

    # ---- P4: скопированные кадры ----
    if args.full_scan:
        md5s = defaultdict(list)
        for iid in sorted(by_id):
            md5s[C.md5_file(os.path.join(C.IMAGES, iid + ".jpg"))].append(iid)
        pairs = [tuple(sorted(v)) for v in md5s.values() if len(v) > 1]
        print(f"  (full-scan: найдено {len(pairs)} пар побайтовых дублей в train)")
    else:
        pairs = []
        for a, b in C.MD5_PAIRS_TRAIN:
            ha = C.md5_file(os.path.join(C.IMAGES, a + ".jpg"))
            hb = C.md5_file(os.path.join(C.IMAGES, b + ".jpg"))
            if ha != hb:
                fail("P4", f"список пар устарел: {a} и {b} не совпадают побайтово")
            else:
                pairs.append((a, b))
    side = lambda ro: "train_fit" if ro == "train_fit" else "val"
    p4_bad = False
    for a, b in pairs:
        sa, sb = side(amap[a]), side(amap[b])
        if sa != sb:
            p4_bad = True
            fail("P4", f"пара скопированных кадров разошлась: {a} ({amap[a]}) vs "
                       f"{b} ({amap[b]}) — один физический кадр по обе стороны")
    if not p4_bad and not any(p == "P4" for p, _ in FAILS):
        ok("P4", f"{len(pairs)} пар побайтовых дублей train — обе половины каждой "
                 f"пары на одной стороне")

    # ---- P5: манифест ----
    mpath = os.path.join(sd, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)
        for name, want in manifest.get("sha256", {}).items():
            got = hashlib.sha256(open(os.path.join(sd, name), "rb").read()).hexdigest()
            if got != want:
                fail("P5", f"{name}: sha256 {got[:12]}… не совпадает с манифестом "
                           f"{want[:12]}…")
        if not any(p == "P5" for p, _ in FAILS):
            ok("P5", f"sha256 всех {len(manifest.get('sha256', {}))} файлов совпадают "
                     f"с манифестом")
    else:
        print("  warn[P5] manifest.json отсутствует — проверка пропущена")

    finish()


def finish():
    if FAILS:
        props = sorted({p for p, _ in FAILS})
        print(f"\nПРОВАЛ: нарушены свойства {', '.join(props)} "
              f"({len(FAILS)} сообщений)")
        sys.exit(1)
    print("\nOK: все свойства выполнены")
    sys.exit(0)


if __name__ == "__main__":
    main()
