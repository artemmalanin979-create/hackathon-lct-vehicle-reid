#!/usr/bin/env python3
"""Готовые иллюстрации для показа: пара кропов, поверх — карта влияния, подпись.

Показывается ВХОД СЕТИ (208x208 после resize) — то, что модель видит на самом деле.
Шкала у обеих половин пары общая, её концы подписаны числом в единицах Δcos.
Каждая подпись несёт число, а не эпитет.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

JOB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JOB / "scripts"))
import occl  # noqa: E402
import render as R  # noqa: E402
from mask_ops import fill, controls  # noqa: E402

PICS = JOB / "pics"
PICS.mkdir(exist_ok=True)
qm, gm = occl.read_split("val_query"), occl.read_split("val_gallery")
S = {s: json.load(open(JOB / f"out/stats_{s}.json"))["rows"] for s in ("cases", "ref")}
M = {s: np.load(JOB / f"out/maps_{s}.npz") for s in ("cases", "ref")}
CN = {s: json.load(open(JOB / f"out/contrib_{s}.json"))["rows"] for s in ("cases", "ref")}
CM = {s: np.load(JOB / f"out/contrib_{s}.npz") for s in ("cases", "ref")}
PL = {s: {(r["case"], r["side"]): r for r in json.load(open(JOB / f"out/plate_{s}.json"))["rows"]}
      for s in ("cases", "ref")}

manifest = []


def to208(g):
    return np.asarray(Image.fromarray(g.astype(np.float32), mode="F")
                      .resize((occl.INPUT, occl.INPUT), Image.BILINEAR)).astype(np.float64)


def save(name, title, subtitle, cells, scale, footer, cb_note="", size=260,
         show_cb=True):
    sh = R.sheet(title, subtitle, cells, scale, footer, size=size, cb_note=cb_note,
                 show_cb=show_cb)
    sh.save(PICS / f"{name}.png")
    manifest.append({"file": f"pics/{name}.png", "title": title, "subtitle": subtitle,
                     "footer": footer, "size": list(sh.size)})
    print("записано", name, sh.size)


def qimg(st, k):
    return occl.net_input(qm[S[st][k]["qi"]])


def gimg(st, k, key="gi"):
    return occl.net_input(gm[S[st][k][key]])


def pair_cells(st, k, size=260, with_contrib=False, qcap="запрос", gcap="кандидат"):
    """[запрос] [запрос+карта] [кандидат+карта] [кандидат] (+ вклад)."""
    r = S[st][k]
    qi_, gi_ = qimg(st, k), gimg(st, k)
    wq, wg = M[st][f"{k}_wq"], M[st][f"{k}_wg"]
    sc = max(np.abs(wq).max(), np.abs(wg).max())
    bq, vq = occl.argmax_window(wq)
    bg, vg = occl.argmax_window(wg)
    cells = [
        (R.tile(qi_, size=size), qcap, f"v{r['vid_q']}, кам. {qm[r['qi']]['camera_id']}"),
        (R.tile(qi_, occl.windows_to_map(wq), sc, box=bq, size=size),
         "карта запроса", f"макс. падение {vq:+.3f} в рамке"),
        (R.tile(gi_, occl.windows_to_map(wg), sc, box=bg, size=size),
         "карта кандидата", f"макс. падение {vg:+.3f} в рамке"),
        (R.tile(gi_, size=size), gcap, f"v{r['vid_g']}, кам. {gm[r['gi']]['camera_id']}"),
    ]
    if with_contrib:
        cq, cg = CM[st][f"{k}_cq"], CM[st][f"{k}_cg"]
        sc2 = max(np.abs(cq).max(), np.abs(cg).max())
        cells.insert(2, (R.tile(qi_, to208(cq), sc2, size=size), "вклад, запрос",
                         f"точное разложение, макс {cq.max():+.4f}"))
        cells.insert(3, (R.tile(gi_, to208(cg), sc2, size=size), "вклад, кандидат",
                         f"точное разложение, макс {cg.max():+.4f}"))
    return cells, sc


# ---------------------------------------------------------------- 00 метод ----
k = 4
r = S["cases"][k]
qi_ = qimg("cases", k)
wq = M["cases"][f"{k}_wq"]
b, v = occl.argmax_window(wq)
one = fill(qi_, b, "ring")
d = ImageDraw.Draw(Image.fromarray(one))
cq = CM["cases"][f"{k}_cq"]
cells = [
    (R.tile(qi_, size=260), "1. вход сети", "кроп по рамке, resize 208×208"),
    (R.tile(one, box=b, size=260, boxcol=(227, 73, 72)), "2. закрываем один участок",
     "48×48, заливка средним цветом кольца"),
    (R.tile(qi_, occl.windows_to_map(wq), np.abs(wq).max(), box=b, size=260),
     "3. карта из 121 такого замера", f"в рамке близость упала на {v:+.3f}"),
    (R.tile(qi_, to208(cq), np.abs(cq).max(), size=260), "4. второй метод: точный вклад",
     "одно прохождение сети, 13×13"),
]
save("00_method", "Как устроено объяснение: чувствительность к перекрытию",
     "закрываем участок входа, смотрим, насколько упала близость именно к этому кандидату; "
     "справа — независимая проверка точным разложением оценки",
     cells, float(np.abs(wq).max()),
     f"cos пары = {r['cos']:.3f}",
     "белая/красная рамка — самое влиятельное окно")

# ------------------------------------------------------------- 01-03 двойники --
c0 = {r["case"]: r for r in CN["cases"]}
for name, k, extra in (("01_twin_camry", 4, "две чёрные Camry, вид в фас"),
                       ("02_twin_chery", 0, "два Chery, вид в корму")):
    r = S["cases"][k]
    cells, sc = pair_cells("cases", k, size=230, with_contrib=True)
    save(name, f"Двойники: почему сервис их путает — {extra}",
         f"{r['note']}; взаимная путаница: запрос A даёт B, запрос B даёт A",
         cells, sc,
         f"cos к двойнику {r['cos']:.3f}, к своей паре {r['cos_alt']:.3f}",
         f"согласие двух методов r = {c0[k]['corr_win_q']:+.2f} (запрос)", size=230)

# 03: пара-специфичность — карта почти не меняется при смене кандидата
k = 4
r = S["cases"][k]
qi_ = qimg("cases", k)
wq, wa = M["cases"][f"{k}_wq"], M["cases"][f"{k}_wq_alt"]
sc = max(np.abs(wq).max(), np.abs(wa).max())
cells = [
    (R.tile(gimg("cases", k), size=230), "двойник (чужой top-1)", f"v{r['vid_g']}"),
    (R.tile(qi_, occl.windows_to_map(wq), sc, box=occl.argmax_window(wq)[0], size=230),
     "карта запроса против двойника", f"макс. {wq.max():+.3f}"),
    (R.tile(qi_, occl.windows_to_map(wa), sc, box=occl.argmax_window(wa)[0], size=230),
     "карта против своей машины", f"макс. {wa.max():+.3f}"),
    (R.tile(gimg("cases", k, "gi_alt"), size=230), "своя пара (верный ответ)", f"v{r['vid_q']}"),
]
_agg = json.load(open(JOB / "out/aggregate.json"))
_ca = json.load(open(JOB / "out/contrib_aggregate.json"))
save("03_twin_specificity",
     "Предел признака внешности: за двойника и за свою машину модель держится за одно и то же",
     "одна и та же картинка запроса, два разных кандидата — карта почти не меняется",
     cells, sc,
     f"здесь корреляция двух карт r = {r['corr_alt']:+.2f}; это верхний край",
     f"по всем {_agg['corr_alt']['n']} таким парам медиана r = {_agg['corr_alt']['median']:+.2f} "
     f"(перекрытие) и {_ca['contrib_corr_alt']['median']:+.2f} (точный вклад)", size=230)

# ------------------------------------------------- 04-05 успешные трудные ------
for name, k, extra in (("04_hardok_viewpoint", 6, "смена ракурса: фас против кормы"),
                       ("05_hardok_light", 8, "смена освещения: светлый кадр против тёмного")):
    r = S["cases"][k]
    cells, sc = pair_cells("cases", k, size=260)
    save(name, f"Трудный случай, найденный верно — {extra}", r["note"], cells, sc,
         f"cos = {r['cos']:.3f} (верный ответ на 1-м месте)",
         f"негативный контроль: против случайной машины cos = {r['cos_null']:.3f}")

# ------------------------------------------------- 06-07 испорченный кроп ------
for name, k, ttl in (
        ("06_badcrop_label", 12, "Рамка стоит на соседней машине — карта это показывает"),
        ("07_badcrop_error", 18, "Ошибка от испорченного кропа: модель смотрит не на ту машину")):
    r = S["cases"][k]
    cells, sc = pair_cells("cases", k, size=260,
                           gcap="кандидат" if k != 18 else "кандидат (чужая машина)")
    save(name, ttl, r["note"], cells, sc,
         f"cos = {r['cos']:.3f}",
         "синее — участок, закрытие которого ПОВЫШАЕТ близость")

# ---------------------------------------------------------------- 08 пластина --
sel = [(k, "q") for k in (4, 6, 9)] + [(2, "g")]
cells = []
dnums = []
for k, side in sel:
    p = PL["cases"].get((k, side))
    if not p:
        continue
    row = qm[S["cases"][k]["qi"]] if side == "q" else gm[S["cases"][k]["gi"]]
    arr = occl.crop_raw(row)
    box = tuple(p["box"])
    ctl = controls(box, arr.shape[:2], 1000 + (S["cases"][k]["qi"] if side == "q"
                                               else S["cases"][k]["gi"]))
    vis = Image.fromarray(arr).resize((260, 260), Image.LANCZOS)
    dd = ImageDraw.Draw(vis)
    kx, ky = 260 / arr.shape[1], 260 / arr.shape[0]
    for bb, col in ((box, (227, 73, 72)), (ctl["shift"], (42, 120, 214)),
                    (ctl["side"], (27, 175, 122))):
        dd.rectangle([bb[0] * kx, bb[1] * ky, (bb[0] + bb[2]) * kx, (bb[1] + bb[3]) * ky],
                     outline=col, width=3)
    cells.append((vis, f"случай {k}, {'запрос' if side == 'q' else 'кандидат'}",
                  f"Δcos: пластина {p['delta']['plate_ring']:+.4f}\n"
                  f"сдвиг {p['delta']['shift_ring']:+.4f} · "
                  f"вбок {p['delta']['side_ring']:+.4f}"))
    dnums.append(p["plate_minus_ctl_mean"])
agg = json.load(open(JOB / "out/aggregate.json"))
save("08_plate_control", "Зона номерной пластины против контролей равной площади",
     "красный — бокс пластины, синий и зелёный — тот же прямоугольник, "
     "перенесённый туда, где пластины нет",
     cells, 0.02,
     f"по {agg['plate']['n_found']} кропам: Δ(пластина) − Δ(контроль) = "
     f"{agg['plate']['paired_diff_mean']:+.4f} "
     f"[{agg['plate']['paired_diff_ci'][0]:+.4f}; {agg['plate']['paired_diff_ci'][1]:+.4f}]",
     f"доля кропов, где пластина дороже контроля: "
     f"{agg['plate']['share_plate_hotter']:.2f} — то есть монетка",
     show_cb=False)

# 09: пластина на карте
cells = []
for k, side in [(4, "q"), (6, "g"), (1, "q"), (5, "g")]:
    p = PL["cases"].get((k, side))
    if not p:
        continue
    st = "cases"
    img = qimg(st, k) if side == "q" else gimg(st, k)
    w = M[st][f"{k}_w{side}"]
    arr_hw = p["crop_hw"]
    bx, by, bw, bh = p["box"]
    kx, ky = occl.INPUT / arr_hw[1], occl.INPUT / arr_hw[0]
    bi = (bx * kx, by * ky, bw * kx, bh * ky)
    t = R.tile(img, occl.windows_to_map(w), float(np.abs(w).max()), size=260)
    dd = ImageDraw.Draw(t)
    kk = 260 / occl.INPUT
    dd.rectangle([bi[0] * kk, bi[1] * kk, (bi[0] + bi[2]) * kk, (bi[1] + bi[3]) * kk],
                 outline=(11, 11, 11), width=3)
    cells.append((t, f"случай {k}, {'запрос' if side == 'q' else 'кандидат'}",
                  f"в зоне пластины {p['map_plate_mean']:+.4f}\n"
                  f"по всей карте {p['map_all_mean']:+.4f}"))
mr = agg["plate"]["map_reading"]
ca = json.load(open(JOB / "out/contrib_aggregate.json"))
save("09_plate_on_map", "Почему по скользящей карте вопрос про пластину НЕ решается",
     f"чёрная рамка — найденная пластина ({agg['plate']['box_area_frac_med']*100:.1f} % площади); "
     "окно карты 48×48 — это 5,3 %, в него всегда попадает бампер вокруг",
     cells, 0.05,
     f"по {agg['plate']['n_found']} кропам: в зоне пластины {mr['map_plate_mean']:+.4f}, "
     f"по карте {mr['map_all_mean']:+.4f} — разность {mr['plate_minus_all']:+.4f}",
     f"на точной карте (ячейка 16 px) разность {ca['contrib_plate']['diff']:+.5f}, "
     f"а точечный тест по боксу — {agg['plate']['paired_diff_mean']:+.4f}")

# ------------------------------------------------------- 10 негативный контроль
k = 0
r = S["ref"][k]
qi_ = qimg("ref", k)
wq, wn = M["ref"][f"{k}_wq"], M["ref"][f"{k}_wq_null"]
sc = max(np.abs(wq).max(), np.abs(wn).max())
cells = [
    (R.tile(gimg("ref", k), size=230), "верный кандидат", f"v{r['vid_g']}"),
    (R.tile(qi_, occl.windows_to_map(wq), sc, box=occl.argmax_window(wq)[0], size=230),
     "карта против верного", f"макс. {wq.max():+.3f}, cos {r['cos']:.3f}"),
    (R.tile(qi_, occl.windows_to_map(wn), sc, box=occl.argmax_window(wn)[0], size=230),
     "карта против случайной машины", f"макс. {wn.max():+.3f}, cos {r['cos_null']:.3f}"),
    (R.tile(occl.net_input(gm[r["null_gi"]]), size=230), "случайный кандидат", "негативный контроль"),
]
save("10_null_control", "Негативный контроль: карта против случайной машины",
     "если бы карта показывала «что вообще заметно на картинке», а не «что связывает пару», "
     "обе карты были бы одинаковыми",
     cells, sc,
     f"корреляция карт r = {r['corr_null']:+.2f}",
     f"по {agg['corr_null']['n']} парам медиана r = {agg['corr_null']['median']:+.2f}", size=230)

# --------------------------------------------------------- 11 обычные случаи ---
cells = []
for k in (2, 5, 9, 14):
    r = S["ref"][k]
    w = M["ref"][f"{k}_wq"]
    cells.append((R.tile(qimg("ref", k), occl.windows_to_map(w), float(np.abs(w).max()),
                         box=occl.argmax_window(w)[0], size=260),
                  f"cos = {r['cos']:.3f}", f"макс. падение {w.max():+.3f}"))
bm = agg["border_minus_center"]
save("11_typical", "Обычные верные совпадения: куда карта ложится в типичном случае",
     "случайная выборка из группы верно найденных пар (rank-1), шкала у каждой своя",
     cells, 0.05,
     f"рамка кропа против центра по карте перекрытия: {bm['mean']:+.4f} "
     f"[{bm['ci'][0]:+.4f}; {bm['ci'][1]:+.4f}] — не отличимо от нуля",
     f"на точной карте вклада рамка холоднее середины: "
     f"{_ca['contrib_frame_vs_inner']['diff']:+.5f} "
     f"[{_ca['contrib_frame_vs_inner']['ci'][0]:+.5f}; "
     f"{_ca['contrib_frame_vs_inner']['ci'][1]:+.5f}]")

(JOB / "out/pics_manifest.json").write_text(
    json.dumps({"n": len(manifest), "pics": manifest}, ensure_ascii=False, indent=1))
print("иллюстраций:", len(manifest))

# -------------------------------------- 12 профиль по высоте кропа (график) ----
import linechart  # noqa: E402

prof = agg["row_profile"]
xs = ["верх"] + [""] * 4 + ["середина"] + [""] * 4 + ["низ"]
ch = linechart.chart(
    [("обычные верные пары", prof["ref"]),
     ("двойники", prof["twin"]),
     ("испорченный кроп", prof["badcrop"])],
    xs,
    "Куда по высоте кропа смотрит модель",
    "средний Δcos по строке карты влияния: насколько падает близость, "
    "если закрыть полосу на этой высоте",
    "Δcos при закрытии участка",
    note=f"строк карты: {occl.NG}; окно 48 px, шаг 16")
ch.save(PICS / "12_row_profile.png")
manifest.append({"file": "pics/12_row_profile.png",
                 "title": "Куда по высоте кропа смотрит модель",
                 "subtitle": "средний Δcos по строке карты влияния",
                 "footer": "", "size": list(ch.size)})
print("записано 12_row_profile", ch.size)

(JOB / "out/pics_manifest.json").write_text(
    json.dumps({"n": len(manifest), "pics": manifest}, ensure_ascii=False, indent=1))
print("иллюстраций:", len(manifest))
