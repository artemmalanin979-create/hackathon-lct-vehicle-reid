#!/usr/bin/env python3
"""Render measured CSV/JSON results with system Matplotlib; no metric formulas."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

# Каталог этапа (scripts/ лежит внутри него): туда же писались REPORT.md,
# графики и results/ в исходном прогоне.
ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results"
SUMMARY = json.loads((OUT / "summary.json").read_text())
ROBUST = json.loads((OUT / "robustness.json").read_text())
REC = json.loads((OUT / "recommendation.json").read_text())
MANIFEST = json.loads((OUT / "manifest.json").read_text())
BASE = SUMMARY["market_absolute_presence"]
GAP = SUMMARY["market_gap_presence"]


def csv_rows(path):
    with path.open() as f:
        return [{k: (None if v == "" else float(v)) for k,v in row.items()}
                for row in csv.DictReader(f)]


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def metric_row(label, row):
    return [label, fmt(row["threshold"],6), *[fmt(row[k]) for k in ("f1","precision","recall","tnr")]]


def render_plots():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.grid":True, "grid.alpha":.18, "figure.dpi":140,
                         "savefig.facecolor":"white"})
    fig, axes = plt.subplots(2,2,figsize=(13,9.5),layout="constrained")
    palette = {"f1":"#2463a0", "precision":"#8c5d99", "recall":"#178572", "tnr":"#d15b2a"}
    for feature,ax,title in (("absolute",axes[0,0],"Абсолютная близость"),("gap",axes[0,1],"Отрыв top-1 от top-2")):
        rows = sorted(csv_rows(OUT / f"curve_market_{feature}_presence.csv"),key=lambda r:r["threshold"])
        for key,label in (("f1","F1"),("precision","Precision"),("recall","Recall"),("tnr","TNR")):
            ax.step([r["threshold"] for r in rows],[r[key] for r in rows],where="pre",lw=1.5,label=label,color=palette[key])
        ax.set(title=title,xlabel="Порог принятия (score ≥ t)",ylabel="Значение метрики",ylim=(-.02,1.02))
        ax.set_xlim((.2,1.) if feature=="absolute" else (0.,.53))
        ax.legend(loc="best",frameon=False,ncol=2)
    for feature,color,label in (("absolute","#2463a0","Абсолютная близость"),("gap","#d15b2a","Отрыв top-1 / top-2")):
        rows = csv_rows(OUT / f"curve_market_{feature}_presence.csv")
        axes[1,0].plot([r["tnr"] for r in rows],[r["f1"] for r in rows],color=color,label=label,lw=1.6)
    r=REC["actual_validation"]
    axes[1,0].scatter([r["tnr"]],[r["f1"]],marker="*",s=160,color="#152f49",zorder=5)
    axes[1,0].annotate(f"t = {r['threshold']:.6f}",(r["tnr"],r["f1"]),xytext=(.29,.54),arrowprops={"arrowstyle":"->","color":"#152f49"})
    axes[1,0].plot([0,1],[0,1],"--",color="#9aa4b0",lw=1)
    axes[1,0].set(title="Цена отказов: все рабочие точки",xlabel="TNR",ylabel="F1",xlim=(0,1),ylim=(0,1))
    axes[1,0].legend(frameon=False,loc="lower left")
    for p,color in (("0.1","#178572"),("0.25","#2463a0"),("0.4","#d15b2a")):
        rows=csv_rows(OUT / f"curve_market_absolute_presence_prior_{p}.csv")
        axes[1,1].plot([r["tnr"] for r in rows],[r["f1"] for r in rows],color=color,label=f"Отказных {float(p):.0%}")
        fixed=BASE["priors"][p]["fixed_robust_balanced"]
        axes[1,1].scatter([fixed["tnr"]],[fixed["f1"]],color=color,s=36,zorder=4)
    axes[1,1].set(title="Меняется только доля отказных",xlabel="TNR",ylabel="F1",xlim=(0,1),ylim=(0,1))
    axes[1,1].legend(frameon=False,loc="lower left")
    fig.suptitle("Калибровка отказа · presence · market · 832 запроса с парой / 278 без пары",fontsize=14)
    for extension in ("png","svg"):
        fig.savefig(ROOT / f"threshold_curves.{extension}")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    for mode,ax in zip(("presence","top1"),axes):
        for feature,color,label in (("absolute","#2463a0","Абсолютная"),("gap","#d15b2a","Отрыв")):
            rows=csv_rows(OUT / f"curve_market_{feature}_{mode}.csv")
            y=[1. if r["precision"] is None else r["precision"] for r in rows]
            auc=SUMMARY[f"market_{feature}_{mode}"]["auc_pr_trapezoid"]
            ax.plot([r["recall"] for r in rows],y,color=color,lw=1.4,label=f"{label}: AUC-PR {auc:.4f}")
        ax.set(title="Наличие пары" if mode=="presence" else "Правильная машина в top-1",
               xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1.02))
        ax.legend(frameon=False,loc="lower left")
    for extension in ("png","svg"):
        fig.savefig(ROOT / f"pr_curves.{extension}")
    plt.close(fig)


def write_report():
    selected=BASE["selected"]
    old=selected["best_f1"]
    rec=selected["robust_balanced"]
    primary_table=table(["Точка", "Порог t*", "F1", "Precision", "Recall", "TNR"],
        [metric_row("Принимать всех",selected["accept_all"]),
         metric_row("Максимум F1",old)] +
        [metric_row(f"TNR ≥ {i/10:.1f}",selected[f"tnr_{i/10:.1f}"]) for i in range(1,11)] +
        [metric_row("Максимум min(F1, TNR)",selected["balanced"]),
         metric_row("Компромисс по трём долям",rec),
         metric_row("Отказать всем",selected["reject_all"])])
    prices=[]
    for i in range(1,11):
        previous,current=selected[f"tnr_{(i-1)/10:.1f}"],selected[f"tnr_{i/10:.1f}"]
        loss=previous["f1"]-current["f1"]
        gain=current["tnr"]-previous["tnr"]
        prices.append(dict(target_from=(i-1)/10,target_to=i/10,
                           actual_tnr_from=previous["tnr"],actual_tnr_to=current["tnr"],
                           f1_from=previous["f1"],f1_to=current["f1"],
                           f1_loss=loss,actual_tnr_gain=gain,f1_cost_per_0_1_tnr=.1*loss/gain))
    with (OUT / "tnr_cost.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(prices[0]));writer.writeheader();writer.writerows(prices)
    price_table=table(["Целевой TNR", "Фактический TNR", "F1 до → после", "Потеря F1", "Цена +0,10 TNR"],
         [[f"{r['target_from']:.1f} → {r['target_to']:.1f}",f"{r['actual_tnr_from']:.4f} → {r['actual_tnr_to']:.4f}",
           f"{r['f1_from']:.4f} → {r['f1_to']:.4f}",f"{r['f1_loss']:+.5f}",f"{r['f1_cost_per_0_1_tnr']:+.5f}"] for r in prices])
    prior_table=table(["Отказных", "Порог max F1", "F1 max", "Precision", "Recall", "TNR", "F1 при старом t"],
        [[p,fmt(b["best_f1"]["threshold"],6),*[fmt(b["best_f1"][k]) for k in ("f1","precision","recall","tnr")],
          fmt(b["fixed_best_f1_actual"]["f1"])] for p,b in BASE["priors"].items()])
    stable_table=table(["Отказных", "Фиксированный t", "F1", "Precision", "Recall", "TNR", "Цена к max F1 этой доли"],
        [[p,fmt(rec["threshold"],6),*[fmt(b["fixed_robust_balanced"][k]) for k in ("f1","precision","recall","tnr")],
          fmt(b["best_f1"]["f1"]-b["fixed_robust_balanced"]["f1"])] for p,b in BASE["priors"].items()])
    with (OUT / "prior_transfer.csv").open() as f:
        transfer=list(csv.DictReader(f))
    actual=[r for r in transfer if r["feature"]=="absolute"]
    transfer_table=table(["Доля при выборе t → реальная доля", "0.10", "0.25", "0.40"],
        [[p]+[f"{float(next(r for r in actual if r['calibration_prior']==p and r['evaluation_prior']==q)['f1']):.4f}" for q in ("0.1","0.25","0.4")]
         for p in ("0.1","0.25","0.4")])
    relative_table=table(["Признак (market, presence)", "t max F1", "F1 max / TNR", "t при TNR ≥ 0.7", "F1 / Recall при TNR ≥ 0.7", "AUC-PR / AP-step"],
        [[label,fmt(b["selected"]["best_f1"]["threshold"],8),f"{b['selected']['best_f1']['f1']:.4f} / {b['selected']['best_f1']['tnr']:.4f}",
          fmt(b["selected"]["tnr_0.7"]["threshold"],8),f"{b['selected']['tnr_0.7']['f1']:.4f} / {b['selected']['tnr_0.7']['recall']:.4f}",
          f"{b['auc_pr_trapezoid']:.4f} / {b['ap_pr_step']:.4f}"] for label,b in (("s₁",BASE),("s₁ − s₂",GAP))])
    gap_priors=table(["Отказных", "t max F1 для gap", "F1", "TNR", "F1 при фиксированном компромиссе gap"],
        [[p,fmt(b["best_f1"]["threshold"],8),fmt(b["best_f1"]["f1"]),fmt(b["best_f1"]["tnr"]),fmt(b["fixed_robust_balanced"]["f1"])]
         for p,b in GAP["priors"].items()])
    semantics=table(["Режим / признак", "t max F1", "F1 max", "TNR при max F1", "F1 при TNR ≥ 0.7", "AUC-PR"],
        [[f"{mode} / {label}",fmt(b["selected"]["best_f1"]["threshold"],8),fmt(b["selected"]["best_f1"]["f1"]),
          fmt(b["selected"]["best_f1"]["tnr"]),fmt(b["selected"]["tnr_0.7"]["f1"]),fmt(b["auc_pr_trapezoid"])]
         for mode in ("presence","top1") for label,feature in (("s₁","absolute"),("s₁−s₂","gap"))
         for b in [SUMMARY[f"market_{feature}_{mode}"]]])
    policies=table(["Камера", "Собственный t компромисса", "F1", "TNR", "F1 при TNR ≥ 0.7", "AUC-PR"],
        [[policy,fmt(b["selected"]["robust_balanced"]["threshold"],8),fmt(b["selected"]["robust_balanced"]["f1"]),
          fmt(b["selected"]["robust_balanced"]["tnr"]),fmt(b["selected"]["tnr_0.7"]["f1"]),fmt(b["auc_pr_trapezoid"])]
         for policy in ("market","all_same_camera","unfiltered") for b in [SUMMARY[f"{policy}_absolute_presence"]]])
    cv_table=table(["Признак", "F1, среднее [min; max]", "TNR, среднее [min; max]", "Пороги, 2.5–97.5 перцентили"],
        [[feature,f"{b['f1']['mean']:.4f} [{b['f1']['min']:.4f}; {b['f1']['max']:.4f}]",
          f"{b['tnr']['mean']:.4f} [{b['tnr']['min']:.4f}; {b['tnr']['max']:.4f}]",
          f"[{b['calibration_thresholds']['lo']:.6f}; {b['calibration_thresholds']['hi']:.6f}]"]
         for feature,b in ROBUST["crossfit_summary"].items()])
    ranking_rows=[]
    for policy in ("market","all_same_camera"):
        b=json.loads((OUT / f"ranking_{policy}.json").read_text())
        ranking_rows.append([policy,"Полная",fmt(b["ranking_full_gallery"]["mAP"]),fmt(b["ranking_full_gallery"]["mINP"]),"—"])
        for order,r in b["ranking_top_k_by_filter_order"].items():
            ranking_rows.append([policy,order,fmt(r["mAP"]),f"не определён / {r['mINP_by_incomplete_policy']['zero_if_incomplete']:.4f}",r["num_incomplete_queries"]])
    ranking_table=table(["Камера", "Галерея/порядок", "mAP", "mINP (undefined / zero)", "Неполных списков"],ranking_rows)
    t7=selected["tnr_0.7"]
    cumulative_price=.1*(old["f1"]-t7["f1"])/(t7["tnr"]-old["tnr"])
    ci=ROBUST["bootstrap_intervals"]["absolute"]["robust_balanced"]
    diff_ci=ROBUST["bootstrap_intervals"]["differences"]
    raw_abs=SUMMARY["unfiltered_absolute_presence"]
    raw_gap=SUMMARY["unfiltered_gap_presence"]
    text=f"""# Калибровка порога отказа

## Итог и рекомендуемая рабочая точка

**Для исходного протокола `presence + market` рекомендую временный порог `{rec['threshold']!r}`:** F1 **{rec['f1']:.4f}**, precision **{rec['precision']:.4f}**, recall **{rec['recall']:.4f}**, TNR **{rec['tnr']:.4f}**. Это явный компромисс по F1 и TNR при трёх возможных долях отказных. Неизвестная формула оценки жюри из этих данных не восстанавливается.

Старый максимум F1 воспроизведён точно: **`{old['threshold']!r}`**, F1 **{old['f1']:.4f}**, TNR **{old['tnr']:.4f}**. Его низкий TNR — свойство выбранной цели оптимизации, а не ошибка перебора. При реальной доле отказных 0.10/0.25/0.40 рекомендуемый фиксированный порог даёт F1 **0.7626/0.7332/0.6930**, TNR **0.6978** во всех трёх сценариях при неизменных условных распределениях.

Если требуется именно **эмпирический TNR ≥ 0.70**, ближайшая точка — `{t7['threshold']!r}`: F1 **{t7['f1']:.4f}**, TNR **{t7['tnr']:.4f}**. Гарантии такого TNR на новом тесте нет. Если F1 требует правильной машины в top‑1, вывод о лучшем признаке меняется (§6).

## 1. Что измерено и какие допущения сделаны

- Векторы: 1110 × 512 и 750 × 512, все значения конечны. Порядок `.ids` точно совпал с CSV; пересечения кадров query/gallery нет. Косинусы рассчитаны `scores_from_embeddings` контура в float64, с его нормировкой.
- По разметке **832** запроса имеют пару и **278** не имеют; это **{100*MANIFEST['actual_refusal_share']:.6f}%** отказных, а не ровно 25%. В query {MANIFEST['known_identities']} известных и {MANIFEST['unknown_identities']} отказных идентичностей. Флаги `has_mate` пересчитаны по галерее, расхождений нет; после камерной фильтрации ни один известный запрос не потерял все верные пары.
- Основной результат использует тот же режим, что исходный порог: `presence`, где положительный класс — **в галерее существует пара**. Он не проверяет правильность принятого top‑1. `top1` дополнительно требует верной идентичности; для него также рассчитаны полные кривые.
- `market` исключает совпадение **vehicle_id И camera_id**; `all_same_camera` исключает все кадры камеры запроса. Это маски оценки по разметке. Ранжирование стабильно по исходному порядку галереи при точном равенстве score.
- Принятие: **score ≥ t**, отказ: **score < t**. Для относительного признака `s₁−s₂` берутся два **изображения** галереи после той же маски, без группировки по неизвестным идентичностям. Малый gap может быть вызван двумя правильными кадрами одной машины.
- F1, precision, recall, TNR, PR-AUC, AP-step и mINP получены существующим контуром. Для быстрого перебора его `_pr_events` даёт события и счётчики, а [адаптер](metric_adapter.py) исполняет неизменённые AST-выражения `positives`, `fn`, `refusal` из `_evaluate_full`. Собственных формул этих метрик нет. Побайтовый [снимок контура](evaluator_snapshot/reid_metrics.py) и [SHA-256 входов](results/manifest.json) приложены.

Трактовка ТЗ взята из предоставленного [context-tz.txt, раздел 9](context-tz.txt): названы F1 и TNR, их веса внутри критерия не указаны. Во внешние источники данные не отправлялись.

## 2. Вся шкала порога: F1, precision, recall, TNR

Ниже точки по всей шкале TNR, включая крайние решения. Для каждого уровня TNR выбран **наименьший представимый порог**, который достигает уровня: `nextafter` соответствующего отказного score вверх. Это сохраняет максимальный recall при таком ограничении; дополнительной подгонки F1 в этих строках нет. Из-за 278 отрицательных запросов TNR меняется шагами по 1/278, поэтому уровни иногда превышены.

{primary_table}

Примечание к t*: для чтения пороги показаны с шестью знаками; метрики посчитаны по полным значениям в CSV/JSON. Последняя строка — `nextafter(max score, +∞)`, а не округлённый максимум. При отсутствии принятых запросов precision не определён, F1 = 0. На этой выборке ниже минимального score 0.2191533 всё принимается, выше максимального 0.9525491 всё отклоняется; интервалы между соседними событиями имеют постоянные метрики.

**Полный перебор:** [1111 рабочих точек абсолютного порога](results/curve_market_absolute_presence.csv), [1111 точек gap](results/curve_market_gap_presence.csv), [равномерная сетка косинуса −1…1](results/grid_market_absolute_presence.csv), [точки для таблицы](results/selected_market_absolute_presence.csv). В полных CSV также есть TP/FP/FN/TN и обе площади PR. Ни одного возможного набора решений между соседними score не пропущено.

![Кривые порога и цена отказов](threshold_curves.png)

**Почему максимум F1 почти всё принимает.** Если принимать всех, TP = 832, FP = 278, FN = 0, F1 = **{selected['accept_all']['f1']:.4f}**, TNR = **0**. Оптимизация F1 улучшает его всего на **{old['f1']-selected['accept_all']['f1']:.5f}**: TP = 819, FP = 238, FN = 13, TN = 40. Получается 1057 принятий, 53 отказа, в том числе 13 ошибочных. **238/278 чужих запросов всё ещё принимаются.** Высокий F1 здесь обеспечен преобладанием положительного класса и не означает хороший режим отказа.

Есть промежуточные точки: при TNR = 0.5000 F1 = 0.8237; при TNR = 0.7014 F1 = 0.7300. Но одновременно высоких значений нет: максимальное `min(F1,TNR)` на текущей смеси — **{min(selected['balanced']['f1'], selected['balanced']['tnr']):.4f}**. Значит, пары F1 ≥ 0.8 и TNR ≥ 0.8 для этого признака и протокола **не существует**. «Разумный баланс» здесь означает осознанную потерю recall, а не бесплатное улучшение.

Важна точность порога: округление старого `{old['threshold']!r}` до `0.349214` исключает ещё один известный запрос: F1 становится **{selected['old_rounded']['f1']:.6f}**, recall **{selected['old_rounded']['recall']:.6f}**. Рекомендуемый порог после округления до `0.549595` даёт те же счётчики на этом вал-сплите; точное значение сохранено в [recommendation.json](results/recommendation.json).

## 3. Сколько F1 стоит каждая десятая доля TNR

Цена = **(F1 до − F1 после) × 0.10 / фактический прирост TNR**. Это разность измеренных метрик, а не новый вариант метрики. Отрицательная цена означает улучшение F1. Столбец нормировки учитывает дискретность TNR.

{price_table}

**Единой цены нет.** Переход от старого максимума F1 к TNR ≥ 0.7 стоит **{old['f1']-t7['f1']:.5f} F1** за **{t7['tnr']-old['tnr']:.5f} TNR**, в среднем **{cumulative_price:.5f} F1 на +0.10 TNR**. На участке 0.8→0.9 цена уже **{prices[8]['f1_cost_per_0_1_tnr']:.5f}**, а 0.9→1.0 — **{prices[9]['f1_cost_per_0_1_tnr']:.5f}**. Все разности и нормировки — [tnr_cost.csv](results/tnr_cost.csv).

## 4. Неизвестная доля запросов без пары

### Метод: чистое изменение доли класса

**Галерея, запросы, модель, scores и распределения внутри классов неизменны.** Для каждой заданной доли всем известным запросам присвоена одна целочисленная кратность, всем неизвестным — другая; `_pr_events` контура получает буквально повторённые события. Кратности known/unknown: **1251/416** для 0.10, **417/416** для 0.25, **417/832** для 0.40. Это точное перевзвешивание, без случайного подвыбора и потери данных. Повторения **не увеличивают статистический размер выборки**.

Предположение — меняется только вероятность класса (prior shift). Генератор `make_split.py --p-refusal` **не запускался**: он меняет также состав query/идентичностей и иногда train_fit, а в задании выданы эмбеддинги только фиксированного сплита. Это исследование доли класса, **не три новых сплита** и не проверка переноса на новые изображения. Эффекты другого состава галереи и сдвига score остаются неизвестными.

### Максимум F1 для каждой доли

{prior_table}

0.25 здесь означает ровно 25%, поэтому F1 немного отличается от основной таблицы с 278/1110 отказных. Порог максимума меняется **0.237711 → 0.349214 → 0.398998**, но TNR остаётся **0.0072 → 0.1439 → 0.2698**. Даже 40% чужих запросов не делают максимум F1 хорошей политикой отказа.

### Насколько плохо ошибиться в доле

В каждой ячейке — F1 на доле из столбца с порогом, выбранным по максимуму F1 на доле из строки:

{transfer_table}

Если выбрать старый порог на 25%, а реально будет 10% или 40%, потери относительно доступного F1-оптимума этих сценариев — **0.00098** и **0.00654**. Худший перенос между любыми двумя из трёх долей — **0.02570 F1** (выбор на 10%, оценка на 40%). Следовательно, сама ошибка в доле умеренно портит F1; главная проблема — низкий TNR сохраняется. Перенос и потери для всех признаков: [prior_transfer.csv](results/prior_transfer.csv).

### Фиксированный рекомендуемый порог, без подстройки к скрытому тесту

{stable_table}

При чистом prior shift TNR и recall фиксированного правила **точно не меняются**, потому что это доли внутри соответствующего класса. Меняются precision и F1. Совпадение recall/TNR проверено для всех событий во всех трёх сценариях, а не только для выбранного порога.

Для сравнения, AUC-PR абсолютного признака при 0.10/0.25/0.40 равен **0.9587/0.8878/0.8030**. Он независим от выбранного порога, но **не от доли положительного класса**; высокий AUC при 10% отказных нельзя напрямую сравнивать с AUC при 40%.

## 5. Абсолютный порог против отрыва первого кандидата

При одинаковой камерной маске и семантике `presence`:

{relative_table}

У gap наблюдаемые AUC-PR ниже на **{BASE['auc_pr_trapezoid']-GAP['auc_pr_trapezoid']:.5f}**, F1 при одном и том же TNR = 0.7014 ниже на **{t7['f1']-GAP['selected']['tnr_0.7']['f1']:.5f}**, recall ниже на **{t7['recall']-GAP['selected']['tnr_0.7']['recall']:.5f}**. В основном протоколе оснований заменять абсолютный score отрывом нет.

{gap_priors}

Порог gap по максимуму F1 почти не меняется, но это устойчивость **почти полного принятия**: TNR 0.0036–0.0216. Его фиксированный компромисс `{GAP['selected']['robust_balanced']['threshold']!r}` даёт TNR **{GAP['selected']['robust_balanced']['tnr']:.4f}** и F1 **0.7493/0.7188/0.6774** при трёх долях — хуже абсолютного компромисса. Максимальная потеря F1 при переносе F1-оптимального gap между долями всего **0.00211**, но такая стабильность не решает задачу отказа.

**Статистическая оговорка:** 95% интервалы кластерного bootstrap для разностей «absolute − gap»: AUC-PR **[{diff_ci['auc_absolute_minus_gap']['lo']:.4f}; {diff_ci['auc_absolute_minus_gap']['hi']:.4f}]**, F1 при фиксированных порогах TNR≈0.7 **[{diff_ci['f1_absolute_minus_gap_fixed_tnr07_cuts']['lo']:.4f}; {diff_ci['f1_absolute_minus_gap_fixed_tnr07_cuts']['hi']:.4f}]**. Они включают ноль. Речь о лучшем наблюдаемом результате, а не о доказанном превосходстве на любой новой выборке.

![PR-кривые для двух трактовок F1](pr_curves.png)

## 6. Развилки протокола, которые меняют ответ

### Если F1 требует правильной машины

{semantics}

При `top1` результат другой: gap повышает AUC-PR **0.4369 → 0.5062**, F1 при TNR≈0.7 **0.5219 → 0.5763**, а максимум F1 **0.5817 → 0.5880**. Если жюри выберет эту семантику, рабочий компромисс gap — **`{SUMMARY['market_gap_top1']['selected']['robust_balanced']['threshold']!r}`**: F1 **{SUMMARY['market_gap_top1']['selected']['robust_balanced']['f1']:.4f}**, TNR **{SUMMARY['market_gap_top1']['selected']['robust_balanced']['tnr']:.4f}**; при долях 0.10/0.25/0.40 F1 **0.6075/0.5829/0.5496**. Это условная рекомендация для score после маски; возможность вычислить такую маску в сервисе ещё требует решения.

Именно поэтому F1=0.733 основного компромисса нельзя называть точностью идентификации машины. В `top1` ошибочно принятый известный запрос даёт FP и остаётся FN; его доля не входит в знаменатель TNR, который относится только к настоящим отказным. Максимально достижимый recall для top1 ограничен Rank-1 = 0.6310; контур не достраивает PR-кривую до недостижимого recall = 1.

### Камерная фильтрация и доступный сервису score

{policies}

Переход `market → all_same_camera` почти не меняет вывод: при том же пороге TNR≥0.7 F1 **0.7300 → 0.7265**. Но отключение маски резко меняет распределение известных запросов: им снова доступны совпадения с той же камеры. Это отдельный протокол, его высокие числа не заменяют кросс-камерную оценку.

Без фильтрации абсолютный признак также лучше отрыва: AUC-PR **{raw_abs['auc_pr_trapezoid']:.4f} против {raw_gap['auc_pr_trapezoid']:.4f}**, максимум F1 **{raw_abs['selected']['best_f1']['f1']:.4f} против {raw_gap['selected']['best_f1']['f1']:.4f}**. Его собственный компромисс — `{raw_abs['selected']['robust_balanced']['threshold']!r}`. Это альтернатива только для сценария, где оценивается пустота **сырого ответа**, а не допустимого после маски списка.

На текущем вал-сплите основной порог `{rec['threshold']!r}` даёт **{REC['raw_service_refusals_on_validation']}** пустых сырых ответов и **{REC['eligible_list_refusals_on_validation']}** пустых ответов после `market`; из последних **194** — верные отказы, **302** — потерянные запросы с парой. Значит, оценочная пустота после фильтрации и отказ, реально показанный клиенту, различаются.

Проверен и **raw-gap**, вычисляемый до любой маски и доступный без меток камер: если им открывать/закрывать весь список, а верность top‑1 затем оценивать после `market`, при TNR=0.7014 получается F1 presence **0.9254**, но F1 top1 **0.5884**, AUC-PR top1 **0.3412**. Это отдельное правило допуска списка, использующее в том числе одноместного двойника. Его нельзя смешивать с gap после маски или считать улучшением ранжирования. Полные результаты — [robustness.json](results/robustness.json), [кривая raw-gap top1](results/curve_market_raw_gap_top1.csv).

Абсолютный **покандидатный** порог можно применить до маски: операции «score ≥ t» и удаления камерных совпадений коммутируют. Для gap это неверно: после удаления первого или второго кадра сам признак меняется. В тесте camera_id не выданы, а `market` требует ещё и истинных vehicle_id — напрямую вычислять такой gap в сервисе нельзя без отдельного приближения.

Число **3 отказа из 1110 на выданном тесте** дано в брифе; здесь оно **не пересчитано**, так как тестовые эмбеддинги не выданы в каталоге задания. Весь численный вывод этого отчёта рассчитан на val. Из распределения тестовых scores нельзя восстановить TNR без меток.

### Полная галерея и top‑10

`evaluate()` выдаёт обе ветки ранжирования, но его `refusal_scope` всегда **full_eligible_gallery**. Поэтому совпадение отказов с top‑10 проверено отдельно по сохранённым спискам: **0 изменений top‑1 и 0 изменений top‑2** при усечении сырой галереи до 10 до маски, обе камерные политики. Минимум оставшихся кандидатов — **9** для market и **4** для all_same_camera. Следовательно, на этих данных абсолютный score и gap, а значит все кривые query-level отказа, совпадают для полной галереи и обоих порядков усечения. Это проверенное свойство этого набора, не общий закон.

Ранжирующие метрики при этом различаются:

{ranking_table}

При усечении mINP по умолчанию не определён из-за неполных списков; второе число — явная альтернативная политика присваивать таким запросам ноль. mINP исходного ранжирования не меняется от выбора порога и не сравнивает качество двух признаков отказа. Для этого в отчёте использован PR-AUC.

## 7. Проверка устойчивости выбора порога

**500 bootstrap-выборок целыми `vehicle_id`**, отдельно внутри классов, seed 20260915. Для фиксированного рекомендуемого порога 95% перцентильные интервалы: F1 **[{ci['f1']['lo']:.4f}; {ci['f1']['hi']:.4f}]**, TNR **[{ci['tnr']['lo']:.4f}; {ci['tnr']['hi']:.4f}]**. Это условная неопределённость при фиксированной галерее; не гарантия на скрытом тесте и не интервал для заново оптимизированного порога. Считать 278 коррелированных кадров 278 независимыми машинами было бы чрезмерно оптимистично.

**5 повторов 5-fold калибровки по идентичностям.** Каждый порог выбран на четырёх folds по тому же устойчивому критерию, решения оценены на пятом; пороги вычтены из scores, и совокупные out-of-fold решения измерены тем же контуром. Идентичности калибровочных и проверочных запросов не пересекаются. Энкодер и галерея остаются фиксированными; это проверка калибровки, не переобучение модели и не новый внешний тест.

{cv_table}

Здесь скобки min/max — разброс пяти повторов, **не доверительный интервал**. Перцентили порогов относятся к 25 обучающим folds. Итог согласуется с оценкой на всём вал-сплите, но не устраняет неизвестность состава закрытой галереи. Все повторы: [crossfit_pooled.csv](results/crossfit_pooled.csv), [crossfit_folds.csv](results/crossfit_folds.csv), [bootstrap.csv](results/bootstrap.csv).

## 8. Обоснование для защиты

**Правило было зафиксировано до просмотра результатов:** выбрать t, максимизирующий

`min(TNR(t), F1(t; p=0.10), F1(t; p=0.25), F1(t; p=0.40))`.

Это прозрачная политика равной защиты двух показателей при неизвестной доле отказных. Она **не является формулой жюри**: равное отношение к F1 и TNR — наше явно заявленное инженерное предпочтение. При иной цене ложного принятия или ином весе TNR нужно выбрать другую точку приложенной кривой. В заданном диапазоне худший F1 приходится на p=0.40; лучший минимум показателей равен **0.6930**.

При переходе от старого порога к предлагаемому:

- F1 **{old['f1']:.4f} → {rec['f1']:.4f}**, потеря **{old['f1']-rec['f1']:.4f}**.
- TNR **{old['tnr']:.4f} → {rec['tnr']:.4f}**, выигрыш **{rec['tnr']-old['tnr']:.4f}**; ложных принятий чужих **238 → 84**, на **154 меньше**.
- Recall **{old['recall']:.4f} → {rec['recall']:.4f}**; принятых запросов с парой **819 → 530**, на **289 меньше**. Цена улучшения отказов существенна и не скрывается.
- При ошибке в доле классов TNR/recall сохраняются только в модели чистого prior shift; ожидаемый F1 фиксированной политики находится в диапазоне **0.6930–0.7626**.

Формулировка для жюри: **«Максимум F1 при 75% запросов с парой принимает 86% чужих запросов. Мы отдельно защищаем долю корректных отказов и выбираем порог по худшему из F1 и TNR в трёх сценариях состава теста. Он отклоняет около 70% чужих, но сохраняет лишь 64% запросов с парой; эти потери показаны явно. Политику проверили на отложенных идентичностях. После уточнения вашей семантики F1 и камерного исключения применим соответствующую заранее измеренную ветку».**

## 9. Что осталось неизвестным

1. Формула объединения F1/TNR и цена разных ошибок у жюри. Максимум итогового балла не установлен.
2. Семантика F1: наличие пары, правильность top‑1 или оценка пар/всего списка; что означает пустой ответ до и после маски. Для pairwise отдельная калибровка здесь не выполнена.
3. Как исключаются камеры на закрытом тесте и как сервис должен вычислять доступный ему признак без camera_id/vehicle_id. Обе оценочные маски рассчитаны, это не реализация определения камер.
4. Реальная доля отказных и сохранение распределений score внутри классов. Перевзвешивание не проверяет другой состав галереи, новые ракурсы, сезон, камеру или изменение модели. Gap также не получил проверки на таком сдвиге.
5. Независимость всех vehicle_id и возможные близкие серии между разными метками; bootstrap группировался по выданному vehicle_id. Ошибки разметки и весь процесс получения эмбеддингов отдельно не аудировались.
6. Результат на закрытых данных. Это калибровка на одном фиксированном вал-сплите с дополнительной внутренней проверкой; ни метрики, ни порог не являются гарантией итогового результата.

## 10. Воспроизведение и проверки

Все команды выполняются синхронно, записи — только в каталоге задания. Обучения/инференса модели не требуется. Основные вычисления используют Python проекта; графики — системный Python с уже установленным Matplotlib.

Входы этапа — векторы валидации бейзлайна и два текстовых контекста; в git их нет
(`*.npy`/`*.ids` не коммитятся), поэтому они кладутся рядом с каталогом `scripts/`
одной командой. Векторы получаются шагом 1 из `baseline/README.md`.

```bash
cd 04-solution/refusal
cp ../baseline/out/val_query.npy ../baseline/out/val_gallery.npy .
cp ../baseline/out/val_query.ids ../baseline/out/val_gallery.ids .
cp ../split/REPORT.md context-split.md          # описание сплита, хешируется в манифест
cp ../../00-task/tz.txt context-tz.txt          # текст ТЗ, хешируется в манифест
```

Дальше — по шагам, из каталога `scripts/`:

```bash
cd scripts
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ../../../.venv/bin/python -B analyze.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ../../../.venv/bin/python -B robustness.py --bootstrap 500 --cv-repeats 5
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B make_report.py
```

Проверено прогоном: все 84 файла `results/` воспроизводятся побайтово (кроме
`manifest.json`, где порядок ключей теперь фиксирован сортировкой — значения
хешей те же).

Выполнены **194 прямые сверки** с немодифицированным публичным `evaluate()` (186 точек основных веток, 6 raw-gap, 2 с повторёнными запросами) и **9 проверок инвариантности** recall/TNR по всей шкале при изменении доли. Максимальное абсолютное расхождение **0**. PR-кривые основных веток совпали с исходным контуром целиком. Подробности: [verification.json](results/verification.json), [robustness.json](results/robustness.json).

Артефакты: [точный порог и ограничения](results/recommendation.json), [полная сводка](results/summary.json), [перезапускаемый анализ](analyze.py), [проверка устойчивости](robustness.py), [графики SVG](threshold_curves.svg), [PR SVG](pr_curves.svg), [журнал](journal.md). SHA-256 закрепляют версии входов и контура. Каталог задания не является Git-репозиторием; исходный проект использован только для чтения.
"""
    (ROOT / "REPORT.md").write_text(text)


if __name__=="__main__":
    render_plots()
    write_report()
    print("REPORT.md, threshold_curves.png/svg, pr_curves.png/svg written")
