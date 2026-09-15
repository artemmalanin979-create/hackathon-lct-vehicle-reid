"""Render measured JSON into the task report; no model-quality formulas here."""
import csv
import json
from pathlib import Path
import numpy as np
import bench

ROOT=Path(__file__).resolve().parent
R=ROOT/"results"
def get(mode,n=750,sigma=.08):
    return json.loads((R/f"{mode}_{n}_{sigma}.json").read_text())
def fmt(x,d=2): return f"{x:,.{d}f}".replace(","," ")
def pct(x): return f"{100*x:.2f}%"
def peak(x): return max(x["resident"]["maxrss_mib"],x["resident"]["rss_mib"])
def chosen(data):
    # Strict exploratory tuning criterion, with separate query-half reporting.
    passing=[s for s in data["settings"] if s["tune_recall10"]>=.999]
    return passing[0] if passing else data["settings"][-1]
def agreement64(n,p,sigma=.08):
    path=R/f"truth64_{n}_{sigma}.npz"
    if not path.exists(): return None
    exact=np.load(path)["ids"]
    ann=np.load(R/f"neighbors_{n}_{sigma}_p{p}.npz")["ids"]
    return float(bench.overlap(ann,exact,10).mean())

sizes=[750,7500,75000,1000000]
large=get("ivf",1000000)
lf=get("flat",1000000)
op=chosen(large)
qual=get("quality")
baseline=qual["baseline"]
full=qual["full_rerank"]
k100=next(r for r in qual["candidates"] if r["K"]==100 and r["source"]=="exact")
ivfk100=next(r for r in qual["candidates"] if r["K"]==100 and r["source"]=="ivf")
env=get("env")
all_results=[json.loads(p.read_text()) for p in R.glob("*.json") if not p.name.startswith("eval_")]
peak_cg=max(x.get("resident",{}).get("memory.peak",0) for x in all_results)
maxrss=max([peak(x) for x in all_results if "resident" in x])
prec=get("truth64",1000000)
lines=[]
def add(s=""): lines.append(s)
add("# ANN на миллион эмбеддингов: измерение масштаба и границ\n")
add(f"Период измерений: {min(x['start'] for x in all_results)} — {max(x['end'] for x in all_results)} (UTC+03:00).\n")
add("## Результат\n")
add(f"Построен и опрошен **реальный индекс на 1 000 000 × 512 float32** в FAISS, "
    f"при лимите 4 ГиБ. Галерея расширена из реальных эмбеддингов контролируемым шумом; "
    f"это **синтетический стресс-тест на основе 750 кадров / {env['data']['gallery_identities']} идентичностей**, а не миллион разных автомобилей.\n")
add(f"Рабочая точка в этом эксперименте: **IVF{large['nlist']}, nprobe={op['nprobe']}**, "
    f"медиана **{fmt(op['performance']['single_median_ms'])} мс**, p95 **{fmt(op['performance']['single_p95_ms'])} мс**, "
    f"Recall@10 **{pct(agreement64(1000000,op['nprobe']))}** против полного перебора в float64 на 256 запросах "
    f"(**{pct(op['recall10'])}** против точного FAISS float32; разница арифметики разобрана ниже). "
    f"Ускорение против точного FAISS: **{fmt(lf['performance']['single_median_ms']/op['performance']['single_median_ms'])}×**. "
    f"Пик RSS процесса индекса **{fmt(peak(large))} МиБ**; сборка **{fmt(large['build_s'])} с** без генерации данных.\n")
add(f"Полное переранжирование воспроизведено: mAP **{baseline['mAP']:.6f} → {full['mAP']:.6f}**, "
    f"прибавка **{full['mAP']-baseline['mAP']:.6f}**. "
    f"Вариант «один запрос + top-100» дал **{k100['mAP']:.6f}**, сохранив "
    f"**{pct(k100['gain_retained_fraction'])} прибавки**. Ниже также показан результат с кандидатами ANN.\n")
add("## 1. Данные и условия\n")
add("- Входы: `04-solution/postproc/out/val_{query,gallery}.npy`, 1110 запросов и 750 объектов галереи, 512 измерений; модель `vehicle-reid-0001` (OSNet-AIN) из `service/model`. SHA256 модели, эмбеддингов, CSV и исходников — в `provenance.json`.")
add("- Порядок `.ids` проверен против `split/files/val_*.csv`; 12 векторов заново получены из исходных изображений тем же кодом извлечения и моделью. У всех cosine ≥0,9999; точные ошибки записаны в `provenance.json`.")
add(f"- Расширение: первые 750 — исходные нормированные векторы; дальше `normalize(g[i % 750] + ε)`, где компоненты ε имеют σ=0,08/√512. Seed={bench.SEED}, независимый seed каждого блока {bench.BLOCK}; префиксы галерей совпадают. Измеренный средний cosine копии с якорем — **{env['data']['anchor_cos_mean']:.6f}**. Шум не моделирует новые ракурсы, камеры или новые идентичности.")
add("- Вычисления: **worker-vm / sespel-worker.local**, Intel i7-6820HQ @ 2,70 ГГц, 8 доступных vCPU, 32083 МиБ RAM. Ограничение эксперимента — **один поток / affinity CPU 1**, `CPUQuota=100%`, `nice=10`, BLAS/OpenMP=1. GPU не используется. Это замер на удалённой Linux-ВМ рабочего ноутбука, не на локальном i5.")
add("- Узел занят: в стартовом снимке соседний `scalper` потреблял около 719% CPU и 6,5 ГиБ RSS. Показаны wall-clock времена с этой конкуренцией. Процессы, CPU, доступная память и swap фиксировались в каждом JSON; изолированного стенда и распределения по независимым повторным запускам машины нет.")
add(f"- Каждый процесс: `MemoryMax=4G`, `MemoryHigh=3500M`, `MemorySwapMax=0`; перед тяжёлыми этапами `free -m`, порог 2500 МиБ. Максимум по всем завершённым процессам: **{fmt(maxrss)} МиБ RSS**, **{fmt(peak_cg/2**20)} МиБ cgroup peak**. Блоки генерируются на лету, индекс и полная исходная галерея одновременно не дублируются. Swap эксперимента остаётся 0; подробности в логах.")
add("- Локально при старте уже было занято 10071 МиБ swap и работали Python, Java, Blender. Поэтому тяжёлое перенесено на worker согласно правилам машины. Это не выдаётся за результат на локальном ноутбуке с 15 ГиБ RAM.")
add("- Поиск: 256 запросов без повторов, детерминированная перестановка реальных 1110 запросов; включая запросы без пары. Recall ANN измеряет совпадение соседей, а не распознавание автомобиля. Первые 128 используются для выбора nprobe, вторые 128 показываются отдельно. Все размеры используют тот же набор запросов.")
add("- Время запроса: один прогрев, затем 64 запроса × 3 повтора; медиана и p95. Пропускная способность отдельно измерена батчами 32 на 256 запросах, 3 повтора. Для исходного сервиса — 16 одиночных запросов × 3 и 32 запроса в батче из-за стоимости нормировки. Это один последовательный клиент, не многопользовательский нагрузочный тест. Исходные времена сохранены.")
add("- Re-ID: только неизменённый `04-solution/eval/reid_metrics.py` + `scope_metrics.py` из `inputs/`. Основной mAP: `market`, исключение same-ID AND same-camera, AP step, все релевантные в знаменателе, 832 валидных запроса. Для mAP@10 сначала top-10, затем фильтр камеры. Никаких новых формул mAP / Rank-1 в эксперименте нет.\n")
add("## 2. Время и память по размеру галереи\n")
add("Таблица: **медиана одиночного top-10 запроса / пик RSS процесса**. Исходный сервис включает нормировку всей галереи в float64 и полную стабильную сортировку; точный FAISS хранит нормированные float32 и выбирает top-10. Поэтому ускорение от ANN оцениваем именно относительно точного FAISS. RSS включает интерпретатор, данные и временные буферы; это не размер одного массива.\n")
add("| Галерея | Текущий точный сервис, мс / МиБ | Точный FAISS, мс / МиБ | IVF (nlist/nprobe), мс / МиБ | Recall@10 / потеря соседей |")
add("|---:|---:|---:|---:|---:|")
for n in sizes:
    c=get("current",n); f=get("flat",n); a=get("ivf",n); s=chosen(a)
    cv=f"{fmt(c['performance']['single_median_ms'])} / {fmt(peak(c),0)}" if c['status']=='ok' else "не запускался: выше бюджета"
    add(f"| {n:,} | {cv} | {fmt(f['performance']['single_median_ms'])} / {fmt(peak(f),0)} | "
        f"{a['nlist']}/{s['nprobe']}: {fmt(s['performance']['single_median_ms'])} / {fmt(peak(a),0)} | {pct(s['recall10'])} / {pct(1-s['recall10'])} |")
add("\nRecall@10 = средняя доля ID из точного top-10, найденная ANN; порядок внутри десятки отдельно этим числом не проверяется. Совпадение top-1 и top-100 — в следующей таблице и JSON. По каждой величине команда: `python -B bench.py {current,flat,ivf} --n N --sigma 0.08`; точный интерпретатор и ограниченный запуск приведены ниже.\n")
add("В последнем столбце этой таблицы эталон — FAISS float32. Значения для float64 отдельно приведены ниже. "
    "Между размерами меняются nlist и nprobe: это сравнение выбранных рабочих точек. "
    "Меньшая задержка на миллионе относительно 75000 объясняется, в частности, другим разбиением индекса "
    "и числом просматриваемых списков; она не доказывает постоянную скорость при росте галереи.\n")
small=get("ivf",750); smallop=chosen(small)
add(f"На настоящей галерее 750 при требовании высокой полноты `nprobe={smallop['nprobe']}` IVF **медленнее** "
    f"точного FAISS: {fmt(smallop['performance']['single_median_ms'],3)} против {fmt(get('flat',750)['performance']['single_median_ms'],3)} мс. "
    "Переход на ANN для этого размера не оправдан. В этих данных выигрыш появляется к 7500; универсальный порог из этого не следует.\n")
add("### Сборка и хранение\n")
add("| N | nlist | Обучение / добавление, с | Генерация, с | Сериализованный индекс, МиБ | RSS после сборки/поиска, МиБ |")
add("|---:|---:|---:|---:|---:|---:|")
for n in sizes:
    a=get("ivf",n)
    add(f"| {n:,} | {a['nlist']} | {fmt(a['train_s'])} / {fmt(a['add_s'])} | {fmt(a['generation_s'])} | {fmt(a['serialized_index_bytes']/2**20)} | {fmt(a['resident']['rss_mib'])} |")
add("\nРазмер сериализации измерен `faiss.write_index` через счётчик записанных байтов, без создания многогигабайтного файла на tmpfs. Память индекса отражена отдельно через RSS: списки IVF имеют запас ёмкости. Полный индекс намеренно не сохранялся; код детерминированной пересборки сохранён. Время `build_s=train_s+add_s` — время вызовов библиотеки обучения и добавления. Генерация блоков для добавления измерена отдельно. Создание массива для обучения и запуск Python не включены; полное wall-clock время подготовки индекса отдельным таймером не измерялось.\n")
add("## 3. Потеря качества ANN и рабочая точка\n")
add(f"Миллион, IVF{large['nlist']}, inner product на L2-нормированных float32, без квантования. "
    f"Coarse k-means: {large['train_rows']} строк из галереи, 15 итераций, фиксированный seed. Все точки ниже используют один построенный индекс.\n")
add("| nprobe | R@10 float32 / float64 | R@100 float32 | Совпадение top-1 | R@10 tune / контроль | медиана / p95, мс | 1 клиент, QPS | batch32, QPS |")
add("|---:|---:|---:|---:|---:|---:|---:|---:|")
for s in large["settings"]:
    p=s["performance"]
    add(f"| {s['nprobe']} | {pct(s['recall10'])} / {pct(agreement64(1000000,s['nprobe']))} | {pct(s['recall100'])} | {pct(s['top1_agreement'])} | {pct(s['tune_recall10'])} / {pct(s['holdout_recall10'])} | {fmt(p['single_median_ms'])} / {fmt(p['single_p95_ms'])} | {fmt(p['single_qps'])} | {fmt(p['batch32_qps'])} |")
add(f"\nВыбор: минимальный измеренный `nprobe`, дающий R@10 ≥99,9% на первых 128 запросах; "
    f"получился **{op['nprobe']}**. На вторых 128 — **{pct(op['holdout_recall10'])}**, "
    f"полностью совпавших десяток по всем 256 — **{pct(op['complete_top10_fraction'])}**. "
    "Это исследовательская рабочая точка на данном распределении, не гарантия SLA или переносимости качества на новые города.\n")
add(f"Точный FAISS на миллионе: {fmt(lf['performance']['single_qps'])} QPS одиночными запросами, "
    f"{fmt(lf['performance']['batch32_qps'])} QPS батчами 32. Это важный сильный exact-baseline; "
    "подмена его медленной нормировкой сервиса дала бы преувеличенное преимущество ANN.\n")
add("### Проверка точности арифметики\n")
add(f"Дополнительно весь миллион пройден блоками по {prec['block_rows']} с **исходным `cosine_scores` в float64**, "
    f"для всех 256 запросов, с точным объединением top-100. Полнота top-10 FAISS float32 относительно этого оракула — "
    f"**{pct(prec['float32_vs_float64_recall10'])}**, top-1 — **{pct(prec['float32_vs_float64_top1'])}**. "
    f"Рабочий ANN относительно float64: **{pct(agreement64(1000000,op['nprobe']))} R@10**. "
    f"Оракул занял {fmt(prec['including_generation_s'])} с с генерацией, пик RSS {fmt(peak(prec))} МиБ. "
    "Это проверка правильности выдачи, а не задержка готового сервиса; онлайн-запросы не должны генерировать галерею заново. Команда: `bench.py truth64 --n 1000000`.\n")
add("Пограничный случай разобран в `precision_detail.json` (`python -B precision_detail.py`): "
    "один запрос, два почти равных кандидата на границе top-10. В точном FAISS float32 их порядок "
    "отличается от float64; в IVF float32 их оценки равны, и разрешение ничьей на этом прогоне "
    "совпало с float64. Полное совпадение ANN top-10 относится к этой выборке, версии библиотеки и среде.\n")
add("### Чувствительность к шуму\n")
add("75 000 объектов, тот же метод, σ=0,25 вместо 0,08; это проверка хрупкости вывода, а не вторая реальная выборка.\n")
add("| nprobe | R@10 σ=0,08 | R@10 σ=0,25 | медиана σ=0,25, мс |")
add("|---:|---:|---:|---:|")
sensitivity=get("ivf",75000,.25)
for old,new in zip(get("ivf",75000)["settings"],sensitivity["settings"]):
    add(f"| {new['nprobe']} | {pct(old['recall10'])} | {pct(new['recall10'])} | {fmt(new['performance']['single_median_ms'])} |")
add("\nКоманды: `bench.py flat --n 75000 --sigma 0.25`, затем `bench.py ivf --n 75000 --sigma 0.25`. Все результаты сетки для 750 / 7500 / 75000 также сохранены в `results/ivf_*.json`; они не скрыты за одной удачной точкой.\n")
add("## 4. Где заканчивается текущая реализация\n")
add("### Исходный полный перебор\n")
c75=get("current",75000)
add(f"Прямо измерено до 75 000: пик {fmt(peak(c75))} МиБ. Миллион текущим вызовом "
    "**не запускался**: только вход float32 и две одновременно живые float64-копии галереи требуют около 9,54 ГиБ; "
    "временная нормировка требует ещё памяти. Консервативный предохранитель (`36*N*512 + 150 MiB`) оценивает "
    f"пик в {fmt(get('current',1000000)['predicted_peak_mib']/1024)} ГиБ. Это **расчёт по массивам, не замер и не реальный OOM**. "
    "Точная граница между 75 000 и миллионом здесь не искалась разрушительным запуском. Блочный exact и заранее нормированная float32-галерея снимают этот барьер — они проверены отдельно.\n")
add("### Полное k-reciprocal переранжирование\n")
add("Импорт неизменённого `service/app/core/rerank.py`, параметры `(6,3,0.3)`, Q=1110 во всех строках. Время — один полный вызов, не мс/одиночный запрос. Галерея расширена тем же шумом.\n")
add("| G | Q+G | Время, с | Пик RSS, МиБ | Статус |")
add("|---:|---:|---:|---:|---|")
for n in [750,1500,3000,6000,7500,10000,1000000]:
    r=get("rerank-scale",n)
    if r["status"]=="ok":
        add(f"| {n:,} | {r['total_nodes']:,} | {fmt(r['seconds'])} | {fmt(peak(r))} | измерено |")
    else:
        add(f"| {n:,} | {r['total_nodes']:,} | — | прогноз {fmt(r['predicted_peak_mib'],0)} | пропущено предохранителем |")
add("\nМассивы `dist_all`, `original_dist`, `initial_rank`, `V`, `V_qe` квадратичны по Q+G. "
    "Для миллиона одна матрица float64 уже около 8 ТБ (7,29 ТиБ); несколько — десятки ТиБ. "
    "Это **экстраполяция размера массивов**, не выделенная память. Предохранитель использует `48*(Q+G)^2 + 3*(Q+G)*512*8 + 120 MiB`; "
    "это консервативная оценка для безопасного отказа, а не подгонка к RSS. Реально доказана работоспособность до G=6000 в заданном бюджете; G=7500 пропущен с запасом до жёсткого лимита. Говорить, что процесс действительно упал на этом размере, нельзя.\n")
add("## 5. Top-K переранжирование: сколько качества остаётся\n")
add("Для каждого запроса выбираем K кандидатов без знания ID/камеры, строим граф **только из одного запроса и этих K векторов**, применяем исходный rerank `(6,3,0.3)` и переставляем этот префикс. Остальная галерея на маленькой проверке сохраняет исходный порядок только для диагностики полного mAP. mAP@10 не получает помощь от хвоста при K≥10. Полный метод имеет доступ к остальным 1109 запросам; локальный вариант этот контекст теряет, даже при K=750.\n")
add("| Вариант | mAP | mAP@10 | Rank-1 | Сохранено прибавки mAP | Медиана rerank/query, мс |")
add("|---|---:|---:|---:|---:|---:|")
add(f"| Косинус | {baseline['mAP']:.6f} | {baseline['mAP@10']:.6f} | {baseline['Rank-1']:.6f} | — | — |")
add(f"| Полный граф Q+G | {full['mAP']:.6f} | {full['mAP@10']:.6f} | {full['Rank-1']:.6f} | 100% | весь пакет {qual['full_rerank_s']:.2f} с |")
for r in qual["candidates"]:
    add(f"| {r['source']}, K={r['K']} | {r['mAP']:.6f} | {r['mAP@10']:.6f} | {r['Rank-1']:.6f} | {pct(r['gain_retained_fraction'])} | {fmt(r['rerank_single_median_ms'])} |")
add(f"\nANN-кандидаты здесь получены IVF16/nprobe=8 на **исходной галерее 750**, без пересортировки точными скорами перед rerank. "
    f"Для K=100 mAP={ivfk100['mAP']:.6f}, сохранено {pct(ivfk100['gain_retained_fraction'])} прибавки; "
    "это end-to-end проверка retrieval + rerank на размеченных данных. Это не mAP городского миллиона: у шумовых копий нет новых достоверных меток.\n")
one_full=next(r for r in qual["candidates"] if r["K"]==750)
three_hundred=next(r for r in qual["candidates"] if r["K"]==300)
add(f"Контроль K=750: mAP={one_full['mAP']:.6f}, сохранено {pct(one_full['gain_retained_fraction'])} прибавки, "
    "хотя вся галерея присутствует. Значит, разницу нельзя объяснить только пропущенными кандидатами: "
    f"меняется контекст других запросов в графе. K=300 дал {three_hundred['mAP']:.6f} и "
    f"{pct(three_hundred['gain_retained_fraction'])} прибавки за {fmt(three_hundred['rerank_single_median_ms'])} мс; "
    "качество по K здесь не монотонно. Выбор K следует подтверждать на отдельной выборке.\n")
add("Ресурсы локального графа замерены отдельным свежим процессом, без пика от предшествующего полного графа:\n")
add("| K | Пик выделений одного rerank/query, МиБ | Пик RSS отдельного процесса, МиБ |")
add("|---:|---:|---:|")
for k in [30,100,300]:
    r=get("candidate-cost",k)
    add(f"| {k} | {fmt(r['single_query_traced_peak_bytes']/2**20,3)} | {fmt(peak(r))} |")
add("\nПик выделений получен `tracemalloc` отдельным вызовом; временные замеры выполнены без трассировки. "
    "При фиксированном K память графа O(K²) и не зависит от G; добыча кандидатов и хранение индекса зависят. "
    "Полные тайминги quality включают извлечение подматрицы и перестановку хвоста для оценки; retrieval и инференс изображения туда не входят.\n")
add("**Практическое решение:** для небольшой конкурсной галереи оставить полный batch-rerank, который измеренно лучше по качеству. "
    "Для большой галереи использовать ANN → извлечение исходных векторов top-K → локальный rerank с отдельной калибровкой. "
    "Сохранение всей прибавки +0,037 не обещать. Кандидатный K=100 — проверенный вариант с показанным компромиссом; "
    "оптимизация разреженного глобального графа в этой работе не реализована.\n")
add("## 6. Что доказано, а что осталось за рамками\n")
add("**Показано:** миллион записей реально добавлен в индекс и реально опрошен; exact-reference на том же миллионе получен; настройка скорости/Recall измерена; память ограничена cgroup без swap; источник векторов подтверждён; потери локального rerank рассчитаны штатным контуром.\n")
add("**Не показано:** миллион независимых городских машин; обобщение на новые камеры/погоду/сезоны; отсутствие деградации скорости при росте G; реальный городской mAP; конкурентные запросы, обновления/удаления и долговременная работа индекса; отказоустойчивость, репликация, распределение по узлам; нагрузка чтения изображений и ONNX; p99/SLA; дисковый векторный store; совместный пик ANN+rerank в промышленном сервисе. Индекс и rerank измерены отдельно. Объединённая память оценивается суммой измеренных частей, а не выдаётся за end-to-end нагрузочный тест.\n")
add("Векторы расширяются вокруг всего 750 якорей: плотные похожие группы могут существенно облегчать поиск. Даже проверка σ=0,25 не заменяет независимую крупную галерею. Рост скорости и точности из этого опыта нельзя переносить на произвольный городской поток.\n")
add("## 7. Пакеты, воспроизведение и перенос\n")
add("В исходную `.venv` добавлен только **faiss-cpu==1.15.0**. Уже были **numpy==2.5.3**, **packaging==26.3**; на worker установлены те же три версии. Python локально 3.13.13, worker 3.13.15. Для перепроверки происхождения использованы существующие onnxruntime==1.30.0 и pillow==12.3.0.\n")
add("FAISS распространяется по **MIT**: проверен [LICENSE официального тега v1.15.0](https://raw.githubusercontent.com/facebookresearch/faiss/v1.15.0/LICENSE), лицензия установленного wheel сохранена в `LICENSE-FAISS`. Формат IVF и значение nprobe сверены с [официальной документацией FAISS](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).\n")
add("Все команды выполняются из каталога `job_17`; переменные ниже обычные shell-пути, ни одна команда не пишет в `service/`:\n")
add("```bash\nREID_PY=/home/artem/projects/hackathon-lct-vehicle-reid/.venv/bin/python\n\"$REID_PY\" -m pip install -r requirements-benchmark.txt\n\"$REID_PY\" -B prepare.py                 # нужно только заново снять входы\n\"$REID_PY\" -B verify.py\n\n# Предпочтительно перенести этот компактный каталог на worker-vm.\n# Там: uv venv --python 3.13 .venv\n#      uv pip install --python .venv/bin/python -r requirements-benchmark.txt\n#      REID_PY=./.venv/bin/python\nfor PHASE in small large rerank sensitivity precision; do\n  \"$REID_PY\" -B run_suite.py \"$PHASE\"\ndone\n\"$REID_PY\" -B make_report.py\n```\n")
add("`run_suite.py` последовательно создаёт ограниченный `systemd-run --user --scope`; не запускает демоны, пулы процессов или фоновые задания. Если MemAvailable ниже 2500 МиБ, останавливается. Каждый файл JSON содержит точную команду и среду; `run-*.log` — команды обёртки, `logs/` — `free -m`, прогресс и предохранители. Для отдельного опыта:\n")
add("```bash\nfree -m\nsystemd-run --user --scope --quiet \\\n  -p MemoryMax=4G -p MemoryHigh=3500M -p MemorySwapMax=0 -p CPUQuota=100% \\\n  taskset -c 1 nice -n 10 \"$REID_PY\" -B bench.py ivf --n 1000000 --sigma 0.08\n```\n")
add("Данные `inputs/` и скрипты являются самодостаточным воспроизводимым пакетом после установки зависимостей. Необязательно повторно извлекать все изображения: сохранены исходные векторы, IDs, сплит, SHA256 и контрольные повторные извлечения. `verify.py` проверяет хеши, префиксы генератора, нормировку, корректность заполнения exact-индекса и преобразования порядка в оценки для evaluator.\n")
add("**Куда переносить:** результаты и скрипты — в новый раздел `04-solution/ann-benchmark/` после обычного проектного ревью; документацию — через принятый процесс заметок/зеркал проекта. При интеграции выделить в `app/core/` отдельный backend ANN с кешированными нормированными векторами и retrieval top-K; для больших галерей вызывать локальный кандидатный rerank, для малого конкурсного batch сохранять полный. Калибровку порогов отказа повторить: шкалы локального и полного rerank могут отличаться. В этом задании рабочий сервис **не изменялся**, публикаций/коммитов исходного репозитория нет. Финальное сравнение хешей — `service_integrity.json`.\n")
(ROOT/"REPORT.md").write_text("\n".join(lines))
rows=[]
for n in sizes:
    flat=get("flat",n)
    ann=get("ivf",n)
    for s in ann["settings"]:
        rows.append({"n":n,"nlist":ann["nlist"],"nprobe":s["nprobe"],
                     "exact_median_ms":flat["performance"]["single_median_ms"],
                     "ann_median_ms":s["performance"]["single_median_ms"],
                     "ann_p95_ms":s["performance"]["single_p95_ms"],
                     "ann_single_qps":s["performance"]["single_qps"],
                     "ann_batch32_qps":s["performance"]["batch32_qps"],
                     "recall10":s["recall10"],"recall100":s["recall100"],
                     "top1_agreement":s["top1_agreement"],
                     "tune_recall10":s["tune_recall10"],"holdout_recall10":s["holdout_recall10"],
                     "recall10_float64":agreement64(n,s["nprobe"]),
                     "build_s":ann["build_s"],"serialized_bytes":ann["serialized_index_bytes"],
                     "peak_rss_mib":peak(ann)})
with (ROOT/"search_summary.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("REPORT.md generated from measured artifacts")
