# B. Как не дать сети выучить камеру — приёмы с замерами

Контекст проекта: 7 248 кадров / 1 171 машина / 96 камер, у каждой машины ≥2 камеры (~6.2 кадра на ID).
`camera_id` есть в обучении, НЕТ на инференсе. Предыдущий прогон: R-1 внутри камеры 0.976, между камерами 0.379.

Правило отчёта: если числа в первоисточнике нет — написано «**замера нет**». Все ссылки открывались, номера таблиц сверены по тексту PDF.

---

## 0. Сводная таблица (главное)

| # | Приём | Датасет | базовое → с приёмом (mAP / R1) | Δ mAP | Источник (табл.) | camera_id на инференсе? |
|---|---|---|---|---|---|---|
| 1 | Hardest-negative только **внутри своей камеры** (Triplet-same) | Duke-SCT | 11.3 / 21.2 → 35.9 / 54.6 | **+24.6** | [arXiv:1909.10848](https://arxiv.org/abs/1909.10848), Табл. 3 | нет |
| 1 | То же | Market-SCT | 18.2 / 39.7 → 28.0 / 51.3 | **+9.8** | там же, Табл. 3 | нет |
| 1 | Hardest-negative из **других** камер (Triplet-other) | Duke-SCT | 11.3 → 3.6 | **−7.7** | там же, Табл. 3 | нет |
| 1 | MCNL (ранжирование cross-cam neg между pos и same-cam neg) | Duke-SCT / Market-SCT | 11.3 → 45.3 / 18.2 → 40.6 | +34.0 / +22.4 | там же, Табл. 3 | нет |
| 1 | Camera-balanced (proxy-balanced) сэмплирование | Market / Duke / MSMT17 | 75.1→79.9 / 59.9→61.6 / 34.0→35.3 | +4.8 / +1.7 / +1.3 | [arXiv:2012.10674](https://arxiv.org/abs/2012.10674), Табл. 1 (CAP3→CAP4) | нет |
| 1 | Выброс «одно-камерных» кластеров из обучения | MSMT17 (CC) | 29.8 → 45.4 | **+15.6** | [arXiv:2502.10195](https://arxiv.org/abs/2502.10195), Рис. 7(a) | нет |
| 2 | Adversarial camera head (GRL) | Market / Duke / MSMT17 (UCC) | 34.4→50.6 / 41.7→46.6 / 10.0→11.2 | +16.2 / +4.9 / **+1.2** | [arXiv:1908.00862](https://arxiv.org/abs/1908.00862), Табл. V | нет |
| 2 | Простой camera-adversarial дискриминатор (+Adv.) | Duke→Mkt (SSG / MMT) | 58.3→62.7 / 79.3→82.2 | +4.4 / +2.9 | [arXiv:1904.01308](https://arxiv.org/abs/1904.01308), Табл. IV | нет |
| 2 | ICAL (adversarial по global-ID между камерами) | MSMT17 / Market | 54.6→58.9 / 89.2→90.1 | +4.3 / **+0.9** | [arXiv:2409.19563](https://arxiv.org/abs/2409.19563), Табл. IV (Order 4→7) | нет |
| 3 | CBN (same-domain) | Market / Duke / MSMT17 | 74.0→77.3 / 66.6→67.3 / 42.3→42.9 | **+3.3 / +0.7 / +0.6** | [arXiv:2001.08680](https://arxiv.org/abs/2001.08680), Табл. 1 | **ДА** |
| 3 | CBN (cross-domain) | Duke→Market | 25.1 → 43.0 | +17.9 | там же, Табл. 1 | **ДА** |
| 3 | AdaBN (без camera_id, статистика по всему тесту) | Duke→Market | 25.1 → 28.1 | **+3.0** | там же, Табл. 6 | нет |
| 4 | R50 → R50-IBN-a | Market / Duke / MSMT17 (fast-reid BoT) | 86.1→87.6 / 77.0→79.6 / 50.2→54.4 | +1.5 / +2.6 / **+4.2** | [fast-reid MODEL_ZOO](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md) | нет |
| 4 | R50 → IBN-Net-a | Market / Duke | 85.9→88.2 / 76.4→79.1 | +2.3 / +2.7 | [IBN-Net README](https://github.com/XingangPan/IBN-Net) | нет |
| 5 | SNR (same-domain) | Market / Duke | 82.8→84.7 / 71.2→72.9 | **+1.9 / +1.7** | [arXiv:2005.11037](https://arxiv.org/abs/2005.11037), Табл. 1 | нет |
| 5 | SNR (cross-domain) | M→D / D→M | 19.8→33.6 / 21.8→33.9 | +13.8 / +12.1 | там же, Табл. 1 | нет |
| 5 | MetaBIN (same-domain supervised) | Market | 67.9 → 68.5 | **+0.6** | [arXiv:2011.14670](https://arxiv.org/abs/2011.14670), Табл. 5 | нет |
| 5 | AIBN (same-domain) | Market | 79.9 → 80.0 | **+0.1** | [arXiv:2103.11658](https://arxiv.org/abs/2103.11658), Табл. 3 | нет |
| 6 | CamStyle | Market / Duke | 65.87→68.72 / 51.83→53.48 | +2.85 / +1.65 | [arXiv:1711.10295](https://arxiv.org/abs/1711.10295), Табл. 4, 5 | нет |
| 6 | CamStyle **без других аугментаций** | Market | 64.10 → 64.86 | **+0.76** | там же, Табл. 3 | нет |
| 7 | SIE (camera embedding) | MSMT17 / VeRi-776 | 61.0→62.4 / 78.2→78.7 | **+1.4 / +0.5** | [arXiv:2102.04378](https://arxiv.org/abs/2102.04378), Табл. 4 | **ДА** |
| 10 | MixStyle (камеры = домены) | M→D / D→M | 19.3→23.4 / 20.4→24.7 | +4.1 / +4.3 | [arXiv:2104.02008](https://arxiv.org/abs/2104.02008), Табл. 2 | нет |
| — | Camera Verification (пост-обработка) | CityFlow-V2 (vehicle) | 66.8 → 72.7 | +5.9 | [arXiv:2105.09701](https://arxiv.org/abs/2105.09701), Табл. 4 | **ДА** |
| — | Inter-Camera Fusion | CityFlow-V2 | 72.7 → 74.7 | +2.0 | там же, Табл. 4 | **ДА** |
| — | Camera-similarity penalty (camera-ReID сеть) | CityFlow val (vehicle) | 47.0 → 50.8 | +3.8 | [arXiv:2004.09164](https://arxiv.org/abs/2004.09164), Табл. 3 | **нет** (камера предсказывается по изображению) |
| — | Camera-specific feature normalization (test-time) | Market (TransReID-SSL) | 53.6 → 62.3 | +8.7 | [arXiv:2502.10195](https://arxiv.org/abs/2502.10195), Табл. 3(a) | **ДА** |
| — | Camera-agnostic centering+scaling (test-time) | CUHK03-NP | 27.2 → 28.8 | **+1.6** | там же, Табл. 4 («Entire») | нет |

---

## 1. Кросс-камерное формирование батчей и триплетов

### 1.1 Самый сильный и самый дешёвый результат: ограничение hard-negative своей камерой

**Single Camera Training for Person Re-identification (MCNL), AAAI 2020** — https://arxiv.org/abs/1909.10848, **Таблица 3**.
Backbone ResNet-50, batch-hard triplet (Hermans et al.), обучение на SCT-версиях (каждый человек — в одной камере).

| Вариант triplet | Duke-SCT R1 / mAP | Market-SCT R1 / mAP |
|---|---|---|
| Triplet (обычный batch-hard, камеры игнорируются) | 21.2 / 11.3 | 39.7 / 18.2 |
| **Triplet-other** (hardest negative берётся из **других** камер) | 9.9 / 3.6 | 25.2 / 8.8 |
| **Triplet-same** (hardest negative берётся **из той же камеры**) | **54.6 / 35.9** | **51.3 / 28.0** |
| MCNL (ранжирование dist+ < dist−,other < dist−,same) | 66.4 / 45.3 | 66.2 / 40.6 |

Формулировка MCNL (Eq. 4 в статье): `[m1 + dist+ − dist−,other]+ + [m2 + dist−,other − dist−,same]+`, т.е. negative из **другой** камеры должен быть дальше позитива, но **ближе**, чем negative из **своей** камеры.

Сопутствующий замер камерной «выученности» — pseudo-F статистика кластеризации фичей по камерам (Рис. 3, меньше = лучше):
Triplet **8.433** → Triplet-other **16.683** → Triplet-same **0.404** → MCNL **0.255**. То есть простое ограничение негатива своей камерой снизило камерную разделимость фич в **20 раз**.

Интерпретация для нас: наш разрыв (0.976 внутри камеры / 0.379 между) — это ровно тот эффект, который MCNL измеряет через pseudo-F. Обычный batch-hard triplet выбирает negative из чужой камеры (он визуально дальше из-за фона/освещения) → сеть учит фон.

### 1.2 Camera-balanced / proxy-balanced сэмплирование

**Camera-aware Proxies (CAP), AAAI 2021** — https://arxiv.org/abs/2012.10674, **Таблица 1** (unsupervised).
«PBsampling» = proxy-balanced (каждый proxy = ID×камера, батч балансируется по proxy), альтернатива — class-balanced (камеры игнорируются).

| Пара | Market mAP | Duke mAP | MSMT17 mAP |
|---|---|---|---|
| CAP1 → CAP2 (intra-cam loss, +PB sampling) | 58.9 → 64.6 (**+5.7**) | 57.0 → 60.9 (+3.9) | 23.0 → 24.8 (+1.8) |
| CAP3 → CAP4 (inter-cam loss, +PB sampling) | 75.1 → 79.9 (**+4.8**) | 59.9 → 61.6 (+1.7) | 34.0 → 35.3 (+1.3) |
| CAP5 → CAP6 (оба loss, +PB sampling) | 75.9 → 79.2 (**+3.3**) | 64.5 → 67.3 (+2.8) | 35.1 → 36.9 (+1.8) |
| CAP2 → CAP6 (добавление inter-camera loss) | 64.6 → 79.2 (**+14.6**) | 60.9 → 67.3 (+6.4) | 24.8 → 36.9 (+12.1) |

Замечание автора CAP: «class-balanced sampling чаще сэмплирует ID из камер, богатых изображениями, из-за чего обучение для бедных камер неэффективно».

### 1.3 Обратный замер: сколько теряется, если убрать камеру из сэмплинга

**Rethinking Sampling Strategies for USL re-ID** — https://arxiv.org/abs/2107.03024, **Таблица VIII**.
Авторы взяли SOTA-методы, которые в коде используют camera_id в сэмплере, и убрали его (строки со звёздочкой):

| Метод | MSMT17 mAP с камерой | без камеры (camera-agnostic) | Δ |
|---|---|---|---|
| HCD (ICCV'21) | 26.9 | 22.1 | **−4.8** |
| ICE (ICCV'21) | 29.8 | 22.7 | **−7.1** |
| (Market-1501) HCD | 78.1 | 77.7 | −0.4 |
| (Market-1501) ICE | 79.5 | 78.9 | −0.6 |

Вывод: вклад camera-aware сэмплинга растёт с числом камер (Market 6 камер: −0.5; MSMT17 15 камер: −5…−7 mAP). У нас 96 камер.

### 1.4 Вред внутрикамерных позитивов (прямые замеры)

**(a) Выброс одно-камерных кластеров.** ICLR 2025, «Exploring the Camera Bias of Person Re-identification» — https://arxiv.org/abs/2502.10195, **Рис. 7(a)** (MSMT17, backbone CC):

| Стратегия | mAP | R1 | Bias (NMI с camera label) |
|---|---|---|---|
| Baseline | 29.8 | 57.1 | 32.5 |
| + debiased pseudo-labeling | 44.6 | 71.9 | 26.5 |
| + **discarding biased clusters** (выброс кластеров из одной камеры) | **45.4** | 73.7 | 25.4 |
| + оба | **49.1** | 76.5 | 23.8 |
| Ground truth | — | — | 19.2 |

**Таблица 6** той же работы, **включая VeRi-776 (vehicle!)**:

| Метод | Market mAP | MSMT17 mAP | VeRi-776 mAP / R1 |
|---|---|---|---|
| CC* (baseline) | 82.6 | 29.8 | 38.2 / 79.8 |
| CC* + Ours | 85.2 | **49.1** | **45.3 / 89.8** |
| PPLR* | 77.4 | 27.2 | 41.5 / 85.6 |
| PPLR* + Ours | 84.6 | 40.7 | 43.2 / 86.7 |
| PPLR-CAM* (уже camera-aware) | 84.1 | 40.7 | 43.3 / 88.1 |
| PPLR-CAM* + Ours | 84.3 | 44.4 | 43.7 / 88.2 |

Комментарий авторов: «gains for PPLR-CAM are relatively small, which is likely because it uses a camera-aware loss function» — т.е. приёмы не аддитивны. Табл. 11: доля выброшенных сэмплов падает с 63.5% (эпоха 0) до 2.8% (эпоха 100).

**(b) Предельный случай — только внутрикамерные позитивы.** CCFP, ACM MM 2021 — https://arxiv.org/abs/2107.13904, **Таблица 2**. Сильные supervised-бейзлайны, обученные на SCT-данных (нет ни одного кросс-камерного позитива):

| Метод | Market-SCT R1 / mAP | Duke-SCT R1 / mAP | MSMT17-SCT R1 / mAP |
|---|---|---|---|
| bagtricks (CVPR'19) | 54.0 / 34.0 | 54.2 / 42.0 | 20.4 / 9.8 |
| AGW (TPAMI'21) | 56.0 / 36.6 | 56.5 / 43.9 | 23.0 / 11.1 |
| MGN-ibn | 45.6 / 26.6 | 46.7 / 32.6 | 27.8 / 11.7 |
| CCFP (кросс-камерное предсказание фич) | 82.4 / 63.9 | 80.3 / 64.5 | 50.1 / 22.2 |

Для сравнения bagtricks на полном Market-1501 даёт ~94.4 R1 / 86.1 mAP (fast-reid). Т.е. отсутствие кросс-камерных позитивов стоит **~50 mAP**.

### 1.5 «Camera Equalization»

Camera-aware re-identification feature for MTMC tracking, *Image and Vision Computing* 2023 (S0262885623002639). Модули: Camera Equalization (в каждом батче на каждый ID — хотя бы по одному кадру из каждой камеры), Camera Position Embedding, Camera Center Loss.
**Замера нет** — статья за paywall (ScienceDirect, 403), ablation-числа в открытом доступе не найдены. Идея CE совпадает с proxy-balanced sampling из CAP (§1.2), там числа есть.

---

## 2. Состязательные головы / GRL / DANN против предсказуемости камеры

### 2.1 ACAN — Adversarial Camera Alignment Network (arXiv:1908.00862)

https://arxiv.org/abs/1908.00862, **Таблица V** («Evaluation of different components»). Настройка — UCC (unsupervised cross-camera): есть только **внутрикамерные** метки, бейзлайн `Only L_Triplet`.

| Датасет (#камер) | Only L_Triplet (mAP/R1) | ACAN-GRL | ACAN-OCE | Δ mAP (GRL) |
|---|---|---|---|---|
| Market1501 (6) | 34.4 / 58.1 | 50.6 / 73.3 | 47.7 / 72.2 | **+16.2** |
| DukeMTMC (8) | 41.7 / 60.1 | 46.6 / 65.1 | 45.1 / 67.6 | **+4.9** |
| MSMT17 (15) | 10.0 / 24.8 | 11.2 / 27.1 | 12.6 / 33.0 | **+1.2** |
| MARS | 34.3 / 45.5 | 49.1 / 59.2 | 47.5 / 57.7 | +14.8 |
| Duke-Tracklet | 30.7 / 38.4 | 43.0 / 52.0 | 40.3 / 50.4 | +12.3 |

**Критическая оговорка:** бейзлайн ACAN не видит НИ ОДНОГО кросс-камерного позитива. У нас кросс-камерные ID-метки есть, поэтому +16 mAP на Market — не наш ориентир. Наш ориентир — MSMT17 (много камер): **+1.2 mAP**.

**Прямые негативные наблюдения в самой ACAN:**
- Табл. VI: альтернативная схема ACE рушится — Duke 16.0 mAP vs OCE 45.1; MSMT17 3.9 vs 12.6. Т.е. выбор формы adversarial-лосса критичнее, чем его наличие.
- Табл. VII + Рис. 10: OCE **сильнее** уменьшает межкамерное расхождение, чем GRL, но на Market1501 **re-ID хуже** (47.7 vs 50.6 mAP). Прямая цитата: «although the proposed OCE scheme is clearly better than the GRL scheme on distribution discrepancy reduction, its Re-ID performance could be less competitive than the latter». **Лучшая камерная инвариантность ≠ лучший re-ID.**
- Рис. 10 (confusion matrices): «on Duke, ... the GRL scheme almost assigns all images into one camera» — GRL вырождается тем сильнее, чем больше камер (Duke 8, MSMT 15). У нас 96.

### 2.2 CANU-ReID (arXiv:1904.01308)

**Таблица IV** — вклад простого camera-adversarial дискриминатора поверх clustering-based UDA:

| База | Mkt→Duke R1/mAP | Duke→Mkt R1/mAP |
|---|---|---|
| SSG | 73.0 / 53.4 | 80.0 / 58.3 |
| SSG + Adv. (обычный GRL-дискриминатор) | 75.4 / **56.4** | 83.8 / **62.7** |
| CANU-SSG (conditional) | 76.1 / 57.0 | 83.3 / **61.9** (хуже простого Adv.) |
| MMT (DBSCAN) | 80.2 / 67.2 | 91.7 / 79.3 |
| MMT + Adv. | 82.6 / **70.3** | 93.6 / **82.2** |
| CANU-MMT | 83.3 / 70.3 | 94.2 / 83.0 |

Итого: простая adversarial-голова даёт **+2.9…+4.4 mAP**; «умная» conditional-версия добавляет сверху **0…+0.8 mAP**, а в одном из четырёх случаев **хуже** простой.

**Таблица III — катастрофа при слишком большом весе adversarial-лосса** (µ — вес):

| µ | CANU-SSG Mkt→Duke R1/mAP | CANU-SSG Duke→Mkt R1/mAP |
|---|---|---|
| 0.01 | 72.8 / 53.3 | 79.7 / 57.2 |
| **0.05 (опт.)** | **76.1 / 57.0** | **83.3 / 61.9** |
| 0.1 | 74.7 / 56.2 | 82.7 / 61.1 |
| 0.4 | 73.3 / 53.5 | 80.4 / 59.2 |
| **1.8** | **7.1 / 2.9** | **39.1 / 17.1** |

При µ=1.8 модель **разваливается** (57.0 → 2.9 mAP). Диапазон рабочих значений — меньше одного порядка.

Причина, названная в статье: negative transfer — «the camera discriminator learns to infer camera from person identity instead of from the input», т.е. дискриминатор эксплуатирует ID-информацию, и adversarial-градиент выжигает ID-сигнал.

### 2.3 CCAFL / CLIP-CAFL (arXiv:2409.19563)

**Таблица IV** (компонентный ablation):

| Order | компоненты | Market mAP/R1 | MSMT17 mAP/R1 |
|---|---|---|---|
| 0 | Baseline | 85.4 / 93.5 | 45.1 / 73.3 |
| 4 | + ICDL + i2tce (**без ICAL**) | 89.2 / 95.5 | 54.6 / 80.0 |
| 5 | Baseline + **только ICAL** | 88.2 / 95.4 | 53.5 / 80.7 |
| 7 | всё, **с ICAL** | 90.1 / 96.1 | 58.9 / 82.9 |

Вклад adversarial-головы (Order 4 → Order 7): **Market +0.9 mAP, MSMT17 +4.3 mAP.**

**Прямой негативный результат (Рис. 8 + текст §, цитата):** «It is evident that **when the classifier and adversarial training start simultaneously, the model performs worse than without ICAL**». По Рис. 8 (MSMT17) mAP при старте adversarial на эпохе 10 ≈ **47.2** против **54.6** без ICAL вовсе (**−7.4 mAP**); оптимум — старт на эпохе 40.

Практический вывод: adversarial-голову нельзя включать с нуля обучения, нужен warm-up (в CCAFL — 40 эпох из ~60).

---

## 3. Camera-Based Batch Normalization (CBN), ECCV 2020

https://arxiv.org/abs/2001.08680 · код https://github.com/automan000/Camera-based-Person-ReID

### 3.1 Числа

**Таблица 1** (ResNet-50, курсивом в статье — same-domain):

| Обучение | Тест | Conventional BN (R1/mAP) | CBN (R1/mAP) | Δ mAP |
|---|---|---|---|---|
| Market | **Market** | 90.2 / 74.0 | 91.3 / 77.3 | **+3.3** |
| Market | Duke | 37.0 / 20.7 | 58.7 / 38.2 | +17.5 |
| Market | MSMT | 17.1 / 5.5 | 25.3 / 9.5 | +4.0 |
| Duke | Market | 53.2 / 25.1 | 72.7 / 43.0 | +17.9 |
| Duke | **Duke** | 81.5 / 66.6 | 82.5 / 67.3 | **+0.7** |
| MSMT | Market | 58.1 / 30.8 | 73.7 / 45.0 | +14.2 |
| MSMT | **MSMT** | 71.5 / 42.3 | 72.8 / 42.9 | **+0.6** |

**Same-domain выигрыш CBN: +3.3 / +0.7 / +0.6 mAP.** Весь «большой» эффект CBN — это cross-domain.

**Таблица 8** (другие backbone, same-domain): MobileNetV2 Market 69.2 → 73.7 mAP; ShuffleNetV2 Market 58.4 → 65.8 mAP. На слабых backbone эффект больше.

### 3.2 КРИТИЧНО: camera_id на инференсе

Прямая цитата (§4.1): «During testing, before employing the learned ReID model to extract features, the above statistics have to be renewed for every testing camera. In short, **we collect several unlabeled images and calculate the camera-related statistics per testing camera**.»
Алгоритм 1 (Приложение 0.A): изображения группируются «according to their camera ID», из каждой группы сэмплируется N мини-батчей, статистики вкалываются в CBN-слои.

**CBN требует camera_id на тесте. Метки ID не нужны, но разбиение теста по камерам — обязательно. У нас его нет → CBN в чистом виде неприменим.**

Табл. 7: достаточно **10 мини-батчей** на камеру (mAP 77.33, дисперсия 0.007); при 1 батче 76.29 (дисперсия 0.032).

### 3.3 Варианты без camera_id на тесте — **Таблица 6**

| Train | Test | Duke→Duke R1/mAP | Duke→Market R1/mAP |
|---|---|---|---|
| BN | BN | 81.5 / 66.6 | 53.2 / 25.1 |
| IBN | IBN | 77.6 / 57.0 | 61.7 / 29.5 |
| BN | **AdaBN** (статистика по всему тесту, **без camera_id**) | 81.2 / 66.2 | 55.8 / **28.1** |
| BN | CBN | 80.2 / 63.7 | 69.5 / 40.6 |
| CBN | CBN | 82.5 / 67.3 | 72.7 / **43.0** |

**Ответ на вопрос «есть ли вариант без camera_id»: да, AdaBN — но он даёт +3.0 mAP вместо +17.9 mAP у CBN (≈17% эффекта).** Плюс AdaBN чуть портит same-domain (66.6 → 66.2).

Обратите внимание: «BN-train / CBN-test» (81.5→80.2 R1, 66.6→63.7 mAP на Duke→Duke) — прикручивать CBN только на инференсе к BN-обученной модели **вредно** для своего домена.

**Таблица 9** — постепенная замена BN на CBN (Duke→Market): AdaBN 55.8/28.1 → first BN 60.6/31.6 → +block1 61.9/32.9 → +block2 65.0/35.3 → +block3 65.7/35.7 → +block4 67.3/37.0 → +last BN **72.7/43.0**. Частичная замена даёт существенно меньше.

---

## 4. IBN-слои (IBN-Net, arXiv:1807.09441)

### 4.1 Что это конструктивно

Код: https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/resnet_ibn.py
`ibn_cfg = ('a','a','a',None)` — IBN ставится в **`bn1` каждого Bottleneck в layer1/layer2/layer3**; layer4 не трогается. `IBN(planes)` делит каналы пополам: половина → `nn.InstanceNorm2d(half, affine=True)`, половина → `nn.BatchNorm2d(half)`. Свёртки не меняются вообще.

### 4.2 ImageNet (Таблица 2 статьи)

| Model | original top1/top5 err | re-impl | IBN-Net-a |
|---|---|---|---|
| ResNet50 | 24.7 / 7.8 | 24.27 / 7.08 | **22.54 / 6.32** (−1.73 / −0.76) |
| ResNet101 | 23.6 / 7.1 | 22.48 / 6.23 | 21.39 / 5.59 |

Т.е. IBN-версия ResNet-50 на ImageNet **лучше** обычной.

### 4.3 Стоимость

Прямая цитата (§4.1 статьи): «Note that our method brings **no additional parameters** while only add **marginal calculations** during inference phase». IN не имеет running-stats (2×C параметров affine вместо 4×C у BN → параметров чуть **меньше**). Отдельного замера FLOPs/латентности/памяти в статье **нет** — **замера нет**.

### 4.4 Person re-ID (README IBN-Net, таблица «Person re-identification», R1 (mAP))

| Backbone | Market-1501 | DukeMTMC-reID |
|---|---|---|
| ResNet50 | 94.5 (85.9) | 86.4 (76.4) |
| ResNet101 | 94.5 (87.1) | 87.6 (77.6) |
| SeResNeXt101 | 95.0 (88.0) | 88.4 (79.0) |
| **IBN-Net-a (R50-ibn-a)** | **95.0 (88.2)** | **90.1 (79.1)** |

Δ к ResNet50: Market **+0.5 R1 / +2.3 mAP**; Duke **+3.7 R1 / +2.7 mAP**. R50-ibn-a обгоняет ResNet101 и SeResNeXt101.

### 4.5 fast-reid MODEL_ZOO — R50 vs R50-ibn

https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md

| Датасет / пайплайн | R50 (R1 / mAP) | R50-ibn (R1 / mAP) | Δ mAP |
|---|---|---|---|
| Market1501 BoT | 94.4 / 86.1 | 94.9 / 87.6 | +1.5 |
| Market1501 AGW | 95.3 / 88.2 | 95.1 / 88.7 | +0.5 (**R1 −0.2**) |
| Market1501 SBS | 95.4 / 88.2 | 95.7 / 89.3 | +1.1 |
| DukeMTMC BoT | 87.2 / 77.0 | 89.3 / 79.6 | **+2.6** |
| DukeMTMC AGW | 89.0 / 79.9 | 90.5 / 80.8 | +0.9 |
| DukeMTMC SBS | 90.3 / 80.3 | 90.8 / 81.2 | +0.9 |
| **MSMT17 BoT** (15 камер) | 74.1 / 50.2 | 77.0 / 54.4 | **+4.2** |
| **MSMT17 AGW** | 78.3 / 55.6 | 81.2 / 59.7 | **+4.1** |
| **MSMT17 SBS** | 81.8 / 58.4 | 83.9 / 60.6 | **+2.2** |

**Чёткая закономерность: выигрыш IBN растёт с числом камер** (Market 6 → +0.5…+1.5; Duke 8 → +0.9…+2.6; MSMT17 15 → +2.2…+4.2).

### 4.6 VeRi / VERI-Wild / VehicleID в fast-reid — **замера R50 vs R50-ibn НЕТ**

В MODEL_ZOO опубликованы **только ibn-конфиги** для vehicle-датасетов, R50-бейзлайна нет:
- VeRi, SBS(R50-ibn): R1 97.0% / mAP 81.9% / mINP 46.3%
- VehicleID, BoT(R50-ibn): Small R1 86.6 / Medium 82.9 / Large 80.6
- VERI-Wild, BoT(R50-ibn): Small 96.4 R1 / 87.7 mAP; Medium 95.1 / 83.5; Large 92.5 / 77.3

Косвенное свидетельство по vehicle: победители AI City 2021 (arXiv:2105.09701, Табл. 6) использовали **только IBN-backbone** (ResNet101-IBN-a, ResNeXt101-IBN-a, SeResNet101-IBN-a, DenseNet169-IBN-a) — но сравнения с не-IBN версиями в статье **нет**.

### 4.7 Можно ли добавить IBN к уже предобученной ResNet-50 без потери весов

- **Свёрточные веса — да, полностью.** IBN меняет только нормализацию.
- **BN-статистики — частично теряются:** половина каналов `bn1` в layer1–3 становится InstanceNorm (running_mean/var этих каналов выбрасываются; affine weight/bias можно скопировать 1:1, т.к. размерности совпадают). Это ~1/6 всех BN-слоёв ResNet-50 (по одному bn1 в каждом из 13 bottleneck-ов layer1–3).
- **Правильный путь — не конвертировать, а взять готовые ImageNet-веса `resnet50_ibn_a`:** `torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)`, либо `MODEL.BACKBONE.WITH_IBN True` в fast-reid. Они лучше обычных на ImageNet (22.54 vs 24.27 top-1 err), так что менять на них ImageNet-инициализацию — чистый выигрыш.
- Замера «конверсия без дообучения vs готовые IBN-веса» **нет** ни в одной из найденных работ.

### 4.8 Когда IBN ВРЕДИТ (см. также §9)

- CBN, Табл. 6: IBN на Duke→Duke **57.0 mAP против 66.6 у BN (−9.6)**, при выигрыше на Duke→Market (+4.4).
- IICS, Табл. 2: «+ IBN» на Market **67.1 → 59.8 mAP (−7.3)**, на Duke **51.4 → 35.4 (−16.0)**.
- MetaBIN, Табл. 5: «BN+IN half» (поканальная смесь, как в IBN-a) на supervised Market **67.9 → 53.9 mAP (−14.0)**.
- CCFP, Табл. 2: MGN-ibn на Market-SCT 26.6 mAP против bagtricks 34.0.

Все четыре — про **обучение IBN-архитектуры с нуля/из ImageNet в конкретном пайплайне**, тогда как fast-reid/IBN-Net показывают выигрыш. Разница, судя по всему, в рецепте (LR для IN-веток, стадия обучения). Вывод: IBN **надо мерить**, а не ставить вслепую.

---

## 5. SNR и другие камера-инвариантные нормализации

### 5.1 SNR — Style Normalization and Restitution, CVPR 2020 (arXiv:2005.11037), Таблица 1

| Сеттинг | Baseline mAP/R1 | Baseline-A-IN | **SNR** | Δ SNR |
|---|---|---|---|---|
| Market → Duke | 19.8 / 35.3 | 24.1 / 42.7 | **33.6 / 55.1** | **+13.8 mAP / +19.8 R1** |
| Duke → Market | 21.8 / 48.3 | 26.5 / 56.0 | **33.9 / 66.7** | **+12.1 / +18.4** |
| **Market → Market (same-domain)** | 82.8 | — | **84.7** | **+1.9** |
| **Duke → Duke (same-domain)** | 71.2 | — | **72.9** | **+1.7** |

Табл. 2a: «SNR w/o L_SNR» (без dual causality loss) даёт 26.1 mAP на M→D против 33.6 у полного SNR — т.е. **7.5 mAP приходится именно на restitution-лосс**, а не на IN.

**Главное для нас: same-domain выигрыш SNR — всего +1.9 / +1.7 mAP.** SNR — метод для unseen-домена.

### 5.2 MetaBIN, CVPR 2021 (arXiv:2011.14670)

**Таблица 5** — сравнение нормализаций, DG (large-scale benchmark) vs supervised (Market-1501):

| Нормализация | DG R1 / mAP | **Supervised Market R1 / mAP** |
|---|---|---|
| BN | 50.9 / 59.5 | **87.2 / 67.9** |
| IN | 54.9 / 63.3 | **71.9 / 46.1** (катастрофа) |
| DualNorm | 57.6 / 61.8 | 82.6 / 57.2 |
| BN+IN half (как IBN-a) | 56.5 / 65.3 | **79.5 / 53.9** |
| BIN | 54.8 / 63.1 | 87.5 / 67.8 |
| **MetaBIN** | **64.7 / 72.3** | **87.9 / 68.5** |

**MetaBIN на same-domain supervised: +0.7 R1 / +0.6 mAP над BN.** На DG: +13.8 R1 / +12.8 mAP.

**Таблица 3** (ablation на DG): BN baseline 50.9/59.5 → MetaBIN+CE 60.6/69.4 → +triplet 62.8/70.8 → +L_scat 63.0/71.0 → +L_shuf 63.1/71.0 → fixed β 63.5/71.3 → **cyclic β 64.7/72.3**.
**Таблица 4:** BIN один 54.8/63.1; BIN+MLDG+cyclic β 58.4/66.3; MetaBIN (с разделением эпизодов) 64.7/72.3.

**Таблица 2** (single-source DG): M→D 55.2 R1 / 33.1 mAP; D→M 69.2 / 35.9 (SNR: 55.1/33.6 и 66.7/33.9).
**Ключевая для нас цитата из §4.2:** «…some identities are simultaneously distributed on different camera domains. …**we divided the entire camera domains in half for each dataset**». Т.е. в single-source-режиме MetaBIN использует **камеры как домены** — ровно наша ситуация (camera_id только в train).

**Стоимость:** Рис. 2 — meta-learning pipeline «does not increase the memory usage» относительно обновления базовой модели (7 895 MiB); преимущество перед MLDG по памяти.

### 5.3 AIBN (Adaptive Instance-and-Batch Norm), IICS CVPR 2021 (arXiv:2103.11658)

**Таблица 2** (unsupervised, ResNet-50):

| Вариант | Market mAP / R1 | Duke mAP / R1 |
|---|---|---|
| Backbone (BN) | 67.1 / 85.5 | 51.4 / 71.3 |
| + IBN | **59.8 / 81.1** | **35.4 / 56.3** |
| + IN | 59.6 / 83.2 | 53.0 / 72.7 |
| + AIBN (α фиксирован 0.5) | 70.7 / 88.0 | 56.9 / 75.2 |
| **+ AIBN (α обучаемый)** | **72.1 / 88.8** | **59.1 / 76.9** |

**Таблица 3** — обобщение backbone:

| Train | Тест | w/o AIBN mAP/R1 | w/ AIBN mAP/R1 | Δ mAP |
|---|---|---|---|---|
| Market | **Market** | 79.9 / 92.0 | 80.0 / 92.0 | **+0.1** |
| Market | Duke | 22.4 / 38.2 | 29.5 / 49.5 | **+7.1** |
| Duke | Market | 21.8 / 48.9 | 27.2 / 54.5 | **+5.4** |
| Duke | **Duke** | 68.7 / 83.9 | 69.2 / 84.8 | **+0.5** |

**Таблица 4** — куда ставить: All 72.1/88.7 (Market mAP/R1) и 58.7/77.1 (Duke); Layer1+2 64.1/86.2 и 53.3/73.2; Layer4 72.1/88.5 и 57.5/76.1; **Layer3+4 72.1/88.8 и 59.1/76.9** (выбрано авторами); Layer2+3+4 71.3/88.7 и 59.8/77.4.

### 5.4 Patch-aware Batch Normalization (arXiv:2304.02848)

**Таблица IX** (единственный retrieval-замер; cross-dataset Market↔GRID, backbone OSNet):

| Метод | Market→GRID mAP / R1 | GRID→Market mAP / R1 |
|---|---|---|
| Baseline | 33.3±0.4 / 24.5±0.4 | 3.9±0.4 / 13.1±1.0 |
| MixStyle | 33.8±0.9 / 24.8±1.6 | 4.9±0.2 / 15.4±1.2 |
| MixStyle + PBN | 35.9±2.3 / 28.0±2.3 | 5.6±0.7 / 16.9±1.6 |
| EFDMix | 35.5±1.8 / 26.7±3.3 | 6.4±0.2 / 19.9±0.6 |
| EFDMix + PBN | 37.0±0.9 / 28.0±2.0 | 6.8±0.3 / 20.5±0.5 |

Разброс (±2.3) сопоставим с эффектом (+2.1). **Same-domain re-ID замера в статье нет.** Ставить PBN в продакшен на этих данных оснований нет.

---

## 6. Camera-style transfer (CamStyle, CVPR 2018, arXiv:1711.10295)

### 6.1 Числа

**Таблица 4 (Market-1501)** и **Таблица 5 (DukeMTMC-reID)**:

| Метод | Market R1 / mAP | Duke R1 / mAP |
|---|---|---|
| IDE* (baseline) | 85.66 / 65.87 | 72.31 / 51.83 |
| IDE* + CamStyle | **88.12 / 68.72** (+2.46 / **+2.85**) | **75.27 / 53.48** (+2.96 / **+1.65**) |
| IDE* + CamStyle + Random Erasing | 89.49 / 71.55 | 78.32 / 57.61 |

**Таблица 3 (Market, комбинации аугментаций)** — CamStyle в изоляции почти ничего не даёт:

| RF+RC | RE | CamStyle | R1 / mAP |
|---|---|---|---|
| — | — | — | 84.15 / 64.10 |
| ✓ | — | — | 85.66 / 65.87 |
| — | ✓ | — | 86.83 / 68.50 |
| — | — | **✓** | **85.01 / 64.86** (+0.86 R1 / **+0.76 mAP**) |
| ✓ | ✓ | — | 87.65 / 69.91 |
| ✓ | — | ✓ | 88.12 / 68.72 |
| ✓ | ✓ | ✓ | 89.49 / 71.55 |

**Random Erasing один даёт больше (+4.40 mAP), чем CamStyle один (+0.76 mAP).**

### 6.2 Стоимость и масштабирование — наш случай прямо противопоказан

Цитата §4.2: «we train a camera-aware style transfer (CycleGAN) model for each pair of cameras. Specifically, **we train C(6,2)=15 and C(8,2)=28 CycleGAN models for Market-1501 and DukeMTMC-reID**». Формула — **C(N,2) = N(N−1)/2**.

**Для наших 96 камер: C(96,2) = 4 560 CycleGAN-моделей.**
Генерация: «for each training image, we generated L−1 (5 for Market-1501)» фейков → для нас **95 фейков на кадр** → 7 248 × 95 = **688 560 сгенерированных изображений** на 7 248 реальных.

**Таблица 2 + Рис. 7 — эффект ПАДАЕТ с ростом числа камер** (vanilla CamStyle без LSR, Rank-1):

| Система | Market Δ R1 |
|---|---|
| 2 камеры | **+17.1** (43.2 → 60.3) |
| 6 камер (полная) | **+0.7** |

Цитата §4.4: «as the number of camera increases in the system, the improvement of vanilla CamStyle becomes smaller. For example, in the 6-camera system on Market-1501, the improvement in rank-1 accuracy is **only +0.7%**. This indicates that 1) the over-fitting problems becomes less severe … and 2) **the noise brought by CycleGAN begins to negatively affect the system accuracy**.»

Плюс Таблица 2: обучение CycleGAN только на 1-й и 2-й камерах даёт 87.20/67.64 против 88.12/68.72 на всех шести — т.е. **85% эффекта получается от 1 из 15 моделей**.

**Оценка реалистичности для нас — прямо: неприменимо.** 4 560 CycleGAN ≈ (при ~6–12 ч на модель на одной GPU) порядка **3–6 GPU-лет**; данных на пару камер у нас в среднем 7 248/96 ≈ 75 кадров на камеру — CycleGAN на 75 изображениях не обучится. Плюс задокументированный тренд: чем больше камер, тем меньше эффект (на 6 камерах уже +0.7 R1).

---

## 7. SIE (Side Information Embedding) из TransReID

https://arxiv.org/abs/2102.04378 (ICCV 2021), **Таблица 4**:

| Вариант | MSMT17 mAP / R1 | VeRi-776 mAP / R1 |
|---|---|---|
| Baseline (ViT, λ=0) | 61.0 / 81.8 | 78.2 / 96.5 |
| + S_C[r] (**только камера**) | **62.4 / 81.9** (**+1.4 mAP** / +0.1 R1) | **78.7 / 97.1** (**+0.5 mAP** / +0.6 R1) |
| + S_V[q] (только viewpoint) | — | 78.5 / 96.9 (+0.3) |
| + S_(C,V) (совместно) | — | **79.6 / 96.9** (+1.4) |

Ablation λ (Рис. 7, текст §4.5): MSMT17 61.0 → **63.0** при λ=2.0; VeRi-776 78.2 → **79.9** при λ=2.5. «Continuing to increase λ, the performance is degraded because the weights for feature embedding and the position embedding are weakened.»
Отдельно: раздельное кодирование `S_C[r] + S_V[q]` вместо совместного `S_(C,V)` даёт только 78.3 mAP на VeRi-776.

**Подтверждение пользовательской оценки «~+1.4 mAP»: да, это MSMT17, camera-only. На vehicle (VeRi-776, 20 камер) camera-only SIE даёт всего +0.5 mAP.**

### camera_id на инференсе — ТРЕБУЕТСЯ

Проверено по коду: https://github.com/damo-cv/TransReID/blob/main/processor/processor.py — и в `do_inference`, и в валидации:
```python
feat = model(img, cam_label=camids, view_label=target_view)
```
SIE — это обучаемый эмбеддинг, **прибавляемый ко входным patch-эмбеддингам**, поэтому без camera_id фичу не извлечь. Обходной путь (подставлять фиксированный/усреднённый camera-эмбеддинг) в статье **не измерен — замера нет**.

**Вывод: SIE у нас неприменим.**

---

## 8. Что делать, когда camera_id есть ТОЛЬКО в обучении

### 8.1 ПРИМЕНИМО (camera_id нужен только на train)

| Приём | Почему применимо | Ожидаемый Δ mAP (по замерам) |
|---|---|---|
| **Cross-camera positive mining** (позитив в триплете — обязательно из другой камеры) | сэмплер — train-only | нет прямого замера; косвенно: SCT-бейзлайны без кросс-кам позитивов теряют ~50 mAP (CCFP Табл. 2) |
| **Same-camera hard negative mining** (`Triplet-same`) | сэмплер/лосс — train-only | **+9.8…+24.6** (MCNL Табл. 3, SCT-режим) |
| **MCNL-ранжирование** dist+ < dist−,other < dist−,same | train-only | +22.4…+34.0 (MCNL Табл. 3, SCT) |
| **Camera-balanced (proxy-balanced) сэмплинг** | train-only | +1.3…+5.7 (CAP Табл. 1) |
| **Выброс/понижение веса одно-камерных «кластеров» ID** | train-only | +15.6 (ICLR'25 Рис. 7a). *У нас каждый ID уже ≥2 камеры → напрямую неприменимо, но применимо к «одно-камерным» позитивным ПАРАМ* |
| **Adversarial camera head (GRL/DANN)** | голова отбрасывается на инференсе | +0.9…+4.3 на много-камерных датасетах (CCAFL Табл. IV; CANU Табл. IV). Требует warm-up и очень аккуратного веса |
| **IBN (R50-ibn-a)** | архитектура, camera_id не нужен нигде | +2.2…+4.2 на MSMT17 (15 камер), fast-reid MODEL_ZOO |
| **SNR** | архитектура | same-domain **+1.9 / +1.7** (SNR Табл. 1) |
| **MetaBIN / MixStyle с камерами как доменами** | camera_id только для формирования meta-эпизодов / для shuffle в батче | MixStyle **+4.1…+4.3** cross-domain (Табл. 2); MetaBIN same-domain **+0.6** (Табл. 5) |
| **AIBN** | архитектура | same-domain **+0.1…+0.5**, cross **+5.4…+7.1** (IICS Табл. 3) |
| **Camera-similarity penalty à la VOC-ReID** | отдельная camera-ReID сеть учится на train (camera_id есть); на инференсе камера **предсказывается по изображению** и её схожесть вычитается из финальной | **+3.8 mAP / +5.0 R1** (VOC-ReID Табл. 3) |
| **CamStyle** | генерация только на train | формально применим; практически — нет, см. §6.2 |
| **Camera-agnostic centering/scaling эмбеддингов** на тесте | статистика по всей галерее, без camera_id | **+1.6 mAP** (ICLR'25 Табл. 4, столбец «Entire») |

### 8.2 НЕПРИМЕНИМО (нужен camera_id на инференсе)

| Приём | Почему нельзя | Сколько теряем |
|---|---|---|
| **CBN** | тестовые изображения надо группировать по камере и пересчитывать per-camera BN-статистики (цитата §4.1 + Алг. 1) | −17.9 mAP на cross-domain; частичная замена — AdaBN (+3.0 вместо +17.9) |
| **SIE / TransReID** | camera-эмбеддинг складывается со входными patch-эмбеддингами: `model(img, cam_label=camids)` | −1.4 mAP (MSMT17), −0.5 (VeRi-776) |
| **Camera Verification** (выкинуть из галереи кандидатов с тем же camera_id) | напрямую требует camera_id запроса и галереи | −5.9 mAP (AI City Табл. 4) |
| **Inter-Camera Fusion** | усреднение фичи с фичами «своей камеры» на инференсе | −2.0 mAP (там же) |
| **Camera-specific feature normalization** (ICLR'25) | центрирование/масштабирование по статистике своей камеры | −8.7 mAP на Market (Табл. 3a); camera-agnostic вариант даёт только +1.6 |
| **Camera-aware re-ranking / camera bias subtraction по camera_id** | — | −1.3 mAP (AI City Табл. 4, «Camera and Orientation Bias») |

**Суммарная цена отсутствия camera_id на инференсе, по vehicle-замерам AI City 2021 (arXiv:2105.09701, Табл. 4): 66.8 → 76.0 mAP, т.е. ≈ 9.2 mAP пост-обработки недоступно.** Единственный частичный заменитель без camera_id — **VOC-ReID-style camera-ReID сеть** (§8.1), которая предсказывает камеру по изображению: +3.8 mAP (Табл. 3 VOC-ReID).

### 8.3 Дополнительно: можно ли вообще восстановить camera_id?

VOC-ReID (arXiv:2004.09164, §3) прямо строит **camera re-identification** сеть: camera_id в train есть → обучают RECT-Net на camera-ID как на классе → на инференсе считают `fc(xi)·fc(xj)` и вычитают из финальной схожести. Табл. 3 (CityFlow val):

| Vehicle ReID | + Orientation ReID | + Camera ReID | mAP | R1 |
|---|---|---|---|---|
| ✓ | | | 44.4 | 65.3 |
| ✓ | ✓ | | 47.0 | 70.5 |
| ✓ | ✓ | ✓ | **50.8** | **75.5** |

Вклад именно camera-ветки: **+3.8 mAP / +5.0 R1**. Табл. 5, VeRi-776: RECT-Net(320) 81.6/96.8 → +Orientation 82.8/97.6 (camera-ветка для VeRi-776 отдельно не замерена — **замера нет**).

---

## 9. Отрицательные результаты (где camera-invariant методы дают мало или вредят)

### 9.1 На сильных backbone / на «своём» домене приросты почти исчезают

**ICLR 2025, arXiv:2502.10195, Таблица 3(a)** — camera-specific feature normalization на **seen-домене (MSMT17, на котором обучались)**:

| Модель | MSMT17 mAP до / после | Δ |
|---|---|---|
| PPLR-CAM (camera-aware) | 42.2 / 41.3 | **−0.9** |
| CAJ (camera-aware) | 44.3 / 42.8 | **−1.5** |
| TransReID (supervised, camera-aware) | 67.8 / 66.7 | **−1.1** |
| PAT (supervised) | 54.8 / 54.1 | **−0.7** |
| SOLIDER (supervised, ViT) | 77.1 / 77.0 | **−0.1** |
| SPCL (слабая USL) | 19.1 / 20.3 | +1.2 |
| CC (USL) | 29.8 / 32.2 | +2.4 |

Чем сильнее модель и чем ближе домен — тем меньше (вплоть до минуса) даёт камерное дебиасирование.

### 9.2 Прирост нормализационных трюков на same-domain ≈ 0

| Метод | Same-domain Δ mAP | Cross-domain Δ mAP | Источник |
|---|---|---|---|
| CBN | +3.3 / +0.7 / +0.6 | +17.5 / +17.9 / +14.2 | arXiv:2001.08680 Табл. 1 |
| SNR | +1.9 / +1.7 | +13.8 / +12.1 | arXiv:2005.11037 Табл. 1 |
| MetaBIN | **+0.6** | +12.8 | arXiv:2011.14670 Табл. 5 |
| AIBN | **+0.1 / +0.5** | +7.1 / +5.4 | arXiv:2103.11658 Табл. 3 |

### 9.3 IN/IBN могут сильно портить supervised-качество

| Замер | mAP |
|---|---|
| MetaBIN Табл. 5, supervised Market: BN 67.9 → **IN 46.1** | **−21.8** |
| MetaBIN Табл. 5, supervised Market: BN 67.9 → **«BN+IN half» 53.9** | **−14.0** |
| IICS Табл. 2, Duke: Backbone 51.4 → **+IBN 35.4** | **−16.0** |
| IICS Табл. 2, Market: 67.1 → **+IBN 59.8** | **−7.3** |
| CBN Табл. 6, Duke→Duke: BN 66.6 → **IBN 57.0** | **−9.6** |
| ICLR'25 Табл. 4: ZCA whitening (entire) на CUHK03-NP 27.2 → **18.7** | **−8.5** |
| fast-reid, Market AGW: R50 95.3 R1 → R50-ibn **95.1 R1** | **−0.2 R1** |

### 9.4 Adversarial: провалы и развалы

| Замер | Результат |
|---|---|
| CANU Табл. III: вес adversarial µ=1.8 | Mkt→Duke **57.0 → 2.9 mAP** (развал) |
| CCAFL Рис. 8 + текст: старт adversarial одновременно с классификатором | MSMT17 **≈47.2 mAP против 54.6 без ICAL вовсе (−7.4)** |
| ACAN Табл. VI: схема ACE вместо OCE | Duke **45.1 → 16.0 mAP**; MSMT17 **12.6 → 3.9** |
| ACAN Табл. V vs VII: OCE лучше выравнивает камеры, но на Market **хуже re-ID** | 50.6 (GRL) vs 47.7 (OCE) mAP |
| ACAN Рис. 10: GRL на Duke «almost assigns all images into one camera» | вырождение GRL растёт с числом камер |
| ACAN Табл. V, MSMT17 (15 камер): вклад GRL-головы | **всего +1.2 mAP** |
| CCAFL Табл. IV, Market (6 камер): вклад ICAL | **+0.9 mAP** |
| CANU §IV: negative transfer — дискриминатор выводит камеру из ID | описано, число не дано |

### 9.5 CamStyle на больших камерных сетях

- Market, 2 камеры: **+17.1 R1**; Market, 6 камер: **+0.7 R1** (arXiv:1711.10295, §4.4 + Рис. 7).
- Без других аугментаций CamStyle даёт **+0.76 mAP**, тогда как Random Erasing — **+4.40 mAP** (Табл. 3).
- Стоимость растёт как C(N,2): 15 моделей для 6 камер, **4 560 для 96**.

### 9.6 Прочее

- MixStyle статья (arXiv:2104.02008, §3.2): **RandomErase «shows a detrimental effect in the cross-dataset re-ID setting»**, DropBlock тоже «unable to show any benefit» — типовые re-ID аугментации не переносятся на камерную/доменную инвариантность.
- PBN (arXiv:2304.02848, Табл. IX): стандартные отклонения ±0.9…±2.3 при эффектах +1.5…+2.1 — эффект на уровне шума; same-domain замера вообще нет.
- CANU Табл. IV: «умная» conditional-версия adversarial по сравнению с простой даёт 0…+0.8 mAP, а на SSG Duke→Mkt **хуже** (61.9 vs 62.7).

---

## 10. Domain-generalization baselines для «одна выборка, много камер»

### 10.1 MixStyle (ICLR 2021, arXiv:2104.02008)

Прямая цитата §3.2: «**As each camera view is itself a distinct domain**, person re-ID is essentially a cross-domain image matching problem» — в re-ID экспериментах «domain label» = **camera ID** (обучение на одном датасете).

**Таблица 2**:

| Модель | Market→Duke mAP / R1 | Duke→Market mAP / R1 |
|---|---|---|
| ResNet-50 | 19.3 / 35.4 | 20.4 / 45.2 |
| + MixStyle (random shuffle, **без** camera_id) | 23.8 / 42.2 (**+4.5**) | 24.1 / 51.5 (+3.7) |
| + MixStyle (**w/ camera label**) | 23.4 / 43.3 (**+4.1**) | 24.7 / 53.0 (**+4.3**) |
| OSNet | 25.9 / 44.7 | 24.0 / 52.2 |
| + MixStyle (random shuffle) | 27.2 / 48.2 | 27.8 / 58.1 |
| + MixStyle (w/ camera label) | 27.3 / 47.5 | 29.0 / 58.2 |

Куда ставить: «MixStyle is inserted **after the 1st and 2nd residual blocks**» (layer1, layer2 в ResNet-50). Стоимость: «only few lines of code», без генерации изображений; отдельного замера FLOPs/времени **нет**.
**Важно: random shuffle (без camera_id) не хуже версии с camera-метками** — на M→D даже лучше (+4.5 vs +4.1). Same-domain замера в статье **нет**.

### 10.2 MetaBIN (CVPR 2021) — см. §5.2

Ключевое: в single-source-режиме камеры используются как домены (делятся пополам на meta-train/meta-test). DG-прирост +13.8 R1 / +12.8 mAP; supervised same-domain +0.7 R1 / **+0.6 mAP**.

### 10.3 DomainMix (BMVC 2021, arXiv:2011.11953)

Метод про «размеченный синтетический + неразмеченный реальный датасет» без человеческой разметки; домены = **датасеты**, не камеры. У нас нет ни синтетики, ни второго неразмеченного корпуса.
**Для нашей задачи неприменим; релевантного замера (камеры как домены) в статье нет — замера нет.**

### 10.4 Сравнительная сводка DG-бейзлайнов (M→D / D→M, mAP)

| Метод | M→D mAP / R1 | D→M mAP / R1 | Источник |
|---|---|---|---|
| IBN-Net | 24.3 / 43.7 | 23.5 / 50.7 | MetaBIN Табл. 2 |
| OSNet | 25.9 / 44.7 | 24.0 / 52.2 | там же |
| OSNet-IBN | 27.6 / 47.9 | 27.4 / 57.8 | там же |
| CrossGrad | 27.1 / 48.5 | 26.3 / 56.7 | там же |
| QAConv | 28.7 / 48.8 | 27.2 / 58.6 | там же |
| L2A-OT | 29.2 / 50.1 | 30.2 / 63.8 | там же |
| OSNet-AIN | 30.5 / 52.4 | 30.6 / 61.0 | там же |
| **SNR** | **33.6 / 55.1** | 33.9 / 66.7 | там же / SNR Табл. 1 |
| **MetaBIN** | 33.1 / 55.2 | **35.9 / 69.2** | MetaBIN Табл. 2 |
| CBN (Market→Duke) | 38.2 / 58.7 | 43.0 / 72.7 | CBN Табл. 1 (**нужен camera_id на тесте**) |

---

## 11. Ранжирование для нашего масштаба (96 камер, 7 248 кадров, camera_id только в train)

Приоритет = (ожидаемый Δ mAP на нашем «честном» кросс-камерном протоколе) / (стоимость внедрения), с поправкой на «много камер, мало данных».

| Ранг | Приём | Опорное число | Почему такой приоритет у нас |
|---|---|---|---|
| **1** | **Same-camera hard-negative + cross-camera hard-positive в batch-hard triplet** (MCNL-схема) | +9.8…+24.6 mAP (MCNL Табл. 3), pseudo-F камерной разделимости 8.43→0.40 | Нулевая стоимость (50 строк в сэмплере/лоссе), напрямую бьёт по механизму, который мы наблюдаем (0.976 vs 0.379). Единственный приём, который **прямо измеряет и снижает** камерную разделимость фич |
| **2** | **Camera-balanced (proxy-balanced) сэмплинг батчей** | +1.3…+5.7 mAP (CAP Табл. 1); удаление camera-aware сэмплинга стоит −4.8…−7.1 на MSMT17 (arXiv:2107.03024 Табл. VIII) | При 96 камерах и 75 кадрах на камеру class-balanced сэмплинг гарантированно перекошен. Стоимость — сэмплер |
| **3** | **R50 → R50-IBN-a (готовые ImageNet-веса)** | MSMT17 (15 камер) +2.2…+4.2 mAP; Duke +0.9…+2.6; Market +0.5…+1.5 | Один флаг (`WITH_IBN`), 0 доп. параметров; выигрыш **растёт с числом камер**. НО: 4 независимых замера показывают провалы (−7…−16 mAP) в других пайплайнах → обязательно A/B |
| **4** | **MixStyle после layer1/layer2, камеры = домены** | M→D +4.1…+4.5 mAP, D→M +3.7…+4.3 (Табл. 2) | Ровно наш сеттинг (camera_id только в train, на инференсе выключается). ~15 строк кода. Оговорка: same-domain замера нет, эффект измерен только cross-dataset |
| **5** | **VOC-ReID-style camera-similarity penalty** (отдельная camera-ReID сеть, камера предсказывается по изображению) | +3.8 mAP / +5.0 R1 (VOC-ReID Табл. 3, vehicle/CityFlow) | Единственный способ вернуть часть из ~9.2 mAP камерной пост-обработки **без camera_id на инференсе**. Стоимость — вторая сеть (96-классовая классификация камер). Vehicle-домен, тот же, что у нас |
| **6** | **SNR-модули** | same-domain +1.9 / +1.7 mAP; cross +13.8 / +12.1 | Архитектурный, camera-free. На своём домене прирост скромный; наш протокол ближе к cross-camera, чем к чистому same-domain, поэтому может быть выше — но замера «96 камер, одна выборка» нет |
| **7** | **Adversarial camera head (GRL) с warm-up** | +0.9 mAP на 6-камерном Market, +1.2 на 15-камерном MSMT17, +4.3 в CCAFL | Низкий ожидаемый выигрыш при высоком риске: развал при µ=1.8 (57.0→2.9 mAP), −7.4 mAP при старте без warm-up, вырождение GRL при большом числе камер. Только после 1–4, с warm-up ≥60% эпох и λ≤0.1 |
| **8** | **AIBN (layer3+4) / MetaBIN** | AIBN same-domain +0.1…+0.5; MetaBIN same-domain +0.6 | Затраты на meta-learning не окупаются same-domain замерами; MetaBIN окупается только если целимся в unseen-камеры |
| **9** | **Camera-agnostic centering/scaling эмбеддингов на тесте** | +1.6 mAP (ICLR'25 Табл. 4, «Entire») | Тривиально (5 строк, пост-обработка), но мало. Взять «бесплатно» вместе с любым из выше |
| **10** | **AdaBN на тесте** | +3.0 mAP cross-domain, −0.4 same-domain (CBN Табл. 6) | Дешёвый суррогат CBN без camera_id; риск ухудшения на своём домене |
| — | **CBN, SIE, Camera Verification, Inter-Camera Fusion, camera-specific normalization** | — | **Неприменимо**: требуют camera_id на инференсе |
| — | **CamStyle** | — | **Неприменимо**: 4 560 CycleGAN, 688 560 фейков, ~75 кадров на камеру для обучения GAN, и измеренный тренд «+17.1 R1 на 2 камерах → +0.7 R1 на 6 камерах» |
| — | **Patch-aware BN** | — | Эффект в пределах ±σ, same-domain замера нет |
| — | **DomainMix** | — | Другой сеттинг (синтетика + неразмеченный реальный корпус) |

### Замечание об аддитивности

ICLR'25 (arXiv:2502.10195, §5.4) прямо предупреждает: «the gains for PPLR-CAM are relatively small, which is likely because it uses a camera-aware loss function» — т.е. **camera-aware лосс и camera-aware дебиасирование перекрываются**. Суммировать Δ из таблиц выше нельзя; после приёма №1 (same-camera negatives) выигрыш от №7 (adversarial) следует ожидать существенно меньшим, чем +4.3 mAP из CCAFL.

---

## Источники (открывались полностью)

1. MCNL / Single Camera Training — https://arxiv.org/abs/1909.10848 (Табл. 3, Рис. 3)
2. CAP — https://arxiv.org/abs/2012.10674 (Табл. 1)
3. Rethinking Sampling Strategies — https://arxiv.org/abs/2107.03024 (Табл. VIII)
4. Exploring the Camera Bias of Person Re-ID (ICLR 2025) — https://arxiv.org/abs/2502.10195 (Табл. 1, 3, 4, 6, 11, Рис. 7)
5. CCFP — https://arxiv.org/abs/2107.13904 (Табл. 2, 3)
6. ACAN — https://arxiv.org/abs/1908.00862 (Табл. V, VI, VII, Рис. 10)
7. CANU-ReID — https://arxiv.org/abs/1904.01308 (Табл. III, IV)
8. CCAFL / CLIP camera-agnostic — https://arxiv.org/abs/2409.19563 (Табл. IV, Рис. 8)
9. CBN — https://arxiv.org/abs/2001.08680 (Табл. 1, 6, 7, 8, 9, Алг. 1)
10. IBN-Net — https://arxiv.org/abs/1807.09441 (Табл. 2, 3) · README https://github.com/XingangPan/IBN-Net · код `ibnnet/resnet_ibn.py`
11. fast-reid MODEL_ZOO — https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md
12. SNR — https://arxiv.org/abs/2005.11037 (Табл. 1, 2a)
13. MetaBIN — https://arxiv.org/abs/2011.14670 (Табл. 2, 3, 4, 5, Рис. 2)
14. IICS / AIBN — https://arxiv.org/abs/2103.11658 (Табл. 2, 3, 4)
15. Patch-aware BN — https://arxiv.org/abs/2304.02848 (Табл. IX)
16. CamStyle — https://arxiv.org/abs/1711.10295 (Табл. 1–5, Рис. 7, §4.2, §4.4)
17. TransReID / SIE — https://arxiv.org/abs/2102.04378 (Табл. 4, 5, Рис. 7) · код https://github.com/damo-cv/TransReID `processor/processor.py`
18. MixStyle — https://arxiv.org/abs/2104.02008 (Табл. 2, §3.2)
19. VOC-ReID — https://arxiv.org/abs/2004.09164 (Табл. 3, 5)
20. AI City 2021 Track2 empirical study — https://arxiv.org/abs/2105.09701 (Табл. 2, 4, 5, 6)
21. DomainMix — https://arxiv.org/abs/2011.11953 (нашего сеттинга не покрывает)
22. Camera-aware re-id feature for MTMC («Camera Equalization») — https://www.sciencedirect.com/science/article/abs/pii/S0262885623002639 (paywall, **замера нет**)
