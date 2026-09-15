# SOTA по vehicle re-identification (2024–2026) и re-ranking

Дата исследования: 2026-09-15. Все числа, полученные открытием источника через WebFetch, помечены `[ПРОВЕРЕНО: url]`; всё остальное — `[ПО ПАМЯТИ / НЕ ПРОВЕРЕНО]`.

---

## 0. Статус Papers With Code и чем он заменён

- **paperswithcode.com закрыт 24 июля 2025**, домен редиректит на Hugging Face «Trending Papers»; **лидерборды не перенесены** (у HF Papers лидербордов нет — это лента трендовых статей). [ПРОВЕРЕНО: https://www.codesota.com/papers-with-code — сайт прямо пишет, что «9 327 benchmark leaderboards … больше не обслуживаются с канонического URL», данные остались только как замороженные JSON-дампы в GitHub `paperswithcode/paperswithcode-data`]. Факт редиректа и дата закрытия дополнительно подтверждаются поисковой выдачей (Coursera, hyper.ai, issue #116 в paperswithcode-data) — [ПО ПАМЯТИ / поисковая выдача, страницы не открывались].
- Зеркало-снапшот `World-Snapshot/papers-with-code` содержит дамп на 16.06.2025 (576 261 статья, 4 451 задача), но **интерактивных лидербордов по конкретным датасетам (VeRi-776) в нём нет**. [ПРОВЕРЕНО: https://github.com/World-Snapshot/papers-with-code]
- web.archive.org из данного окружения недоступен (WebFetch блокирует домен).
- **Чем заменил лидерборды**: сравнительные таблицы из свежих статей 2025–2026, которые сами агрегируют SOTA: CLIP-SENet (февраль 2025, arXiv:2502.16815), «Rethinking Multi-Branch and Cross-Backbone Fusion for Vehicle Re-ID in the Foundation-Model Era» (июль 2026, arXiv:2607.22068 — с **верификацией протоколов**), «Generalization Limits in Vehicle Re-Identification» (июнь 2026, arXiv:2606.01981), LKA-ReID (сентябрь 2024, arXiv:2409.17908), плюс сортировка arXiv по дате (поиск "vehicle re-identification", проверено — за 2025–2026 новых заявок SOTA на VeRi-776/VERI-Wild, кроме перечисленных, нет). [ПРОВЕРЕНО: https://arxiv.org/search/?query=%22vehicle+re-identification%22&searchtype=all&order=-announced_date_first&size=50]

---

## 1. Топ-методы: VeRi-776

### 1.1. Сводная таблица (supervised, БЕЗ re-ranking, если не указано иное)

| Метод (год) | mAP | Rank-1 | Источник числа | Код |
|---|---|---|---|---|
| **CLIP-SENet (2025)** ⚠ | **92.9** | **98.7** | [ПРОВЕРЕНО: https://arxiv.org/html/2502.16815v1] | нет |
| MBR-4B-LAI + k-recip. RR (2023) | 92.09 | 98.03 | [ПРОВЕРЕНО: https://github.com/videturfortuna/vehicle_reid_itsc2023 README] | да |
| RPTM (ResNet-101) + RR (WACV 2023) | 88.00 | 97.30 | [ПРОВЕРЕНО: https://arxiv.org/html/2110.07933] | да |
| LKA-ReID* (2024, с метаданными) | 86.65 | 98.03 | [ПРОВЕРЕНО: https://arxiv.org/html/2409.17908v1] | нет |
| MBR-4B-LAI (2023, без RR) | 85.63 | 97.74 | [ПРОВЕРЕНО: https://arxiv.org/html/2310.01129v1] | да |
| CLIP-ReID ViT + SIE + OLP (AAAI 2023) | 84.5 | 97.3 | [ПРОВЕРЕНО: https://arxiv.org/html/2211.13977v3, «all data … without re-ranking»] | да |
| CLIP-ReID ViT (AAAI 2023) | 83.3 | 97.4 | [ПРОВЕРЕНО: там же] | да |
| TransReID* ViT-B (ICCV 2021) | 82.1 | 97.4 | [ПРОВЕРЕНО: https://github.com/damo-cv/TransReID README; в таблице CLIP-SENet — 82.3/97.1] | да |
| FastReID baseline (2023) | 81.9 | 97.0 | [ПРОВЕРЕНО: таблицы в 2502.16815v1 и 2409.17908v1] | да |
| ASSEN (2022) | 81.7 | 97.3 | [ПРОВЕРЕНО: https://arxiv.org/html/2502.16815v1, таблица I] | — |
| ANet (2021) | 81.2 | 96.8 | [ПРОВЕРЕНО: там же] | — |
| GiT (2023) | 80.3 | 96.8 | [ПРОВЕРЕНО: там же] | — |
| DCAL (2022) | 80.2 | 96.9 | [ПРОВЕРЕНО: там же] | — |
| PVEN (CVPR 2020) | 79.5 | 95.6 | [ПРОВЕРЕНО: там же] | да |

⚠ **Важные оговорки по «лидеру» CLIP-SENet (92.9 mAP)**:
1. Кода и весов нет («The paper does not provide a public code link») [ПРОВЕРЕНО: https://arxiv.org/html/2502.16815v1].
2. Статья arXiv:2607.22068 (июль 2026) прямо помечает CLIP-SENet как **«unverified protocols and no public implementation»** и показывает, что пропуск официального junk-фильтра (same-ID/same-camera) **завышает mAP на VeRi-776 на +3.04, на VERI-Wild на +3.94** [ПРОВЕРЕНО: https://arxiv.org/html/2607.22068v1].
3. В таблице CLIP-SENet метод CLIP-ReID указан с 91.7 mAP, тогда как оригинальная статья CLIP-ReID сообщает 84.5 mAP без re-ranking [ПРОВЕРЕНО: обе ссылки выше] — т.е. таблица CLIP-SENet, судя по всему, смешивает числа с re-ranking/другим протоколом. Аналогично MBR в ней указан как 91.9 (это его результат С re-ranking, без RR — 85.63).
4. Ablation CLIP-SENet стартует с baseline 86.7 mAP (ResNeXt101-IBN, 320×320, A40) [ПРОВЕРЕНО: 2502.16815v1] — сам по себе выше всех опубликованных baseline, что тоже указывает на нестандартный протокол.

**Честный вывод по VeRi-776**: воспроизводимый верхний уровень без re-ranking — **~84.5–86.7 mAP / 97–98 Rank-1** (CLIP-ReID SIE+OLP, MBR-4B-LAI, LKA-ReID); с k-reciprocal re-ranking — **~88–92 mAP**. Заявка 92.9 без RR (CLIP-SENet) невоспроизводима и оспорена в литературе 2026 г.

Независимая перекрёстная проверка (важно для планирования): при **строгом** протоколе и чужом переобучении авторы 2606.01981 получили на VeRi-776 лишь TransReID 81.3 / CLIP-ReID 84.1 / RotTrans 80.4 mAP, а при кросс-датасетном переносе (train VERI-Wild → test VeRi-776) все методы падают до ~63 mAP [ПРОВЕРЕНО: https://arxiv.org/html/2606.01981 — «Generalization Limits in Vehicle Re-Identification», код: https://github.com/deejey674/Generalisation-limits-in-Vehicle-Re-ID/].

### 1.2. VERI-Wild (тест-сеты Small/Medium/Large)

Таблица из CLIP-SENet [ПРОВЕРЕНО: https://arxiv.org/html/2502.16815v1]:

| Метод | Small mAP/R1 | Medium mAP/R1 | Large mAP/R1 |
|---|---|---|---|
| PVEN (2020) | 79.8 / 94.0 | 73.9 / 92.0 | 66.2 / 88.6 |
| ANet (2021) | 86.9 / 96.5 | 82.5 / 95.2 | 75.9 / 92.5 |
| FastReID (2023) | 87.7 / 96.4 | 83.5 / 95.1 | 77.3 / 92.5 |
| MBR (2023) | 88.1 / 96.3 | 83.8 / 95.1 | 77.4 / 92.4 |
| CLIP-SENet (2025) ⚠ | 89.1 / 97.9 | 85.2 / 97.0 | 79.5 / 95.4 |

**Протокольно-верифицированные** числа 2026 г. (arXiv:2607.22068, один backbone DINOv3-distilled ConvNeXt-Base, «Option A») [ПРОВЕРЕНО: https://arxiv.org/html/2607.22068v1]:

| Метод | Small (3k ID, галерея 38 861) mAP | Large (10k ID, галерея 128 517) mAP |
|---|---|---|
| MBR-4B (перепроверен) | 87.97 | 77.15 |
| MBR-4B-LAI (перепроверен) | 88.12 | 77.41 |
| Option A (2026, без метаданных) | 88.19 | 77.47 |
| **Option A + k-reciprocal RR** | **92.38** | **83.68** |

Код 2607.22068: https://gitee.com/robertwangwang/Dinov3ForMBR (на Gitee, не GitHub) [ПРОВЕРЕНО: там же]. Обучение — 8× NVIDIA RTX PRO 6000 (батч 512) [ПРОВЕРЕНО: там же] — для маленькой команды тяжело, но это верхняя граница; MBR обучался на одном RTX 4090 (см. ниже).

---

## 2. Код, веса, лицензии, воспроизводимость за 2 недели

| Метод | Репозиторий | Лицензия | Веса | Оценка воспроизводимости (2 нед., малая команда) |
|---|---|---|---|---|
| TransReID | https://github.com/damo-cv/TransReID | **MIT** | да, VeRi-776 (Google Drive; mAP 82.1 ViT, 82.4 DeiT) | **Высокая**: готовые веса + конфиги; обучение — 1× V100 16/32 ГБ [ПРОВЕРЕНО: README] |
| CLIP-ReID | https://github.com/Syliz517/CLIP-ReID | **MIT** | да, вкл. VeRi-776 (ViT-CLIP-ReID-SIE-OLP) | **Высокая**: веса и логи тестов выложены [ПРОВЕРЕНО: README] |
| MBR (ITSC 2023) | https://github.com/videturfortuna/vehicle_reid_itsc2023 | **не указана** (юр. риск для продукта) | частично (Google Drive) | **Высокая технически**: обучение на 1× RTX 4090, CUDA 11.3/PyTorch 1.11; в README числа с RR (VeRi 92.09 mAP) [ПРОВЕРЕНО: README] |
| RPTM (WACV 2023) | https://github.com/adhirajghosh/RPTM_reid | **MIT** | нет (надо обучать) | **Средняя**: ResNet-101, 240×240, фикс. гиперпараметры; обучение реалистично за дни [ПРОВЕРЕНО: README + https://arxiv.org/html/2110.07933] |
| FastReID | https://github.com/JDAI-CV/fast-reid | **Apache-2.0** | model zoo (вкл. vehicle re-id: VeRi/VehicleID/VERI-Wild) | **Высокая**: промышленный фреймворк, есть re-ranking [ПРОВЕРЕНО: README; состав model zoo по vehicle-конфигам — ПО ПАМЯТИ, README ссылается на MODEL_ZOO.md] |
| CLIP-SENet | — | — | **нет кода** | **Невоспроизводим** [ПРОВЕРЕНО: 2502.16815v1 + флаг в 2607.22068v1] |
| LKA-ReID | — | — | **нет кода** | Невоспроизводим напрямую [ПРОВЕРЕНО: 2409.17908v1 — «No code repository link»] |
| Option A / Dinov3ForMBR (2026) | https://gitee.com/robertwangwang/Dinov3ForMBR | не проверена; базируется на DINOv3 (лицензия Meta DINOv3 — ограничительная) [ПО ПАМЯТИ / НЕ ПРОВЕРЕНО] | заявлен публичный релиз | **Средняя**: код есть, но обучение 8×GPU; inference/файнтюн частично реалистичны |
| GNN re-ranking | https://github.com/Xuanmeng-Zhang/gnn-re-ranking | не указана в README | не нужны (плагин поверх фич) | **Высокая**: Paddle + PyTorch-версия в layumi/Person_reID_baseline_pytorch [ПРОВЕРЕНО: README] |
| VehicleX (синтетика) | https://github.com/yorkeyao/VehicleX | не указана в README | 3D-модели (.fbx), Unity-движок, готовые адаптированные картинки (GDrive/Baidu) | **Средняя**: готовые адаптированные датасеты скачиваются [ПРОВЕРЕНО: README] |

**Рекомендация для «2 недели, малая команда»**: базовый трек — FastReID (Apache-2.0) или CLIP-ReID/TransReID (MIT) с готовыми весами + собственный k-reciprocal/GNN re-ranking; MBR — как источник идей (multi-branch + LAI), но лицензии нет.

---

## 3. Устойчивость к ракурсу и условиям съёмки

### 3.1. Смена ракурса (перёд/бок/зад) — что документированно работает

| Приём | Прирост | Источник |
|---|---|---|
| **VANet** (ICCV 2019): две метрики — для пар «одинаковый ракурс» и «разные ракурсы», выбор метрики по предсказанному ракурсу | VeRi-776: **58.75 → 66.34 mAP (+7.59), 84.68 → 89.78 R1**; прирост в основном за счёт различения «positive с другого ракурса vs negative с того же ракурса» | [ПРОВЕРЕНО: https://ar5iv.labs.arxiv.org/html/1910.04104]; код (репродукция): https://github.com/emdata-ailab/VANET |
| **SIE в TransReID** (side information embedding: обучаемые эмбеддинги ID камеры и ракурса, добавляются к патч-токенам) | VeRi-776: baseline 78.2 → +SIE(cam) 78.7 → +SIE(view) 78.5 → **+SIE(cam+view) 79.6 mAP (+1.4)** | [ПРОВЕРЕНО: https://arxiv.org/html/2102.04378v2] |
| **CLIP-ReID + SIE + OLP** | VeRi-776: 83.3 → **84.5 mAP (+1.2)** от SIE+overlapping patches | [ПРОВЕРЕНО: https://arxiv.org/html/2211.13977v3] |
| **Part/parsing-based: PVEN** (CVPR 2020, парсинг на 4 вида-части, view-aware выравнивание признаков) | VeRi-776: 79.5 mAP / 95.6 R1 (уровень SOTA-2020); принцип уверенно цитируется во всех обзорах | [ПРОВЕРЕНО (числа): https://arxiv.org/html/2502.16815v1] |
| **RPTM** (WACV 2023): pose-aware выбор триплетов через feature matching — не тянуть в одну точку разные натуральные под-группы (ракурсы) одного ID | VeRi-776: 80.8 mAP (R101, без RR) при 240×240; «view consistency» заложена в майнинг | [ПРОВЕРЕНО: https://arxiv.org/html/2110.07933] |
| **Метаданные камеры/ракурса в MBR (LAI-модуль)** | VeRi-776: 84.72 → **85.63 mAP (+0.91)** (MBR-4B → MBR-4B-LAI) | [ПРОВЕРЕНО: https://arxiv.org/html/2310.01129v1] |
| **Синтетика VehicleX** (ECCV 2020, attribute descent: подгонка атрибутов рендера под целевой датасет, joint training real+synthetic) | VeRi-776: **66.54 → 70.62 mAP (+4.08)**; VehicleID(large): +2.84; CityFlow: **+6.95** | [ПРОВЕРЕНО: https://ar5iv.labs.arxiv.org/html/1912.08855]; код: https://github.com/yorkeyao/VehicleX |
| Синтетика 2025: **CLIPVehicle + SynVS-Day/SynVS-All** (vehicle search) | датасеты и фреймворк заявлены, чисел не проверял | [ПРОВЕРЕНО (существование): arXiv:2508.04120 в выдаче поиска arXiv] |

Практический вывод: наибольший документированный прирост от «ракурсных» приёмов давали viewpoint-aware метрики на слабых baseline (+7.6 mAP у VANet); на сильных трансформерных baseline остаточный прирост от SIE/метаданных — **+0.9…+1.4 mAP**. Синтетика (VehicleX) даёт +3…+7 mAP как аугментация к реальным данным, особенно на малых датасетах. Внимание: SIE требует знать ID камеры/ракурс на инференсе (для ракурса — отдельный классификатор).

### 3.2. Ночь / дождь / блики

| Факт | Числа | Источник |
|---|---|---|
| **Day-Night vehicle re-id — отдельная задача с CVPR 2024**: датасеты **DN-Wild** (2 286 ID; 85 945 дневных + 54 952 ночных изображений) и DN-348; фреймворк **DNDM**: подавление бликов фар (glare suppression), усиление структуры в low-light, cross-domain class awareness | численные mAP из статьи не извлечены (openaccess.thecvf.com отдаёт 403) — [ЧИСЛА НЕ ПРОВЕРЕНЫ] | [ПРОВЕРЕНО (датасет, модули, авторы Li и др.): https://mlanthology.org/cvpr/2024/li2024cvpr-daynight/ и https://github.com/chenjingong/DN-ReID] |
| **Дождь бьёт сильнее тумана** (UAV-бенчмарк с синтетической погодой, июль 2026): VRU-Large mAP — AdaSP: 95.9 (чисто) → 93.0 (туман) → 88.5 (дождь); CLIP-ReID(RN50): 95.2 → 91.2 → 86.8; CLIP-ReID **ViT-B/16 деградирует сильнее всех**: 91.4 → 81.7 → 78.7; UAV-VeID: AdaSP 92.7 → 88.7 → 76.2 | падение до **−13…−17 mAP под дождём**; самый устойчивый — **AdaSP** (adaptive sparse pairwise loss) | [ПРОВЕРЕНО: https://arxiv.org/html/2607.10583] |
| VERI-Wild сам содержит ночь/погоду (174 камеры, месяц съёмки) — деградация Small→Large (88 → 77 mAP у лучших) частично отражает сложные условия + масштаб | см. таблицу §1.2 | [ПРОВЕРЕНО: 2502.16815v1, 2607.22068v1] |
| Прочие направления: реставрация/disentanglement день-ночь («Disentangled reflectance-ambient feature learning», Appl. Soft Computing 2025), universal adverse-weather ReID (ScA-UniReID, Sensors 2026), мультиспектральные (RGB+NIR+TIR, RGBNT100) | числа не проверены | [ПО ПАМЯТИ / выдача поиска; страницы не открывались] |

Практический вывод: для ночи главное — не общий low-light enhancement, а **подавление бликов фар** и структурные признаки (DNDM); по устойчивости к погоде лучший документированный выбор лосса — **AdaSP**; ViT без спец-мер деградирует под погодой сильнее CNN.

---

## 4. k-reciprocal re-ranking (Zhong et al., CVPR 2017)

Механика: для каждого запроса строится множество k-reciprocal-соседей, кодируется в вектор, финальная дистанция = λ·исходная + (1−λ)·Jaccard. Параметры по умолчанию **k1=20, k2=6, λ=0.3** [ПРОВЕРЕНО: https://ar5iv.labs.arxiv.org/html/1701.08398].

### 4.1. Прирост «до/после» (конкретные пары из литературы)

| Датасет | До RR (mAP) | После RR (mAP) | Δ | Источник |
|---|---|---|---|---|
| **VeRi-776** (baseline из GNN-статьи) | 78.94 | **88.44** | **+9.50** | [ПРОВЕРЕНО: https://arxiv.org/html/2012.07620v2] |
| **VeRi-776**, RPTM ResNet-101 | 80.80 | **88.00** | **+7.20** | [ПРОВЕРЕНО: https://arxiv.org/html/2110.07933] |
| **VeRi-776**, MBR-4B-LAI | 85.63 (статья) | **92.09** (README) | **+6.46** | [ПРОВЕРЕНО: https://arxiv.org/html/2310.01129v1 + https://github.com/videturfortuna/vehicle_reid_itsc2023] |
| **VERI-Wild Small** (галерея 38 861) | 88.19 | **92.38** | **+4.19** | [ПРОВЕРЕНО: https://arxiv.org/html/2607.22068v1] |
| **VERI-Wild Large** (галерея 128 517) | 77.47 | **83.68** | **+6.21** | [ПРОВЕРЕНО: там же] |
| Market-1501 (персоны, для калибровки) | 88.26 | 94.38 | +6.12 | [ПРОВЕРЕНО: 2012.07620v2] |
| Market-1501, IDE(R)+KISSME (ориг. статья) | 46.00 | 63.63 | +17.63 | [ПРОВЕРЕНО: ar5iv 1701.08398] |

Итого по vehicle re-id: **k-reciprocal стабильно даёт +4…+10 mAP**, причём прирост сохраняется даже на очень сильных моделях 2026 г. При этом Rank-1 растёт слабо (96.4→96.4–97.3, иногда ≈0) — весь эффект в хвосте ранжирования (mAP).

### 4.2. Сложность и реальное время

- Теоретическая сложность: **O(N²) на попарные дистанции + O(N² log N) на сортировки**; при офлайн-предрасчёте галереи онлайновая часть на один запрос — O(N) + O(N log N) [ПРОВЕРЕНО: ar5iv 1701.08398].
- Замеры на реальных галереях:
  - галерея ~19.7k (Market-1501): классическая CPU-реализация — **89.2 с** на весь query-сет [ПРОВЕРЕНО: https://arxiv.org/html/2012.07620v2];
  - галерея 38 861 (VERI-Wild Small): классическая — **534 с**, точная разреженная (sparse) реализация — **29 с (×18)** [ПРОВЕРЕНО: https://arxiv.org/html/2607.22068v1];
  - галерея 128 517 (VERI-Wild Large): классическая **падает по памяти (~190 ГБ)**; sparse-версия — **137 с** в «десятках ГБ» [ПРОВЕРЕНО: там же];
  - ECN-статья: k-reciprocal на Duke — 124.6 с [ПРОВЕРЕНО: https://ar5iv.labs.arxiv.org/html/1711.10378].
- Практический ориентир для галереи 10k–100k: классический CPU-код — от десятков секунд до «не работает» на 100k+; **sparse/GPU-реализации переводят задачу в диапазон миллисекунды–десятки секунд** (см. §5).

---

## 5. Современные альтернативы re-ranking (2018–2026)

| Метод | Точность (mAP) | Время | Источник / код |
|---|---|---|---|
| **GNN re-ranking** (Zhang et al., 2020; «Understanding Image Retrieval Re-Ranking: A GNN Perspective») — переформулировка k-reciprocal как разреженного GNN на k-NN-графе, полностью на GPU | Market: base 88.26 → KR 94.38 → **GNN 94.65**; **VeRi-776: base 78.94 → KR 88.44 → GNN 88.61 (R1 96.42)** | Market-1501: **89.2 с (KR, CPU) → 9.4 мс (GNN, GPU K40m)**, ~×9 500 | [ПРОВЕРЕНО: https://arxiv.org/html/2012.07620v2]; код: https://github.com/Xuanmeng-Zhang/gnn-re-ranking (Paddle + PyTorch-версия в layumi/Person_reID_baseline_pytorch) [ПРОВЕРЕНО: README] |
| **ECN** (Sarfraz et al., CVPR 2018, expanded cross neighborhood) — агрегирование дистанций соседей без вычисления reciprocal-списков на пару | Market: 55.0 → 71.1 (+16.1); Duke: 50.3 → 62.0 | Duke: KR 124.6 с vs ECN(rank-dist) 115.3 с vs **ECN(orig-dist) 73.2 с**; та же O(N² log N) | [ПРОВЕРЕНО: https://ar5iv.labs.arxiv.org/html/1711.10378] |
| **LBR** (Local Blurring Re-ranking, из Luo et al., «Spectral feature transformation», ICCV 2019) | слабее KR: Market BoT 85.9 → LBR 91.3 (KR: 94.2); MSMT17: 45.1 → 45.5 (почти ноль) | — | [ПРОВЕРЕНО: https://arxiv.org/html/2306.08792v1, таблицы GCR] |
| **GCR / GCR_Local** (Graph Convolution based Re-ranking, Zhang et al., IEEE TMM 2023) — свёртки на графе по фичам, параллелится | Market: 85.9 → **94.7** (KR 94.2, ECN 93.2); MSMT17: 45.1 → **58.5** (KR 57.7) | Market, 24-ядерный CPU: KR 76 с, ECN 72 с, **GCR 24 с, GCR_Local 10 с** | [ПРОВЕРЕНО: https://arxiv.org/html/2306.08792v1]; код: https://github.com/WesleyZhang1991/GCN_rerank |
| **Cheb-GR** (CVPR 2025, Yang, Li, Du, Ye) — адаптивный выбор соседей по неравенству Чебышёва вместо k-NN + графовая свёртка; «cluster-aware»-семейство | в абстракте чисел нет: «significantly reduces the computation costs while maintaining relatively strong performance» | — | [ПРОВЕРЕНО (абстракт): https://mlanthology.org/cvpr/2025/yang2025cvpr-chebgr/; PDF на openaccess.thecvf.com — 403] |
| **Sparse exact k-reciprocal** (arXiv:2607.22068, 2026) — математически точная разреженная реализация классики | идентичные классике приросты (+4.19/+6.21 на VERI-Wild) | 38.9k галерея: 534→**29 с**; 128.5k: **137 с** (классика падает по памяти ~190 ГБ) | [ПРОВЕРЕНО: https://arxiv.org/html/2607.22068v1]; код: gitee.com/robertwangwang/Dinov3ForMBR |
| K-nearest weighted fusion (сент. 2025, arXiv:2509.04050) | не проверялось детально | — | [ПО ПАМЯТИ / только выдача поиска] |

Резюме: (а) точность всех сильных re-ranking-методов на уровне k-reciprocal ±0.3–1 mAP, выигрыш современных — **скорость и память** (GNN на GPU: миллисекунды; GCR: ×3–7 на CPU; sparse-KR: ×18 и линейная по памяти); (б) LBR заметно слабее; (в) для vehicle re-id проверенные числа есть у GNN-RR (VeRi-776) и sparse-KR (VERI-Wild).

---

## 6. Вывод: применять ли re-ranking в соревновании

Условие: организатор отдельно измеряет (1) время формирования эмбеддинга при батче 1 и (2) пропускную способность; **ранжирование участник делает в своём пайплайне**.

**Да, re-ranking применять почти обязательно.** Аргументация:

1. **Метрики времени его не видят.** Замеряется только стадия «изображение → эмбеддинг» (batch-1 latency и throughput). k-reciprocal/GNN-RR — постпроцессинг матрицы дистанций уже готовых эмбеддингов, он не добавляет ни миллисекунды к измеряемым величинам. Это самый дешёвый способ купить +4…+10 mAP (см. §4.1, все пары проверены).
2. **Прирост устойчив и на сильных моделях**: даже DINOv3-бэкбон 2026 г. получает +4.2/+6.2 mAP на VERI-Wild [ПРОВЕРЕНО: 2607.22068v1]; на VeRi-776 типично +6.5…+9.5 mAP.
3. **Вычислимость на целевых размерах галереи**: 10k–40k — секунды–минуты даже классикой; 100k+ — обязательно sparse-реализация (137 с на 128.5k галерее) или GNN-GPU (миллисекунды на ~20k, память O(N·k) благодаря разреженному графу). Классический «плотный» код на 100k+ галерее падает (~190 ГБ RAM) — закладывать sparse/GPU сразу.
4. **Риски и оговорки**:
   - если в регламенте ранжирование всё же входит в какой-то общий тайм-бюджет или есть жёсткий SLA на «запрос → выдача», выбирать GNN-RR/GCR (мс–секунды), а не классический KR;
   - re-ranking требует доступа ко всей галерее сразу (батч-режим запросов); в стриминговом сценарии «один запрос за раз» нужен офлайн-предрасчёт k-NN-списков галереи (тогда онлайн-часть O(N log N) на запрос [ПРОВЕРЕНО: ar5iv 1701.08398]);
   - прирост идёт в mAP; если организатор считает только Rank-1 — эффект близок к нулю (96.36 vs 95.59 на VeRi-776 [ПРОВЕРЕНО: 2012.07620v2]), и приоритет смещается на саму модель;
   - гиперпараметры k1/k2/λ подбирать на валидации: дефолт k1=20, k2=6, λ=0.3 работает на VeRi/VERI-Wild [ПРОВЕРЕНО: 1701.08398, 2607.22068v1].
5. **Рекомендуемый стек**: эмбеддер CLIP-ReID/TransReID/FastReID (MIT/Apache, веса есть) → L2-нормировка → GPU-GNN-re-ranking (репо Xuanmeng-Zhang, PyTorch-порт у layumi) с fallback на sparse k-reciprocal для галерей 100k+.

---

## Приложение: ключевые источники

- CLIP-SENet: https://arxiv.org/abs/2502.16815 (html: /html/2502.16815v1)
- Foundation-model era MBR rethink + sparse RR + верификация протоколов: https://arxiv.org/abs/2607.22068
- Generalization Limits in Vehicle Re-ID: https://arxiv.org/html/2606.01981
- MBR: https://arxiv.org/abs/2310.01129, код: https://github.com/videturfortuna/vehicle_reid_itsc2023
- RPTM: https://arxiv.org/abs/2110.07933, код: https://github.com/adhirajghosh/RPTM_reid
- TransReID: https://arxiv.org/abs/2102.04378, код: https://github.com/damo-cv/TransReID
- CLIP-ReID: https://arxiv.org/abs/2211.13977, код: https://github.com/Syliz517/CLIP-ReID
- LKA-ReID: https://arxiv.org/abs/2409.17908
- VANet: https://arxiv.org/abs/1910.04104
- VehicleX: https://arxiv.org/abs/1912.08855, код: https://github.com/yorkeyao/VehicleX
- Day-Night ReID (CVPR 2024): https://github.com/chenjingong/DN-ReID
- UAV weather benchmark: https://arxiv.org/abs/2607.10583
- k-reciprocal: https://arxiv.org/abs/1701.08398
- ECN: https://arxiv.org/abs/1711.10378
- GNN re-ranking: https://arxiv.org/abs/2012.07620, код: https://github.com/Xuanmeng-Zhang/gnn-re-ranking
- GCR (TMM 2023): https://arxiv.org/abs/2306.08792, код: https://github.com/WesleyZhang1991/GCN_rerank
- Cheb-GR (CVPR 2025): https://mlanthology.org/cvpr/2025/yang2025cvpr-chebgr/
- Судьба PwC: https://www.codesota.com/papers-with-code, зеркало: https://github.com/World-Snapshot/papers-with-code
