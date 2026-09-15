# Foundation-модели как экстракторы признаков для vehicle re-id (VeRi-776 / VERI-Wild / VehicleID)

Дата исследования: 2026-09-15. Все числа снабжены пометкой статуса:
- **[ПРОВЕРЕНО: url]** — источник открыт через WebFetch, число процитировано из него;
- **[ПО ПАМЯТИ / НЕ ПРОВЕРЕНО]** — источник открыть не удалось, утверждение из общих знаний.

---

## TL;DR

1. **Прямых замеров zero-shot DINOv2/CLIP/SigLIP2 на VeRi-776 в литературе НЕ найдено** (это само по себе результат). Ближайшие данные: (а) frozen DINOv3 без всякого обучения даёт ~14–19 mAP на VERI-Wild (arXiv 2607.22068); (б) на person re-id zero-shot foundation-эмбеддинги катастрофически плохи — 0.1–6 % mAP (arXiv 2601.20598).
2. **С лёгким дообучением картина резко меняется**: frozen DINOv2-B + LoRA даёт 61.1 mAP на VeRi-776 (community-модель VehicleDINO); полное дообучение DINOv3-ConvNeXt-B даёт 88.2 mAP на VERI-Wild и 66.0 mAP zero-shot cross-dataset на VeRi-776 (2607.22068).
3. **CLIP-ReID (AAAI 2023)**: 80.3 mAP (CNN) / 83.3 mAP (ViT) / 84.5 mAP (ViT+SIE+OLP) на VeRi-776. Лучший CLIP-последователь — CLIP-SENet (2025): 92.9 mAP на VeRi-776.
4. **Подводный камень подтверждён публикациями**: frozen-признаки отлично ловят тип/марку/модель (DINOv3 linear probe: 97.7 % тип, >93 % марка/модель), но «не приоритизируют identity-specific признаки» — экземпляры одной модели/цвета не различают; даже дообученные reid-модели теряют ~24 п.п. mAP на невиданных типах кузова.
5. **VehicleMAE (AAAI 2024)** — специализированное предобучение (ViT-B, 1M изображений машин): после дообучения 85.6 mAP на VeRi-776 против 76.7 у MAE-ImageNet — специализация даёт ~+9 mAP. Веса открыты, но только на Baidu Netdisk и без файла лицензии.
6. **Лицензии**: DINOv2 — Apache-2.0; DINOv3 — собственная «DINOv3 License» Meta (коммерческое использование РАЗРЕШЕНО, но: запрет военного применения, атрибуция «Built with DINOv3», gated-доступ); CLIP — MIT; OpenCLIP/LAION — MIT; SigLIP/SigLIP2 — Apache-2.0.

---

## 1. Zero-shot / frozen признаки: что есть в литературе (arXiv 2023–2026)

### 1.1. Главный отрицательный результат

Целенаправленный поиск («DINOv2 vehicle re-identification», «foundation models re-identification zero-shot», «CLIP vehicle reid VeRi-776», «DINOv3 re-identification», «frozen backbone VeRi», «training-free vehicle re-identification» и т.п.) **не выявил ни одной статьи, где DINOv2/DINOv3/CLIP/SigLIP2 замерялись бы как чистые zero-shot экстракторы на VeRi-776 / VehicleID со стандартным протоколом**. Всё, что есть — косвенные и смежные замеры ниже. Это пробел в литературе: бенчмарк «frozen foundation features на VeRi-776» фактически свободен.

### 1.2. Frozen DINOv3 на VERI-Wild — единственный найденный «vehicle» замер

Статья **«Rethinking Multi-Branch and Cross-Backbone Fusion for Vehicle Re-Identification in the Foundation-Model Era»**, Yu Wang, Hongyu Yang, arXiv:2607.22068 (24.07.2026):

> «Nor is the ViT weak per se — frozen, it is the strongest bare backbone we test (**18.56 zero-shot vs. ConvNeXt 14.63, ResNet50 3.77**)»

- Контекст: замер «голых» замороженных бэкбонов без какого-либо обучения на задаче VERI-Wild; метрика по контексту статьи — mAP (в самом предложении датасет/метрика явно не подписаны — осторожно). Разрешение 256×256.
  - frozen DINOv3-ViT-L: **18.56**
  - frozen DINOv3-ConvNeXt-Base: **14.63**
  - frozen ResNet50 (ImageNet): **3.77**
- Для сравнения, после дообучения: DINOv3-ConvNeXt-Base — **88.19 mAP** (VERI-Wild test_3000), **77.47** (test_10000); с re-ranking 92.38/83.68. ViT-L при полном дообучении ~**67 mAP**, лучший LoRA-вариант **75.19 mAP** — т.е. на этой задаче дообученный ConvNeXt обходит ViT-L, хотя frozen ViT-L был сильнее.
- Cross-dataset zero-shot (обучение на VERI-Wild → тест на VeRi-776 без адаптации): **66.02 mAP / 87.07 R1** (с re-ranking 69.27), при in-domain-референсе на VeRi-776 **84.07 mAP**.
- Основной вывод статьи: в эпоху foundation-моделей мультибранч- и кросс-бэкбон-фьюжн перестают окупаться (выигрыш ≤ +0.11 mAP), достаточно одного хорошо настроенного бэкбона.

[ПРОВЕРЕНО: https://arxiv.org/html/2607.22068v1 — три отдельных фетча, цитата дословная]

### 1.3. Zero-shot foundation-эмбеддинги на re-id (person) — систематический бенчмарк

**«Person Re-ID in 2025: Supervised, Self-Supervised, and Language-Aligned. What Works?»**, Lakshman Balasubramanian, arXiv:2601.20598 (28.01.2026). 11 моделей × 9 датасетов, **только person re-id (vehicle-датасетов нет)**, протокол — zero-shot: только image-энкодер, замороженный. Точные mAP:

| Модель (zero-shot) | MSMT17 | Market-1501 | CelebReID |
|---|---|---|---|
| CLIP-B/32 (151M) | 0.10 % | 0.37 % | 0.61 % |
| CLIP-B/16 (150M) | 0.11 % | 0.43 % | — |
| CLIP-L/14 (428M) | 0.14 % | 0.50 % | 0.82 % |
| SigLIP2-256 (375M) | **5.64 %** | 5.67 % | 14.23 % |
| SigLIP2-384 (376M) | 4.56 % | **5.97 %** | — |
| DINOv2-B/14 (87M) | 0.37 % | 1.71 % | 3.68 % |
| DINOv2-L/14 (304M) | 0.39 % | 1.40 % | 3.79 % |
| PE-Core-L14 (671M) | 0.91 % | 1.25 % | — |
| PE-Spatial-S16 (22M) | 0.09 % | 0.54 % | — |
| OSNet-x1.0 (supervised, 2.5M) | 3.37 % | 83.57 % (in-domain) | — |
| CLIP-ReID (fine-tuned, 87.5M) | 66.22 % (in-domain) | 50.59 % | — |

Выводы авторов (цитаты):
> «Despite strong performance on generic vision benchmarks, DINOv2 exhibits limited effectiveness in zero-shot ReID, achieving only 0.3 %–4.7 % mAP across datasets… purely visual self-supervised pretraining lacks the explicit semantic structure required for identity matching.»

> «Language-aligned models… learn disentangled semantic attributes… match individuals based on high-level semantic appearance rather than brittle, low-level visual patterns» — поэтому SigLIP2 zero-shot на порядок лучше CLIP и DINOv2, и language-aligned модели неожиданно робастны cross-domain.

[ПРОВЕРЕНО: https://arxiv.org/abs/2601.20598 и https://arxiv.org/html/2601.20598v1 — числа процитированы из таблиц]

Экстраполяция на vehicle-домен (наша, не из статьи): порядок величин zero-shot mAP на VeRi-776 у сырых CLIP/DINOv2, скорее всего, единицы процентов; frozen DINOv3 в vehicle-домене показывает 14–19 mAP (п. 1.2), т.е. заметно лучше, но всё равно в 4–6 раз хуже дообученных моделей.

### 1.4. Frozen DINOv2 + LoRA на VeRi-776 (community, не peer-reviewed)

**VehicleDINO** (Hugging Face, автор Wei Meng Soh / wms2537, 2026): единая мультизадачная модель на **замороженном DINOv2 ViT-B/14 с LoRA-адаптерами** (SimpleFPN + HybridEncoder + 6 голов):
- Re-ID на VeRi-776: **mAP 61.1 %, Rank-1 86.1 %**
- Классификация: тип 95.6 %, марка 98.4 %, модель 87.7 % (top-1)
- Варианты: FP32 450 MB, INT8 139 MB; лицензия Apache-2.0.
Показательно: та же frozen-модель почти идеально решает марку/тип, но по re-id сильно отстаёт от специализированных (~85+ mAP) — прямая иллюстрация подводного камня из §4.

[ПРОВЕРЕНО: https://huggingface.co/wms2537/VehicleDINO]

### 1.5. Смежное: zero-shot re-id через VLM-описания

**«Zero-Shot Semantic Re-Identification for Autonomous Driving: A VLM Baseline Study»**, arXiv:2606.09362 (2026): vehicle+pedestrian re-id на KITTI-ReID полностью без обучения — VLM генерирует текстовое описание, матчинг по текстовым эмбеддингам. Лучшая связка Qwen3.5-27B + EmbeddingGemma-300M: **mAP 0.717**, Rank-1 до 0.831 (с Qwen3-Embedding-4B); supervised-бейзлайн ResNet50: mAP 0.6724. Сравнения с сырыми CLIP/DINOv2-эмбеддингами в статье нет. KITTI-ReID — маленький и нестандартный бенчмарк, на VeRi-776 не переносится, но показывает жизнеспособность training-free-подхода через семантику.
[ПРОВЕРЕНО: https://arxiv.org/html/2606.09362v1]

### 1.6. Контекст: frozen DINOv2 умеет instance-level retrieval — но на ландмарках

Оригинальная статья DINOv2 (arXiv:2304.07193, TMLR 2024) отдельно замеряет frozen-признаки на instance recognition: ViT-L/14 — Oxford-Medium **75.1 mAP** / Oxford-Hard **54.0**, Paris-M 92.7/H 83.5, Met 40.0, AmsterTime 71.6; ViT-g/14 — Oxford-M 73.6/H 52.3, Met 36.8, AmsterTime 76.5. Для сравнения OpenCLIP-G: Oxford-M 50.7/H 19.7, iBOT ViT-L: 39.0/12.7. Авторы: «our features significantly outperform both SSL (+41 % mAP on Oxford-Hard) and weakly-supervised (+34 %) ones». То есть frozen DINOv2 в принципе способен на instance-матчинг (и много лучше CLIP-семейства), но здания/картины отличаются друг от друга сильнее, чем два серых Nissan одной модели — на транспорте этот потенциал напрямую не реализуется (см. §4).
[ПРОВЕРЕНО: https://arxiv.org/html/2304.07193v2]

---

## 2. CLIP-ReID (AAAI 2023) и последователи: числа на VeRi-776

### 2.1. CLIP-ReID — оригинальные числа

«CLIP-ReID: Exploiting Vision-Language Model for Image Re-Identification without Concrete Text Labels», Li, Sun, Li; AAAI 2023, arXiv:2211.13977. Таблица 3 оригинала (все числа без re-ranking):

| Метод (VeRi-776) | Бэкбон | mAP | R1 |
|---|---|---|---|
| baseline (CNN) | RN50 (CLIP) | 79.3 | 95.7 |
| **CLIP-ReID (CNN)** | RN50 (CLIP) | **80.3** | **96.8** |
| TransReID | ViT | 80.6 | 96.9 |
| TransReID! (доп. трюки) | ViT | 82.0 | 97.1 |
| **CLIP-ReID (ViT)** | ViT-B/16 (CLIP) | **83.3** | **97.4** |
| **CLIP-ReID! (ViT + SIE + OLP)** | ViT-B/16 | **84.5** | **97.3** |

VehicleID (small): CNN CLIP-ReID R1 85.2 / R5 97.1; ViT CLIP-ReID R1 **85.3** / R5 97.6. Энкодеры: RN50 → 1024-мерный эмбеддинг (attention pooling), ViT-B/16 → hidden 768, проекция в 512.
[ПРОВЕРЕНО: https://ar5iv.labs.arxiv.org/html/2211.13977 — таблица процитирована построчно]

Механика: двухэтапное обучение — на 1-й стадии для каждого ID выучиваются текстовые prompt-токены (CoOp-стиль), на 2-й стадии image-энкодер CLIP **дообучается целиком** (это НЕ frozen-подход).

### 2.2. Последователи 2024–2026

- **CLIP-SENet** («CLIP-based Semantic Enhancement Network for Vehicle Re-identification», Lu, Fu, Chu, Wang, Xu; arXiv:2502.16815, февраль 2025, IEEE TITS): image-энкодер TinyCLIP (ViT-B/32-дистиллят) как источник семантики + модуль AFEM + CNN-ветка внешности; итоговый признак 2048-мерный; текстовый энкодер выброшен. Результаты (SOTA на момент публикации): **VeRi-776 92.9 mAP / 98.7 R1; VehicleID (small) 90.4 R1 / 98.7 R5; VERI-Wild 89.1 mAP / 97.9 R1**. [ПРОВЕРЕНО: https://arxiv.org/abs/2502.16815 + PDF]
  - ВНИМАНИЕ, расхождение: в сравнительной таблице CLIP-SENet строки «CLIP-ReID» на VeRi-776 стоят как 88.3 и 91.7 mAP (а MBR — 91.9), что противоречит оригинальным 80.3/83.3/84.5 из §2.1; судя по всему, в таблице использованы числа с re-ranking или иной протокол — при цитировании опирайтесь на оригинал CLIP-ReID. [ПРОВЕРЕНО: текст PDF 2502.16815, таблица I]
- **PCL-CLIP** («Prototypical Contrastive Learning-based CLIP Fine-tuning for Object Re-identification», Li, Gong; arXiv:2310.17218, ревизия января 2026): дообучение CLIP через prototypical contrastive loss без prompt-стадии, supervised и unsupervised режимы, person+vehicle; в абстракте конкретных чисел на VeRi нет (полный текст не открывался). [ПРОВЕРЕНО (abs): https://arxiv.org/abs/2310.17218]
- **CLIPVehicle** («A Unified Framework for Vision-based Vehicle Search», Wang, Han, Zhang, Feng; arXiv:2508.04120, август 2025): CLIP-RN50 + CoOp для сквозного «поиска машин» (детекция+re-id) на новых бенчмарках CityFlowVS (mAP 14.1/Top-1 83.8), SynVS-Day (32.4/84.1), SynVS-All (24.6/83.0); текстовый энкодер заморожен; на VeRi-776 не замерялись. [ПРОВЕРЕНО: https://arxiv.org/html/2508.04120]
- **CLIP-based Partial-wise Prompt Learning for Unsupervised Vehicle Re-identification** (Expert Systems with Applications, 2025) — global-to-partial image-text contrastive для unsupervised vehicle reid, VeRi-776; текст закрыт (ScienceDirect), числа не проверены. [ПО ПАМЯТИ / НЕ ПРОВЕРЕНО]

### 2.3. Насколько всё это «дообучение» тяжёлое

Во всех перечисленных работах image-энкодер CLIP дообучается полностью (CLIP-ReID, PCL-CLIP) либо используется как frozen-источник семантики при обучаемой CNN-ветке (CLIP-SENet). «Лёгкое» дообучение (LoRA/адаптеры) в vehicle-домене найдено в 2607.22068 (LoRA на DINOv3-ViT-L: 75.19 mAP VERI-Wild — хуже полного FT ConvNeXt 88.19) и VehicleDINO (61.1 mAP VeRi-776).

---

## 3. Сводная таблица моделей

Размеры — файл model.safetensors (fp32) с вкладки Files соответствующего репозитория HF, «≈МБ» в десятичных мегабайтах. «Эмб.» — размерность выходного глобального эмбеддинга (CLS/pooled). Все размеры/лицензии проверены через HF API, кроме отмеченных.

| Модель | Вариант (HF id) | Размер весов | Эмб. | Лицензия | Числа на vehicle-reid | Источник | Статус |
|---|---|---|---|---|---|---|---|
| DINOv2 | facebook/dinov2-small | 88 249 960 Б ≈ 88 МБ (22.1M пар.) | 384 [станд.] | Apache-2.0 | прямых нет | HF API | ПРОВЕРЕНО (размер/лиц.); эмб. по памяти |
| DINOv2 | facebook/dinov2-base | 346 345 912 Б ≈ 346 МБ (86.6M) | 768 (config) | Apache-2.0 | frozen+LoRA: VeRi-776 61.1 mAP / 86.1 R1 (VehicleDINO, community) | HF API; huggingface.co/wms2537/VehicleDINO | ПРОВЕРЕНО |
| DINOv2 | facebook/dinov2-large | 1 217 522 888 Б ≈ 1.22 ГБ (304.4M) | 1024 (config) | Apache-2.0 | zero-shot person-reid 0.39–1.40 % mAP (2601.20598) | HF API; arXiv:2601.20598 | ПРОВЕРЕНО |
| DINOv2 | facebook/dinov2-giant | 4 546 005 432 Б ≈ 4.55 ГБ (1136.5M) | 1536 (config) | Apache-2.0 | прямых нет | HF API | ПРОВЕРЕНО |
| DINOv3 | facebook/dinov3-vits16-pretrain-lvd1689m | 86 406 384 Б ≈ 86 МБ (21.6M) | 384 [по памяти] | DINOv3 License (gated) | прямых нет | HF API | ПРОВЕРЕНО (размер/лиц.) |
| DINOv3 | facebook/dinov3-vitb16-pretrain-lvd1689m | 342 662 192 Б ≈ 343 МБ (85.7M) | 768 [по памяти] | DINOv3 License (gated) | прямых нет | HF API | ПРОВЕРЕНО (размер/лиц.) |
| DINOv3 | facebook/dinov3-vitl16-pretrain-lvd1689m | 1 212 559 808 Б ≈ 1.21 ГБ (303.1M) | 1024 [по памяти] | DINOv3 License (gated) | frozen: ~18.56 (VERI-Wild, mAP по контексту); LoRA-FT: 75.19 mAP VERI-Wild | HF API; arXiv:2607.22068 | ПРОВЕРЕНО |
| DINOv3 | facebook/dinov3-vith16plus-pretrain-lvd1689m | 3 362 432 800 Б ≈ 3.36 ГБ (840.6M) | 1280 [по памяти] | DINOv3 License (gated) | прямых нет | HF API | ПРОВЕРЕНО (размер/лиц.) |
| DINOv3 | facebook/dinov3-vit7b16-pretrain-lvd1689m | 26 864 210 088 Б ≈ 26.9 ГБ (6716M) | 4096 [по памяти] | DINOv3 License (gated) | прямых нет | HF API | ПРОВЕРЕНО (размер/лиц.) |
| DINOv3 | facebook/dinov3-convnext-base-pretrain-lvd1689m | 350 302 312 Б ≈ 350 МБ (87.6M) | 1024 (после pool) [по памяти] | DINOv3 License (gated) | frozen: ~14.63; full-FT: 88.19 mAP VERI-Wild-small, 77.47 large; cross-dataset zero-shot VeRi-776: 66.02 mAP / 87.07 R1 | HF API; arXiv:2607.22068 | ПРОВЕРЕНО |
| CLIP | openai/clip-vit-large-patch14 | 1 710 540 580 Б ≈ 1.71 ГБ (427.6M, обе башни) | proj 768 (vision hidden 1024, config) | MIT (GitHub openai/CLIP); на HF-карточке лицензия не проставлена | zero-shot person-reid 0.14–0.50 % mAP (2601.20598); базис CLIP-ReID | HF API; raw.githubusercontent.com/openai/CLIP/main/LICENSE | ПРОВЕРЕНО |
| CLIP (для reid) | RN50 / ViT-B/16 из CLIP-ReID | — | 1024 (RN50) / 512 (ViT-B/16 proj) | MIT | VeRi-776: 80.3 / 83.3 / 84.5 mAP (см. §2.1) | ar5iv 2211.13977 | ПРОВЕРЕНО |
| OpenCLIP | laion/CLIP-ViT-B-32-laion2B-s34B-b79K (пример) | — | 512 [станд.] | MIT | прямых нет | HF API | ПРОВЕРЕНО (лиц.) |
| SigLIP | google/siglip-so400m-patch14-384 | 3 511 950 624 Б ≈ 3.51 ГБ (878M) | 1152 [станд.] | Apache-2.0 | прямых нет | HF API | ПРОВЕРЕНО (размер/лиц.) |
| SigLIP2 | google/siglip2-base-patch16-224 | 1 500 800 904 Б ≈ 1.50 ГБ (375.2M, обе башни) | 768 [по памяти] | Apache-2.0 | zero-shot person-reid 4.6–6.0 % mAP — лучший zero-shot в 2601.20598 | HF API; arXiv:2601.20598 | ПРОВЕРЕНО (размер/лиц./числа); эмб. по памяти |
| SigLIP2 | google/siglip2-so400m-patch14-384 | 4 544 143 072 Б ≈ 4.54 ГБ (1136M, обе башни) | 1152 (config) | Apache-2.0 | прямых нет | HF API + config.json | ПРОВЕРЕНО |
| VehicleMAE | ViT-B/16 (GitHub Event-AHU/VehicleMAE, веса на Baidu Netdisk, код 6zkx) | не указан (~330 МБ для ViT-B fp32 [оценка]) | 768 [станд. ViT-B] | лицензия НЕ указана (файла LICENSE нет) | после FT (протокол TransReID): VeRi-776 **85.6 mAP / 97.9 R1** | github.com/Event-AHU/VehicleMAE; arXiv:2312.09812 (PDF) | ПРОВЕРЕНО (числа/наличие весов); размер файла не опубликован |

Пояснения:
- Для CLIP/SigLIP2 указан размер полного чекпойнта (image+text башни); для feature-extraction нужна только vision-башня (~половина или меньше).
- DINOv3 доступен только после ручного gated-запроса на HF (форма: имя, дата рождения, страна, аффилиация, должность) либо через портал Meta. [ПРОВЕРЕНО: HF API, поле gated=manual]
- Полный список DINOv3-вариантов на HF: ViT S/S+/B/L/H+/7B (LVD-1689M), ViT-L/7B (SAT-493M, спутниковый), ConvNeXt T/S/B/L. [ПРОВЕРЕНО: HF API author=facebook]

### Лицензии подробно

- **DINOv2**: Apache-2.0 (изначально была CC-BY-NC-4.0, сменена на Apache-2.0 ещё в 2023 — [ПО ПАМЯТИ] в части истории; текущее состояние Apache-2.0 [ПРОВЕРЕНО: HF API facebook/dinov2-base]).
- **DINOv3**: «DINOv3 License» (license: other). Разобрана по тексту [ПРОВЕРЕНО: https://ai.meta.com/resources/models-and-libraries/dinov3-license]:
  - коммерческое использование **разрешено**: «non-exclusive, worldwide, non-transferable and royalty-free limited license… use, reproduce, distribute, copy, create derivative works of, and make modifications»; порогов по выручке/MAU (как у Llama) **нет**;
  - **запрещено**: военные применения/оружие, ядерная отрасль, шпионаж («military or warfare purposes, nuclear industries or applications, espionage…»); reverse engineering; нарушение экспортного контроля (OFAC/ITAR);
  - **обязательства**: при распространении производных — копия соглашения + заметная надпись «Built with DINOv3» на сайте/в документации; упоминание в публикациях.
  - Плюс gated-доступ к весам (персональные данные при запросе).
- **CLIP (OpenAI)**: MIT, «Copyright (c) 2021 OpenAI» [ПРОВЕРЕНО: raw LICENSE в openai/CLIP]. На HF-карточке openai/clip-vit-large-patch14 тег лицензии отсутствует [ПРОВЕРЕНО: HF API].
- **OpenCLIP (LAION-веса)**: MIT [ПРОВЕРЕНО: HF API laion/CLIP-ViT-B-32-laion2B-s34B-b79K].
- **SigLIP и SigLIP2**: Apache-2.0 [ПРОВЕРЕНО: HF API google/siglip-so400m-patch14-384 и google/siglip2-*].
- **VehicleMAE**: лицензия не указана ни в репо, ни в README → формально «все права защищены», для коммерческого продукта рискованно. [ПРОВЕРЕНО: github.com/Event-AHU/VehicleMAE]

---

## 4. Подводные камни: «тип/цвет ловят, экземпляры — нет»

Тезис подтверждается сразу несколькими независимыми публикациями:

1. **Прямое подтверждение для object re-id**: «Generalizable Object Re-Identification via Visual In-Context Prompting» (VICP), Huang & Liu (MSU), arXiv:2508.21222 (август 2025):
   > «DINOv2 predominantly retrieves images sharing shape similarity or semantic attributes… but fails to prioritize identity-specific features, resulting in frequent false positives.»
   Числа: frozen DINOv2 на ShopID10K — mAP 34.1 / R1 45.7; на PetFace — mAP 6.5. Их метод (DINOv2-S + LLM-guided in-context prompting) на VeRi-776 даёт 81.2 mAP / 97.1 R1, «превосходя TransReID (79.6/97.0)» — но протокол для VeRi-таблицы в тексте явно не описан (in-domain или cross-domain — неясно), цитировать с осторожностью. [ПРОВЕРЕНО: https://arxiv.org/html/2508.21222v1]

2. **Атрибуты — да, идентичность — нет**: «Evaluating 2D and 3D-Aware Vision Foundation Models for Vehicle Attribute Recognition», Delazeri и др. (UFPR), arXiv:2608.29929 (август 2026): 14 foundation-моделей, frozen linear probing на UFPR-VeSV (24 945 снимков с камер, 14 типов / 26 марок / 136 моделей). **DINOv3 — лучший: 97.7 % Ma-Acc по типу и >93 % по марке и модели**. То есть frozen-признаки почти полностью решают «какая это машина» (тип/марка/модель) — а vehicle re-id требует отличать экземпляры ВНУТРИ этих классов, что как раз и остаётся узким местом. CLIP/SigLIP2 в этой работе не замерялись. [ПРОВЕРЕНО: https://arxiv.org/html/2608.29929]

3. **Zero-shot провал на re-id при отличных «общих» признаках** — arXiv:2601.20598 (§1.3): DINOv2 0.3–4.7 % mAP zero-shot; авторская формулировка причины: «purely visual self-supervised pretraining lacks the explicit semantic structure required for identity matching». Language-aligned (SigLIP2) на порядок лучше в zero-shot и робастнее cross-domain, но и его 5–6 % mAP — непригодно для продакшена без дообучения. [ПРОВЕРЕНО]

4. **Даже дообученные reid-модели опираются на тип, а не на экземплярные детали**: «Generalization Limits in Vehicle Re-Identification», Ben Mabrouk и др., arXiv:2606.01981 (июнь 2026). Проблема: в стандартных сплитах VeRi-776/VERI-Wild визуально почти одинаковые машины (та же марка/модель/цвет) есть и в train, и в test → модели «запоминают» типы. На предложенных сплитах по невиданным типам: CLIP-ReID падает с 84.1 → **60.2 mAP** на VeRi-776 (≈−24 п.п.); на VERI-Wild с 80.3 → 67.7 (mixed view) и до **51.8 mAP** при смене ракурса; разрывы до 30 п.п. Вывод авторов: «most state-of-the-art methods struggle with unseen vehicle types, and their robustness to viewpoint changes and attention to detail are limited to vehicle types seen during training». [ПРОВЕРЕНО: https://arxiv.org/abs/2606.01981 + html]

5. **Косвенно**: VehicleDINO (§1.4) — frozen DINOv2+LoRA: марка 98.4 % acc, но re-id только 61.1 mAP. [ПРОВЕРЕНО]

6. Контрапункт: frozen DINOv2/v3 всё же умеют instance-retrieval на ландмарках (Oxford-Hard 54 mAP, §1.6) и frozen DINOv3 заметно лучше CLIP/ResNet как «голый» бэкбон на VERI-Wild (18.6 vs 3.8, §1.2) — т.е. сигнал об экземпляре в признаках есть, просто его недостаточно без метрического дообучения на identity-суперзвиции.

**Практический итог**: zero-shot foundation-эмбеддинги пригодны как грубый префильтр (цвет/тип/марка/ракурс) и как инициализация, но не как готовый vehicle-reid матчер; минимально необходимое — метрическое дообучение (хотя бы LoRA — даёт скачок с ~15–19 до 60–75 mAP; полное FT — до 85–92 mAP).

---

## 5. VehicleMAE и специализированные предобучения

### 5.1. VehicleMAE (AAAI 2024)

«Structural Information Guided Multimodal Pre-training for Vehicle-centric Perception», Xiao Wang и др. (Anhui University), arXiv:2312.09812. MAE + Structural Prior (скетч-контуры кузова) + Semantic Prior (дистилляция из CLIP по image-text парам). Предобучение: **Autobot1M** — 1 026 394 изображений машин (включая CompCars и VERI-Wild; 732 112 surveillance + 294 282 web), 12 693 текст-описания. Архитектура: ViT-Base/16.

Downstream vehicle re-id: **VeRi-776** (train 37 715 / test 11 579), протокол — дообучение в рамках TransReID. Таблица 1 (mAP / R1):

| Инициализация | mAP | R1 |
|---|---|---|
| Scratch (без предобучения) | 35.3 | 57.3 |
| MoCov3 (ImageNet-1K) | 75.5 | 94.4 |
| DINO (ImageNet-1K) | 64.3 | 91.5 |
| iBOT (ImageNet-1K) | 68.9 | 92.6 |
| MAE (ImageNet-1K) | 76.7 | 95.8 |
| MAE (Autobot1M) | 75.5 | 95.4 |
| **VehicleMAE (Autobot1M)** | **85.6** | **97.9** |

Ключевые наблюдения: (а) доменное предобучение само по себе (MAE на Autobot1M) НЕ помогает (75.5 vs 76.7 у MAE-IN1K) — помогают именно структурные/семантические prior'ы (+8.9 mAP); (б) VehicleMAE-инициализация почти догоняет специализированные reid-методы. Прочие задачи: атрибуты (VeRi) mA 92.21; fine-grained (Stanford Cars) acc 94.5; сегментация частей mIoU 73.29.
[ПРОВЕРЕНО: PDF https://arxiv.org/pdf/2312.09812 — таблица извлечена pdftotext]

**Веса**: открыто выложен ViT-B чекпойнт, но только на Baidu Netdisk (код 6zkx) — для скачивания вне Китая нужен аккаунт Baidu; зеркала на HF нет (проверено поиском и HF API author=Event-AHU — пусто). Файла LICENSE в репо нет. [ПРОВЕРЕНО: github.com/Event-AHU/VehicleMAE]

### 5.2. VehicleMAE-V2 (журнальное расширение, дек. 2025)

«Vehicle-centric Perception via Multimodal Structured Pre-training», Wu, Wang, Li, Tang, Luo; arXiv:2512.19934 (22.12.2025): датасет **Autobot4M** (~4M изображений, 12 693 описания), модули Symmetry-guided Mask / Contour-guided / Semantics-guided, 5 downstream-задач. Конкретные числа и статус весов из abstract не видны; полный текст не разбирался. [ПРОВЕРЕНО (abs): https://arxiv.org/abs/2512.19934; числа НЕ ПРОВЕРЕНЫ]

### 5.3. Одноимённый VehicleMAE с ICCV 2025 (другая группа!)

«VehicleMAE: View-asymmetry Mutual Learning for Vehicle Re-identification Pre-training via Masked AutoEncoders», Qi Wang и др., ICCV 2025, стр. 4701–4711 — предобучение специально под vehicle re-id: диффузионно сгенерированный датасет **DiffVERI** (>1.7M изображений с multi-view аннотациями), модули VMIM (view-asymmetry masked image modeling) и PPMD (past-to-present mutual distillation). Не путать с AAAI-2024 VehicleMAE. Численные результаты недоступны: openaccess.thecvf.com отдаёт 403, arXiv-версия не найдена. [ЧАСТИЧНО ПРОВЕРЕНО: абстракт через mlanthology.org/iccv/2025/wang2025iccv-vehiclemae/; числа НЕ ПРОВЕРЕНЫ]

---

## 6. Выводы для практики

1. **Zero-shot (совсем без обучения)**: ни одна из универсальных моделей не даёт рабочего качества на vehicle re-id. Ожидаемые mAP: единицы процентов (CLIP/DINOv2, по аналогии с person-замерами 2601.20598) до ~15–19 (DINOv3 на VERI-Wild, 2607.22068). Годится только для префильтрации по типу/цвету/марке — это frozen-модели решают почти идеально (97+ % — 2608.29929).
2. **Лёгкое дообучение (LoRA/адаптеры при frozen бэкбоне)**: скачок до 61 mAP (DINOv2-B, VeRi-776, VehicleDINO) и 75 mAP (DINOv3-ViT-L, VERI-Wild, 2607.22068). Уже пригодно для многих применений, но до SOTA далеко.
3. **Полное дообучение**: DINOv3-ConvNeXt-B 88.2 mAP (VERI-Wild) и 66 mAP cross-dataset на VeRi-776 без адаптации; CLIP-инициализация (CLIP-ReID→CLIP-SENet) 83–93 mAP на VeRi-776. Специализированное предобучение (VehicleMAE) даёт +9 mAP к MAE-IN1K при одинаковом FT.
4. **Выбор модели**: (а) максимум качества при полном FT — DINOv3 (но лицензия Meta: gated, «Built with DINOv3», запрет military; коммерция разрешена); (б) максимально чистая лицензия — DINOv2 (Apache-2.0) или SigLIP2 (Apache-2.0); (в) лучший zero-shot из коробки — SigLIP2 (language-aligned признаки робастнее cross-domain); (г) VehicleMAE — интересная инициализация, но веса за Baidu Netdisk и без лицензии.
5. **Открытая ниша**: систематический бенчмарк «frozen DINOv2/v3/CLIP/SigLIP2 на VeRi-776/VehicleID/VERI-Wild» в литературе отсутствует — при необходимости придётся замерять самим (это дёшево: только инференс + retrieval).

## Источники (открывались через WebFetch)

- arXiv:2211.13977 (CLIP-ReID, AAAI 2023) — https://arxiv.org/abs/2211.13977 ; таблицы: https://ar5iv.labs.arxiv.org/html/2211.13977
- arXiv:2502.16815 (CLIP-SENet, TITS 2025) — https://arxiv.org/abs/2502.16815
- arXiv:2601.20598 (Person Re-ID in 2025) — https://arxiv.org/html/2601.20598v1
- arXiv:2607.22068 (Vehicle ReID in Foundation-Model Era) — https://arxiv.org/html/2607.22068v1
- arXiv:2608.29929 (Foundation Models for Vehicle Attribute Recognition) — https://arxiv.org/html/2608.29929
- arXiv:2606.01981 (Generalization Limits in Vehicle Re-ID) — https://arxiv.org/abs/2606.01981
- arXiv:2606.09362 (Zero-Shot Semantic Re-ID, VLM) — https://arxiv.org/html/2606.09362v1
- arXiv:2508.21222 (VICP, generalizable object re-id) — https://arxiv.org/html/2508.21222v1
- arXiv:2304.07193 (DINOv2) — https://arxiv.org/html/2304.07193v2
- arXiv:2508.10104 (DINOv3) — https://arxiv.org/abs/2508.10104 ; https://arxiv.org/html/2508.10104v1
- arXiv:2312.09812 (VehicleMAE, AAAI 2024; PDF) — https://arxiv.org/pdf/2312.09812
- arXiv:2512.19934 (VehicleMAE-V2) — https://arxiv.org/abs/2512.19934
- arXiv:2310.17218 (PCL-CLIP) — https://arxiv.org/abs/2310.17218
- arXiv:2508.04120 (CLIPVehicle) — https://arxiv.org/html/2508.04120
- ICCV 2025 VehicleMAE (view-asymmetry): https://mlanthology.org/iccv/2025/wang2025iccv-vehiclemae/ (openaccess.thecvf.com — 403)
- GitHub: Syliz517/CLIP-ReID; Event-AHU/VehicleMAE; facebookresearch/dinov3 (README); openai/CLIP (LICENSE)
- Hugging Face API/страницы: facebook/dinov2-{small,base,large,giant}; facebook/dinov3-{vits16,vitb16,vitl16,vith16plus,vit7b16,convnext-base}-pretrain-lvd1689m; openai/clip-vit-large-patch14; google/siglip2-base-patch16-224; google/siglip2-so400m-patch14-384; google/siglip-so400m-patch14-384; laion/CLIP-ViT-B-32-laion2B-s34B-b79K; wms2537/VehicleDINO
- Лицензия DINOv3: https://ai.meta.com/resources/models-and-libraries/dinov3-license
