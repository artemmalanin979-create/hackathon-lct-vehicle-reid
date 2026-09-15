# Обзор моделей и весов для повторной идентификации автомобилей (vehicle re-id)

Дата: 2026-09-15. Подготовлено по заданию `BRIEF.md` с учётом ТЗ заказчика (`context-tz.txt`): open-set retrieval по кропу ТС, основная метрика mAP (+Rank-1/5), суммарные веса ≤ 2 ГБ, офлайн-инференс, замер (а) времени на объект при батче 1 и (б) пропускной способности; заказчик коммерческий, предпочтение свободному ПО.

**Методика и статус проверки.** Каждый факт добывался открытием первоисточника (GitHub/GitHub API, Hugging Face API, arXiv, PyPI, официальные страницы датасетов, MLPerf-логи) 15.09.2026. Если не указано иное — факт **проверен открытием источника**; всё, что взято по памяти или из недоступного источника, помечено «⚠ не проверено». Детальные выкладки с точными URL по каждой строке — в `notes/01-weights.md` … `notes/05-speed-licenses.md`; здесь — сводка с ключевыми ссылками. Замечание об экосистеме: paperswithcode.com закрыт 24.07.2025 (редирект на HF Papers), лидерборды утрачены — SOTA восстановлен по сравнительным таблицам статей 2025–2026.

---

## 1. Готовые предобученные веса (проверены скачиваемость, лицензия, числа)

| Модель | URL весов | Лицензия | Размер | Архитектура | Обучено на | mAP / R1 (бенчмарк) | Источник числа |
|---|---|---|---|---|---|---|---|
| fast-reid SBS(R50-ibn) VeRi | github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth | Apache-2.0 | 189 МБ | ResNet50-IBN (SBS) | VeRi-776 | **81.9 / 97.0** (VeRi-776) | MODEL_ZOO.md fast-reid |
| fast-reid BoT(R50-ibn) VehicleID | …/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth | Apache-2.0 | 578 МБ | ResNet50-IBN (BoT) | VehicleID | R1 **86.6** (small) … 80.6 (large) | MODEL_ZOO.md |
| fast-reid BoT(R50-ibn) VERI-Wild | …/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth | Apache-2.0 | 988 МБ | ResNet50-IBN (BoT) | VERI-Wild | **87.7 / 96.4** (small) … 77.3 / 92.5 (large) | MODEL_ZOO.md |
| **Intel OMZ vehicle-reid-0001** | storage.openvinotoolkit.org/…/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx | MIT | **8,4 МБ** (ONNX) | OSNet-AIN x1.0, эмб. 512-d, вход 208×208 | в README не названо | **85.15 / 96.31** (VeRi-776) | README open_model_zoo |
| TransReID ViT-B VeRi | Google Drive (см. README damo-cv/TransReID) | MIT | ⚠ размер не виден (GDrive); ~340 МБ ⚠ по памяти | ViT-B/16 + JPM + SIE | VeRi-776 | **82.1 / 97.4**; DeiT-B: 82.4 / 97.1 | README TransReID |
| CLIP-ReID ViT VeRi | Google Drive (см. README Syliz517/CLIP-ReID) | MIT | ⚠ не виден (GDrive) | ViT-B/16 (иниц. CLIP) | VeRi-776 | **83.3 / 97.4**; вариант SIE+OLP: **84.5 / 97.3** | статья arXiv:2211.13977 (Table 3) |
| HF occurra/vehicle_vit_clip_reid (зеркало CLIP-ReID) | huggingface.co/occurra/vehicle_vit_clip_reid | MIT | .pth 505 МБ / **.onnx 345 МБ** | CLIP ViT-B/16 + проекция, эмб. 512-d | VeRi-776 | на карточке метрик нет | — |
| HF occurra/vehicle_reid_siglip2_naflex_512d | huggingface.co/occurra/vehicle_reid_siglip2_naflex_512d | Apache-2.0 | 373 МБ | SigLIP2 NaFlex, эмб. 512-d | не названо | 64.5 / 84.4 (VeRi-776, авторская оценка) | карточка модели |
| VehicleNet ft-VeRi (layumi/AICIty-reID-2020) | Google Drive (README) | MIT (репо), но обучено на CityFlow → см. §6 | ⚠ не виден | ResNet-50 | VehicleNet (VeRi+VehicleID+CompCars+CityFlow) | **83.41** mAP (VeRi-776) | README репо |
| PaddleDetection PP-LCNet vehicle | paddledet.bj.bcebos.com/models/mot/deepsort/deepsort_pplcnet_vehicle.tar | Apache-2.0 | ⚠ не проверен | PP-LCNet (лёгкий) | VERI-Wild | метрик в доке нет | — |
| HF umair894/KAT-ReID-Veri776 | huggingface.co/umair894/KAT-ReID-Veri776 | Apache-2.0 | 345 МБ | ViT + GR-KAN | VeRi-776 | 59.5 / 88.0 — слабая | карточка модели |

Где весов **нет**: PaddleClas (только конфиги обучения на VERI-Wild), regob/vehicle_reid (MIT, «обучайте сами»), решения победителей AI City (DMT-2021: веса есть, но без лицензии и на запрещённом для коммерции CityFlow), поиск «veri-wild» на HF — 0 результатов. Полная таблица и лог неудачных запросов — `notes/01-weights.md`.

**Важно про числа fast-reid/OMZ:** это числа авторов моделей на стандартных сплитах; независимая перепроверка 2026 г. (arXiv:2606.01981) при строгом протоколе получает близкие, но чуть меньшие значения у SOTA-методов (TransReID 81.3, CLIP-ReID 84.1 на VeRi-776).

---

## 2. Библиотеки (состояние на 15.09.2026)

| Библиотека | Активность | Версия | PyTorch | Лицензия | ONNX | Vehicle-датасеты | За 2 недели? |
|---|---|---|---|---|---|---|---|
| **fast-reid** | коммит 15.07.2024 — заброшен | v1.3.0 (2021), PyPI 1.4.0 (2023) | писан под 1.6; безопасно ~1.10–2.1, requirements не запинены (issue #758) | Apache-2.0 | да (`tools/deploy`: ONNX+TRT) | **да: VeRi/VehicleID/VERI-Wild из коробки** | **Да**, +1–2 дня на заморозку окружения |
| **BoxMOT** | релиз v25.0.0 от 09.09.2026 — самый живой | 25.0.0 | torch ≥2.2.1 <3.0 | **AGPL-3.0** — блокер для закрытого продукта | лучший: ONNX/TRT/OpenVINO/TFLite одной командой | `train-reid` «person or vehicle», свой датасет через YAML | Да технически; юр. вопрос главный |
| **torchreid (deep-person-reid)** | код заморожен с ~02.2023 | GitHub v1.0.6; PyPI-имя `torchreid` **перехвачено** чужой сборкой | баги с torch 2.6 (#592), ONNX-экспорта (#584/585) | MIT | скрипт есть, баги открыты | нет (person) | Только как источник OSNet |
| **layumi/Person_reID_baseline** | push 30.08.2026, DINOv3-бэкбон с 02.2026 — живой | без релизов | до ~2.0+ | MIT | нет из коробки; модели простые, экспорт руками ⚠ не проверен | VeRi с правками prepare.py/test.py (issue #107) | **Да** — лёгкая MIT-альтернатива |
| **PaddleClas** | v2.6.0 (11.2024), режим поддержки | 2.6.0 | нет (PaddlePaddle) | Apache-2.0 | Paddle2ONNX | конфиги VERI-Wild ReID (ArcMargin) | Только для команды со стеком Paddle |
| TransReID / CLIP-ReID (иссл. код) | 2024 / 2023, мертвы | — | 1.6 / 1.8 | MIT | нет | VeRi + VehicleID + претрейны | Как доноры весов/рецептов — да |
| pytorch-metric-learning / open-metric-learning | 08.2025 / 11.2025 — живые | — | актуальный | MIT / Apache-2.0 | штатный torch.onnx | нет (собирать самим) | Запасной путь «reid как retrieval» |
| reid-strong-baseline, mmpretrain/mmtracking, deep-object-reid, roboflow/trackers | мертвы или без reid | — | — | — | — | — | Не брать |

Подробности (даты по GitHub API, цитаты issues) — `notes/02-libraries.md`.

---

## 3. Вариант без обучения: foundation-модели как экстракторы

**Главный вывод: чистый zero-shot непригоден; замеров DINOv2/CLIP/SigLIP2 именно на VeRi-776 в литературе нет** (проверенный пробел — при желании дёшево замерить самим на выданном датасете).

Что найдено (все числа — из открытых статей):
- Frozen-бэкбоны на **VERI-Wild** (arXiv:2607.22068, 07.2026): DINOv3-ViT-L **18.6**, DINOv3-ConvNeXt-B 14.6, ResNet50-ImageNet 3.8 (mAP по контексту статьи, 256×256). После полного дообучения тот же ConvNeXt-B — **88.19 mAP** (small) / 77.47 (large); перенос на VeRi-776 без адаптации — 66.0 mAP.
- Zero-shot на **person re-id** (систематический бенчмарк arXiv:2601.20598): CLIP-L/14 — 0.14–0.50 % mAP, DINOv2-L — 0.39–1.40 %, лучший SigLIP2 — лишь 4.6–6.0 %. Авторы: у чисто визуального SSL «нет явной семантической структуры для матчинга идентичностей».
- Лёгкое дообучение спасает: frozen DINOv2-B + LoRA — **61.1 mAP / 86.1 R1** на VeRi-776 (community-модель wms2537/VehicleDINO, Apache-2.0, FP32 450 МБ); LoRA на DINOv3-ViT-L — 75.2 mAP VERI-Wild. Полное дообучение CLIP-инициализации (CLIP-ReID) — 83.3–84.5 mAP VeRi-776.
- Подводный камень подтверждён публикациями: frozen-признаки почти идеально решают тип/марку/модель (DINOv3 linear probe 97.7 %/93 %+, arXiv:2608.29929), но «fail to prioritize identity-specific features» (arXiv:2508.21222); даже дообученный CLIP-ReID падает 84.1→60.2 mAP на невиданных типах кузова (arXiv:2606.01981).
- VehicleMAE (AAAI 2024, arXiv:2312.09812): доменное предобучение ViT-B на 1M кропов машин → после FT **85.6 mAP** VeRi-776 (против 76.7 у MAE-ImageNet). Но веса только на Baidu Netdisk и **без лицензии**.

| Модель (HF id) | Веса (fp32) | Эмб. | Лицензия |
|---|---|---|---|
| facebook/dinov2-small / base / large | 88 МБ / 346 МБ / 1.22 ГБ | 384 / 768 / 1024 | Apache-2.0 |
| facebook/dinov3-vitb16 / vitl16 / convnext-base (LVD-1689M) | 343 МБ / 1.21 ГБ / 350 МБ | 768 ⚠ / 1024 ⚠ / 1024 ⚠ (конфиги gated) | DINOv3 License (Meta): коммерция разрешена, gated-доступ, запрет military/ITAR, надпись «Built with DINOv3» |
| openai/clip-vit-large-patch14 (обе башни) | 1.71 ГБ | proj 768 | MIT |
| google/siglip2-base-patch16-224 (обе башни) | 1.50 ГБ | 768 ⚠ | Apache-2.0 |

**Практически:** zero-shot годится только как префильтр по типу/цвету; рабочий минимум — метрическое дообучение (LoRA → 61–75 mAP, полное FT → 85–92 mAP). Подробно — `notes/03-foundation.md`.

---

## 4. Что реально улучшает качество (2024–2026) и устойчивость к ракурсу/условиям

**Честная верхушка VeRi-776 без re-ranking: ~84.5–86.7 mAP / 97–98 R1** (CLIP-ReID SIE+OLP 84.5 — код+веса MIT; MBR-4B-LAI 85.63 — код есть, лицензии нет; LKA-ReID 86.65 — кода нет). С re-ranking — 88–92 mAP. Заявка CLIP-SENet 92.9 mAP (TITS 2025) — **без кода, протокол оспорен** в arXiv:2607.22068 (пропуск junk-фильтра завышает mAP на +3–4 п.п.; в её сравнительной таблице числа конкурентов взяты с RR). VERI-Wild (протокольно верифицировано, 2607.22068): 88.19/77.47 mAP (small/large), с RR 92.38/83.68. Вывод той же работы: в эпоху foundation-бэкбонов мульти-бранч-фьюжн почти не окупается (≤ +0.11 mAP) — важнее один хорошо дообученный бэкбон.

Воспроизводимо за 2 недели малой командой: fast-reid (Apache-2.0, 1 GPU), CLIP-ReID / TransReID (MIT, обучение на 1×V100, веса есть), MBR (1×RTX 4090, но лицензии нет), RPTM (MIT, обучать самим). Невоспроизводимо: CLIP-SENet, LKA-ReID (нет кода).

**Ракурс** (перёд/бок/зад): на слабых бейзлайнах viewpoint-aware метрики давали много (VANet: 58.75→66.34 mAP), на сильных трансформерах остаточный прирост скромный: SIE(cam+view) в TransReID +1.4 mAP, LAI в MBR +0.91, SIE+OLP в CLIP-ReID +1.2. Синтетика **VehicleX** (attribute descent): +4.08 mAP на VeRi, +6.95 на CityFlow — рабочий приём при малых данных. ⚠ Нюанс под наше ТЗ: организатор **не передаёт id камеры/ракурс** — SIE по камере недоступен, по ракурсу потребуется свой классификатор ракурса.

**Ночь/дождь/блики:** отдельное направление Day-Night re-id (CVPR 2024, DNDM + датасеты DN-Wild/DN-348: подавление бликов фар; числа не проверены — openaccess отдаёт 403). Замеры деградации по погоде (arXiv:2607.10583): дождь бьёт сильнее тумана (−13…−17 mAP), самый устойчивый лосс — AdaSP, ViT без спец-мер деградирует сильнее CNN. Практика: аугментации ночь/дождь/блики + структурные признаки важнее общего low-light enhancement. Подробно — `notes/04-sota-reranking.md`.

---

## 5. Бюджет 2 ГБ и скорость

**Бюджет — не ограничение** для одиночной модели: ResNet50 97.8 МБ fp32 (torchvision docs), OSNet x1.0 ~8.8 МБ (2.2M парам.), EfficientNet-B0…B4 20.5–74.5 МБ, ViT-B/16 330 МБ, Swin-B 335 МБ, ConvNeXt-B 338 МБ, ViT-L ~1.16–1.22 ГБ; fp16 — вдвое меньше. В 2 ГБ влезает даже ViT-L + вспомогательные модели; проблема возникает только у ансамблей крупных моделей.

Скоростные ориентиры (все со ссылками в `notes/05-speed-licenses.md`):
- **Батч 1, PyTorch eager**: ResNet50 — 10.4 мс (V100 fp32) / 10.7 мс (T4) на изображение (офиц. таблицы NVIDIA DeepLearningExamples).
- **Пропускная способность, TensorRT int8 (MLPerf Offline)**: ResNet50 — ~5 622 img/s на T4, 13 158 на L4, 42 379 на A100.
- **PyTorch AMP, большой батч (timm benchmarks)**: RTX 3090 — R50 3 219 / ViT-B 1 571 img/s; RTX 4090 — 4 218 / 10 113 img/s.
- **Экспорт**: Torch-TensorRT — «до 6x» к eager (блог NVIDIA, EffNet-B0, A100, fp16, батч 1); ONNX Runtime + fp16 — «2.88x» к PyTorch (блог Microsoft, T4). Реалистичное обещание: **2–6x**; главный источник ускорения — fp16/int8 + слияние графа. ⚠ Прямых опубликованных замеров батч-1 TensorRT-латентности R50 не нашли; типичные ~1–2 мс — по памяти.

Следствие для выбора: и R50-IBN, и ViT-B укладываются в «реальное время» на любом разумном GPU организатора; OSNet (2.6 GFLOPs) — страховка, если железо окажется слабым, а ViT-B — самый дорогой по батчу-1 без TensorRT.

---

## 6. Лицензионная чистота (коммерческий заказчик)

Датасеты (условия открыты и процитированы в `notes/05-speed-licenses.md`):
| Датасет | Доступ | Условия |
|---|---|---|
| VeRi-776 | письмо (ФИО+аффилиация) | «non-commercial purposes» |
| VehicleID (PKU) | подписанное соглашение, только академическая почта | «ACADEMIC PURPOSES. NO COMERCIAL USE» |
| VERI-Wild | письмо | «non-commercial purposes» |
| CityFlow / AI City (NVIDIA) | форма + лицензия | запрет «use the DATASET **or any models developed using the DATASET** for any commercial or production purpose» |
| VehicleX (синтетика) | открытый GitHub | LICENSE-файла нет; «for your research purpose» — статус не определён |

Веса: **чистые для коммерции** — DINOv2, SigLIP/SigLIP2, timm-веса (Apache-2.0), CLIP (MIT), Paddle (Apache-2.0), Intel OMZ vehicle-reid-0001 (MIT); **DINOv3** — спец-лицензия Meta: коммерция разрешена, но gated-доступ, запрет military/ITAR, обязательная надпись «Built with DINOv3». **Ловушки**: SWAG-веса ViT в torchvision — CC-BY-NC 4.0 (некоммерческие); веса без лицензии (MBR, VehicleMAE, UFDN, DMT) — формально «all rights reserved»; PyPI-пакет `torchreid` — не официальный.

**Веса, обученные на research-only датасетах** (весь fast-reid vehicle-zoo, TransReID/CLIP-ReID VeRi-веса): для CityFlow запрет на модели явный; для VeRi/VehicleID/VERI-Wild публичные тексты про веса молчат — **серая зона, нужен юрист** (позиция подтверждена: torchvision прямо перекладывает ответственность на пользователя; arXiv:2303.15715 фиксирует правовую неопределённость). Для хакатона внешние источники разрешены при перечислении в README, но: доступ к VeRi/VehicleID — по заявке/соглашению, т.е. жюри не сможет свободно их скачать — риск по критерию воспроизводимости; для промышленной эксплуатации безопасный путь — дообучение чистых баз (DINOv2/CLIP/SigLIP2/timm) на данных организатора и своей синтетике.

---

## 7. Переранжирование

k-reciprocal (CVPR 2017, дефолт k1=20, k2=6, λ=0.3) на vehicle re-id даёт **+4…+10 mAP** (проверенные пары: 78.94→88.44 и 80.8→88.0 на VeRi-776; 85.63→92.09 MBR; 88.19→92.38 и 77.47→83.68 на VERI-Wild) — прирост сохраняется даже у DINOv3-бэкбонов 2026 г. **Rank-1 почти не растёт** — весь эффект в mAP. Цена: O(N²); классический CPU-код — 89 с на галерее ~20k и падение по памяти (~190 ГБ) на 128k; sparse-реализация — 29–137 с; **GNN-re-ranking на GPU — 9.4 мс** на галерее ~20k при том же качестве (arXiv:2012.07620; код Xuanmeng-Zhang/gnn-re-ranking + PyTorch-порт у layumi); GCR (TMM 2023) — на CPU в 3–7 раз быстрее классики и чуть точнее.

**Применимость в нашем регламенте: да, почти обязательно.** Организатор меряет только «изображение → эмбеддинг» (батч 1 и throughput); re-ranking — постпроцессинг матрицы дистанций и в замер не попадает. Оговорки: (а) в submission.csv/candidates.csv идут уже переранжированные списки — это легально по ТЗ; (б) для режима отказа пороги калибровать на переранжированных дистанциях (Jaccard-компонента меняет масштаб оценок); (в) на галерее ~10⁶ (демо масштабируемости) — только sparse/GNN-варианты после ANN-предфильтра.

---

## Что мы НЕ смогли проверить (честно)

1. Размеры файлов весов на Google Drive (TransReID, CLIP-ReID, VehicleNet, UFDN, DMT) — Drive не отдаёт размер без входа; и размер PP-LCNet vehicle .tar.
2. Причину крупных .pth у fast-reid (578/988 МБ) — гипотеза «внутри состояние оптимизатора, после стрипа меньше» не проверена.
3. Числа из закрытых PDF: DNDM (CVPR 2024) и Cheb-GR (CVPR 2025) — openaccess.thecvf.com отдаёт 403; ICCV-2025 VehicleMAE (view-asymmetry); статьи на ScienceDirect.
4. Zero-shot mAP DINOv2/CLIP/SigLIP2 именно на VeRi-776 — таких замеров в литературе нет (наша экстраполяция с person-reid — оценка, не факт).
5. Батч-1 латентность ResNet50 в TensorRT — прямого опубликованного замера не нашли (только амортизированная нижняя граница из MLPerf).
6. Размерности эмбеддингов DINOv3 — конфиги за gated-доступом (указаны по памяти).
7. Полные тексты подписываемых соглашений VeRi-776/VehicleID (высылаются по запросу) — могут содержать доп. условия, в т.ч. про модели.
8. Наличие/цену коммерческой лицензии BoxMOT (вместо AGPL) и юридический статус весов, обученных AGPL-кодом.
9. Верхнюю границу совместимости fast-reid с PyTorch (оценка «до ~2.1» — по issues, не по тестам); практический экспорт layumi-моделей в ONNX.
10. ModelScope (страницы на JS не читаются), web.archive.org (недоступен из среды), лидерборды Papers with Code (сервис закрыт).
11. Метрики зеркала occurra/vehicle_vit_clip_reid — карточка их не приводит (оригинальные числа CLIP-ReID к зеркалу применимы лишь предположительно).
12. Конфигурация GPU организатора не объявлена — все скоростные выводы даны диапазоном по T4/V100/L4/A100/RTX.

---

## Рекомендация

1. **Старт (дни 1–2):** бейзлайн без обучения на выданных данных — Intel OMZ `vehicle-reid-0001` (MIT, 8,4 МБ ONNX, заявлено 85.15 mAP на VeRi-776): сразу даёт рабочий пайплайн «кроп → 512-d вектор → косинус → submission» и точку отсчёта.
2. **Основной трек качества:** дообучение ViT-B/16 с CLIP-инициализацией по рецепту CLIP-ReID (MIT) либо fast-reid R50-IBN (Apache-2.0, окружение запинить: torch ~1.13–2.1, numpy<2) **на train-данных организатора** — это одновременно и лучший результат по литературе (83–86 mAP на VeRi-классе задач), и юридически чистый путь (без research-only весов в проде).
3. **Обязательно:** L2-нормировка эмбеддингов; k-reciprocal/GNN re-ranking в своём пайплайне (+4–10 mAP бесплатно к замеряемой скорости); аугментации ночь/дождь/блики; для ракурсов — сэмплинг пар «разные виды одного ID» и, при ресурсах, синтетика типа VehicleX.
4. **Скорость/сдача:** экспорт в ONNX Runtime (fp16) с опцией TensorRT — ожидаемое ускорение 2–6x; и R50, и ViT-B укладываются в реальное время на любом кандидатном GPU; бюджет 2 ГБ позволяет даже пару «лёгкая + тяжёлая модель».
5. **Запасные пути:** (а) если обучение буксует — готовые веса fast-reid VeRi/VERI-Wild как драфт для хакатона (с перечислением источников и пометкой о research-only происхождении); (б) если железо организатора слабое — OSNet/PP-LCNet-класс (≤10 МБ, миллисекунды на CPU/GPU); (в) DINOv2-B + LoRA — если захочется foundation-трека (но по литературе он пока проигрывает CLIP-ReID-рецепту).
6. **Не делать:** не брать CityFlow/AI City данные и веса (прямой запрет на коммерческие модели); не использовать SWAG-веса torchvision (CC-BY-NC); не ставить `torchreid` с PyPI (перехвачено имя); не гнаться за CLIP-SENet (92.9 mAP — без кода, протокол оспорен); BoxMOT — только если AGPL приемлема.
7. **Юридически:** до промышленного релиза получить заключение юриста по весам, обученным на research-only датасетах; в README хакатона перечислить абсолютно все внешние источники.
