# A. С каких весов стартовать — замеры из первоисточников

Дата сбора: 2026-09-16. Все ссылки проверены (HTTP 200 / прямое открытие) в день сбора.
Контекст задачи: цель = 7 248 кадров / 1 171 ID / 96 камер, вход 208×208; ImageNet-R50 30 эпох → mAP 0,433;
OSNet-AIN `vehicle-reid-0001` без обучения → mAP 0,657.

Обозначения: «проверено» = да, если я открыл сам источник (raw-файл, PDF, arXiv HTML, API) и вижу число своими глазами;
«косвенно» = число взято из текста источника, но таблица в PDF/HTML-фигуре.

---

## 1. Инициализация с vehicle-re-id весов vs ImageNet при дообучении на малом датасете

**Прямого замера «дообучение с re-id весов vs с ImageNet при N изображений цели» для vehicle re-id я не нашёл.**
Есть три близких замера, которые вместе закрывают вопрос.

### 1.1 VehicleNet (TMM 2021) — двухстадийное обучение, цель = CityFlow (26 803 train-изображения, ~333 ID)

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| Baseline: ImageNet-init, обучение только на CityFlow (26 803 изобр.) | **37,65 mAP / 73,65 R@1** | https://arxiv.org/pdf/2004.06305 (Table II) | да (pdftotext) |
| + со-обучение с VeRi-776 (+49 357 изобр.) | **43,47 mAP / 79,48 R@1** (+5,82 mAP) | там же | да |
| + со-обучение с CompCar (+136 713) | **48,71 mAP / 83,37 R@1** (+11,06) | там же | да |
| + со-обучение с VehicleID (+221 567) | **47,56 mAP / 83,37 R@1** (+9,91) | там же | да |
| VehicleNet целиком (434 440 изобр. / 31 805 ID) | **57,35 mAP / 88,77 R@1** (+19,70 mAP к ImageNet-baseline) | там же | да |
| Stage-I (обучен на VehicleNet) → Stage-II (finetune на CityFlow), private test | **68,21 → 75,60 mAP** (+7,39 mAP, +4,75 R@1) | там же (Table V) | да |
| Тот же приём на VeRi-776 (576 ID / 37 778 train) | Stage-I **80,91 mAP** → Stage-II **83,41 mAP** (+2,50) | там же (Table III / Table IX) | да |
| Backbone / вход / оптимизатор | ResNet-50, 256×256, SGD, bs=36, 60 эпох Stage-I + 12 эпох Stage-II | там же (Sec. V-A) | да |
| Размер VehicleNet | 434 440 изобр. / 31 805 ID / 62 камеры | там же (Table I) | да |

**Ключевой вывод замера:** выигрыш от домен-подходящих данных **растёт, когда цель мала**: +19,7 mAP на CityFlow (26,8k изобр.) против +2,5 mAP на VeRi-776 (37,8k изобр., уже насыщенный).

### 1.2 AI City Challenge 2021 empirical study — цель = CityFlow-V2 (52 717 изобр. / 440 ID)

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| Размер CityFlow-V2 | 85 058 изобр. / 880 ID; train = **52 717 изобр. / 440 ID** | https://arxiv.org/pdf/2105.09701 (Sec. 4.1) | да (pdftotext) |
| Type1 = только CityFlow-V2, ImageNet-init, ResNet50-IBN-a, 256×256 | **36,0 mAP / 51,7 R-1** | там же (Table 1, 2) | да |
| Type1, ResNet101-IBN-a | **38,8 / 54,8** | там же (Table 2) | да |
| Type1, TransReID (ViT-B/16) | **42,1 / 59,8** | там же (Table 2) | да |
| Type2 = + синтетика VehicleX, ResNet50-IBN-a | **46,2 / 64,8** (+10,2 mAP) | там же | да |
| Type2, TransReID | **45,5 / 61,0** — **ViT проиграл CNN**, авторы: «TransReID is easier to overfit to the training data than CNN-based backbones» | там же (Sec. 4.4) | да |
| Type3 = + crop | **49,1 / 65,9** | там же (Table 1) | да |

### 1.3 Cross-dataset transfer без дообучения (нижняя граница «готовых весов»)

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| VERI-Wild → VeRi-776 zero-shot, TransReID | in-domain 79,5 (Wild-3k) / 81,3 (VeRi) → **63,5 mAP** cross | https://arxiv.org/html/2606.01981v1 (Table II) | косвенно (WebFetch по HTML, таблица распознана) |
| VERI-Wild → VeRi-776 zero-shot, CLIP-ReID | 80,3 / 84,1 → **63,1 mAP** | там же | косвенно |
| VERI-Wild → VeRi-776 zero-shot, RotTrans | 78,5 / 80,4 → **63,9 mAP** | там же | косвенно |
| Авторская формулировка провала | «**~15 % mAP performance drop**» при переносе Wild→VeRi | там же | косвенно |
| VeRi-Wild → VeRi-776 zero-shot, DINOv3-ConvNeXt (2026) | **66,02 mAP / 87,07 R1**; с re-ranking 69,27 | https://arxiv.org/html/2607.22068v1 | да (текст извлечён локально) |
| Провал на **невиданных типах машин** (новый протокол) | seen 90–99 mAP → unseen **57–70 mAP** (mixed view, −~30); different-view: seen 88–97 → unseen **44–56** (−~40) | https://arxiv.org/html/2606.01981v1 (Tables III–V) | косвенно |

**Вывод для проекта:** ваши **0,657 mAP** от готового `vehicle-reid-0001` — это ровно уровень опубликованного
cross-dataset переноса vehicle re-id (63–66 mAP). То есть ваш zero-shot результат «нормальный», а 0,433 после
дообучения с ImageNet — аномально плохой, не «нормальный потолок малого датасета».

### 1.4 SnP (CVPR 2023): сколько исходных ID нужно, цель = VeRi и приватный AliceVehicle

Source pool = 15 060 ID / 399 715 изображений (10 датасетов), task model IDE, **direct transfer** (без меток цели).

| Цель / бюджет | Число | URL | Проверено |
|---|---|---|---|
| VeRi: весь source pool | R1 55,90 / **mAP 25,03** | https://arxiv.org/pdf/2303.16186 (Table 1) | да (pdftotext) |
| VeRi: SnP-searched, 2 % ID | R1 69,96 / **mAP 31,10** | там же | да |
| VeRi: SnP, 5 % ID | R1 72,05 / **mAP 36,01** | там же | да |
| VeRi: SnP, 20 % ID | R1 73,48 / **mAP 40,75** | там же | да |
| VeRi: SnP без бюджета | R1 74,13 / **mAP 41,71** | там же | да |
| AliceVehicle (реальная приватная цель): обучение только на VeRi | R1 30,69 / R5 43,66 / **mAP 11,05** | там же (Table 2) | да |
| AliceVehicle: только CityFlow | 23,95 / 36,00 / **7,45** | там же | да |
| AliceVehicle: только VehicleID | 16,3 / 29,13 / **4,73** | там же | да |
| AliceVehicle: только VehicleX (синт.) | 18,85 / 32,18 / **8,89** | там же | да |
| AliceVehicle: весь source pool | 30,47 / 48,33 / **14,64** | там же | да |
| AliceVehicle: SnP-searched подмножество | 46,78 / 64,41 / **25,46** | там же | да |

---

## 2. Person-reid веса как старт для vehicle-reid (и наоборот)

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| Прямой замер «person-reid веса → инициализация vehicle-reid, сравнение с ImageNet» | **замера не нашёл** — прогнал 5 поисковых формулировок, ни одна не дала публикации с таким ablation | — | — |
| Ближайший суррогат: OSNet-AIN `vehicle-reid-0001` = person-reid кодовая база torchreid, переобученная на vehicle; исходная архитектура person-reid | архитектурная совместимость 1:1 (см. п.4) | https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/README.md | да (raw) |
| Косвенный аргумент «против»: перенос person-методов в другой домен (животные) работает, но веса переучивают заново; ImageNet-init обязателен, «prior experiments without transfer learning reported consistently worse results» | качественно, без числа | https://arxiv.org/html/2410.00204v1 | косвенно (через поиск) |
| Косвенный аргумент «за»: домен-специфичный re-id-претрейн бьёт generic-претрейн на **всех 29** датасетах (аналог: person→animal) | см. п.6.3 (MegaDescriptor) | https://openaccess.thecvf.com/content/WACV2024/papers/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.pdf | да (pdftotext) |
| Обратное направление (vehicle→person) | **замера не нашёл** | — | — |

**Практическая интерпретация:** vehicle-re-id веса доступны и бесплатны (п.3, п.4), поэтому вопрос «person вместо
ImageNet» академический. Единственный случай, когда он важен — если вы берёте `osnet_ain_x1_0_msmt17` (п.5) как
старт для OSNet-AIN: числа переноса person→vehicle для этого не опубликованы.

---

## 3. fast-reid `veri_sbs_R50-ibn.pth` и `veriwild_bot_R50-ibn.pth`

### 3.1 Ссылки, размеры, живость (проверено `curl -I`, 2026-09-16)

| Файл | mAP / R@1 (заявлено) | Размер | Ссылка | Живость |
|---|---|---|---|---|
| `veri_sbs_R50-ibn.pth` | VeRi-776: **81,9 mAP / 97,0 R@1 / 46,3 mINP** | **198 261 759 B (189,1 MiB)** | https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth | HTTP 200, скачан локально |
| `veriwild_bot_R50-ibn.pth` | VERI-Wild S/M/L: **87,7 / 83,5 / 77,3 mAP**, R@1 96,4 / 95,1 / 92,5 | **1 036 401 141 B (988,4 MiB)** | https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth | HTTP 200 |
| `vehicleid_bot_R50-ibn.pth` | VehicleID S/M/L R@1: **86,6 / 82,9 / 80,6** | **606 149 904 B (578,1 MiB)** | https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth | HTTP 200 |

Источник чисел: https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md — проверено.
Лицензия репозитория: **Apache-2.0** (GitHub API `license.spdx_id`) — проверено.
Репозиторий не архивирован, последний push **2024-07-30**, 3 990 звёзд — проверено (GitHub API).

### 3.2 Конфиги: вход и эмбеддинг (проверено по raw YAML/Python)

| Параметр | `configs/VeRi/sbs_R50-ibn.yml` | `configs/VERIWild/bagtricks_R50-ibn.yml` | URL |
|---|---|---|---|
| `INPUT.SIZE_TRAIN` / `SIZE_TEST` | **[256, 256]** | **[256, 256]** | https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/configs/VeRi/sbs_R50-ibn.yml , .../configs/VERIWild/bagtricks_R50-ibn.yml |
| Backbone | R50, `WITH_IBN: True`, `WITH_NL: True`, `LAST_STRIDE: 1`, `FEAT_DIM: 2048` | R50, `WITH_IBN: True`, без NL | .../configs/Base-SBS.yml , .../configs/Base-bagtricks.yml |
| `MODEL.HEADS.EMBEDDING_DIM` | **не задан → default 0** | **не задан → default 0** | https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/fastreid/config/defaults.py (стр. ~72) |
| Что значит 0 | в `EmbeddingHead`: `if embedding_dim > 0: neck.append(Conv2d(feat_dim, embedding_dim,...))` — при 0 проекции нет ⇒ **эмбеддинг = FEAT_DIM = 2048** | то же | https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/fastreid/modeling/heads/embedding_head.py (стр. 70–74) |
| Потери | CE(ε=0,1) + Triplet(margin 0, hard mining), `CircleSoftmax` s=64 m=0,35, GeM-P pooling, `NECK_FEAT: after` | CE + Triplet(margin 0, **без** hard mining), GeM, `NECK_FEAT: before` | там же |
| Оптимизатор / эпохи / батч | SGD, lr 0,01, 60 эпох, **IMS_PER_BATCH 64**, WARMUP 3000, **FREEZE_ITERS 3000** (+ `FREEZE_LAYERS: [backbone]`) | Adam, lr 3,5e-4, 120 эпох, **IMS_PER_BATCH 512 (на 4 GPU)** | там же |
| Sampler | `BalancedIdentitySampler`, `NUM_INSTANCE: 16` | `NaiveIdentitySampler`, `NUM_INSTANCE: 4` | там же |
| AMP | `SOLVER.AMP.ENABLED: True` | то же | .../configs/Base-bagtricks.yml |

### 3.3 Что реально лежит внутри `veri_sbs_R50-ibn.pth` (проверено: скачал 189 MiB и распаковал zip)

| Утверждение | Число | Как проверено |
|---|---|---|
| Формат | torch-zip, **657 тензоров**, 198,1 MB несжатого | `zipfile` над файлом |
| Наличие состояния оптимизатора | строк `optimizer` / `exp_avg` / `param_groups` в `data.pkl` **не найдено** ⇒ плоский state_dict (большой размер — это NL-блоки + IBN) | pickle-строки |
| Классификатор | тензор ровно **575 × 2048 fp32** (1 177 600 элементов) ⇒ num_classes=**575**, эмбеддинг **2048** | сопоставление размеров zip-записей |
| Имена ключей головы в релизе | `heads.classifier.weight`, `heads.bnneck.*`, `heads.bottleneck.0.*`, `heads.pool_layer.p` | строки в `data.pkl` |
| Имена в **текущем master** | `heads.weight` (а не `heads.classifier.weight`), `heads.bottleneck` = `nn.Sequential` | https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/fastreid/modeling/heads/embedding_head.py (стр. 78, 83) |

⚠ **Замеренная проблема:** ключи головы в релизных весах v0.1.1 **не совпадают** с master. Подтверждено пользовательским
багрепортом: «`'heads.classifier.weight'` has shape (575, 2048) in the checkpoint but (0, 2048) in the model» —
https://github.com/JDAI-CV/fast-reid/issues/617 (открыт, закрыт как stale, **ответа мейнтейнера нет**). Проверено.

### 3.4 Есть ли готовый путь «загрузить и дообучить на своём датасете»

| Утверждение | Факт | URL | Проверено |
|---|---|---|---|
| Флаг `--finetune` | **отсутствует**. `default_argument_parser` содержит только `--config-file`, `--resume`, `--eval-only`, `--num-gpus`, `--num-machines`, `--machine-rank`, `--dist-url`, `opts` | https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/fastreid/engine/defaults.py (стр. 39–69) | да |
| `MODEL.WEIGHTS` | есть: `_C.MODEL.WEIGHTS = ""` (defaults.py стр. 125). Механизм: `trainer.resume_or_load(resume=False)` → «load weights from `cfg.MODEL.WEIGHTS` (but will not load other states) and start from iteration 0» | defaults.py; fastreid/engine/defaults.py стр. 241–255 | да |
| Обработка несовпадения `num_classes` | **работает автоматически**: `Checkpointer._load_model` сравнивает shape, кладёт несовпавшие в `incorrect_shapes`, `pop`-ает их из state_dict и делает `load_state_dict(..., strict=False)`; логирует «Skip loading parameter ... due to incompatible shapes» | https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/fastreid/utils/checkpoint.py стр. 236–262 | да |
| Документация finetune | **в `GETTING_STARTED.md` описан только `--eval-only MODEL.WEIGHTS ...`**; про дообучение с чужих весов ни слова | https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/GETTING_STARTED.md | да |

**Итог п.3:** путь существует и работает (`MODEL.WEIGHTS=/path/veri_sbs_R50-ibn.pth` + свой датасет + `MODEL.HEADS.NUM_CLASSES` ставится автоматически по датасету), но он **не задокументирован** и **упрётся в несовпадение имён ключей головы** между v0.1.1 и master — нужен либо тег v0.1.1/v1.0, либо ручной ремап 4 ключей.

---

## 4. Intel OMZ `vehicle-reid-0001` → обучаемая torch-модель

Модель лежит в `models/**public**/`, не в `models/intel/` (в `models/intel/` её нет — проверил все теги 2020.4…2022.3.0 и master, везде 404).

### 4.1 Паспорт модели (raw README + model.yml — проверено)

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| VeRi-776 rank-1 | **96,31 %** | https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/README.md | да (raw) |
| VeRi-776 mAP | **85,15 %** | там же | да |
| GFlops / MParams | **2,643 / 2,183** | там же | да |
| Вход / выход | `1,3,208,208` NCHW RGB → `1,512` | там же | да |
| Лицензия исходной модели | **MIT** (Copyright 2018 Kaiyang Zhou) | там же | да |
| ONNX-файл | `osnet_ain_x1_0_vehicle_reid.onnx`, **8 836 743 B** | https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx (HTTP 200) | да (скачал) |
| Оригинальный источник в model.yml | google_drive id `1MEtaIr_9mWuntGD9edydFl_T5waloNRm`; `license: https://raw.githubusercontent.com/sovrasov/deep-person-reid/vehicle_reid/LICENSE` | https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/public/vehicle-reid-0001/model.yml | да |

**Замечание, важное для проекта:** `vehicle-reid-0001` даёт **85,15 mAP на VeRi-776** — это выше, чем fast-reid `veri_sbs_R50-ibn` (81,9) и сильно выше VehicleDINO (61,1). При 2,18 M параметров.

### 4.2 (а) Есть ли исходные torch-веса

| Утверждение | Факт | URL | Проверено |
|---|---|---|---|
| Репозиторий с **определением архитектуры** | `sovrasov/deep-person-reid`, ветка **`vehicle_reid`** (head `ea27fd23c962addbd24d8586c5aaf8afe60db0db`), 204 файла. Содержит `torchreid/models/osnet_ain.py` c параметрами `input_IN`, `pool2`, `pool3`, `feature_dim=256`, `self.fc = nn.ModuleList()`, и датасеты `veri.py`, `veriwild.py`, `vehicle1m.py` | https://github.com/sovrasov/deep-person-reid/tree/vehicle_reid (HTTP 200) | да (GitHub trees API + raw) |
| Опубликованы ли там `.pth` | **Нет.** В дереве ветки нет ни одного `.pth`/`.onnx`, нет vehicle-конфига в `configs/` (только `configs/person/*`), в `docs/MODEL_ZOO.md` нет строк vehicle/veri | тот же trees API | да |
| `openvinotoolkit/deep-object-reid` | **archived: true**, default branch `ote`, последний push **2023-02-02**, лицензия NOASSERTION, 57 звёзд. Torch-весов `vehicle-reid-0001` там не опубликовано | https://github.com/openvinotoolkit/deep-object-reid (GitHub API) | да |
| Вывод | **Готовых torch-весов не существует публично; существует только ONNX + исходный код архитектуры (MIT)** | — | — |

### 4.3 (б) Реально ли получить обучаемый граф — **замерено локально**

Скачал ONNX (8,84 MB), распарсил через `onnx==1.20.x`:

| Утверждение | Число | Как проверено |
|---|---|---|
| IR version / opset / producer | **ir_version 4, opset 9 (domain ''), producer `pytorch 1.3`** | `onnx.load` |
| Узлов в графе | **482** | `len(graph.node)` |
| Уникальных типов операций | **18**: Conv(187), Relu(100), BatchNormalization(76), GlobalAveragePool(25), Sigmoid(24), Mul(24), Add(24), InstanceNormalization(6), Pad(2), AveragePool(2), Constant(2), Unsqueeze(2), Concat(2), Gemm(2), MaxPool(1), Shape(1), Gather(1), Reshape(1) | `collections.Counter` |
| Параметров в инициализаторах | **2 189 686 (2,190 M)** — совпадает с заявленным MParams 2,183 | суммирование `dims` |
| Форма входа | `('batch_size','channels','height','width')` — **полностью динамическая**, 208×208 не зашито | `graph.input` |
| **Покрытие onnx2torch 1.5.15** | **18 из 18 операций поддержаны** (реестр конвертеров содержит 106 типов; все наши — `OK`) | распаковал wheel, `grep add_converter(operation_type=...)` |
| **Имена инициализаторов = имена torch state_dict** | **559 инициализаторов** с именами вида `conv1.bn.weight`, `conv2.0.IN.bias`, `conv2.0.conv2.0.layers.0.conv1.weight`, `fc.0.0.weight`, `input_IN.weight`, `pool3.0.conv.weight` | перечисление `graph.initializer` |
| Прямое доказательство, что это torch-экспорт | **76 инициализаторов `*.num_batches_tracked`, которые НЕ используются ни одним узлом графа** (`unused initializers: 76`) — чистая BN-бухгалтерия PyTorch, сохранённая при экспорте | сверка с `node.input` |
| Структура головы | GAP → 512-d → `Gemm(fc.0.0: 256×512)`+`BN1d(fc.0.1)` **и** `Gemm(fc.1.0: 256×512)`+`BN1d(fc.1.1)` → `Concat` → **512-d output** | последние 12 узлов графа |
| Дополнительно | `input_IN` = `InstanceNorm2d(3, affine=True)` на входе (есть в `sovrasov/.../osnet_ain.py` стр. 466); `pool2`/`pool3` = `Conv1x1 + AvgPool2d(2)` (стр. 474, 477) | raw-файл ветки | 

**Замеренный вывод по п.4:** onnx2torch формально сработает (100 % покрытие операций), **но он не нужен**.
Правильный путь дешевле и надёжнее: взять определение модели из `sovrasov/deep-person-reid@vehicle_reid`
(`torchreid/models/osnet_ain.py`, MIT) с `input_IN=True, feature_dim=256`, и **загрузить веса напрямую из ONNX-инициализаторов
по их именам** — они уже являются валидными ключами torch `state_dict`. Это даёт нативный `nn.Module`
(с работающими BN/IN в train-режиме), тогда как onnx2torch даёт функционально-эквивалентный, но «плоский»
граф-обёртку, где BN/IN восстановлены как модули не всегда с корректной train-семантикой.

Статус пакетов (PyPI JSON API, проверено):

| Пакет | Версия | Последняя загрузка | Лицензия | Репозиторий |
|---|---|---|---|---|
| `onnx2torch` | **1.5.15** | **2024-08-07** | Apache-2.0 (ENOT LLC) | https://github.com/ENOT-AutoDL/onnx2torch |
| `onnx2pytorch` | **0.6.0** | **2026-08-12** (активен) | Apache-2.0 | https://github.com/ToriML/onnx2pytorch |
| `onnx-pytorch` | 0.1.5 | 2022-08-03 (мёртв) | Apache-2.0 | https://github.com/fumihwh/onnx-pytorch |

---

## 5. torchreid OSNet-AIN веса

| Утверждение | Число / факт | URL | Проверено |
|---|---|---|---|
| Лицензия torchreid | **MIT**, Copyright (c) 2018 Kaiyang Zhou | https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/LICENSE | да |
| `osnet_ain_x1_0` (ImageNet) — ссылка в коде | `https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo` | https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/torchreid/models/osnet_ain.py (стр. 11–20) | да |
| Размер файла ImageNet-весов | **10 929 757 B (10,4 MiB)**, HTTP 200 | тот же gdrive id | да (curl -I) |
| Другие ImageNet-варианты в коде | `osnet_ain_x0_75` (`1apy0hpsMypqstfencdH-jKIUEFOW4xoM`), `x0_5` (`1KusKvEYyKGDTUBVRxRiz55G31wkihB6l`), `x0_25` (`1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt`) | osnet_ain.py стр. 11–20 | да |
| **`osnet_ain_x1_0_msmt17` в official MODEL_ZOO** | **В таблице «Same-domain ReID» строки `osnet_ain_x1_0` НЕТ вообще** — там только `osnet_x1_0`, `osnet_x0_75`, `osnet_x0_5`, `osnet_x0_25`, `resnet50`, `mlfn`, `hacnn`, `mobilenetv2` | https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/docs/MODEL_ZOO.md (стр. 33–45) | да (raw markdown) |
| ⚠ Ловушка | Цифры «94.2 (82.6) / 87.0 (70.2) / 74.9 (43.8)», которые часто приписывают osnet_ain — это строка **`osnet_x1_0`**, а не AIN. Проверил по raw-разметке | там же | да |
| Что такое `osnet_ain_x1_0_msmt17.pt` на практике | BoxMOT резолвит это имя в gdrive id `1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal` | https://github.com/mikel-brostrom/boxmot/blob/master/boxmot/reid/core/catalog.py (стр. 37) | да (клонировал репо) |
| Этот же id в torchreid MODEL_ZOO | раздел **«MSMT17 (`combineall=True`) → Market1501 & DukeMTMC-reID»**, строка `osnet_ain_x1_0`: **msmt17→market1501 = 70,1 R-1 (43,3 mAP)**, **msmt17→dukemtmcreid = 71,1 R-1 (52,7 mAP)** | MODEL_ZOO.md стр. 70–78 | да |
| Размер файла | **17 293 009 B (16,5 MiB)**, HTTP 200 | тот же gdrive id | да (curl -I) |
| Прочие AIN-веса (cross-domain) | Market→Duke: **52,4 R-1 / 30,5 mAP** (`14bNFGm0FhwHEkEpYKqKiDWjLNhXywFAd`); Duke→Market: **61,0 / 30,6** (`1hypJvq8G04SOby6jvF337GEkg5K_bmCw`) | MODEL_ZOO.md стр. 52–64 | да |
| Multi-source DG (`osnet_ain_x1_0`) | MS+D+C→M **73,3 (45,8)**; MS+M+C→D **65,6 (47,2)**; MS+D+M→C **27,4 (27,1)**; D+M+C→MS **40,2 (16,2)** | MODEL_ZOO.md стр. 93 | да |
| Конфиг обучения этих весов | `configs/im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml`, `max_epoch=50`, вход **(256,128)**, softmax, cosine distance | MODEL_ZOO.md стр. 87 | да |
| Параметры / GFLOPs | **2,2 M / 0,98 GFLOPs** | MODEL_ZOO.md | да |

**Важно:** `osnet_ain_x1_0_msmt17` — это **person** re-id веса, обучены на MSMT17 с `combineall=True`, вход 256×128 (не квадрат). Числа переноса на vehicle не опубликованы (см. п.2).

---

## 6. Foundation-модели + LoRA / лёгкая адаптация

### 6.1 wms2537/VehicleDINO — **перепроверено, цифры подтверждены частично**

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| Re-ID на VeRi-776 | **61,1 % mAP / 86,1 % Rank-1** | https://huggingface.co/wms2537/VehicleDINO | да |
| Backbone | «DINOv2 ViT-B/14 (frozen, with LoRA adapters)» + SimpleFPN + RT-DETR-декодер (300 queries), 6 голов (detect/type/make/model/ReID/OCR) | там же | да |
| Вход / эмбеддинг | **560×560**, Re-ID эмбеддинг **256-d, L2-normalized** | там же | да |
| Обучающие данные | VeRi-776 (776 ID, 49 360 изобр.) + CompCars + CCPD-Green | там же | да |
| Лицензия | **Apache-2.0** | HF API `cardData.license` | да |
| LoRA rank | **не указан на карточке** | там же | да |
| Число обучаемых параметров | **не указано** | там же | да |
| Эпохи / железо / batch / lr | **не указаны** | там же | да |
| Файлы | `vehicledino_dinov2.onnx` (FP32, ~450 MB), `_int8.onnx` (~139 MB), `vehicledino_dinov2_best.pt`, `_coco.onnx`, `_coco_checkpoint.pt` | HF API `siblings` | да |
| Обновлён | 2026-03-09, downloads 0, likes 0 | HF API | да |
| **«DINOv3-ViT-L + LoRA = 75,2 mAP на VERI-Wild» у wms2537** | **ОПРОВЕРГНУТО.** На профиле wms2537 всего 2 модели: `VehicleDINO` и `qwen3-0.6b-malaysia-moderation-cot`. DINOv3-модели нет | https://huggingface.co/wms2537 | да |

### 6.2 Откуда на самом деле «75,2 mAP» — arXiv 2607.22068 (Wang & Yang, 24 июл 2026)

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| **LoRA-ViT ветка: rank 8, α=16, блоки 20–23** (DINOv3 ViT-L) | **75,19 mAP** | https://arxiv.org/html/2607.22068v1 (Table 6) | да (текст извлечён локально) |
| Full fine-tuning того же ViT-L, **3 независимых рецепта** | потолок **~73 mAP** | там же | да |
| Standalone ViT-L (полный FT вне fusion) | плато **67,0 mAP** (72,1 без cam) | там же | да |
| **Frozen / zero-shot «голые» backbone'ы** | **ViT-L 18,56**, ConvNeXt **14,63**, ResNet50 **3,77** mAP | там же (Sec. 5) | да |
| Одиночный DINOv3-ConvNeXt, полный рецепт | **88,19 mAP VeRi-Wild Small**, **77,47 Large**; с re-ranking **92,38 / 83,68** | там же (abstract, Table 2) | да |
| Разрыв ViT vs ConvNeXt | **13–15 mAP** (~15 при full FT, ~13 при лучшей LoRA) — устойчив ко всем режимам адаптации | там же | да |
| **Кривая «заморозка → разморозка»** | frozen Phase 1 = **39,46 mAP** → разморозка даёт **56,25 за одну эпоху (+16,8)** → 85,98 к эпохе 31 → LR-decay на эпохе 40 даёт **+2,21** → **88,19 на эпохе 42** | там же (Sec. 4.1) | да |
| Железо / батч / вход / эпохи | **8× NVIDIA RTX PRO 6000**, P16K4 (16 ID/GPU × 4 изобр. = **effective batch 512**), **256×256**, **42 эпохи** | там же | да |
| Размер данных | VeRi-Wild: official `test_3000` (3 000 query ID, 38 861 gallery) и `test_10000` (128 517 gallery); train ~**277 597 изобр. / 30 671 ID** | там же (Sec. 3) | да |
| Bootstrap CI | ±0,66 mAP на Small (88,19, CI [87,51; 88,83]); ±0,51 на Large | там же | да |
| Дообучаемые параметры LoRA | **не указано в процентах/штуках** — только rank 8, α=16, 4 блока | там же | да |
| Fusion LoRA-ViT с ConvNeXt | **oracle score-fusion выбирает вес 0** для LoRA-ViT; любой ненулевой вес хуже (−0,06 при α=0,3; −0,33 при 0,5; −2,34 при равных весах) | там же (Prop. 1) | да |

**Критично:** LoRA здесь применена на датасете в **277 тыс. изображений / 30 тыс. ID** — это в 38 раз больше вашего.
Замера LoRA-адаптации на re-id-датасете вашего размера (<10k изобр., ~1k ID) я **не нашёл** (проверил 4 формулировки поиска).

### 6.3 Frozen foundation features для re-id — сильный отрицательный замер

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| **DINOv2 zero-shot на person re-id** | **0,3–4,7 % mAP** по 9 датасетам (B14: 0,37 MSMT / 1,71 Market / 4,12 LasT; L14: 0,39 / 1,40 / 4,70) | https://arxiv.org/html/2601.20598 (Balasubramanian, 28 янв 2026) | косвенно (WebFetch по HTML, таблица распознана) |
| CLIP zero-shot (B32/B16/L14) | **0,10–2,70 % mAP** | там же | косвенно |
| SigLIP2 zero-shot | 1,3–15,3 % mAP (лучше всех generic на CelebReID: **14,23** против CLIP-ReID 7,93) | там же | косвенно |
| Обученный CLIP-ReID | **66,22 MSMT / 50,59 Market / 58,28 Duke** mAP | там же | косвенно |
| Supervised OSNet-x1.0 in-domain vs cross | Market **83,57** → MSMT **3,37**, PKU **1,90**, IUS **0,86** | там же | косвенно |
| Покрывает ли vehicle re-id | **нет, только person** | там же | косвенно |

### 6.4 Домен-специфичный re-id-претрейн vs generic foundation (аналог из animal re-id, 29 датасетов, все frozen)

MegaDescriptor-L (Swin-L/p4-w12-384, обучен на всех публичных animal re-id) vs ImageNet-1k Swin-B / CLIP ViT-L/14-336 / DINOv2 ViT-L/14-518:

| Датасет | ImageNet | CLIP | DINOv2 | MegaDesc. |
|---|---|---|---|---|
| HyenaID2022 | 46,83 | 45,71 | 49,52 | **78,41** |
| LeopardID2022 | 61,13 | 59,94 | 57,50 | **75,58** |
| SeaTurtleIDHeads | 43,84 | 33,57 | 46,08 | **91,18** |
| SealID | 41,73 | 34,05 | 29,26 | **78,66** |
| WhaleSharkID | 28,26 | 26,37 | 22,02 | **62,02** |
| NyalaData | 10,28 | 10,51 | 14,72 | **36,45** |
| BelugaID | 19,58 | 11,20 | 14,64 | **66,48** |

Источник: https://openaccess.thecvf.com/content/WACV2024/papers/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.pdf (Table 4) — **проверено (pdftotext)**.
Авторская формулировка: «DINOv2 — with a much higher input size (518×518) and larger backbone — **performs poorly in animal re-identification**»; MegaDescriptor-L «outperforms all methods on **all 29 datasets**».

---

## 7. Отрицательные результаты: full fine-tuning проигрывает на малых целях

| Утверждение | Число | URL | Проверено |
|---|---|---|---|
| **LP-FT (Kumar et al., ICLR 2022):** full FT даёт **+2 %** in-distribution, но **−7 %** OOD относительно linear probing (10 OOD-датасетов) | +2 / −7 п.п. | https://arxiv.org/abs/2202.10054 | да |
| LP-FT против full FT | **+10 % OOD, +1 % ID** | там же | да |
| Условие, когда LP выигрывает | «fine-tuning can achieve worse accuracy than linear probing out-of-distribution (OOD) **when the pretrained features are good and the distribution shift is large**» | там же (abstract) | да |
| Средние показатели LP-FT | ID **85,7 %**, OOD **68,9 %**; лучший на 5/6 ID и **10/10 OOD** | там же | да |
| CIFAR-10 → CIFAR-10.1 | FT 92,3 / LP 82,7 / **LP-FT 93,5** | там же | да |
| **ViT переобучается на малом vehicle re-id:** TransReID на CityFlow-V2 (52,7k изобр., 440 ID) | Type1 42,1 mAP (лучше CNN), но Type2 **45,5 — худший из трёх** backbone'ов; «TransReID is easier to overfit» | https://arxiv.org/pdf/2105.09701 (Table 2, Sec. 4.4) | да |
| **Full FT self-supervised ViT нестабилен:** 3 рецепта дают потолок ~73 mAP, LoRA — 75,19 | +2,2 mAP в пользу PEFT | https://arxiv.org/html/2607.22068v1 (Table 6) | да |
| Тот же источник: деградация full FT | layer-wise LR decay: 67,4 (E2) → 68,9 (E3 peak) → **59,7 (E8)** — обвал на 9,2 mAP при продолжении обучения | там же | да |
| Симметричное разморожение двух foundation-backbone'ов | пик 85,30 (E7) → **падение 10 эпох подряд до 83,36** | там же | да |
| Полная заморозка (frozen Phase 1) **на большом** датасете | **39,46 mAP** vs 88,19 при разморозке ⇒ на 277k изобр. заморозка — плохая идея | там же | да |
| Публикация «на малой vehicle-цели заморозка/linear probe бьёт full FT» с числами | **замера не нашёл** | — | — |
| «On Combining Animal Re-ID Models to Address Small Datasets», IJCV 2025 | источник **gated**: `link.springer.com` отдаёт HTTP 303 → `idp.springer.com/authorize`, полный текст недоступен | https://link.springer.com/article/10.1007/s11263-025-02708-9 | нет (403/gated) |

---

## 8. Зависимость mAP от числа обучающих идентичностей

**Для vehicle re-id опубликованной кривой «#ID → mAP» с фиксированным методом я не нашёл.**
Есть две кривые для person re-id и одна бюджетная кривая по исходным ID для vehicle-цели.

### 8.1 LUPerson (Fu et al., CVPR 2021) — **прямая кривая «% ID → mAP»**, метод MGN

Протокол: «Small-scale. We randomly select a certain percentage of IDs and all the images belonging to the sampled IDs».
10 % на Market1501 = **75 персон / 1 170 изображений** (авторская формулировка) — это практически ваш масштаб по изображениям.

**Market1501, small-scale (доля ID), mAP/cmc1:**

| pre-train | 10 % (75 ID / 1 170 изобр.) | 30 % | 50 % | 70 % | 90 % |
|---|---|---|---|---|---|
| IN sup. (ImageNet) | **53,1 / 76,9** | 75,2 / 90,8 | 81,5 / 93,5 | 84,8 / 94,5 | 86,9 / 95,2 |
| IN unsup. | 58,4 / 81,7 | 76,6 / 91,9 | 82,0 / 94,1 | 85,4 / 94,5 | 87,4 / 95,5 |
| **LUP unsup. (домен-претрейн)** | **64,6 / 85,5** | 81,9 / 93,7 | 85,8 / 94,9 | 88,8 / 95,9 | 90,5 / 96,4 |
| **Δ (LUP − ImageNet)** | **+11,5** | +6,7 | +4,3 | +4,0 | **+3,6** |

**DukeMTMC, small-scale:**

| pre-train | 10 % | 30 % | 50 % | 70 % | 90 % |
|---|---|---|---|---|---|
| IN sup. | 45,1 / 65,3 | 64,7 / 80,2 | 71,8 / 84,6 | 75,5 / 86,8 | 78,0 / 88,3 |
| LUP unsup. | **53,5 / 72,0** | 69,4 / 81,9 | 75,6 / 86,7 | 78,9 / 88,2 | 81,1 / 90,0 |
| Δ | **+8,4** | +4,7 | +3,8 | +3,4 | +3,1 |

**MSMT17, small-scale:**

| pre-train | 10 % | 30 % | 50 % | 70 % | 90 % |
|---|---|---|---|---|---|
| IN sup. | 23,2 / 50,2 | 41,9 / 70,8 | 50,3 / 76,9 | 56,9 / 81,2 | 61,9 / 84,2 |
| LUP unsup. | **25,5 / 51,1** | 44,6 / 71,4 | 53,0 / 77,7 | 59,5 / 81,8 | 63,7 / 85,0 |
| Δ | +2,3 | +2,7 | +2,7 | +2,6 | +1,8 |

**Few-shot (все ID, доля изображений на ID), Market1501:** IN sup. 10 % = **21,1 / 41,8**; LUP = **26,4 / 47,5** (Δ +5,3).

Полнотекстовая цитата авторов: «for the “small-scale” setting on Market1501, which contains only **1,170 images for 75 persons**
(percentage=10 %), MGN with our pre-training model achieves **64.6 mAP** on the testing set, which is **11.5 mAP higher** than
the ImageNet supervised counterpart».

Также Table 5 (полные датасеты, MGN): CUHK03 IN sup **70,5** → LUP **74,7**; Market IN sup **87,5** → LUP **91,0**;
Duke **79,4** → **82,1**; MSMT17 **63,7** → **65,7**.

Источник: https://arxiv.org/pdf/2012.03753 (Tables 5, 6) — **проверено (pdftotext)**.

### 8.2 Форма кривой (важнее чем абсолютные значения)

По Market1501/ImageNet-init: 10 %→53,1; 30 %→75,2; 50 %→81,5; 70 %→84,8; 90 %→86,9.
Прирост от 10→30 % = **+22,1 mAP**, от 70→90 % = **+2,1 mAP**. Кривая логарифмическая, **насыщается после ~30–50 % ID**.
Δ(домен-претрейн − ImageNet) монотонно **падает** с ростом данных: 11,5 → 6,7 → 4,3 → 4,0 → 3,6.

### 8.3 Vehicle-цель: кривая по бюджету исходных ID (SnP, direct transfer)

VeRi как цель, SnP-отбор из пула 15 060 ID: 2 % ID → **31,10 mAP**; 5 % → **36,01**; 20 % → **40,75**; без бюджета → **41,71**.
Источник: https://arxiv.org/pdf/2303.16186 (Table 1) — проверено.
SnP также замерили корреляции: FID↔rank-1 **≤ −0,636** (сильная отрицательная), «#ID ↔ rank-1: there exists a positive
correlation but **such a correlation is not stable**», FID↔#ID — слабая.
**Вывод замера: соответствие домена важнее количества ID.**

---

## Чего в литературе нет

1. **Прямого замера «дообучение с vehicle-re-id весов vs с ImageNet» на vehicle-цели заданного размера.**
   Все найденные работы делают либо со-обучение (VehicleNet Table II, AI City Table 1), либо двухстадийку без
   контрольной точки «ImageNet-init на той же цели» в одной таблице.
2. **Никаких чисел person→vehicle и vehicle→person переноса весов.** Проверил 5 формулировок.
3. **Никаких чисел LoRA-адаптации на re-id-датасете <10k изображений / ~1k ID.** Единственные vehicle+LoRA замеры
   (2607.22068) сделаны на VeRi-Wild, 277k изображений.
4. **Кривой «#обучающих ID → mAP» для vehicle re-id нет вообще.** Ближайшее — person (LUPerson) и бюджет исходных ID (SnP).
5. **Нет публикации «на малой vehicle-цели заморозка backbone / linear probe бьёт full fine-tuning»** с числами.
   LP-FT (2202.10054) — общий CV-результат, не re-id; на re-id частичная заморозка есть только как `FREEZE_ITERS`
   в рецепте fast-reid (3 000 итераций для VeRi SBS) без ablation.
6. **Нет torch-весов `vehicle-reid-0001`.** Только ONNX + MIT-код архитектуры.
7. **Нет данных о LoRA rank / обучаемых параметрах / эпохах / железе для VehicleDINO** — карточка HF их не содержит.
8. **«DINOv3-ViT-L + LoRA = 75,2 mAP на VERI-Wild» у wms2537 — такой модели не существует**; число принадлежит
   arXiv 2607.22068 (Wang & Yang), и там это ветка ViT в fusion-модели, а не самостоятельная модель.
9. Gated/недоступно: `link.springer.com/article/10.1007/s11263-025-02708-9` (IJCV, «small datasets» для animal re-id) — HTTP 303 на SSO.
10. `docs.openvino.ai/2024/omz_models_model_vehicle_reid_0001.html` — **HTTP 404**; актуальная страница модели живёт только в GitHub-репозитории OMZ (`models/public/`).

---

## Практический вывод: с каких весов стартовать при 1 171 ID / 7 248 кадров / 4 ГБ VRAM

Опорные замеры для калибровки ожиданий:
- Ваш zero-shot OSNet-AIN = **0,657 mAP** — это уровень опубликованного cross-dataset vehicle-переноса (63–66 mAP; п.1.3).
- Ваши 0,433 после 30 эпох с ImageNet — **ниже** уровня «ничего не делать», что согласуется с LP-FT (full FT под большим
  сдвигом домена хуже, чем не трогать признаки; −7 п.п. OOD, п.7) и с 42,1→45,5 деградацией TransReID (п.7).
- Δ(домен-претрейн − ImageNet) при вашем масштабе по LUPerson-кривой ≈ **+8…+11,5 mAP** (п.8.1, столбец 10 %).
- Прирост от Stage-II дообучения поверх домен-весов: **+2,5 mAP** (VeRi-776) … **+7,4 mAP** (CityFlow, малая цель) — п.1.1.

### Вариант 1 (рекомендую) — OSNet-AIN `vehicle-reid-0001`, восстановленный в torch, + мягкое дообучение

- **Веса:** ONNX-инициализаторы (559 тензоров, имена = torch state_dict) → `sovrasov/deep-person-reid@vehicle_reid`
  `osnet_ain_x1_0(input_IN=True, feature_dim=256)`. Лицензия **MIT**. 2,19 M параметров.
- **Почему:** это единственный старт, у которого уже **измерен** ваш собственный baseline 0,657 на вашем домене;
  вход 208×208 совпадает 1:1 (граф динамический, п.4.3); 2,19 M параметров при 4 ГБ VRAM позволяют P=8/K=4 без AMP-трюков.
- **Ожидаемый mAP:** **0,70–0,76**. Обоснование: 0,657 стартовая точка + Stage-II-подобный прирост +2,5…+7,4 mAP (п.1.1),
  умеренно вниз за меньший объём цели. **Это экстраполяция, не замер.**
- **Риск: низкий.** Худший исход — вернуться к замороженной модели (0,657), потеряв только время.
- **Митигация переобучения (все три пункта имеют опору в замерах):**
  (а) LP-FT-порядок: сначала только голова (2202.10054: +10 % OOD против full FT);
  (б) `FREEZE_ITERS`-аналог — fast-reid для VeRi SBS замораживает backbone на 3 000 итераций (п.3.2);
  (в) ранняя остановка по отложенным ID — в 2607.22068 деградация начинается уже на E3–E8 (п.7).
- **Главный риск реализации:** нужно написать ~30 строк маппинга ONNX→state_dict. Проверено, что имена валидны;
  onnx2torch (18/18 операций) — запасной путь.

### Вариант 2 — fast-reid `veri_sbs_R50-ibn.pth` с `MODEL.WEIGHTS`

- **Веса:** 189 MiB, Apache-2.0, VeRi-776 **81,9 mAP**, эмбеддинг **2048**, вход конфига **256×256**, классификатор 575×2048.
- **Почему:** самый сильный публично измеренный vehicle-re-id чекпойнт с открытым тренировочным пайплайном;
  рецепт SBS (GeM + CircleSoftmax + hard-mining triplet + AutoAug + `FREEZE_ITERS`) уже настроен против переобучения.
- **Ожидаемый mAP:** **0,66–0,74**. Обоснование: домен-претрейн на VeRi даёт +8…+11,5 mAP к ImageNet при вашем масштабе
  (LUPerson-кривая, 10 %), то есть 0,433 + ~0,10 ≈ 0,53 при наивном переносе кривой, но с более сильным исходным
  чекпойнтом (81,9 против 43,3/52,7 у person-весов) верхняя часть диапазона реалистичнее. **Экстраполяция, не замер.**
  Риск не превзойти 0,657 от варианта 1 — существенный.
- **Риск: средний.**
  - **4 ГБ VRAM — узкое место.** Замеренный факт: конфиг предписывает 256×256 и `IMS_PER_BATCH 64`
    (`configs/VeRi/sbs_R50-ibn.yml`), а веса — 189 MiB / ~49 M fp32-параметров. Оценка (не замер): в 4 ГБ
    этот батч не поместится, придётся 208×208 + batch ≤32 + `SOLVER.AMP.ENABLED: True` (в конфиге уже включён) —
    то есть **отход от рецепта**, под который замерены 81,9.
  - **Несовпадение ключей головы v0.1.1 ↔ master** (issue #617, без ответа мейнтейнера) — нужен чекаут тега или ремап.
  - 2048-d эмбеддинг против ваших 512 — в 4 раза дороже по индексу/памяти на inference.

### Вариант 3 — DINOv2/DINOv3 + LoRA

- **Ожидаемый mAP:** **0,45–0,65**, широкий разброс. **Не рекомендую при вашем бюджете.**
- **Риск: высокий**, и он подтверждён замерами, а не соображениями:
  - Frozen DINOv2 на re-id = **0,3–4,7 % mAP** (п.6.3) ⇒ без обучения головы это ноль.
  - Frozen DINOv3 ViT-L на vehicle re-id = **18,56 mAP** (п.6.2) ⇒ 3,5× хуже вашего текущего zero-shot 0,657.
  - Лучшая опубликованная LoRA-конфигурация (rank 8, α=16, блоки 20–23) даёт 75,19 mAP, но **на 277k изображений
    и 8× RTX PRO 6000**; замера на 7k изображениях нет вообще (п.6.2).
  - VehicleDINO (DINOv2-B + LoRA, 560×560) даёт **61,1 mAP на VeRi-776** — это на **20,8 mAP хуже** fast-reid
    и на **24,1 хуже** OSNet-AIN `vehicle-reid-0001` (85,15) на том же бенчмарке.
  - ViT-L/560×560 в 4 ГБ VRAM — нереалистично даже с LoRA и gradient checkpointing.
- Единственный осмысленный подвариант при желании попробовать: DINOv2 **ViT-S/14** на 210×210 (кратно 14),
  замороженный + обучаемая BNNeck-голова. Чисел для этого нет ни у кого.

### Что измерить у себя в первую очередь (дешёвые эксперименты, закрывающие пробелы литературы)

1. Zero-shot `veri_sbs_R50-ibn` на вашем hold-out (без обучения) — прямое сравнение с 0,657. 20 минут.
2. LP-only: заморозить OSNet-AIN, обучить только BNNeck+голову. Прямая проверка LP vs full FT на ваших данных.
3. Кривая по ID: 10/30/50/100 % ваших 1 171 ID при фиксированном рецепте — это будет **первая** такая кривая для vehicle re-id.
