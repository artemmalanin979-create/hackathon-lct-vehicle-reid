# D. Как совмещать чужие данные со своими — замеры, расписание обучения, лицензии

Дата сбора: 2026-09-16. Все PDF открывались напрямую (`curl` + `pdftotext -layout`), конфиги — сырые файлы с raw.githubusercontent.com.
Колонка «проверено» = да, если число прочитано мной из первоисточника (PDF/конфиг/страница), а не из вторичного пересказа.

Контекст задачи: 7 248 кадров / 1 171 ид. / 96 камер, вход 208×208 → **≈6,2 кадра на идентичность**. Это режим «мало ID-шников И мало кадров на ID» одновременно. Держите это в голове при чтении: часть приёмов ниже ломается именно при K<8 кадров на ID в батче.

---

## 0. Сводка: что даёт числа

| Приём | Δ / число | Источник (URL + таблица) | Проверено |
|---|---|---|---|
| Двухстадийно (pretrain на чужих → FT на своих) вместо одностадийного | **68,21 → 75,60 mAP (+7,39)**, R1 82,70 → 87,45 (+4,75) | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. V (CityFlow private test) | да |
| То же на VeRi-776 (ResNet-50) | **80,91 → 83,41 mAP (+2,50)**, R1 95,95 → 96,78 (+0,83) | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. III, две последние строки | да |
| Наивное смешивание синтетики с реальными («Solution-1») | **76,7 → 72,9 mAP (−3,8)** | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2 | да |
| Pretrain только на чужих → FT на своих («Solution-2») | **76,7 → 71,5 mAP (−5,2)** | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2 | да |
| Правильная схема: pretrain на (свои ∪ *дозированная порция* чужих) → FT только на своих | **76,7 → 79,4 mAP (+2,7)**; на CityFlow 59,7 → 65,3 (+5,6) | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2 и Табл. 3 | да |
| Заморозка первых двух слоёв на стадии FT (vehicle re-id) | **+1,4 mAP** (без заморозки −1,4) | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) §4.4, текст | да |
| BNNeck | **+4,0 mAP Market / +5,3 mAP Duke** (in-domain); **+3,4 / +6,2 mAP** (cross-domain) | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 1 и Табл. 3 | да |
| Label smoothing | **+1,0 mAP Market / +1,0 mAP Duke** | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 1 | да |
| Center loss | **+0,2 mAP Market / +0,5 mAP Duke** | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 1 | да |
| Warmup 10 эпох | **+1,2 mAP Market / +1,4 mAP Duke** | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 1 | да |
| Balanced sampling (равные шансы классам) вместо naive | **−3,56 mAP** (43,65 → 40,09) | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. VIII | да |
| SSL-предобучение на своём же малом наборе (8k картинок, MIM) vs ImageNet-sup | **92,7–92,8 vs 92,2** (ViT-S, Stanford Cars) | [arXiv:2112.10740](https://arxiv.org/pdf/2112.10740) Табл. 6 | да |
| SSL-предобучение на большом in-domain корпусе при 10 % целевых меток | **53,1 → 64,6 mAP (+11,5)** | [arXiv:2012.03753](https://arxiv.org/pdf/2012.03753) Табл. 6(a) | да |
| Пересчёт статистик BatchNorm на целевом домене (без меток) | error **40,8 % → 33,1 %** и **64,4 % → 52,5 %** | [arXiv:2105.07576](https://arxiv.org/pdf/2105.07576) Табл. 4 | да |
| LP-FT (сначала линейный пробинг головы, потом полный FT) vs FT | OOD avg **59,3 → 68,9** (+9,6), ID avg 85,1 → 85,7 | [arXiv:2202.10054](https://arxiv.org/pdf/2202.10054) Табл. 1 и Табл. 2 | да |
| Хирургический FT (разморозить один блок) vs full FT | FMoW OOD **38,9 → 44,9**; Camelyon17 **92,3 → 95,6** | [arXiv:2210.11466](https://arxiv.org/pdf/2210.11466) Табл. 1 | да |

---

## 1. Одновременно (joint) vs последовательно (pretrain → finetune)

### 1.1 VehicleNet — перепроверено по PDF

Источник: **Zheng et al., «VehicleNet: Learning Robust Visual Representation for Vehicle Re-identification», IEEE TMM**, [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305).
Постановка: VehicleNet = объединение CityFlow + VeRi-776 + CompCars + VehicleID = **434 440 изображений / 31 805 классов**. Stage-I — обучение на всём VehicleNet; Stage-II — дообучение только на целевом датасете.

**Табл. V (CityFlow, private test):**

| Стадия | Rank@1 (%) | mAP (%) |
|---|---|---|
| Stage I (только joint на объединении) | 82,70 | 68,21 |
| Stage II (+ FT на целевом) | 87,45 | **75,60** |

Δ = **+7,39 mAP, +4,75 R@1**. Цифры заказчика подтверждены.

**Табл. III (последние две строки), ResNet-50:**

| Стадия | VeRi-776 mAP | VeRi-776 R@1 | VehicleID Small R@1 | Medium R@1 | Large R@1 |
|---|---|---|---|---|---|
| Ours (Stage-I) | 80,91 | 95,95 | 83,26 | 81,13 | 79,06 |
| Ours (Stage-II) | **83,41** | **96,78** | 83,64 | 81,35 | 79,46 |

Δ на VeRi-776 = +2,50 mAP / +0,83 R@1. На VehicleID прирост **почти нулевой** (+0,2…+0,4 R@1) — важное ограничение: вторая стадия даёт много там, где целевой домен сильно отличается от смеси (CityFlow: +7,39), и почти ничего там, где целевой датасет и так доминирует в смеси.

**Табл. II (сколько даёт само добавление данных, до второй стадии; SE-ResNeXt101, валидационный сплит CityFlow):**

| Обучающие данные | # изображений | R@1 | mAP |
|---|---|---|---|
| CityFlow† | 26 803 | 73,65 | 37,65 |
| CityFlow + VeRi-776 | +49 357 | 79,48 | 43,47 |
| CityFlow + CompCars | +136 713 | 83,37 | 48,71 |
| CityFlow + VehicleID | +221 567 | 83,37 | 47,56 |
| VehicleNet (всё) | 434 440 | **88,77** | **57,35** |

Т.е. только добавление чужих данных (без второй стадии) = **+19,70 mAP**. Суммарный вклад «чужие данные» ≫ вклад «двухстадийность».

**Почему вторая стадия нужна — цитата авторов (§V.C):** «In the Stage I, the target training set, i.e., CityFlow, only occupy **6 %** of VehicleNet. The learned model, therefore, is sub-optimal for the target environment.» Это единственная явная пропорция в статье.

**Табл. IX — альтернатива через style-transfer хуже:**

| Метод (VeRi-776) | R@1 | mAP |
|---|---|---|
| w CycleGAN data | 92,91 | 75,23 |
| Stage I | 95,95 | 80,91 |
| Stage II | 96,78 | 83,41 |
| Stage II + PCB | 97,26 | 83,54 |

### 1.2 Vehicle re-id, AICity-2020 (DMT) — обе наивные схемы провалились

Источник: **He et al., «Multi-Domain Learning and Identity Mining for Vehicle Re-Identification»**, [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547), **Табл. 2**.
Split-test = внутренний сплит CityFlow: train 26 272 кадров / 233 ид., test 10 663 кадров / 100 ид. Бэкбон ResNet101-IBN-a, 320×320.

| Модель | mAP | R@1 | Δ mAP |
|---|---|---|---|
| Baseline (BoT-BS, только свои данные) | 76,7 | 89,7 | — |
| **Solution-1**: прямое слияние своих + чужих (VehicleX) | 72,9 | 85,7 | **−3,8** |
| **Solution-2**: pretrain только на чужих → FT на своих | 71,5 | 83,0 | **−5,2** |
| MDL (50 чужих ID) | 77,1 | 88,7 | +0,4 |
| **MDL (100 чужих ID)** | **79,4** | **90,7** | **+2,7** |
| MDL (200 чужих ID) | 76,6 | 88,3 | −0,1 |
| MDL (300 чужих ID) | 76,8 | 87,0 | +0,1 |

**Табл. 3** (CityFlow, официальный тест): Baseline (BoT-BS + WF-TRR) 59,7 mAP → + MDL **65,3 mAP** (+5,6).

Рецепт MDL (цитата §3.2): pre-train на `R ∪ S`, где S — подвыборка чужих ID, причём **«the number of identities of S is not larger than the one of R»**; затем FT только на R, **«first two layers of the pre-trained model are frozen»**, **«Reducing the learning rate is also necessary»**.

> **Поправка к исходной вводной.** Число **«+3,62 mAP»** в arXiv:2004.10547 я не нашёл — ни в тексте, ни в Табл. 1–5. Проверяемые дельты двухстадийности там: **+2,7 mAP** (Split-test, Табл. 2) и **+5,6 mAP** (CityFlow, Табл. 3). Рекомендую заменить цифру в отчёте.

### 1.3 Person re-id: наивный joint на 5 датасетах *ухудшает* результат

Источник: **«Multi-Domain Joint Training for Person Re-Identification»**, [arXiv:2201.01983](https://arxiv.org/pdf/2201.01983), **Табл. II**. Смешаны 5 датасетов, гиперпараметры: 120 эпох, LR 3,5e-4, /10 на 40 и 90 эпохе, triplet margin 0,3.

mAP, одиночное обучение → наивный joint:

| Датасет | BoT† (соло) | BoT (Joint) | Δ | AGW† (соло) | AGW (Joint) | Δ |
|---|---|---|---|---|---|---|
| CUHK03(L) | 67,3 | 65,7 | **−1,6** | 73,4 | 69,1 | **−4,3** |
| CUHK-SYSU | 86,0 | 89,3 | +3,3 | 88,7 | 90,8 | +2,1 |
| Market1501 | 86,1 | 84,6 | **−1,5** | 88,5 | 86,2 | **−2,3** |
| Duke | 76,8 | 75,9 | **−0,9** | 79,4 | 78,5 | **−0,9** |
| MSMT17 | 50,2 | 49,5 | **−0,7** | 54,3 | 52,6 | **−1,7** |

Т.е. **4 из 5 датасетов деградируют** от простого смешивания. Помогает только добавление domain-conditioned модуля (DCSD): BoT-DCSD (Joint) даёт 78,0 / 91,3 / 89,5 / 80,6 / 61,5, «Joint Training Gain» до **+12,3 mAP** — но это уже архитектурное изменение, а не просто смешивание.

### 1.4 Multi-source joint vs single-source: помогает не всегда

Источник: **torchreid MODEL_ZOO**, раздел «Multi-source domain generalization», <https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md> (модели обучены по [arXiv:1910.06827](https://arxiv.org/pdf/1910.06827), `max_epoch=50`).

`osnet_ain_x1_0`, R@1 (mAP):

| Схема | → Market1501 | → Duke |
|---|---|---|
| Один источник MSMT17 (combineall) | 70,1 (43,3) | 71,1 (52,7) |
| Три источника (MS+D+C → M) / (MS+M+C → D) | **73,3 (45,8)** | **65,6 (47,2)** |
| Δ | +3,2 R@1 / +2,5 mAP | **−5,5 R@1 / −5,5 mAP** |

Остальные multi-source: MS+D+M→C 27,4 (27,1); D+M+C→MS 40,2 (16,2).
**Вывод:** «больше источников» ≠ «лучше обобщение». Один большой близкий по домену источник может побить смесь.

### 1.5 Joint и pretrain — не альтернативы, а слагаемые

Источник: **ReMix**, [arXiv:2410.21938](https://arxiv.org/html/2410.21938), **Табл. 4** (обучение на MSMT17-merged + неразмеченные single-camera данные LUPerson; тест кросс-датасетный, R@1/mAP):

| SSL-pretrain | Joint | Market1501 | Duke |
|---|---|---|---|
| ✗ | ✗ | 78,4 / 51,7 | 75,8 / 58,7 |
| ✓ | ✗ | 81,7 / 54,9 | 75,1 / 59,2 |
| ✗ | ✓ | 81,3 / 57,0 | 76,9 / 60,7 |
| ✓ | ✓ | **84,0 / 61,0** | **77,6 / 61,6** |

SSL-предобучение на неразмеченных данных (+3,2 mAP Market) и joint-использование их же в обучении (+5,3 mAP) складываются почти аддитивно (+9,3 mAP).

---

## 2. Пропорции смешивания и сэмплирование

| Приём | Число | Источник | Проверено |
|---|---|---|---|
| Доля целевого датасета в объединении (VehicleNet) | целевой = **6 %** смеси; Stage-II компенсирует | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) §V.C | да |
| Правило дозирования чужих ID | **#ID(чужих) ≤ #ID(своих)**; оптимум 100 чужих ID при 233 своих (≈43 % по ID) | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) §3.2 + Табл. 2 | да |
| По кадрам в смеси | 14 536 чужих + 26 272 своих = **35,6 % чужих кадров** | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) §4.2 | да |
| Ablation по числу чужих ID | 50 → 77,1; **100 → 79,4**; 200 → 76,6; 300 → 76,8 mAP | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2 | да |
| Class-balanced sampling при длинном хвосте | **вредит: 43,65 → 40,09 mAP** (R@1 77,97 → 76,03) | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. VIII | да |
| Размер батча P×K (person re-id) | 8×3: 79,2 mAP; 16×4: **83,7**; 16×6: 82,8; 8×8: 82,0 (Market) | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 5 | да |
| Инстансов на ID в батче (fast-reid SBS) | `NUM_INSTANCE: 16` | [Base-SBS.yml](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-SBS.yml) | да |
| Сэмплер в fast-reid по умолчанию (bagtricks) | `NaiveIdentitySampler`, `NUM_INSTANCE: 4` | [Base-bagtricks.yml](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-bagtricks.yml) | да |
| Сэмплер в официальном VeRi-конфиге fast-reid (SBS) | `BalancedIdentitySampler` | [configs/VeRi/sbs_R50-ibn.yml](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | да |

Критично для нас: **прямого ablation «доля целевого домена в батче» (domain-balanced batch) с числами я не нашёл** — ни в vehicle, ни в person re-id. Ближайшее по смыслу — Табл. 2 из 2004.10547 (дозирование по *числу ID*, а не по доле батча) и Табл. VIII VehicleNet (класс-балансировка вредна).
Обратите внимание на противоречие: VehicleNet (Табл. VIII) говорит «naive sampling лучше balanced», базовый конфиг fast-reid ставит `NaiveIdentitySampler`, а официальный VeRi-конфиг (SBS) — `BalancedIdentitySampler`. **Замера, который бы их рассудил, нет.**

Полный проверенный дефолт `Base-bagtricks.yml` (raw-файл прочитан целиком): ResNet-50, LAST_STRIDE 1, `WITH_BNNECK: True`, `NECK_FEAT: before`, losses = CrossEntropy (`EPSILON: 0.1` = label smoothing) + Triplet (`MARGIN: 0.3`, `HARD_MINING: True`), вход 256×128, `REA ENABLED, PROB: 0.5`, `NaiveIdentitySampler`, `NUM_INSTANCE: 4`, Adam, `BASE_LR: 0.00035`, `WEIGHT_DECAY: 0.0005`, `IMS_PER_BATCH: 64`, MultiStepLR `STEPS: [40, 90]`, `GAMMA: 0.1`, `WARMUP_FACTOR: 0.1`, `WARMUP_ITERS: 2000`, `MAX_EPOCH: 120`. Отметьте `NUM_INSTANCE: 4` — это K=4, наши 6,2 кадра/ID такое выдерживают; K=16 из SBS — нет. И отметьте, что REA включён по умолчанию с prob 0,5 — при нашей проблеме обобщения на невиданные ID это подозреваемый номер один (см. §9).

Предостережение для нашего случая: при 6,2 кадрах на ID вариант K=16 (fast-reid SBS) физически недостижим; BoT Табл. 5 показывает, что при K=3 против K=4 теряется 79,2 vs 80,0 mAP (P=8). Т.е. наш режим по построению находится в худшей точке этой таблицы.

---

## 3. Скорость обучения на этапе дообучения

### 3.1 Конкретные re-id-рецепты

| Источник | Стадия | Оптимизатор | LR | Расписание | Эпох |
|---|---|---|---|---|---|
| VehicleNet, [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) §V.A | Stage-I | SGD mom 0,9, bs 36, wd 1e-4 | **0,02** | /10 на эп. 40 | **60** |
| VehicleNet, там же | **Stage-II** | тот же | **0,02** (тот же!) | /10 на эп. 8 | **12** |
| VehicleNet **код** [`train_2020.py`](https://github.com/layumi/AICIty-reID-2020/blob/master/pytorch/train_2020.py) | Stage-I | SGD nesterov, wd 5e-4 | head `--lr 0.02`, **backbone = 0,1×lr = 0,002** | MultiStepLR [60, 75], γ=0,1 | 80 |
| VehicleNet **код** [`train_ft_2020.py`](https://github.com/layumi/AICIty-reID-2020/blob/master/pytorch/train_ft_2020.py) | **Stage-II** | тот же | head 0,02, backbone 0,002 | MultiStepLR [30], γ=0,1, `--warm_epoch 5` | **40** |
| DMT, [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) §4.2 | Stage-I | SGD, bs 8×12 | **1e-2**, warmup 10 эп. 1e-3→1e-2 | →1e-3 на 40, →1e-4 на 70 | 100 |
| DMT, там же | **Stage-II (FT)** | SGD | **backbone 1e-3, FC 2e-3** (т.е. ×0,1 от stage-I) | — | **40** |
| BoT, [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) §3.1 | — | Adam | **3,5e-4**, warmup 10 эп. с 3,5e-5 | →3,5e-5 на 40, →3,5e-6 на 70 | 120 |
| fast-reid [`Base-bagtricks.yml`](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-bagtricks.yml) | — | Adam, wd 5e-4, bs 64 | **3,5e-4**, WARMUP_ITERS 2000, factor 0,1 | STEPS [40, 90], γ=0,1 | 120 |
| fast-reid [`Base-SBS.yml`](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-SBS.yml) | — | Adam | 3,5e-4, WARMUP_ITERS 2000 | CosineAnnealingLR, DELAY_EPOCHS 30, ETA_MIN_LR 7e-7 | **60** |
| fast-reid [`VeRi/sbs_R50-ibn.yml`](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | — | **SGD** | **0,01**, ETA_MIN_LR 7,7e-5, WARMUP_ITERS 3000 | cosine | **60** |
| fast-reid [`VehicleID/bagtricks_R50-ibn.yml`](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VehicleID/bagtricks_R50-ibn.yml) | — | Adam, bs 512 | 3,5e-4, WARMUP_ITERS 2000 | STEPS [30, 50] | 60 |

**Два несовместимых прецедента по LR на второй стадии, оба из победивших решений AICity:**
* VehicleNet **не снижает** базовый LR на Stage-II (0,02 в обоих стадиях и в статье, и в коде) — компенсирует коротким расписанием (12/40 эпох против 60/80) плюс warmup 5 эпох.
* DMT **снижает LR в 10 раз** (1e-2 → 1e-3) и явно пишет «Reducing the learning rate is also necessary».
Прямого A/B «тот же LR vs /10» на стадии FT в re-id **замера нет**.

**Разный LR для головы и бэкбона — устойчивый паттерн:**
* VehicleNet-код: `base_params` при `0.1*lr`, `classifier` при `lr` → **голова в 10× быстрее бэкбона** на обеих стадиях.
* DMT: backbone 1e-3, FC 2e-3 → **голова в 2× быстрее**.
* fast-reid `defaults.py`: `_C.SOLVER.HEADS_LR_FACTOR = 1.` с комментарием в коде «*This LR is applied to the last classification layer if you want to 10x higher than BASE_LR*».

### 3.2 Общая литература по FT-LR

**LP-FT — «Fine-Tuning can Distort Pretrained Features»**, [arXiv:2202.10054](https://arxiv.org/pdf/2202.10054):

| Метод | ID acc (avg, Табл. 1) | OOD acc (avg, Табл. 2) |
|---|---|---|
| Fine-tuning | 85,1 | 59,3 |
| Linear probing | 82,9 | 66,2 |
| **LP-FT** | **85,7** | **68,9** |

Наиболее показательные OOD-строки (Табл. 2): DomainNet FT 55,5 / LP 79,7 / LP-FT 80,7; ImageNet-R FT 52,4 / LP 70,6 / LP-FT 72,9; STL FT 82,4 / LP 85,1 / LP-FT 90,7.
Протокол: свип **6 значений LR**, cosine, bs 64, early stop по ID-валидации; при равном бюджете каждая стадия LP-FT = половина эпох обычного FT.
Механизм (§4.3): «LP-FT indeed changes both ID and OOD features **10×−100× less** than fine-tuning does».
**Прямое применение к нам:** «переобучение + провал на невиданных ID» = ровно ID-vs-OOD tradeoff из этой статьи. Случайно инициализированная голова при старте FT рвёт предобученные фичи; сперва обучить только голову (BNNeck+classifier) на замороженном бэкбоне, потом размораживать.

**«How to Fine-Tune Vision Models with SGD»**, [arXiv:2211.09359](https://arxiv.org/pdf/2211.09359):
* Свип LR для SGD: **[3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]** (App. A) — практический диапазон поиска.
* SGD (freeze-embed) OOD avg **76,7 %** vs SGD 71,9 % vs AdamW 76,0 %; на CLIP-ViT-B/16 разница ≈ **+8 %** OOD, на CLIP-ViT-L/14 **+14,3 %**.
* AdamW требует на **36 %** больше памяти, чем SGD (freeze-embed, no momentum).

---

## 4. Замораживание слоёв и BN-freeze

### 4.1 Замеры на re-id / vehicle

| Приём | Число | Источник | Проверено |
|---|---|---|---|
| Заморозить **первые два слоя** бэкбона на стадии FT (CityFlow, ResNet101-IBN-a) | **+1,4 mAP** («If the first two layers are not frozen…the performance will be reduced by 1,4 % mAP») | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) §4.4 | да |
| Заморозить **весь бэкбон** на первые 1000 итераций (SBS-рецепт) | конфиг-дефолт `MODEL.FREEZE_LAYERS: [backbone]`, `SOLVER.FREEZE_ITERS: 1000` | [Base-SBS.yml](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-SBS.yml) | да |
| То же для VeRi (vehicle) | `FREEZE_ITERS: 3000` = ровно длина warmup (`WARMUP_ITERS: 3000`) | [VeRi/sbs_R50-ibn.yml](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | да |
| Реализация заморозки в fast-reid | замороженные модули переводятся в `module.eval()` → **статистики BN тоже заморожены** | [fastreid/engine/hooks.py](https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/engine/hooks.py), `LayerFreeze.freeze_specific_layer` | да |

Т.е. дефолт SOTA-рецепта fast-reid для vehicle re-id = «первые 3000 итераций обучается только голова (BNNeck + классификатор), бэкбон + его BN заморожены; это совпадает с окном warmup». Это ровно LP-FT, зашитый в конфиг.

**Замеров «заморозить stem+layer1 / только последний блок / всё кроме головы» с mAP-числами конкретно на re-id я не нашёл.** Всё, что есть по re-id, — это два числа выше (+1,4 mAP и конфиг-дефолты).

### 4.2 Замеры вне re-id (перенос по аналогии)

**Surgical fine-tuning**, [arXiv:2210.11466](https://arxiv.org/pdf/2210.11466), **Табл. 1** (CLIP ViT-B/16, WILDS, OOD accuracy):

| Что размораживаем | Camelyon17 | FMoW |
|---|---|---|
| Ничего (no fine-tuning) | 86,2 | 35,5 |
| **Всё (full FT)** | 92,3 (1,7) | 38,9 (0,5) |
| Только embedding (первый слой) | **95,6 (0,4)** | 36,0 (0,1) |
| Первые три блока | 92,5 (0,5) | 39,8 (1,0) |
| Последние три блока | 87,5 (4,1) | **44,9 (2,6)** |
| Только последний слой | 90,1 (1,5) | 36,9 (5,5) |

Аннотация: «fine-tuning only one block of layers and freezing the others **outperforms full fine-tuning on all parameters by almost 3 %**».
Правило выбора блока (§2, Рис. 1): **input-level сдвиг → первый блок; feature-level → средний; output-level → последний слой**. У нас смена камер/освещения = input-level → кандидат: размораживать ранние слои, а не только голову. Это прямо противоречит «классическому» FT-совету.
**Табл. 4** (7 реальных задач, средний ранг): Auto-RGN 1,29 < Full FT 2,71 < L1-reg 3,28 < Gradual Unfreeze(Last→First) 4,0 < Auto-SNR 4,14 < Gradual Unfreeze(First→Last) 4,71. **Gradual unfreezing проиграл обычному full FT** — популярный приём не подтверждается.
**Табл. 2** (CIFAR-10-C, unsupervised online adaptation): No adaptation 67,8; All 69,7; **Layer 1: 75,4; Layer 1-2: 75,5**; Last 67,8.
Протокол: Adam, свип 3 LR, **early stop по held-out целевым данным**.

### 4.3 BN-freeze / пересчёт статистик BN — числа

Источник: **«Rethinking "Batch" in BatchNorm»**, [arXiv:2105.07576](https://arxiv.org/pdf/2105.07576), **Табл. 4**. ResNet-50, обученный на ImageNet; переоценка population statistics BN на данных целевого домена (**без меток**, 1000 картинок):

| Домен оценки | Данные для статистик BN | Error rate (%) |
|---|---|---|
| IN-C-contrast | ImageNet | 40,8 |
| IN-C-contrast | **IN-C-contrast** | **33,1** (−7,7 п.п.) |
| IN-C-gaussian | ImageNet | 64,4 |
| IN-C-gaussian | **IN-C-gaussian** | **52,5** (−11,9 п.п.) |
| IN-C-jpeg | ImageNet | 40,4 |
| IN-C-jpeg | **IN-C-jpeg** | **36,7** (−3,7 п.п.) |
| ImageNet (чистый) | ImageNet | 23,4 |
| ImageNet (чистый) | IN-C-contrast | 39,7 (**+16,3 п.п. хуже**) |

Практика FrozenBN (§4.4): «When applied in fine-tuning, it's common to **freeze both normalization statistics and the subsequent affine transform**, so that they can be fused into a single affine transform». Также: «FrozenBN … getting **25,2 %** error rate even when the model is trained with a small normalization batch size of 2 for its first 80 epochs», но «**When normalization batch size is large enough, tuning with FrozenBN underperforms regular BN**». PreciseBN даёт **+3,5 %** при малом normalization batch size.

**Практический вывод для нас (дёшево, без обучения):** пересчитать running_mean/var BN на своих неразмеченных кадрах и/или заморозить BN при дообучении на 7k кадров. Замера именно на re-id — нет, но механизм («модель учит камеру») тот же.

---

## 5. Сколько эпох и warmup

| Число | Источник | Проверено |
|---|---|---|
| Warmup **10 эпох**, линейно 3,5e-5 → 3,5e-4 (Adam); decay →3,5e-5 на 40 эп., →3,5e-6 на 70; всего **120 эпох** | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) §3.1, ур. (1) | да |
| Warmup даёт **+1,2 mAP** Market / **+1,4 mAP** Duke (87,7/74,0 → 88,7/75,2) | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 1 | да |
| Warmup в cross-domain даёт **+1,2 mAP** M→D / **+2,9 mAP** D→M | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 3 | да |
| VehicleNet: Stage-I **60** эпох → Stage-II **12** эпох (×0,2). Время: 30 ч vs **1,5 ч** | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) §V.A и §V.C «Time Cost» | да |
| В коде VehicleNet: Stage-I 80 эпох, Stage-II **40** эпох + `--warm_epoch 5` | [train_2020.py](https://github.com/layumi/AICIty-reID-2020/blob/master/pytorch/train_2020.py) / [train_ft_2020.py](https://github.com/layumi/AICIty-reID-2020/blob/master/pytorch/train_ft_2020.py) | да |
| DMT: Stage-I **100** эпох (warmup 10) → Stage-II **40** эпох | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) §4.2 | да |
| fast-reid SBS: **60** эпох (vs 120 у bagtricks), cosine, DELAY_EPOCHS 30 | [Base-SBS.yml](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-SBS.yml) | да |
| torchreid multi-source DG модели: `max_epoch=50` | [MODEL_ZOO.md](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md) | да |
| «Attribute to the trained weight of the first stage, the second stage **converge early**» (Рис. 8) | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Рис. 8 | да |
| Early stopping по held-out целевым данным — стандартный протокол в FT-исследованиях | [arXiv:2210.11466](https://arxiv.org/pdf/2210.11466) §3.1; [arXiv:2202.10054](https://arxiv.org/pdf/2202.10054) §4.1 | да |

**Прямого ablation «N эпох vs N/3 эпох при малом целевом датасете» на re-id с mAP-числами я не нашёл — замера нет.** Косвенно: все двухстадийные победители AICity сокращают вторую стадию в 2–5 раз относительно первой, а LP-FT явно оговаривает деление бюджета эпох пополам между LP и FT.

Наши 30 эпох на 7 248 кадрах = порядка 7 248/64 ≈ 113 итераций/эпоху × 30 ≈ **3 400 итераций всего**, что **меньше**, чем один только период заморозки бэкбона в VeRi-конфиге fast-reid (FREEZE_ITERS 3000). Это отдельный сигнал: у нас не «слишком долго учим», а «схема расписания вообще не соответствует масштабу данных».

---

## 6. Label smoothing, BNNeck, center loss — точные Δ из Bag-of-Tricks

Источник: **Luo et al., «Bag of Tricks and A Strong Baseline for Deep Person Re-identification»**, [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071). ResNet-50, Adam, 120 эпох.

**Табл. 1 (in-domain, приёмы добавляются кумулятивно):**

| Модель | Market R@1 | Market mAP | Δ mAP | Duke R@1 | Duke mAP | Δ mAP |
|---|---|---|---|---|---|---|
| Baseline-S | 87,7 | 74,0 | — | 79,7 | 63,7 | — |
| +warmup | 88,7 | 75,2 | +1,2 | 80,6 | 65,1 | +1,4 |
| +REA (random erasing) | 91,3 | 79,3 | +4,1 | 81,5 | 68,3 | +3,2 |
| **+LS (label smoothing)** | 91,4 | 80,3 | **+1,0** | 82,4 | 69,3 | **+1,0** |
| +stride=1 | 92,0 | 81,7 | +1,4 | 82,6 | 70,6 | +1,3 |
| **+BNNeck** | 94,1 | 85,7 | **+4,0** | 86,2 | 75,9 | **+5,3** |
| **+center loss** | 94,5 | 85,9 | **+0,2** | 86,4 | 76,4 | **+0,5** |

**Табл. 3 (cross-domain — важнее для нас, т.к. у нас провал на невиданных ID):**

| Модель | M→D R@1 | M→D mAP | D→M R@1 | D→M mAP |
|---|---|---|---|---|
| Baseline | 24,4 | 12,9 | 34,2 | 14,5 |
| +warmup | 26,3 | 14,1 | 39,7 | 17,4 |
| **+REA** | 21,5 | **10,2** | 32,5 | **13,5** |
| +LS | 23,2 | 11,3 | 36,5 | 14,9 |
| +stride=1 | 23,1 | 11,8 | 37,1 | 15,4 |
| **+BNNeck** | 26,7 | **15,2** | 47,7 | **21,6** |
| +center loss | 27,5 | 15,0 | 47,4 | 21,4 |
| **−REA (убрать random erasing)** | **41,4** | **25,7** | **54,3** | **25,5** |

**Табл. 2 (изолированный эффект BNNeck):** f без BNNeck (Euclid) 92,0 / **81,7** mAP Market, 82,6 / **70,6** Duke → f_i с BNNeck + cosine 94,1 / **85,7**, 86,2 / **75,9**. **+4,0 / +5,3 mAP**. Инференс — по `f_i` (после BN) с cosine-расстоянием.
Прочее: center loss β = **0,0005**. Табл. 6 (размер входа): 256×128 → 83,7 mAP; 224×224 → 83,3; 384×128 → 82,7; 384×192 → 83,1 — «image size is not a pretty importance factor». Для нашего 208×208 это хорошая новость.

**Ранжирование приёмов по нашей задаче:** BNNeck (+4…+5,3 in-domain, +3,4…+6,2 cross-domain) ≫ warmup ≈ label smoothing (+1,0…+1,4) ≫ center loss (+0,2…+0,5, в cross-domain даже −0,2 mAP M→D). Random erasing — **отдельный разговор, см. §9**.

---

## 7. Что нам недоступно по лицензии/доступу

Проверка URL выполнена 2026-09-16 из этой среды (`curl -sIL`).

| Датасет | Размер | Лицензия / доступ | Коммерческое использование | Статус URL |
|---|---|---|---|---|
| **VeRi-776** | 776 ид. / 20 камер | Запрос по email, xinchenliu@bupt.cn. Цитата: «We ask for your information only to make sure the dataset is used for **non-commercial** purposes» | ❌ запрещено | <https://vehiclereid.github.io/VeRi/> HTTP 200 |
| **VehicleID (PKU)** | 221 763 изобр. / 26 267 ТС | **Подписанное соглашение** (PDF) на академический email (gmail/outlook/qq отклоняются). Цитата: «can only be used for **ACADEMIC PURPOSES. NO COMERCIAL USE is allowed**» | ❌ запрещено | <https://pkuml.org/resources/pku-vehicleid.html> HTTP 200 |
| **PKU-VD (VD1/VD2)** | VD1 1 097 649 изобр./1 232 модели; VD2 807 260/1 112 | **Соглашение с рукописной подписью PI**, та же формулировка «NO COMERCIAL USE» | ❌ запрещено | <https://pkuml.org/resources/pku-vds.html> HTTP 200 |
| **VERI-Wild** | 416 314 изобр. / 40 671 ид. / 174 камеры | Запрос по email yanbai@pku.edu.cn; «terms of access in the email», non-commercial | ❌ запрещено | <https://github.com/PKU-IMRE/VERI-Wild> HTTP 200 |
| **CityFlow / AI City** | — | NVIDIA Dataset License Agreement. §2a дословно: «**You may not use the DATASET or any models developed using the DATASET for any commercial or production purpose**». §1(ii): после челленджа — «non-commercial, academic purposes only» | ❌❌ запрещены **и данные, и модели** | [Dataset-License-AIC2022.pdf](http://www.aicitychallenge.org/wp-content/uploads/2022/02/Dataset-License-AIC2022.pdf) HTTP 200, PDF прочитан |
| **CompCars** | 136 726 full-car изобр. / 1 716 моделей / 163 марки + 50 000 surveillance | «The CompCars database is available for **non-commercial research purposes only**»; «You agree not to … exploit for any commercial purposes, any portion of the images» | ❌ запрещено | <https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/> HTTP 200 |
| **VRAI** (UAV) | train 66 113 изобр./6 302 ид.; test 71 500/6 720 | «Our VRAI is **prohibited for any commercial using**» | ❌ запрещено | <https://github.com/JiaoBL1234/VRAI-Dataset> HTTP 200 |
| **Vehicle-1M** | ~1M изобр. | Требуется подписанное соглашение (Vehicle-1MAGREEMENT.pdf), copyright NLPR/CASIA «all rights reserved» | ❌ нет явного разрешения | nlpr.ia.ac.cn — **ECONNREFUSED** из нашей сети |
| **VRIC** | 60 430 изобр. / 5 622 ид. / 60 камер | Формы-соглашения нет, прямая ссылка Google Drive (129 МБ). НО: «All the images were collected from **UA-DETRAC** and the **copyright belongs to the original owners**» → лицензия наследуется от UA-DETRAC, явного коммерческого гранта нет | ⚠️ серая зона, **не считать безопасным** | <https://qmul-vric.github.io/> HTTP 200; detrac-db.rit.albany.edu редиректит на общую страницу лаборатории |
| **BoxCars116k** | 116k изобр. / 27 496 ТС / 137 камер (число камер — из вторичного обзора, первоисточник не открылся) | **LICENSE-файла в репозитории нет.** README кода: «This code is for **research only** purposes». Лицензии самого датасета не заявлено | ⚠️ лицензия не заявлена ⇒ по умолчанию все права защищены | Основной хост `medusa.fit.vutbr.cz` — **DNS не резолвится** (ENOTFOUND). Google Drive-бэкап <https://drive.google.com/file/d/19LHLOmmVyUS1R4ypwByfrV8KQWnz2GDT/view> HTTP 200 |
| **Stanford Cars** | 8 144 train / 196 классов | Оригинальные условия — research only | ⚠️ | `http://ai.stanford.edu/~jkrause/car196/cars_train.tgz` → **HTTP 500 (мёртвая)**. Неофициальные зеркала: HF `Multimodal-Fatima/StanfordCars_train` HTTP 200; часть Kaggle-зеркал уже 404 |
| **VehicleX** | 136 726 синт. изобр. / 1 362 ид. | **LICENSE-файла в репозитории нет** (raw LICENSE → 404 и на master, и на main). CityFlow-адаптированная версия требует регистрации в AI City ⇒ наследует запрет NVIDIA | ⚠️/❌ | <https://github.com/yorkeyao/VehicleX> HTTP 200 |
| **Synthehicle** (синтетика CARLA, 64 сцены / 340 видео / 17 ч) | — | **LICENSE-файла в репозитории не найдено** (raw → 404); README говорит «freely available», но без юридической формулировки | ⚠️ надо уточнять у авторов | <https://github.com/fubel/synthehicle> HTTP 200 |
| HuggingFace `dgwon/vehicle-re-identification-data` | 10K–100K изобр., imagefolder | **Лицензия не указана, карточки датасета нет, происхождение неизвестно** | ❌ не использовать | HTTP 200, 20 скачиваний / 0 лайков |
| Roboflow Universe «Vehicle ReID / example1» и аналогичные | десятки–тысячи кадров | Помечены CC BY 4.0, но это пользовательская загрузка без проверки происхождения | ⚠️ юридически ненадёжно | найдено поиском; не проверял содержимое |

### Отдельный риск: сами предобученные веса

* Код **torchreid / deep-person-reid — MIT** (проверено: raw LICENSE, «MIT License, Copyright (c) 2018 Kaiyang Zhou»).
* Но чекпойнты из `MODEL_ZOO.md` (в т.ч. cross-domain и multi-source `osnet_ain_x1_0`) обучены на **MSMT17, Market-1501, DukeMTMC-reID, CUHK03** — все non-commercial research-датасеты (а DukeMTMC вдобавок отозван авторами). **Лицензия кода MIT не отмывает веса.**
* Поэтому нынешний «бесплатный» baseline 0,657 mAP на готовых весах OSNet-AIN — **сам по себе юридическая мина**, если продукт коммерческий. Нужно выяснить, какой именно чекпойнт используется и на чём он обучен, до того как строить на нём продакшн.
* Аналогично ImageNet: ILSVRC-условия — «for non-commercial research and/or educational purposes»; на практике все используют ImageNet-веса коммерчески, но это тот же класс риска, что и выше. **Замера/прецедента я здесь не искал.**

**Итог по §7: среди публичных vehicle re-id датасетов не нашлось ни одного с явным разрешением коммерческого использования.** Самое близкое к «доступно прямо сейчас без формы» — VRIC (прямая ссылка Drive) и BoxCars116k (Drive-бэкап), но у обоих лицензия либо унаследована, либо не заявлена — т.е. по умолчанию «все права защищены».

---

## 8. Если чужие данные недоступны — что остаётся (с числами)

У нас есть неразмеченные: 1 110 + 750 + 448 кадров валидации + тест-выборка без меток. Это порядка **10 тыс. неразмеченных кадров**. Ниже — что об этом масштабе известно.

### 8.1 SSL на большом in-domain корпусе — эталон возможного выигрыша

**LUPerson**, [arXiv:2012.03753](https://arxiv.org/pdf/2012.03753) (MoCo-v2 на 4,18M неразмеченных кадров людей), **Табл. 6(a)**, Market1501, MGN, формат mAP/cmc1, «small-scale» = процент оставленных ID:

| Доля | 10 % (= **1 170 кадров / 75 ид.**) | 30 % | 50 % | 70 % | 90 % |
|---|---|---|---|---|---|
| ImageNet sup. | 53,1 / 76,9 | 75,2 / 90,8 | 81,5 / 93,5 | 84,8 / 94,5 | 86,9 / 95,2 |
| ImageNet unsup. | 58,4 / 81,7 | 76,6 / 91,9 | 82,0 / 94,1 | 85,4 / 94,5 | 87,4 / 95,5 |
| **LUPerson unsup.** | **64,6 / 85,5** | **81,9 / 93,7** | **85,8 / 94,9** | **88,8 / 95,9** | **90,5 / 96,4** |
| Δ (LUP − IN sup.) | **+11,5 mAP** | +6,7 | +4,3 | +4,0 | +3,6 |

Ключевая закономерность: **чем меньше целевых данных, тем больше выигрыш от in-domain SSL-предобучения** (11,5 → 3,6 mAP). Наш режим (7 248 кадров / 1 171 ид.) по числу кадров близок к «10 % Market1501» — т.е. к точке максимального выигрыша.
**Табл. 7** (масштаб SSL-корпуса, MGN → Market1501, mAP/cmc1): 12,5 % LUPerson (~522k) 89,5/95,9; 25 % 90,1/96,2; 50 % 90,8/96,4; 100 % 91,0/96,4 — **насыщение уже на ~0,5–1 M кадров**.

### 8.2 А если SSL-корпус = только наши 10 тыс. кадров?

**Контрастивный SSL — нет.** «When Does Contrastive Visual Representation Learning Work?», [arXiv:2105.05837](https://arxiv.org/pdf/2105.05837), §4.1: «The gap between the **500k** and **1M** pretraining image curves is typically **less than 1-2 %** top-1 … the difference between **50k** and **250k** pretraining images is substantial for each dataset, **often in excess of 10 %** top-1 accuracy». **10 тыс. кадров — на порядок ниже даже нижней точки их развёртки (50k).** SimCLR/MoCo на наших данных с нуля — почти наверняка пустая трата.

**Masked image modeling (MIM/BEiT) на своём же маленьком наборе — да, работает.** «Are Large-scale Datasets Necessary for Self-Supervised Pre-training?», [arXiv:2112.10740](https://arxiv.org/pdf/2112.10740), **Табл. 6**, Stanford Cars = **8 144 изображений** (почти ровно наш масштаб!), top-1 после дообучения:

| Метод | Данные предобучения | ViT-S | ViT-B |
|---|---|---|---|
| Random init (без предобучения) | — | 35,3 | 36,9 |
| DeiT (supervised ImageNet) | ImageNet | 92,2 | **92,1** |
| BEiT (SSL) | ImageNet | 92,4 | **93,9** |
| **BEiT (SSL)** | **только target 8k** | **92,7** | 92,7 |
| **SplitMask (SSL)** | **только target 8k** | **92,8** | 93,1 |

Интерпретация без приукрашивания: на ViT-S предобучение MIM **только на 8 тыс. своих картинок** бьёт supervised-ImageNet (**92,8 vs 92,2**, +0,6) и SSL-ImageNet (92,8 vs 92,4). На ViT-B SSL-ImageNet всё ещё впереди (93,9 vs 93,1). Т.е. выигрыш реален, но **скромный (<1 п.п.)** и исчезает с ростом ёмкости модели.
Цена (Табл. 3): для Stanford Cars предобучение шло **5 000 эпох**. Плюс критичная деталь (§5.2): «we **reduce the maximal size of the crop from 100 % to 25 %** of the raw image size» — стандартный RandomResizedCrop на маленьком наборе не работает.
**Табл. 5** (ADE20k, 20 тыс. изображений, сегментация, mIoU): Random init 25,4; DeiT (sup. IN) 46,1; BEiT (IN) 45,6; **BEiT (только ADE20k) 45,6**; SplitMask (только ADE20k) 45,7 — паритет с ImageNet при 20k изображений.

### 8.3 Ещё более дешёвые опции без обучения

* **Пересчёт статистик BatchNorm на своих неразмеченных кадрах** — см. §4.3, Табл. 4 в [arXiv:2105.07576](https://arxiv.org/pdf/2105.07576): −7,7 / −11,9 / −3,7 п.п. ошибки на трёх видах сдвига, стоимость — один прямой проход по 1000 картинкам. Замера на re-id нет.
* **Псевдо-метки на неразмеченной тест-выборке**: [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 4 (CityFlow): Baseline (BoT-BS+MDL) **65,3** → + k-means псевдо-метки **63,9 (−1,4, хуже!)** → + Identity Mining **68,5 (+3,2)**. Точность псевдо-меток: k-means 84,0 % vs IM 98,7 %. Т.е. кластеризация в лоб вредит; нужен алгоритм с порогами по расстоянию (dn=0,49, dp=0,23).
* **Слияние фич supervised-ImageNet + in-domain SSL**: [arXiv:2105.05837](https://arxiv.org/pdf/2105.05837) Табл. 3 — добавление ImageNet-supervised фич поверх in-domain SimCLR даёт **+4,7 %** на iNat21 (обратное направление даёт −4,2 %).
* **SSL + joint складываются**: ReMix Табл. 4 (см. §1.5): pretrain-only +3,2 mAP, joint-only +5,3 mAP, оба вместе **+9,3 mAP** (Market).

---

## 9. Отрицательные результаты (где добавление чужих данных / «улучшений» НЕ помогло)

| Что не сработало | Число | Источник | Проверено |
|---|---|---|---|
| Прямое слияние реальных и синтетических (Solution-1) | **76,7 → 72,9 mAP (−3,8)** | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2 | да |
| Предобучение **только** на чужих → FT на своих (Solution-2) | **76,7 → 71,5 mAP (−5,2)**. Цитата: «the pre-trained model on D_S **may not be better than the pre-trained model on ImageNet**» | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2 + §3.2 | да |
| Слишком много чужих ID | 100 ID → 79,4 mAP; **200 ID → 76,6** (хуже baseline 76,7); 300 ID → 76,8 | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2 | да |
| Наивный joint на 5 person-датасетах | **4 из 5 датасетов деградируют**: BoT −1,6 / −1,5 / −0,9 / −0,7 mAP; AGW −4,3 / −2,3 / −0,9 / −1,7 mAP | [arXiv:2201.01983](https://arxiv.org/pdf/2201.01983) Табл. II | да |
| Multi-source вместо одного близкого источника | MS+M+C→Duke **65,6 (47,2)** против MSMT17→Duke **71,1 (52,7)**: **−5,5 R@1 / −5,5 mAP** | [torchreid MODEL_ZOO](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md) | да |
| Class-balanced sampling при длинном хвосте | **43,65 → 40,09 mAP (−3,56)**. Причина по авторам: «balanced sampling have more chance to select the same image in the classes with fewer images. Thus the model is prone to **over-fit the class with limited samples**» | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. VIII | да |
| **Random Erasing губит обобщение на новый домен** | M→D: с REA **15,2 mAP**, без REA **25,7 mAP** (**+10,5**); D→M: 21,6 → 25,5 (**+3,9**). При этом in-domain REA даёт +4,1 mAP. Авторы: «REA masking the regions of training images lets the model **learn more knowledge in the training domain**. It causes the model to perform worse in [other domains]» | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 3, строки «+REA» и «−REA» | да |
| Center loss в cross-domain | M→D: 15,2 → **15,0 mAP (−0,2)**; D→M: 21,6 → 21,4 (−0,2) | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Табл. 3 | да |
| Instance Norm (OSNet-AIN) ухудшает in-domain | Market 86,7 → **84,4 mAP (−2,3)**; Duke 76,7 → **74,2 (−2,5)**; но cross-domain +4,6 / +6,6 mAP | [arXiv:1910.06827](https://arxiv.org/pdf/1910.06827) Табл. 7 и Табл. 8 | да |
| CycleGAN-адаптация домена вместо двух стадий | VeRi-776: **75,23 mAP** против 83,41 у Stage-II (**−8,2**) | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. IX | да |
| Более мощный бэкбон ≠ лучше | SENet-154 (ImageNet top-5 95,53) → **45,14 mAP**; SE-ResNeXt101 (95,04) → **48,71 mAP** | [arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. VII | да |
| Объединение SSL-корпусов из разных доменов | In-Domain(250k) **0,451 / 0,608 / 0,485** бьёт pooled-1M (4×250k) **0,410 / 0,574 / 0,482**; In-Domain(500k) 0,477/0,629/0,499 бьёт все pooled-500k. «Adding pretraining data from a different domain **typically hurts** performance» | [arXiv:2105.05837](https://arxiv.org/pdf/2105.05837) Табл. 2 | да |
| Добавление in-domain supervised фич поверх ImageNet SimCLR | **−4,2 %** | [arXiv:2105.05837](https://arxiv.org/pdf/2105.05837) Табл. 3 | да |
| Gradual unfreezing (популярный приём) | средний ранг 4,71 (First→Last) и 4,0 (Last→First) против **2,71** у обычного full FT | [arXiv:2210.11466](https://arxiv.org/pdf/2210.11466) Табл. 4 | да |
| Полный FT при сильном сдвиге домена | OOD avg **59,3** у FT против **66,2** у простого linear probing | [arXiv:2202.10054](https://arxiv.org/pdf/2202.10054) Табл. 2 | да |
| MoCo-на-ImageNet вместо supervised-ImageNet (при малых данных) | MSMT17 10 %: IN sup **23,2** mAP vs IN unsup **22,6** (−0,6) | [arXiv:2012.03753](https://arxiv.org/pdf/2012.03753) Табл. 6(c) | да |
| k-means псевдо-метки на неразмеченном тесте | 65,3 → **63,9 mAP (−1,4)** | [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 4 | да |
| FrozenBN при достаточном normalization batch size | «tuning with FrozenBN **underperforms regular BN**» | [arXiv:2105.07576](https://arxiv.org/pdf/2105.07576) §4.3 | да |
| Неверные статистики BN | ImageNet clean: 23,4 % → **39,7 %** ошибки, если статистики BN взяты с другого домена | [arXiv:2105.07576](https://arxiv.org/pdf/2105.07576) Табл. 4 | да |

---

## 10. Поправки к ранее установленному

1. **«+3,62 mAP от двухстадийности в arXiv:2004.10547» — не подтверждается.** В Табл. 1–5 этой статьи такого числа нет. Проверяемые значения: **+2,7 mAP** (Split-test, Табл. 2: 76,7 → 79,4) и **+5,6 mAP** (CityFlow, Табл. 3: 59,7 → 65,3). В самом тексте §4.4 есть внутреннее расхождение: «improves the mAP accuracy from **59,3 %** to 65,3 % in Table 3», тогда как в Табл. 3 стоит 59,7.
2. **VehicleNet: +7,39 mAP (68,21 → 75,60) — подтверждено** дословно, Табл. V, private test CityFlow. Но это **лучший** случай; на VeRi-776 та же схема даёт только **+2,50 mAP**, а на VehicleID — **+0,2…+0,4 R@1**.
3. **«Одностадийное смешивание даёт меньше» — уточнение.** В VehicleNet одностадийное смешивание (Табл. II) даёт **+19,70 mAP** — это самый крупный источник прироста; двухстадийность добавляет поверх ещё +7,39. Формулировка «одностадийное даёт меньше» верна, но может создать ложное впечатление, что смешивание малополезно. Главный выигрыш — от данных, а не от расписания.
4. **Наивное смешивание проваливается не только с синтетикой.** См. §1.3 (5 person-датасетов, 4/5 деградаций) и §1.4 (multi-source −5,5 mAP). Это общий эффект, а не артефакт sim-to-real разрыва.
5. **VehicleNet не снижает LR на второй стадии** — ни в статье (0,02 в обеих стадиях), ни в коде. Это противоречит DMT («Reducing the learning rate is also necessary», 1e-2 → 1e-3). Универсального правила по LR второй стадии в re-id-литературе нет.

---

## 11. Чего замера нет (честный список пробелов)

* Ablation **доли целевого домена в батче** (domain-balanced batch / oversampling целевого домена) с mAP-числами — **не найдено** ни в vehicle, ни в person re-id. Есть только дозирование по числу ID ([arXiv:2004.10547](https://arxiv.org/pdf/2004.10547) Табл. 2) и класс-балансировка ([arXiv:2004.06305](https://arxiv.org/pdf/2004.06305) Табл. VIII).
* A/B «тот же LR vs LR/10» на стадии дообучения в re-id — **замера нет**, есть только два противоречащих друг другу прецедента.
* Систематическая развёртка «заморозить stem+layer1 / только layer4 / всё кроме головы» **на re-id** — **замера нет**. Есть одно число (+1,4 mAP за заморозку первых двух слоёв) и конфиг-дефолты fast-reid. Развёртка есть только вне re-id ([arXiv:2210.11466](https://arxiv.org/pdf/2210.11466) Табл. 1).
* BN-freeze **на re-id** — **замера нет**. Числа только на классификации ImageNet-C ([arXiv:2105.07576](https://arxiv.org/pdf/2105.07576) Табл. 4).
* «Меньше эпох лучше при малых данных» на re-id — **прямого ablation нет**; косвенно из практики двухстадийных решений (вторая стадия в 2–5 раз короче первой) и из протоколов early stopping.
* LP-FT **на re-id** — **замера нет** (LP-FT меряли на классификации). Ближайший аналог в re-id — `FREEZE_ITERS` в fast-reid, но без ablation «с/без».
* SSL на ~10 тыс. неразмеченных кадров **именно для re-id** — **замера нет**. Ближайшая аналогия — Stanford Cars 8k в [arXiv:2112.10740](https://arxiv.org/pdf/2112.10740) Табл. 6 (классификация, ViT+MIM).
* Число камер в BoxCars116k (137) — из вторичного источника (обзор), первоисточник недоступен; **не подтверждено**.
* Лицензионный статус Vehicle-1M и Synthehicle — **не подтверждён** (сайт NLPR недоступен; LICENSE в репозитории Synthehicle не найден).
