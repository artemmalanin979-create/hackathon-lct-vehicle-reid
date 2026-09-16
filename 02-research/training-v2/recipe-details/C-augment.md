# C — Аугментации и приёмы, ломающие связь «объект ↔ фон». Замеры.

Дата сбора: 2026-09-16. Все числа извлечены из PDF первоисточников через `pdftotext -layout`
(вербатим из таблиц), не из аннотаций и не из вторичных обзоров.

**Наш контекст:** 7 248 кадров / 1 171 ID / 96 камер, кропы 208×208, ID-loss + triplet.
Rank-1 внутри камеры 0,976 / между камерами 0,379. Это ровно тот разрыв, который в литературе
называют *background bias* (Tian CVPR'18) и *camera bias* (Song ICLR'25).

> ⚠️ Главный вывод раздела вперёд: **Random Erasing — самый популярный «дефолтный» приём
> в re-ID — в нашей ситуации, скорее всего, ВРЕДИТ.** Два независимых источника измеряют
> падение кросс-доменного mAP от REA. Подробности в §1 и «Отрицательные результаты».

---

## Сводная таблица

| Приём | Датасет | Базовое → с приёмом (mAP) | Δ mAP | Δ Rank-1 | URL + таблица | Цена (CPU/GPU/доп. метки) |
|---|---|---|---|---|---|---|
| **Random Erasing (в домене)** | Market-1501, IDE R50 | 63,56 → 68,28 | **+4,72** | +2,10 (83,14→85,24) | [arXiv:1708.04896](https://arxiv.org/pdf/1708.04896) Table 6 | ~0 CPU, меток нет |
| Random Erasing (в домене) | Market-1501, TriNet R50 (triplet — как у нас) | 65,79 → 68,67 | **+2,88** | +1,34 (82,60→83,94) | там же, Table 6 | ~0 |
| Random Erasing (в домене) | DukeMTMC, IDE R50 | 51,29 → 56,17 | **+4,88** | +2,25 (71,99→74,24) | там же, Table 6 | ~0 |
| Random Erasing (в домене) | **CUHK03-labeled** (малый: 767 train ID), IDE R50 | 27,37 → 36,77 | **+9,40** | +11,17 (30,29→41,46) | там же, Table 6 | ~0 |
| Random Erasing (в домене) | Market-1501, BoT-пайплайн | 75,2 → 79,3 | **+4,1** | +2,6 (88,7→91,3) | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) Table 1 | ~0 |
| Random Erasing (в домене) | DukeMTMC, BoT | 65,1 → 68,3 | **+3,2** | +0,9 (80,6→81,5) | там же, Table 1 | ~0 |
| 🔴 **Random Erasing (КРОСС-домен)** | Market→Duke, BoT | 14,1 → 10,2 | **−3,9** | −4,8 (26,3→21,5) | [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071) **Table 3** | ~0 |
| 🔴 **Удаление REA из полной модели** | Market→Duke, BoT | 15,0 → **25,7** | **+10,7** | +13,9 (27,5→41,4) | там же, **Table 3** (строка `-REA`) | ~0 |
| 🔴 Удаление REA из полной модели | Duke→Market, BoT | 21,4 → **25,5** | **+4,1** | +6,9 (47,4→54,3) | там же, Table 3 | ~0 |
| 🔴 Random Erasing (кросс-домен) | Market→Duke, ResNet-50 | 19,3 → 14,3 | **−5,0** | −7,6 (35,4→27,8) | [arXiv:2104.02008](https://arxiv.org/pdf/2104.02008) **Table 2** | ~0 |
| 🔴 Random Erasing (кросс-домен) | Duke→Market, ResNet-50 | 20,4 → 16,1 | **−4,3** | −6,7 (45,2→38,5) | там же, Table 2 | ~0 |
| 🔴 Random Erasing (кросс-домен) | Market→Duke, OSNet | 25,9 → 20,5 | **−5,4** | −8,5 (44,7→36,2) | там же, Table 2 | ~0 |
| **Замена фона (online, p=0,5)** | Market-1501, main-net | 75,6 → 79,5 (top-1) | *mAP не даётся* | **+3,9 top-1** | [Tian CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.pdf) **Table 4** | сегментация 1× офлайн + 100 фонов |
| Замена фона (online, p=0,25) | Market-1501, main-net | 75,6 → 79,2 (top-1) | — | +3,6 top-1 | там же, Table 4 | то же |
| Замена фона (online, p=0,75) | Market-1501, main-net | 75,6 → 78,8 (top-1) | — | +3,2 top-1 | там же, Table 4 | то же |
| 🔴 Замена фона (**offline** 1:1) | Market-1501, main-net | 75,6 → 76,1 (top-1) | — | **+0,5 top-1** | там же, Table 4 | то же |
| 🔴 Замена фона (offline 1:2) | Market-1501, main-net | 75,6 → 77,3 (top-1) | — | +1,7 top-1 | там же, Table 4 | то же |
| **BIR: смесь сегм./ориг., k=0,2** | **VeRi-776 (ТС!)** | 65,78 → **70,74** | **+4,96** | **+4,17** (86,29→90,46) | [arXiv:1910.06613](https://arxiv.org/pdf/1910.06613) **Table 2** | сегм.-сеть на каждый кадр (офлайн) |
| 🔴 **Жёсткое удаление фона (Seg)** | VeRi-776 (ТС) | 65,78 → **57,53** | **−8,25** | −4,59 (86,29→81,70) | там же, **Table 1** | то же |
| 🔴 Удаление фона + постобр. (Seg+Post) | VeRi-776 (ТС) | 65,78 → 64,79 | **−0,99** | +0,60 (86,29→86,89) | там же, Table 1 | то же |
| Обучение на сегм., тест на ориг. (TrainS+TestN) | VeRi-776 (ТС) | 65,78 → 66,08 | **+0,30** | +0,60 | там же, Table 1 | то же |
| **Слабо-supervised тесная обрезка** | **CityFlow val (ТС!)** | 39,5 → **44,4** | **+4,9** | +1,3 (64,0→65,3) | [VOC-ReID CVPRW'20](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zhu_VOC-ReID_Vehicle_Re-Identification_Based_on_Vehicle-Orientation-Camera_CVPRW_2020_paper.pdf) **Table 2** | 1 forward своей же модели, офлайн |
| **Штраф за схожесть камеры (inference)** | CityFlow val (ТС) | 47,0 → **50,8** | **+3,8** | **+5,0** (70,5→75,5) | там же, **Table 3** | нужна camera-ReID модель + метки камер |
| Штраф за ориентацию (inference) | CityFlow val (ТС) | 44,4 → 47,0 | +2,6 | +5,2 (65,3→70,5) | там же, Table 3 | нужна orientation-модель |
| Camera Verification (post-proc.) | CityFlow Split-Test (ТС) | 66,8 → **72,7** | **+5,9** | +6,4 (66,9→73,3) | [arXiv:2105.09701](https://arxiv.org/pdf/2105.09701) **Table 4** | метки камер, 0 обучения |
| Inter-Camera Fusion (post-proc.) | CityFlow Split-Test (ТС) | 72,7 → 74,7 | +2,0 | +2,1 | там же, Table 4 | метки камер |
| Camera Verification (post-proc.) | CityFlow val (ТС) | 47,66 → 49,06 | +1,40 | — | [Zheng CVPRW'20](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zheng_Going_Beyond_Real_Data_A_Robust_Visual_Representation_for_Vehicle_CVPRW_2020_paper.pdf) Table 3 | метки камер |
| **Camera-specific feature norm** (test-time) | Market-1501 (unseen), TransReID | 43,1 → **52,5** | **+9,4** | +6,6 (69,5→76,1) | [arXiv:2502.10195](https://arxiv.org/pdf/2502.10195) **Table 3a** | **0 обучения**, нужны метки камер на тесте |
| Camera-specific feature norm | Market-1501 (unseen), CC | 22,5 → 30,0 | **+7,5** | +9,1 (47,3→56,4) | там же, Table 3a | 0 обучения |
| Camera-specific feature norm | Market-1501 (unseen), SOLIDER | 72,4 → 79,2 | **+6,8** | +2,7 (89,0→91,7) | там же, Table 3a | 0 обучения |
| 🔴 Camera-specific norm, **seen domain, supervised** | MSMT17 (seen), SOLIDER | 77,1 → 77,0 | **−0,1** | −0,1 | там же, Table 3a | — |
| **MixStyle (res12, random shuffle)** | Market→Duke, ResNet-50 | 19,3 → **23,8** | **+4,5** | +6,8 (35,4→42,2) | [arXiv:2104.02008](https://arxiv.org/pdf/2104.02008) **Table 2** | ~0, без меток |
| MixStyle (w/ domain label) | Duke→Market, ResNet-50 | 20,4 → **24,7** | **+4,3** | +7,8 (45,2→53,0) | там же, Table 2 | ~0, нужны метки домена |
| MixStyle (random shuffle) | Duke→Market, OSNet | 24,0 → 27,8 | **+3,8** | +5,9 (52,2→58,1) | там же, Table 2 | ~0 |
| 🔴 **MixStyle не в том месте (res1234)** | cross-dataset re-ID, ResNet-50 | 19,3 → **10,2** | **−9,1** | — | там же, **Table 3(b)** | ~0 |
| 🔴 DropBlock | Market→Duke, ResNet-50 | 19,3 → 18,2 | **−1,1** | −2,2 (35,4→33,2) | там же, Table 2 | ~0 |
| **VehicleX синтетика (attr. descent)** | **VeRi-776 (ТС)** | 66,54 → **70,62** | **+4,08** ✅ | +0,71 (92,73→93,44) | [arXiv:1912.08855](https://arxiv.org/pdf/1912.08855) **Table 7** | 192k изобр., 2-стадийное обучение |
| **VehicleX синтетика (attr. descent)** | **CityFlow (ТС)** | 30,21 → **37,16** | **+6,95** ✅ | +7,32 (56,75→64,07) | там же, **Table 8** | то же |
| VehicleX (random attributes) | VeRi-776 (ТС) | 66,54 → 69,28 | +2,74 | +0,48 | там же, Table 7 | то же |
| VehicleX синтетика | CityFlow-V2 Split-Test (ТС) | 36,0 → **46,2** | **+10,2** | **+13,1** (51,7→64,8) | [arXiv:2105.09701](https://arxiv.org/pdf/2105.09701) **Table 1** | то же |
| VehicleX синтетика | CityFlow val (ТС) | 43,87 → 46,90 | +3,03 | +1,08 (79,78→80,86) | [Zheng CVPRW'20](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zheng_Going_Beyond_Real_Data_A_Robust_Visual_Representation_for_Vehicle_CVPRW_2020_paper.pdf) Table 2 | то же |
| VehicleX 2-стадийно vs 1-стадийно | CityFlow (ТС) | 33,54 → 37,16 | **+3,62** | — | [arXiv:1912.08855](https://arxiv.org/pdf/1912.08855) Table 5 | обязательно! |
| DNDM (день↔ночь, архитектура) | DN-348 Day→Night (ТС) | 0,456 → 0,475 | +1,9 | +3,3 (0,674→0,707) | [Li CVPR'24](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Day-Night_Cross-domain_Vehicle_Re-identification_CVPR_2024_paper.pdf) Table 2 | метки день/ночь, смена архитектуры |
| DNDM glare suppression (NGS) один | DN-348 Day→Night (ТС) | 0,456 → 0,467 | +1,1 | +1,6 (0,674→0,690) | там же, Table 3 | то же |
| CutOut | Market-1501 | 84,7 (нет baseline) | н/д | — | [arXiv:2007.08779](https://arxiv.org/pdf/2007.08779) Table 3 | ~0 |
| 🔴 CutMix vs CutOut | Market-1501 | 84,7 → 82,4 | **−2,3** | −1,3 (94,1→92,8) | там же, Table 3 | ~0 |
| 🔴 CutMix vs CutOut | DukeMTMC | 74,5 → 72,0 | **−2,5** | −1,3 (87,1→85,8) | там же, Table 3 | ~0 |
| 🔴 Attentive CutMix vs CutOut | Market-1501 | 84,7 → 78,1 | **−6,6** | −2,5 (94,1→91,6) | там же, Table 3 | ~0 |
| **GridMask на re-ID** | — | — | **замера не найдено** | — | — | — |
| **RandAugment на re-ID** | — | — | **замера не найдено** | — | — | — |
| 🔴 AugMix на vehicle re-ID | CityFlow | «не работает» (без цифр) | н/д | — | [VOC-ReID CVPRW'20](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zhu_VOC-ReID_Vehicle_Re-Identification_Based_on_Vehicle-Orientation-Camera_CVPRW_2020_paper.pdf) §4.1 | — |
| **Copy&Paste (Mask R-CNN+inpaint)** | CityFlow (ТС) | **изолированного замера НЕТ** | н/д | — | [Zheng CVPRW'20](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zheng_Going_Beyond_Real_Data_A_Robust_Visual_Representation_for_Vehicle_CVPRW_2020_paper.pdf) §3.1 | 3 модели: MaskRCNN + DeepFill v2 + seamless cloning |

---

## §1. Random Erasing (arXiv:1708.04896)

### Точные числа из статьи Zhong, Table 6 (вербатим)

Гиперпараметры для re-ID: `p=0,5; s_l=0,02; s_h=0,2; r1=1/r2=0,3`, вход 256×128.
(Замечание: для классификации авторы берут `s_h=0,4` — для re-ID область стирания **вдвое меньше**.)

```
                    Market-1501      DukeMTMC-reID    CUHK03(lab)     CUHK03(det)
Method Model    RE  Rank-1   mAP     Rank-1   mAP     Rank-1  mAP     Rank-1  mAP
IDE   R-18      No  79.87  57.37     67.73  46.87     28.36  25.65    26.86  25.04
IDE   R-18      Yes 82.36  62.06     70.60  51.41     36.07  32.58    34.21  31.20
IDE   R-34      No  82.93  62.34     71.63  49.71     31.57  28.66    30.14  27.55
IDE   R-34      Yes 84.80  65.68     73.56  54.46     40.29  35.50    36.36  33.46
IDE   R-50      No  83.14  63.56     71.99  51.29     30.29  27.37    28.36  26.74
IDE   R-50      Yes 85.24  68.28     74.24  56.17     41.46  36.77    38.50  34.75
TriNet R-50     No  82.60  65.79     72.44  53.50     49.86  46.74    50.50  46.47
TriNet R-50     Yes 83.94  68.67     72.98  56.60     58.14  53.83    55.50  50.74
SVDNet R-50     No  84.41  65.60     76.82  57.70     42.21  38.73    41.85  38.24
SVDNet R-50     Yes 87.08  71.31     79.31  62.44     49.43  45.07    48.71  43.50
```

**Наблюдение, важное для нас:** прирост REA **тем больше, чем меньше датасет**.
CUHK03 (767 train ID) → +9,40 mAP для IDE-R50, против +4,72 на Market (751 ID, но
12 936 изображений против 7 368). То есть на малых данных REA работает как сильный
регуляризатор *внутри домена*.

### Bag of Tricks (arXiv:1903.07071) — сколько даёт REA отдельно

**Table 1 (в домене), вербатим:**
```
Model          Market1501 r=1  mAP    DukeMTMC r=1  mAP
Baseline-S          87.7      74.0        79.7     63.7
+warmup             88.7      75.2        80.6     65.1
+REA                91.3      79.3        81.5     68.3      <- вклад REA: +4.1 / +3.2 mAP
+LS                 91.4      80.3        82.4     69.3
+stride=1           92.0      81.7        82.6     70.6
+BNNeck             94.1      85.7        86.2     75.9
+center loss        94.5      85.9        86.4     76.4
```
Вклад REA отдельно (разница строк `+warmup` → `+REA`): **Market +4,1 mAP / +2,6 R-1;
Duke +3,2 mAP / +0,9 R-1**. Это второй по величине вклад после BNNeck.

### Вредит ли REA cross-domain generalization? — ДА, подтверждено дважды

**(a) Bag of Tricks, Table 3 (вербатим):**
```
                    M→D            D→M
Model          r = 1   mAP    r = 1   mAP
Baseline       24.4   12.9    34.2   14.5
+warmup        26.3   14.1    39.7   17.4
+REA           21.5   10.2    32.5   13.5     <- добавление REA: -3.9 mAP / -3.9 mAP
+LS            23.2   11.3    36.5   14.9
+stride=1      23.1   11.8    37.1   15.4
+BNNeck        26.7   15.2    47.7   21.6
+center loss   27.5   15.0    47.4   21.4
-REA           41.4   25.7    54.3   25.5     <- УБРАЛИ REA: +10.7 mAP / +4.1 mAP
```
Цитата авторов: *«However, REA does harm to models in cross-domain ReID task... We infer
that REA masking the regions of training images lets the model learn more knowledge in the
training domain. It causes the model to perform worse in [the unseen target domain].»*

Убирание REA из финальной модели даёт **M→D: mAP 15,0 → 25,7 (+10,7), Rank-1 27,5 → 41,4
(+13,9)**. Это крупнейший единичный эффект во всей их кросс-доменной таблице.

**(b) MixStyle (arXiv:2104.02008), Table 2 (вербатим):**
```
                                 Market1501→Duke          Duke→Market1501
Model                          mAP   R1   R5   R10      mAP   R1   R5   R10
ResNet-50                      19.3 35.4 50.3 56.4      20.4 45.2 63.6 70.9
+ RandomErase                  14.3 27.8 42.6 49.1      16.1 38.5 56.8 64.5   <- -5.0 / -4.3 mAP
+ DropBlock                    18.2 33.2 49.1 56.3      19.7 45.3 62.1 69.1
+ MixStyle w/ random shuffle   23.8 42.2 58.8 64.8      24.1 51.5 69.4 76.2
+ MixStyle w/ domain label     23.4 43.3 58.9 64.7      24.7 53.0 70.9 77.8
OSNet                          25.9 44.7 59.6 65.4      24.0 52.2 67.5 74.7
+ RandomErase                  20.5 36.2 52.3 59.3      22.4 49.1 66.1 73.0   <- -5.4 / -1.6 mAP
+ DropBlock                    23.1 41.5 56.5 62.5      21.7 48.2 65.4 71.3
+ MixStyle w/ random shuffle   27.2 48.2 62.7 68.4      27.8 58.1 74.0 81.0
+ MixStyle w/ domain label     27.3 47.5 62.0 67.1      29.0 58.2 74.9 80.9
```
Цитата: *«RandomErase shows a detrimental effect in the cross-dataset re-ID setting. Indeed,
similar to DropBlock, randomly erasing pixels offers no guarantee to improve the robustness
when it comes to domain shift.»*

**Вывод для нас.** Кросс-камерный разрыв — это domain shift в миниатюре. Обе работы измеряют
падение от REA именно в этом режиме. Наш первый и самый дешёвый эксперимент: **обучить
контрольную модель без REA и сравнить кросс-камерный Rank-1.** Стоимость — один прогон,
ожидаемый эффект по литературе — единицы-десятки пунктов mAP.
Оговорка: обе работы меряют кросс-*датасетный*, а не кросс-*камерный* перенос внутри одного
датасета. Прямого замера «REA vs кросс-камерный Rank-1 на одном датасете» я не нашёл.

---

## §2. Замена / удаление фона

### 2.1. Person: Tian et al., CVPR 2018 «Eliminating Background-bias» — ключевая работа

**Table 4 (Market-1501, только main-branch — изолирует эффект самой аугментации), вербатим:**
```
Aug. strategies    Top-1   Top-5   Top-10  Top-20
Main-net only      75.6    91.9    95.6    97.0
online (0.25)      79.2    93.7    96.3    97.9
online (0.5)       79.5    93.9    96.5    98.1   <- лучший, +3.9 top-1
online (0.75)      78.8    93.6    96.5    98.1
offline (1:1)      76.1    92.4    95.6    97.6   <- офлайн почти не работает
offline (1:2)      77.3    93.1    96.0    97.7
```
Три вывода, критичных для реализации:
1. **Online (замена фона на лету, каждую эпоху новый фон) даёт +3,9 top-1; offline
   (заранее сгенерированный фиксированный набор) — только +0,5.** Разница в 7,8×.
2. **Вероятность замены p некритична**: 0,25 / 0,5 / 0,75 укладываются в 1% (79,2 / 79,5 / 78,8).
   Цитата: *«Different replacing probability p of online strategies have little impact (1%)
   on the final results»*.
3. **Достаточно 100 фоновых картинок.** Цитата: *«one background image B will be randomly
   selected from 100 background images collected from real surveillance scenes»*.

**Стоимость (цитата из §3.1):** *«Before the training process, for one input image I_in, its
background-foreground binary mask M is first generated.»* — **маска считается ОДИН РАЗ
до обучения**, не каждую эпоху. В рантайме остаётся только композит `fg*M + bg*(1-M)`.
Плюс: *«we do not use segmentation annotation information on person ReID datasets.
State-of-art deep learning segmentation models are used to generate foreground-background
masks automatically»* — ручной разметки не нужно.

Table 1: полная модель 80,5 → с аугментацией 81,2 top-1 (+0,7) — на сильной модели прирост
меньше, чем на голом main-net (+3,9).

### 2.2. Vehicle: Wu et al., arXiv:1910.06613 «Background Segmentation for Vehicle Re-ID»

Единственная найденная работа с прямым замером удаления фона для **ТС**.

**Table 1 (VeRi-776), вербатим:**
```
Method          mAP     top1    top5    top10
Seg             57.53   81.70   82.01   94.76
Seg+Post        64.79   86.89   95.47   97.85
TrainS+TestN    66.08   86.89   94.87   97.14
Baseline        65.78   86.29   94.76   96.96
```
**Жёсткое удаление фона ХУЖЕ базовой линии на 8,25 mAP.** Даже с постобработкой (заливка дыр,
крупнейшая связная компонента) — всё ещё −0,99 mAP. Причина по авторам: несовершенная
сегментация (дыры в кузове, номера, блики).

**Table 2 (BIR — случайная смесь сегментированных и оригинальных), вербатим:**
```
k        mAP     top1    top5    top10
0.1     70.12   90.28   97.08   98.81
0.2     70.74   90.46   96.96   98.69    <- лучший: +4.96 mAP / +4.17 top1 над baseline
0.3     69.61   89.03   95.89   98.15
0.4     70.10   89.98   96.48   98.15
0.5     69.65   89.27   96.42   98.51
0.6     68.59   88.67   96.00   98.15
0.7     70.34   89.45   96.06   98.39
0.8     67.69   88.73   95.59   97.79
0.9     67.22   87.72   95.89   97.79
Baseline 65.78  86.29   94.76   96.96
```
**Все k от 0,1 до 0,9 бьют baseline.** Оптимум k=0,2 (20% изображений с удалённым фоном).
Это та же закономерность, что у Tian: *смешивать*, а не *заменять целиком*.

Детали реализации и стоимость:
- Сегментатор: **DilatedResNet-50 (encoder) + PPM (decoder)** + детектор границ + постобработка
  (flood fill → крупнейшая связная компонента).
- Пришлось **собрать свой датасет сегментации**: Vehicle-Segmentation = 27 896 изображений
  (2 686 CS-vehicle + 5 000 Cityscapes + 20 210 ADE20K). Это отдельный проект.
- Отбрасывают сегменты, где площадь машины < 0,60 кадра.
- ⚠️ **Random-k применяется и на train, и на test** (см. §4.3: *«Random-k randomly uses k percent
  images processed by Seg+Post and others original images for both training and test»*).
- Честное наблюдение авторов: *«background removal results for white vehicles are better than
  that for black vehicles, because the road is mostly dark gray»* — на тёмных машинах
  сегментация деградирует.

### 2.3. Vehicle, дёшево и без сегментатора: VOC-ReID (CVPRW 2020), 2-е место AI City 2020

**«Weakly supervised detection augmentation»:** обучают начальную re-ID модель → берут её
собственную heatmap-реакцию → порог → bbox → **более тесная обрезка**. Сегментационная сеть
не нужна вообще.

**Table 2 (CityFlow validation), вербатим:**
```
Data             mAP      Rank1
Real             29.7%    50.8%
Real+Syn         39.5%    64.0%
Real+Syn+Aug     44.4%    65.3%     <- тесная обрезка: +4.9 mAP
```
Цитата: *«weakly supervised data augmentation brings almost 5% gains because of relatively
loose cropping in CityFlow. The scale factor of vehicle and background would obviously
deteriorate the performance.»*

**Table 3 — прямая атака на связь «фон = камера», вербатим:**
```
Method          mAP      Rank1
RECT-Net        44.4%    65.3%
+Orientation    47.0%    70.5%
+Camera         50.8%    75.5%
```
Формула: `D = Dr − λ1·Dc − λ2·Do` (λ1=0,1, λ2=0,05), где Dc — расстояние camera-ReID модели,
Do — orientation. То есть **из финальной метрики вычитают «похожесть камеры»**.
Идея авторов дословно: *«the similarity of background can be interpreted as camera
re-identification and similarity of shape interpreted as orientation re-identification»*.
Итог +6,4 mAP / +10,2 Rank-1. У нас есть метки 96 камер — это применимо напрямую.

### 2.4. Copy-paste для ТС — измерения НЕТ

Zheng et al. CVPRW'20 (Baidu-UTS, 1-е место AI City 2020) описывают Copy&Paste:
Mask R-CNN (маска ТС) → DeepFill v2 (inpainting фона) → seamless image cloning.
**Изолированной абляции Copy&Paste в статье нет** — в Table 2 меряется только эффект
синтетики (43,87 → 46,90 mAP). Числа для Copy&Paste привести не могу.

---

## §3. Насколько сильно фон вообще влияет — количественные замеры

**Tian et al., CVPR 2018, Table 2 (вербатим):**
```
Method    Test on CUHK03        Top-1  Top-5  Top-10
DGD       original              87.2   97.4   98.3
DGD       mean-background       77.2   93.7   97.0     <- -10.0
DGD       random-background     64.0   85.5   91.7     <- -23.2
Ours      original              92.5   98.4   98.9
Ours      mean-background       91.7   97.7   98.6
Ours      random-background     91.1   97.6   98.5     <- всего -1.4

Method    Test on Market-1501   Top-1  Top-5  Top-10
DGD       original              79.1   94.1   96.6
DGD       mean-background       71.1   90.3   94.2     <- -8.0
DGD       random-background     58.4   80.1   86.4     <- -20.7
Ours      original              81.2   94.6   97.0
Ours      mean-background       80.5   93.9   98.2
Ours      random-background     79.8   93.5   98.0     <- всего -1.4
```

**Это прямой измеритель background bias.** Обычная модель, обученная на оригинальных кадрах,
при подмене фона на тесте теряет **20,7–23,2 пункта Rank-1**. Модель, обученная с
random-background аугментацией, теряет **1,4 пункта**.

Дополнительные числа из §2.2 той же статьи (вербатим по тексту):
- *«the top-1 accuracies decreases by 10.3% on the mean-background dataset and by 28.4% on
  the random-background dataset»* (CUHK03).
- **Тест на «только фон» (объект залит средним пикселем): top-1 = 5,3%** — *«much higher than
  random guess»*. То есть **один фон, без машины/человека, несёт опознаваемый сигнал личности**.
- Цена устойчивости: *«There are 2%-3% performance drop on the original test sets compared with
  the deep model trained with original images»* — обучение на random-background стоит 2–3 пункта
  на «родном» тесте, но убирает 20+ пунктов провала при смене фона. **Ровно наш размен.**

**Camera bias, Song et al., ICLR 2025 (arXiv:2502.10195), Table 3** — измеряет смещение по
камерам через NMI кластеризации (чем выше, тем сильнее модель группирует по камере, а не по ID):
Market-1501 ground truth bias = 6,4; CC = 17,1; PPLR = 15,6; TransReID = 13,6.
Их лекарство — **camera-specific feature normalization на этапе теста** (вычесть покамерное
среднее, поделить на покамерное std), приросты на unseen domain:
CC 22,5→30,0 mAP; TransReID 43,1→52,5; TransReID-SSL 53,6→62,3; SOLIDER 72,4→79,2; PAT 43,8→52,9.

> ⚠️ **Важная оговорка против наивного применения у нас.** Цитата: *«For the seen domain, the
> camera-agnostic unsupervised models show slight improvement, while the camera-aware or
> supervised models exhibit no improvement.»* Наша модель — supervised на своих же 96 камерах,
> т.е. это «seen domain, supervised» — в таблице это красная зона (SOLIDER на MSMT17 seen:
> 77,1 → 77,0). Ожидаемый прирост у нас **близок к нулю**. Но стоимость эксперимента — 20 строк
> кода и один прогон инференса, поэтому проверить дёшево.

---

## §4. CutMix / Mixup / GridMask / Cutout на re-ID

Изолированных абляций «baseline → +CutMix» на re-ID я **не нашёл**. Единственное прямое
сравнение с одинаковым бэкбоном — PMM (arXiv:2007.08779), **Table 3, вербатим**:
```
Methods       Market-1501     DukeMTMC      CUHK03
              Rank-1  mAP     Rank-1  mAP   Rank-1  mAP
CutOut         94.1   84.7     87.1  74.5    73.5  69.8
CutMix         92.8   82.4     85.8  72.0    74.3  69.9
A-CutMix       91.6   78.1     83.8  68.7    65.6  62.8
A-Hard-Mix     94.1   85.2     88.0  74.6    76.3  73.0
```
⚠️ **Строки «без аугментации» в таблице нет**, поэтому Δ относительно чистого baseline
посчитать нельзя. Можно утверждать только относительное: **CutMix хуже CutOut на 2,3 mAP
(Market) и 2,5 mAP (Duke)**; Attentive CutMix хуже CutOut на 6,6 mAP (Market).

Объяснение авторов: *«it can be hard to use the limited features to make precise prediction,
thus the mixed label can bring some misleading information in the training process»*.
Фундаментальная причина для нас: **CutMix/Mixup производят дробные метки, а triplet loss
их не принимает** — придётся либо отключать triplet на смешанных батчах, либо переделывать лосс.

- **GridMask на re-ID: замера не найдено.**
- **Mixup на re-ID с чистым baseline: замера не найдено.**
- Cutout в re-ID фактически = Random Erasing с постоянным значением заливки; см. §1.

---

## §5. Стилевые аугментации против доменного сдвига

**MixStyle (arXiv:2104.02008), Table 2** — числа в §1(b) выше. Сводка приростов над ванильным
бэкбоном: ResNet-50 **M→D +4,5 mAP / +6,8 R1**, **D→M +3,7…+4,3 mAP / +6,3…+7,8 R1**;
OSNet M→D +1,3 mAP, D→M +3,8 mAP.

**Table 3(b) — куда ставить MixStyle (cross-dataset re-ID), вербатим:**
```
Model                    mAP
ResNet-50               19.3
+ MixStyle (res1)       22.6
+ MixStyle (res12)      23.8    <- оптимум для re-ID
+ MixStyle (res123)     22.0
+ MixStyle (res1234)    10.2    <- КАТАСТРОФА: -9.1 mAP от baseline
+ MixStyle (res14)      11.1
+ MixStyle (res23)      20.6
```
Цитата: *«on the re-ID datasets res12 is the best... the performance plunges when applying
MixStyle to the last residual block»*. **Неправильно поставленный MixStyle хуже, чем его
отсутствие, почти вдвое.** Ставить только после res1 и res2.

Стоимость: *«MixStyle can be implemented with only few lines of code»*, параметров нет,
вероятность активации 0,5, *«At test time, no MixStyle is applied»*, градиенты через µ/σ
заблокированы. Версия `random shuffle` **не требует меток домена** (23,8 против 23,4 с метками —
практически одинаково). У нас есть естественные метки домена — ID камеры.

- **AugMix на re-ID: числового замера не найдено.** Есть только качественное отрицательное
  утверждение для ТС (см. «Отрицательные результаты»).
- **RandAugment на re-ID: замера не найдено.**

---

## §6. Погода / освещение

**arXiv:2607.10583 «Benchmarking UAV-based Vehicle Re-Identification under Simulated Weather
Conditions»** — деградация по погоде, Tables II и III:

| Метод | VRU: норма | туман | дождь | UAV-VeID: норма | туман | дождь |
|---|---|---|---|---|---|---|
| CLIP-ReID R50 | 95,2/92,3 | 91,2/86,5 | 86,8/81,6 | 88,1/83,7 | 83,5/77,7 | 72,4/66,9 |
| CLIP-ReID ViT-B/16 | 91,4/86,3 | 81,7/72,7 | 78,7/70,4 | 80,1/73,0 | 70,4/61,4 | 63,3/56,3 |
| MSINet | 93,8/90,0 | 89,9/84,5 | 84,1/77,9 | 87,7/82,6 | 82,5/76,0 | 71,0/64,8 |
| AdaSP R50+IBN | 95,9/93,3 | 93,0/89,2 | 88,5/83,7 | 92,7/89,4 | 88,7/84,1 | 76,2/70,9 |

(формат mAP/Rank-1)

Деградация: **туман −2,9…−9,7 mAP; дождь −7,4…−16,8 mAP.** Дождь бьёт сильнее тумана везде.
ViT-бэкбоны деградируют заметно сильнее CNN (CLIP-ReID ViT теряет 9,7 mAP на тумане VRU против
4,0 у R50). Симуляция: LIME (оценка освещённости) + **MiDaS (оценка глубины)** + Бер-Ламберт для
тумана + гауссов шум → motion blur → alpha-blending для дождя.

> ⚠️ **Дают ли аугментации погоды измеримый прирост — в этой статье НЕ измеряется.**
> Цитата: *«In our experiments, models are trained and evaluated within the same weather
> condition»*, и в ограничениях: *«the condition-matched training and evaluation protocol does
> not assess cross-weather robustness or domain-shift generalization. Future work may explore
> clean-to-weather and weather-to-weather transfer»*. **Числа прироста от погодной аугментации
> отсутствуют.**

**Day-Night (Li et al., CVPR 2024), Table 2/3** — архитектурное решение, не аугментация:
DN-348 Day→Night baseline 0,674/0,456 → DNDM 0,707/0,475 (**+1,9 mAP**);
Night→Day 0,723/0,439 → 0,803/0,462 (**+2,3 mAP, +8,0 R-1**).
Модуль подавления бликов фар (NGS) в одиночку: 0,456 → 0,467 mAP (+1,1).
Требует меток день/ночь и изменения бэкбона.

---

## §7. Синтетика VehicleX

### Проверка твоих чисел — ОБА ПОДТВЕРЖДЕНЫ ✅

**Table 7 (VeRi-776), вербатим:**
```
Experiment        Method          Data   Rank-1  Rank-5   mAP
Joint Training    IDE             R      92.73   96.78   66.54
                  Ran. Attr.      R+S    93.21   96.20   69.28
                  Attr. Desc.     R+S    93.44   97.26   70.62     <- 70.62 - 66.54 = +4.08 ✅
                  Attr.Desc.(PCB) R+S    94.34   97.91   74.51
```
**+4,08 mAP на VeRi — подтверждено точно.**

**Table 8 (CityFlow), вербатим:**
```
Method            Data   R-1     R-20    mAP
BA                R      49.62   80.04   25.61
BS                R      49.05   78.80   25.57
PAMTRI            R+S    59.7    80.13   33.81
IDE(CE+Tri.)      R      56.75   72.24   30.21
Ran. Attr.        R+S    63.59   82.60   35.96
Attr. Desc.       R+S    64.07   83.27   37.16     <- 37.16 - 30.21 = +6.95 ✅
```
**+6,95 mAP на CityFlow — подтверждено точно.**

Дополнительно: на VehicleID (Large) 78,51 → 81,35 mAP (+2,84), Table 3.
Независимые замеры того же эффекта: +10,2 mAP (arXiv:2105.09701 Table 1, CityFlow-V2
Split-Test, 36,0 → 46,2) и +3,03 mAP (Zheng CVPRW'20 Table 2, 43,87 → 46,90).

### Сколько синтетики надо и как её применять

- **Объём: 1 362 ID / 192 150 изображений** (подтверждено в arXiv:2105.09701 §3 и VOC-ReID §4.2).
- **У нас 7 248 реальных кадров. Отношение синтетика:реальность = 26:1.** Это опасно.
  Прямое подтверждение риска — arXiv:2004.10547: *«these two solutions do not work in the
  challenge. Because the number of data in D_S is much larger than the one in D_R, Solution-1
  will cause the model to be more biased towards D_S»* (Solution-1 = просто смешать; Solution-2 =
  предобучить на синтетике и дообучить на реальных).
- **Обязательно двухстадийное обучение**, Table 5 (вербатим):
  ```
             VehicleID   VeRi    CityFlow
  Stage I      77.54     69.39    33.54
  Stage II     81.35     70.62    37.16      <- +3.62 mAP на CityFlow
  ```
  Stage I — обучение на смеси real+synthetic с общим классификатором (333+1362 = 1695 классов
  для CityFlow); Stage II — замена классификатора на новый и дообучение только на реальных.
- **Attribute descent vs random attributes**: +1,34 mAP на VeRi (69,28 → 70,62), +1,20 на
  CityFlow (35,96 → 37,16). То есть львиная доля выигрыша — от самого факта синтетики,
  а подгонка атрибутов добавляет ~1–1,3 mAP.
- **Style DA (SPGAN) критичен только при обучении на одной синтетике**, Table 2:
  VehicleID 24,36 → 35,33 mAP; VeRi 12,35 → 21,29 mAP.

### Лицензия и реалистичность скачивания

- **Лицензии в репозитории НЕТ.** GitHub Licenses API для `yorkeyao/VehicleX` возвращает 404,
  поле `license` = `None`. В README есть только фраза про код: *«You are welcomed to use every
  single part of the code for your research purpose»* — это (а) про код, не про данные,
  (б) research purpose. **Для коммерческого проекта юридический статус неясен — нужен запрос
  к авторам (yue.yao@anu.edu.au).**
- **Скачивание реалистично для VeRi-776 и VehicleID**: готовые адаптированные (content+style)
  наборы лежат прямыми ссылками на Google Drive и Baidu.
  **CityFlow-версия закрыта** — нужна регистрация на aicitychallenge.org.
- Формат имён `id_cam_num.jpg` (например `0001_c001_33.jpg`) — **ID камеры уже в имени файла**.
- В XML идут метки: ориентация, интенсивность света, направление света, расстояние до камеры,
  высота камеры, тип ТС, цвет ТС.
- Опубликован Unity-проект и модели в .fbx → **можно рендерить своё, в том числе на своих фонах**,
  но это отдельный крупный проект (Unity + ML-Agents).
- Размер репозитория ~119 МБ (без наборов изображений); сами наборы — отдельные архивы, их размер
  в README не указан.

---

## Требует данных / меток, которых у нас нет

| Приём | Чего не хватает | Можно ли обойти |
|---|---|---|
| Замена фона (Tian, BIR) | Маски ТС. Нет ни масок, ни датасета сегментации ТС | **Да**: готовые COCO-модели (класс `car`/`truck`/`bus`) дают маску без разметки. Tian прямо пишет, что использует автоматические сегментаторы без ручной разметки |
| Замена фона | Банк фоновых изображений | **Да, дёшево**: Tian хватило **100** кадров с реальных камер наблюдения. У нас 96 камер — можно взять медиану/фон каждой камеры |
| BIR (arXiv:1910.06613) | Датасет сегментации ТС (они собрали 27 896 изобр.) | **Да**: брать готовую COCO/Cityscapes-модель вместо обучения своей |
| Штраф за камеру (VOC-ReID), Camera Verification | Метки камер на train и test | **Есть** — 96 камер известны |
| Штраф за ориентацию (VOC-ReID) | Метки ориентации ТС | Нет. Либо обучать классификатор ориентации, либо взять метки из VehicleX (там 36 бинов) |
| Camera-specific feature norm | Метки камер на тесте + ≥несколько образцов на камеру | **Есть** |
| MixStyle w/ domain label | Метки домена | **Есть** (камера как домен). Но `random shuffle` вообще без меток даёт почти то же (23,8 vs 23,4) |
| VehicleX | 192k синтетических изображений + двухстадийный пайплайн + **неясная лицензия** | Скачать VeRi-адаптированную версию можно; лицензию надо выяснять |
| Day-Night DNDM | Метки день/ночь + смена архитектуры | Частично: время суток обычно есть в метаданных кадра |
| Погодная симуляция (2607.10583) | Карты глубины (MiDaS) + LIME | Модели публичные, но это ещё одна сеть на каждый кадр |
| Copy&Paste (Zheng) | Mask R-CNN + DeepFill v2 + seamless cloning | Тяжело, и **изолированного замера выигрыша нет** |

---

## Отрицательные результаты

1. **Random Erasing разрушает кросс-доменный перенос.**
   - BoT Table 3: удаление REA даёт **M→D +10,7 mAP / +13,9 R-1**, D→M +4,1 mAP / +6,9 R-1.
     Добавление REA поверх warmup: −3,9 mAP на обоих направлениях.
     [arXiv:1903.07071](https://arxiv.org/pdf/1903.07071)
   - MixStyle Table 2: −5,0 / −4,3 mAP (ResNet-50), −5,4 / −1,6 mAP (OSNet).
     [arXiv:2104.02008](https://arxiv.org/pdf/2104.02008)
   - При этом **в домене** REA даёт +2,9…+9,4 mAP. Это и есть ловушка: метрика внутри камеры
     растёт, между камерами падает — в точности наш симптом (0,976 vs 0,379).

2. **Жёсткое удаление фона у ТС хуже, чем ничего.** VeRi-776: baseline 65,78 → Seg 57,53 mAP
   (**−8,25**); Seg+Post 64,79 (**−0,99**). Работает только вероятностная смесь (k=0,2 → +4,96).
   [arXiv:1910.06613](https://arxiv.org/pdf/1910.06613) Table 1 vs Table 2.

3. **Offline-замена фона почти бесполезна против online.** Market-1501: offline 1:1 даёт
   +0,5 top-1, online p=0,5 — +3,9 top-1. Заранее сгенерированный фиксированный набор
   вариантов не работает; нужна свежая случайная подстановка каждую эпоху.
   [Tian CVPR'18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.pdf) Table 4.

4. **MixStyle в неправильном слое хуже отсутствия MixStyle почти вдвое.** res1234 → 10,2 mAP
   против baseline 19,3 (**−9,1**); res14 → 11,1. Только res1/res12 полезны.
   [arXiv:2104.02008](https://arxiv.org/pdf/2104.02008) Table 3(b).

5. **CutMix и Attentive CutMix хуже простого CutOut на re-ID.** Market −2,3 / −6,6 mAP,
   Duke −2,5 / −5,8 mAP. Причина — дробные метки конфликтуют с метрическим обучением.
   [arXiv:2007.08779](https://arxiv.org/pdf/2007.08779) Table 3.

6. **DropBlock не даёт ничего в кросс-домене**: 19,3 → 18,2 mAP (M→D), 20,4 → 19,7 (D→M).
   [arXiv:2104.02008](https://arxiv.org/pdf/2104.02008) Table 2.

7. **AugMix, random patch, random rotate/scale, random blur не работают на vehicle re-ID.**
   Дословно: *«We also try other data augmentation methods like random patch, augmix [29],
   random rotate/scale and random blur, but all of them turn out not working well on this task.»*
   Работали только: random horizontal flip, random crop, color jitter, random erase.
   ⚠️ Числа для отрицательных приёмов **не приводятся**.
   [VOC-ReID CVPRW'20](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zhu_VOC-ReID_Vehicle_Re-Identification_Based_on_Vehicle-Orientation-Camera_CVPRW_2020_paper.pdf) §4.1.

8. **Наивное подмешивание синтетики проваливается при большом перекосе объёмов.**
   *«these two solutions do not work in the challenge. Because the number of data in D_S is much
   larger than the one in D_R, Solution-1 will cause the model to be more biased towards D_S.»*
   Спасает только двухстадийная схема (+3,62 mAP, VehicleX Table 5).
   [arXiv:2004.10547](https://arxiv.org/pdf/2004.10547).

9. **Camera-specific normalization не помогает supervised-моделям на своих (seen) камерах.**
   *«the camera-aware or supervised models exhibit no improvement»*; SOLIDER на MSMT17 seen:
   77,1 → 77,0 mAP. Все крупные приросты (+6,8…+9,4) — только на unseen domain.
   [arXiv:2502.10195](https://arxiv.org/pdf/2502.10195) Table 3.

10. **Замер «аугментация погоды помогает» отсутствует.** Единственный benchmark по погоде для
    ТС (arXiv:2607.10583) обучает и тестирует в одинаковых условиях и явно перечисляет
    cross-weather transfer в нерешённых задачах.

11. **Про «тяжёлые аугментации вредят на малом датасете» — прямого замера на re-ID не найдено.**
    Более того, данные Zhong Table 6 указывают в **обратную** сторону: на самом малом датасете
    (CUHK03, 767 train ID) REA даёт наибольший внутридоменный прирост (+9,40 mAP для IDE-R50
    против +4,72 на Market). Утверждение «сильная аугментация вредит малым данным» я
    подтвердить числами не могу; подтверждается другое — **сильная аугментация типа
    стирания вредит переносу между доменами/камерами независимо от размера датасета**.

---

## §9. Цена по времени

Опорный замер для сегментации на CPU: **DeepLabv3+ с MobileNetV3-large-0,5x = 135 мс/кадр**,
PP-LCNet-0,5x = 82 мс, MobileNetV3-large-0,75x = 151 мс — на Intel Xeon Gold 6148, batch=1,
MKLDNN, 10 потоков, Cityscapes val ([arXiv:2109.15099](https://arxiv.org/pdf/2109.15099) Table 5).
Это на разрешении Cityscapes (≈1024×512 ≈ 524 тыс. пикселей). Наши кропы 208×208 = 43 тыс.
пикселей, т.е. **≈12× меньше** → *оценочно* **10–20 мс/кадр** (экстраполяция по пикселям,
не замер — реальную цифру надо мерить).

| Приём | Разовая стоимость | Стоимость за эпоху | Вердикт |
|---|---|---|---|
| Убрать Random Erasing | 0 | 0 (станет быстрее) | **Бесплатно.** Один контрольный прогон |
| MixStyle (res1+res2) | ~10 строк кода | ≈0 (две статистики на канал, без параметров; на тесте выключен) | **Практически бесплатно** |
| Random Erasing | 0 | ≈0 (numpy-заливка прямоугольника) | Бесплатно, но см. отрицательные результаты |
| **Маски ТС для замены фона** | 7 248 × ~15 мс ≈ **2 минуты CPU** одним проходом; на GPU — секунды | **0** — маски считаются один раз и кешируются на диск (~7 248 PNG 208×208 ≈ 30 МБ) | **Дёшево.** Главный миф — что сегментатор нужен каждый кадр каждую эпоху; у Tian он нужен один раз |
| Замена фона online | сбор ~100 фонов (минуты) | композит `fg*M + bg*(1−M)` на 208×208 ≈ **десятки микросекунд**/кадр, укладывается в DataLoader | **Дёшево**, лучший кандидат |
| Слабо-supervised тесная обрезка (VOC-ReID) | 7 248 forward своей же модели: на GPU секунды, на CPU ~2–5 мин | 0 (кропы кешируются) | **Дёшево**, сегментатор не нужен |
| Camera Verification / штраф за камеру | обучить camera-ReID модель (те же 7 248 кадров, 96 классов) | 0 — только инференс | **Дёшево**, метки камер уже есть |
| Camera-specific feature norm | ~20 строк | 0 — чисто test-time | **Бесплатно**, но ожидаемый эффект у нас ~0 (см. отрицательный результат №9) |
| CutMix/Mixup | переделка triplet loss под дробные метки | ≈0 вычислительно | Дорого по инженерии, отрицательный замер |
| **VehicleX** | скачать ~192k изображений (несколько ГБ) + выяснить лицензию | **+192k изображений в Stage I** — обучение вырастет в ~26 раз по объёму данных; обязательны 2 стадии | **Дорого**, но самый большой измеренный Δ (+4,08…+10,2 mAP) |
| Погодная симуляция | MiDaS (depth) + LIME на каждый кадр — ещё одна сеть, тяжелее сегментации | 0 если кешировать | Средне, **и выигрыш не измерен** |
| Copy&Paste (Mask R-CNN + DeepFill v2 + cloning) | три модели, inpainting генеративный и медленный | 0 если офлайн | **Дорого**, и изолированного замера нет |
| BIR-сегментатор (DilatedResNet-50 + PPM) | тяжелее MobileNet-варианта в разы | 0 если кешировать | Средне; брать MobileNet-DeepLab вместо их модели |

**Ключевой вывод по стоимости:** сегментация фона **не** является «моделью сегментации на каждый
кадр во время обучения». И Tian (*«Before the training process... mask M is first generated»*),
и схема BIR считают маски **офлайн, один раз**. Для 7 248 кадров 208×208 это единицы минут CPU.
Дорогая часть — не сегментация, а VehicleX.

---

## Ранжирование по «Δ на единицу затрат»

1. **Убрать REA / обучить контрольную модель без него** — Δ до +10,7 mAP кросс-домен, цена 0.
2. **MixStyle после res1 и res2** — +3,7…+4,5 mAP кросс-домен, цена ≈0, меток не нужно.
3. **Онлайн-замена фона, p≈0,2–0,5** — +3,9 top-1 (person) / +4,96 mAP (VeRi, k=0,2);
   маски офлайн ≈2 мин CPU, 100 фонов.
4. **Штраф за схожесть камеры на инференсе** — +3,8…+5,9 mAP, метки камер уже есть.
5. **Тесная обрезка по собственной heatmap модели** — +4,9 mAP, без сегментатора.
6. **VehicleX** — +4,08…+10,2 mAP, но дорого и лицензия неясна.
7. Camera-specific feature norm — бесплатно попробовать, но ожидаемый Δ у нас ≈0.
8. CutMix/Mixup/GridMask/AugMix/RandAugment — **не рекомендую**: либо отрицательные
   замеры, либо замеров нет вовсе.

---

## Источники

- Zhong et al., Random Erasing Data Augmentation — https://arxiv.org/pdf/1708.04896 (Table 6)
- Luo et al., Bag of Tricks and A Strong Baseline — https://arxiv.org/pdf/1903.07071 (Tables 1, 3)
- Zhou et al., Domain Generalization with MixStyle, ICLR 2021 — https://arxiv.org/pdf/2104.02008 (Tables 2, 3)
- Tian et al., Eliminating Background-bias for Robust Person Re-ID, CVPR 2018 — https://openaccess.thecvf.com/content_cvpr_2018/papers/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.pdf (Tables 1, 2, 4)
- Wu et al., Background Segmentation for Vehicle Re-Identification — https://arxiv.org/pdf/1910.06613 (Tables 1, 2, 3)
- Zhu et al., VOC-ReID, CVPRW 2020 — https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zhu_VOC-ReID_Vehicle_Re-Identification_Based_on_Vehicle-Orientation-Camera_CVPRW_2020_paper.pdf (Tables 2, 3, §4.1)
- Zheng et al., Going Beyond Real Data, CVPRW 2020 — https://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Zheng_Going_Beyond_Real_Data_A_Robust_Visual_Representation_for_Vehicle_CVPRW_2020_paper.pdf (Tables 2, 3)
- Yao et al., Simulating Content Consistent Vehicle Datasets with Attribute Descent, ECCV 2020 — https://arxiv.org/pdf/1912.08855 (Tables 2, 3, 5, 7, 8)
- Luo et al., An Empirical Study of Vehicle Re-ID on the AI City Challenge — https://arxiv.org/pdf/2105.09701 (Tables 1, 4)
- Song et al., Exploring the Camera Bias of Person Re-identification, ICLR 2025 — https://arxiv.org/pdf/2502.10195 (Table 3)
- He et al., Multi-Domain Learning and Identity Mining — https://arxiv.org/pdf/2004.10547
- Zhao et al., Progressive Multi-stage Feature Mix (PMM) — https://arxiv.org/pdf/2007.08779 (Table 3)
- Li et al., Day-Night Cross-domain Vehicle Re-identification, CVPR 2024 — https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Day-Night_Cross-domain_Vehicle_Re-identification_CVPR_2024_paper.pdf (Tables 2, 3)
- Benchmarking UAV-based Vehicle Re-ID under Simulated Weather Conditions — https://arxiv.org/abs/2607.10583 (Tables II, III)
- Cui et al., PP-LCNet: A Lightweight CPU Convolutional Neural Network — https://arxiv.org/pdf/2109.15099 (Table 5, латентность CPU)
- VehicleX repo (лицензия/скачивание) — https://github.com/yorkeyao/VehicleX
