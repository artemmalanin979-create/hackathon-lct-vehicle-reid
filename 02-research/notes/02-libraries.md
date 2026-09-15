# Обзор библиотек re-identification для vehicle re-id (состояние на 15.09.2026)

Критерии задачи: обучение/дообучение на своём датасете за ~2 недели, экспорт в ONNX, офлайн-инференс, коммерческое использование.

Методика: каждая дата/версия/лицензия проверена открытием источника через WebFetch (GitHub API `api.github.com/repos/...` даёт `pushed_at` — дату последнего пуша в любую ветку; страницы commits — дату последнего коммита в основной ветке; PyPI JSON API — версии и даты загрузки). Непроверенные утверждения помечены [НЕ ПРОВЕРЕНО].

## Сводная таблица

| Библиотека | Последняя активность | Версия | PyTorch | Лицензия | ONNX-экспорт | Vehicle-датасеты | Оценка «за 2 недели» |
|---|---|---|---|---|---|---|---|
| **fast-reid** (JDAI-CV) | коммит в master 15.07.2024; push 30.07.2024 | GitHub-релиз v1.3.0 (31.05.2021); PyPI `fastreid` 1.4.0 (15.02.2023) | заявлен ≥1.6 (пример установки — 1.6.0); requirements без пинов, свежие зависимости ломаются (issue #758) | Apache-2.0 | Да: `tools/deploy/onnx_export.py` + `trt_export.py` + Caffe | Да, из коробки: конфиги `VeRi`, `VehicleID`, `VERIWild` | **Да, с оговорками**: код заброшен 2+ года, потребуется день-два на пиннинг окружения (torch/numpy) |
| **torchreid / deep-person-reid** | коммит 09.01.2026, но это только README; код реально не менялся с ~02.2023 | теги GitHub до v1.0.6; PyPI `torchreid` 0.2.5 — **чужая переупаковка**, не официальная | инструкции древние (cudatoolkit 9.0); открыт баг с torch 2.6 (`weights_only`, issue #592) | MIT | Да: `tools/export.py` (ONNX/OpenVINO/TFLite, добавлен 08.2022); открыты баги экспорта #584/#585 | **Нет** — только person/video датасеты; vehicle придётся писать свой dataset-класс | **Условно**: как источник OSNet — да; полный vehicle-пайплайн — впритык, плюс патчи под torch 2.6+ |
| **reid-strong-baseline** | push 23.04.2020 — мёртв 6+ лет | нет релизов [НЕ ПРОВЕРЕНО наличие тегов] | pytorch>=0.4 (эпоха 2019) | MIT | Нет | Нет (Market1501, DukeMTMC) | **Нет** — не поднимать; идеи живут в fast-reid/TransReID |
| **PaddleClas (PP-ShiTu)** | push в репо 15.09.2026, но в release/2.6 последний коммит 01.07.2025 (доки); режим поддержки | v2.6.0 (GitHub и PyPI, 05.11.2024) | **Не PyTorch** — PaddlePaddle | Apache-2.0 | Да, через Paddle2ONNX | Частично: `configs/Vehicle/ResNet50_ReID.yaml` и `PPLCNet_2.5x_ReID.yaml` (датасет-классы `VeriWild`, `CompCars`); VeRi-776 — через конвертацию разметки в txt-списки | **Да, если принять стек Paddle**; для PyTorch-команды — потеря времени на экосистему |
| **mmpretrain / OpenMMLab** | push 01.11.2024; PyPI-релиз 1.2.0 от 04.01.2024 | 1.2.0 | torch 1.8+, но жёсткий пин `mmcv>=2.0,<2.4` тянет назад | Apache-2.0 | Через mmdeploy (сам mmdeploy: push 30.09.2024) | **Нет**: из metric learning только `arcface` (InShop); reid был в mmtracking — мёртв (push 19.09.2023) | **Нет** для vehicle re-id — нет задачи из коробки, стек стагнирует |
| **BoxMOT** (mikel-brostrom) | push 15.09.2026; релиз v25.0.0 от 09.09.2026 — очень живой | 25.0.0 (PyPI, 09.09.2026) | `torch>=2.2.1,<3.0.0`, Python 3.10–3.13 | **AGPL-3.0** | Да, лучший в списке: `boxmot export` → ONNX, TensorRT, OpenVINO, CoreML, TFLite | Частично: `train-reid` заявлен «для person или vehicle»; встроены Market1501/Duke/CUHK03/MSMT17, свой датасет — через YAML (path/train/query/gallery); VeRi лёг бы как Market-подобная структура | **Да, технически проще всех**, но AGPL — стоп-фактор для закрытого коммерческого продукта |
| **roboflow/trackers** | push 14.09.2026 — живой | версия не зафиксирована в README | torch опционален (только для McByte) | Apache-2.0 | — | **Нет reid вообще**: «appearance/ReID branches are not included» | **Нет** — это не reid-библиотека |
| **TransReID** (damo-cv) | push 12.06.2024; по сути исследовательский код 2021 г. | нет релизов [НЕ ПРОВЕРЕНО] | ≥1.6 (эксперименты на 1.6.0) | MIT | Нет | **Да**: VeRi-776 + VehicleID, есть претрейны для обоих | **Условно**: обучить можно, но деплой (ONNX для ViT+SIE) и окружение — на вас |
| **CLIP-ReID** (Syliz517) | push 21.11.2023 — мёртв ~3 года | нет релизов [НЕ ПРОВЕРЕНО] | pytorch==1.8.0 | MIT | Нет | **Да**: VeRi-776 + VehicleID, претрейны | **Скорее нет** — сильные веса, но код заморожен на torch 1.8 |
| **layumi/Person_reID_baseline_pytorch** | push 30.08.2026; DinoV3 добавлен 16.02.2026 — **живой** | нет формальных релизов [НЕ ПРОВЕРЕНО] | тестировался вплоть до torch 2.0+, новые фичи требуют >1.7 | MIT | Нет из коробки (в README — TensorRT/JIT); модели простые, ручной `torch.onnx.export` тривиален [НЕ ПРОВЕРЕНО на практике] | Частично: VeRi поддержан «с правками `prepare.py`/`test.py`» (README, issue #107) | **Да** — простой живой MIT-бейзлайн; ONNX руками |

Источники по строкам (все открыты 15.09.2026):
- fast-reid: [ПРОВЕРЕНО: https://api.github.com/repos/JDAI-CV/fast-reid], [ПРОВЕРЕНО: https://github.com/JDAI-CV/fast-reid/commits/master], [ПРОВЕРЕНО: https://api.github.com/repos/JDAI-CV/fast-reid/releases/latest], [ПРОВЕРЕНО: https://pypi.org/pypi/fastreid/json], [ПРОВЕРЕНО: https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/INSTALL.md], [ПРОВЕРЕНО: https://github.com/JDAI-CV/fast-reid/tree/master/tools/deploy], [ПРОВЕРЕНО: https://github.com/JDAI-CV/fast-reid/tree/master/configs], [ПРОВЕРЕНО: https://api.github.com/repos/JDAI-CV/fast-reid/issues?state=all&sort=created&direction=desc&per_page=12]
- torchreid: [ПРОВЕРЕНО: https://api.github.com/repos/KaiyangZhou/deep-person-reid], [ПРОВЕРЕНО: https://github.com/KaiyangZhou/deep-person-reid/commits/master], [ПРОВЕРЕНО: https://pypi.org/pypi/torchreid/json], [ПРОВЕРЕНО: https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/master/README.rst], [ПРОВЕРЕНО: https://github.com/KaiyangZhou/deep-person-reid/tree/master/tools], [ПРОВЕРЕНО: https://api.github.com/repos/KaiyangZhou/deep-person-reid/tags], [ПРОВЕРЕНО: https://api.github.com/search/issues?q=repo:KaiyangZhou/deep-person-reid+torch]
- reid-strong-baseline: [ПРОВЕРЕНО: https://api.github.com/repos/michuanhaohao/reid-strong-baseline], [ПРОВЕРЕНО: https://github.com/michuanhaohao/reid-strong-baseline]
- PaddleClas: [ПРОВЕРЕНО: https://api.github.com/repos/PaddlePaddle/PaddleClas], [ПРОВЕРЕНО: https://github.com/PaddlePaddle/PaddleClas/commits/release/2.6], [ПРОВЕРЕНО: https://api.github.com/repos/PaddlePaddle/PaddleClas/releases/latest], [ПРОВЕРЕНО: https://pypi.org/pypi/paddleclas/json], [ПРОВЕРЕНО: https://raw.githubusercontent.com/PaddlePaddle/PaddleClas/release/2.6/ppcls/configs/Vehicle/ResNet50_ReID.yaml], [ПРОВЕРЕНО: https://github.com/PaddlePaddle/PaddleClas/tree/release/2.6/ppcls/configs/Vehicle], [ПРОВЕРЕНО: https://raw.githubusercontent.com/PaddlePaddle/PaddleClas/release/2.6/ppcls/data/dataloader/vehicle_dataset.py], упоминание Paddle2ONNX: [ПРОВЕРЕНО: https://github.com/PaddlePaddle/PaddleClas]
- mmpretrain/OpenMMLab: [ПРОВЕРЕНО: https://api.github.com/repos/open-mmlab/mmpretrain], [ПРОВЕРЕНО: https://pypi.org/pypi/mmpretrain/json], [ПРОВЕРЕНО: https://github.com/open-mmlab/mmpretrain/tree/main/configs], [ПРОВЕРЕНО: https://api.github.com/repos/open-mmlab/mmtracking], [ПРОВЕРЕНО: https://api.github.com/repos/open-mmlab/mmdeploy]
- BoxMOT: [ПРОВЕРЕНО: https://api.github.com/repos/mikel-brostrom/boxmot], [ПРОВЕРЕНО: https://pypi.org/project/boxmot/], [ПРОВЕРЕНО: https://api.github.com/repos/mikel-brostrom/boxmot/releases/latest], [ПРОВЕРЕНО: https://raw.githubusercontent.com/mikel-brostrom/boxmot/master/pyproject.toml], [ПРОВЕРЕНО: https://raw.githubusercontent.com/mikel-brostrom/boxmot/master/docs/modes/train.md], [ПРОВЕРЕНО: https://raw.githubusercontent.com/mikel-brostrom/boxmot/master/docs/modes/export.md], [ПРОВЕРЕНО: https://github.com/mikel-brostrom/boxmot/tree/master/boxmot/reid]
- roboflow/trackers: [ПРОВЕРЕНО: https://api.github.com/repos/roboflow/trackers], [ПРОВЕРЕНО: https://raw.githubusercontent.com/roboflow/trackers/develop/README.md]
- TransReID: [ПРОВЕРЕНО: https://api.github.com/repos/damo-cv/TransReID], [ПРОВЕРЕНО: https://raw.githubusercontent.com/damo-cv/TransReID/main/README.md]
- CLIP-ReID: [ПРОВЕРЕНО: https://api.github.com/repos/Syliz517/CLIP-ReID], [ПРОВЕРЕНО: https://raw.githubusercontent.com/Syliz517/CLIP-ReID/master/README.md]
- layumi: [ПРОВЕРЕНО: https://api.github.com/repos/layumi/Person_reID_baseline_pytorch], [ПРОВЕРЕНО: https://raw.githubusercontent.com/layumi/Person_reID_baseline_pytorch/master/README.md]

## Примечания и нюансы

### 1. fast-reid — лучший функциональный матч, но проект де-факто заброшен
- Последний коммит в master — 15.07.2024 («Update README.md (#730)»), релизов нет с 2021 (v1.3.0), на PyPI лежит 1.4.0 от 15.02.2023. Репозиторий не архивирован, 3990 звёзд, всего 18 открытых issues (их чистят, но код не развивают).
- Это единственный из «классических» тулбоксов с **vehicle re-id из коробки**: готовые конфиги и бейзлайны для VeRi-776, VehicleID, VERI-Wild (`configs/VeRi/sbs_R50-ibn.yml` и т.п.), плюс официальные скрипты экспорта ONNX/TensorRT/Caffe с проверкой численного соответствия.
- Известные грабли: requirements не запинены — issue #758 (21.11.2025, закрыт): «安装最新的报错较多» («со свежими версиями зависимостей много ошибок»). Issue #719 (закрыт 01.2024): у пользователя на torch 2.1.2 зависало обучение. Прямых issue «не работает с torch 2.x» в выдаче поиска по issues не нашёл — но проект писался под torch 1.6, так что безопасная зона — torch 1.10–2.1 + numpy<2 [оценка; точная верхняя граница НЕ ПРОВЕРЕНА].
- Открыты «модернизационные» PR/issues от октября 2025 (#755 pyproject, #756 model zoo compatibility, #757 docs) — висят без merge, т.е. мейнтейнер не реагирует.
- Поддерживаемых форков с обновлением под torch 2.x найти не удалось (поиск по «fast-reid fork maintained 2025» ничего живого не дал) [проверено поиском, отсутствие ≠ доказано].

### 2. torchreid: имя на PyPI перехвачено
- `pip install torchreid` ставит **не библиотеку Kaiyang Zhou**, а стороннюю переупаковку `torchreid-pip` от автора kadirnar, версия 0.2.5 от 16.10.2022 (homepage — github.com/goksenin-uav/torchreid-pip). Официальный torchreid ставится только из исходников GitHub. Это важно для воспроизводимости и供应 цепочки (supply chain).
- Репозиторий формально «дышит» (README-коммит 09.01.2026), но код заморожен с начала 2023. Актуальные открытые баги: #592 (04.2025) — torch 2.6 сломал загрузку чекпойнтов из-за `weights_only=True` по умолчанию; #594 (07.2025) — низкий mAP на RTX 5090 (несовместимость CUDA-сборки); #584/#585 (09–10.2024) — баги ONNX-экспорта (динамический batch даёт неверные эмбеддинги; ошибка загрузки checkpoint в экспортере). Все открыты, никто не чинит.
- Vehicle-датасетов нет; API позволяет зарегистрировать свой `ImageDataset` (день работы), но mAP-протокол VeRi (junk-фильтры по камере) придётся контролировать самим.
- Главная ценность в 2026 — предобученные OSNet-веса, которые все (включая BoxMOT прошлых версий) переиспользуют.

### 3. reid-strong-baseline — исторический артефакт
Последний push 23.04.2020. Наследники по прямой линии: fast-reid (тот же круг авторов JD AI) и TransReID (первый автор M. He / H. Luo — авторы Bag of Tricks). Использовать сам репозиторий в 2026 смысла нет.

### 4. PaddleClas — жив, но это другой мир
- Активность в репо есть вплоть до 15.09.2026 (push), однако в основной ветке release/2.6 последний коммит — 01.07.2025 и почти вся активность 2025 — документация и мелкие фиксы; последний релиз v2.6.0 — 05.11.2024. Фокус PaddlePaddle сместился на PaddleX/PaddleOCR.
- Vehicle re-id реально есть: конфиги `ResNet50_ReID.yaml` (VERI-Wild, ArcMargin-голова, CELoss+SupConLoss) и `PPLCNet_2.5x_ReID.yaml`; датасет-классы `VeriWild` и `CompCars`. Класс `VeriWild` читает произвольные txt-списки «путь метка», так что VeRi-776/свой датасет подключаются конвертацией разметки, без нового кода.
- Экспорт: Paddle → инференс-модель → Paddle2ONNX. Работает, но это второй конвертер в цепочке; PyTorch-веса из других зоопарков не переносятся.
- Совместимость с PyTorch — не применима (фреймворк PaddlePaddle); это главный организационный риск: обучение, отладка и профилирование в незнакомом стеке легко съедают отведённые 2 недели.

### 5. mmpretrain / OpenMMLab — reid-задачи нет, экосистема стагнирует
- В mmpretrain из «метрического» — только конфиг `arcface` (retrieval на InShop); ни reid-голов, ни reid-датасетов. Re-id в OpenMMLab жил в mmtracking (модуль reid + MOT) — последний push 19.09.2023, проект де-факто умер. mmpretrain: последний push 01.11.2024, последний PyPI-релиз 1.2.0 от 04.01.2024 с пином `mmcv>=2.0,<2.4` — привязка к старым сборкам mmcv/torch. mmdeploy (путь к ONNX) — последний push 30.09.2024.
- Вывод: собирать vehicle re-id на OpenMMLab в 2026 — против течения.

### 6. BoxMOT — самый живой, но AGPL
- Взрывная активность: релиз v25.0.0 от 09.09.2026, push 15.09.2026, 8293 звезды, всего 2 открытых issue. Требования: Python 3.10–3.13, `torch>=2.2.1,<3.0.0` — единственный проект списка с актуальным пином под свежий PyTorch.
- В v25 появился полный reid-цикл: `boxmot train-reid` (документация прямо говорит «fits ReID backbones on **person or vehicle** re-identification datasets»), `eval-reid`, `compare-reid`, `boxmot export`. Обучаемые архитектуры: `osnet_x0_25`, `lmbn_n`, собственные `csl_tinyvit_7m/11m/23m`, `mobilenetv4_conv/hybrid_medium`, `hi_afa`. Встроенные датасеты — person (Market1501/Duke/CUHK03/MSMT17); свой (в т.ч. VeRi-подобный Market-формат `bounding_box_train/query/bounding_box_test`) подключается YAML-конфигом с полями path/train/query/gallery. Готового пресета именно «VeRi-776» в доках не видел [наличие пресета VeRi НЕ ПРОВЕРЕНО — в docs/modes/train.md не упомянут].
- Экспорт эталонный: ONNX, TensorRT (через ONNX), OpenVINO, CoreML (FP16 MLProgram), TFLite (int8 с калибровкой) — одной командой, включая свои обученные чекпойнты.
- **Лицензия AGPL-3.0(+)** — для закрытого коммерческого продукта означает обязанность открыть исходники производного кода (включая сетевое использование) либо договариваться с автором об отдельной коммерческой лицензии. Наличие/цену коммерческой лицензии не проверял [НЕ ПРОВЕРЕНО]. Отдельный юр. вопрос — считаются ли веса, обученные AGPL-кодом, производной работой (консенсуса нет).
- v25 — «breaking release»; API нестабилен между мажорными версиями (сам проект переименовывался yolo_tracking → boxmot).

### 7. roboflow/trackers — не про reid
Живой (push 14.09.2026), Apache-2.0, но README прямо заявляет: appearance/ReID-ветки алгоритмов сознательно не реализованы. Как источник reid-модели бесполезен; годится только как трекер поверх вашего внешнего эмбеддера.

### 8. Исследовательские репо с vehicle-SOTA (веса ценные, код мёртвый)
- **TransReID** (MIT, push 12.06.2024): VeRi-776 и VehicleID из коробки + претрейны ViT/DeiT для обоих; писан под torch 1.6, ONNX нет. Годится как донор весов/рецептов для fast-reid-подобного пайплайна.
- **CLIP-ReID** (MIT, push 21.11.2023): VeRi-776/VehicleID + претрейны, pytorch==1.8.0, ONNX нет. Поднять окружение можно, но всё руками.
- Экосистемный факт: paperswithcode.com мёртв — редиректит на huggingface.co/papers [ПРОВЕРЕНО: редирект 302 с https://paperswithcode.com/sota/vehicle-re-identification-on-veri-776], так что старые ссылки на лидерборды VeRi не работают.

### 9. layumi/Person_reID_baseline_pytorch — живой MIT-бейзлайн «на каждый день»
Push 30.08.2026; в феврале 2026 добавлена поддержка DinoV3-бэкбона. README: работает на torch вплоть до 2.0+ (новые фичи требуют >1.7). Vehicle: «For some vehicle re-ID datasets. e.g. VeRi, you also need to modify the prepare.py and test.py» (ссылка на issue #107) — т.е. VeRi заводится с небольшими правками. ONNX-экспорта в комплекте нет (упомянуты TensorRT/JIT), но модели — простые torchvision-бэкбоны с линейной головой, ручной `torch.onnx.export` элементарен [НЕ ПРОВЕРЕНО на практике]. Это скорее учебный бейзлайн, чем производственный тулбокс (нет конфиг-системы уровня fast-reid), но он единственный «классический» reid-репозиторий, который реально сопровождается в 2026.

### 10. Обходной путь: универсальные metric-learning библиотеки (обе живые)
Если ни один reid-тулбокс не устраивает, reid — это retrieval-задача, которую можно собрать на живых библиотеках:
- **pytorch-metric-learning** (KevinMusgrave): push 17.08.2025, MIT, 6341 звезда [ПРОВЕРЕНО: https://api.github.com/repos/KevinMusgrave/pytorch-metric-learning] — лоссы (triplet, ArcFace, SupCon), майнеры, к любому torchvision/timm-бэкбону; ONNX — штатный torch.onnx, т.к. модель ваша.
- **open-metric-learning (OML)**: push 26.11.2025, Apache-2.0, 998 звёзд [ПРОВЕРЕНО: https://api.github.com/repos/OML-Team/open-metric-learning] — готовые retrieval-пайплайны и метрики (CMC/mAP), формат датасета — простая таблица; reid-датасеты нужно конвертировать самому.
Такой путь дороже по коду (нет reid-специфики: camera-aware сэмплинг, junk-протокол VeRi, re-ranking), но свободен от мёртвых зависимостей.

### 11. Проверенные тупики (не тратить время)
- **openvinotoolkit/deep-object-reid** (Intel-форк torchreid) — **архивирован**, последний push 02.02.2023 [ПРОВЕРЕНО: https://api.github.com/repos/openvinotoolkit/deep-object-reid].
- **regob/vehicle_reid** — push 16.12.2022, мёртв [ПРОВЕРЕНО: https://api.github.com/repos/regob/vehicle_reid].
- Поиск «vehicle reid framework/toolbox 2025–2026» по GitHub topics и вебу нового сопровождаемого тулбокса не выявил: топ «recently updated» в топике vehicle-reidentification — студенческие репо на 0–2 звезды и awesome-списки [ПРОВЕРЕНО: https://github.com/topics/vehicle-reidentification?o=desc&s=updated]. Ниша фактически схлопнулась в BoxMOT (тулбокс) + мёртвые академические репо.

## Итоговые рекомендации под критерии задачи

1. **Основной вариант — fast-reid** (Apache-2.0): единственный тулбокс с VeRi-776/VehicleID/VERI-Wild и ONNX/TRT-экспортом из коробки. План: заморозить окружение (Python 3.8–3.10, torch ~1.13–2.1, numpy<2 — подобрать за день), обучить SBS/BoT R50-ibn на своём датасете, экспортировать `onnx_export.py`. Риск: проект не сопровождается — любые баги чините сами; заложите 1–2 дня буфера на окружение.
2. **Если AGPL приемлема (или продукт открытый/внутренний с соблюдением AGPL) — BoxMOT v25**: самый живой, свежий torch, `train-reid` + экспорт во все форматы одной командой; свой vehicle-датасет через YAML. Юридический вопрос — главный.
3. **Лёгкая MIT-альтернатива — layumi baseline**: жив (08.2026), VeRi с мелкими правками, ONNX руками; хорош, если нужен простой понятный код, а не «комбайн».
4. **torchreid** — только как источник OSNet-весов/архитектуры или для команды, готовой дописать vehicle-датасет и патчи под torch 2.6; помнить, что PyPI-имя перехвачено — ставить из GitHub.
5. **PaddleClas** — рабочий вариант только для команды, уже знакомой с Paddle; иначе 2 недели уйдут на борьбу со стеком.
6. **mmpretrain, reid-strong-baseline, CLIP-ReID/TransReID (как фреймворки), deep-object-reid, roboflow/trackers** — для данной задачи не брать (нет reid-задачи, мертвы или нет vehicle/ONNX); TransReID/CLIP-ReID держать в уме как доноров претрейнов VeRi.
