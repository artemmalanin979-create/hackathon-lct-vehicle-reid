# Результат CrossViewCore, 25.09.2026

**Решение: оставить в релизе `d1_j48`.** Самостоятельное ядро `RGB crop → 512D`, обученное на наших изображениях, действительно работает и на измеренной CPU-границе batch-1 имеет p95 **2.190 мс** против **71.642 мс** у двухветвевого release baseline (≈**32.7× быстрее**). Но на одинаковой val оно даёт **KR mAP 0.311518** против **0.774092**. Dev-выбранное объединение с baseline даёт **0.755009** и также уступает ему; его p95 **93.175 мс** превышает установленный предел. `PASS` в [JSON-результатах](results/) означает успешное выполнение измерения, **не принятие кандидата**. Условия качества замороженного протокола не выполнены, поэтому ни ONNX ядра, ни 1024D fusion не заменяют релизную модель.

## Что именно обучено

[Исходники и команды](README.md) описывают один MobileNetV3-Small image encoder с публичной ImageNet-инициализацией и нашими обучаемыми global-256/top-128/bottom-128 проекциями. Этап 1 обучал головы при замороженном backbone 2 эпохи, этап 2 — весь backbone и головы 12 эпох; 600 P8K4 batch на эпоху, CE + triplet с позитивами только между камерами. Это не прежний student над замороженными векторами `combined_v1`. Фактический основной CPU-прогон завершил **14/14 эпох за 43:08.89**, pilot `PASS`; лучший checkpoint выбран на последней эпохе по dev cosine mAP **0.443993**. Пик RSS внутри обучения **2259.8 MiB** при лимите **4096 MiB**. `training.json.status=PASS`, `checkpoint-status.json.status=COMPLETE`.

Из-за исчерпания доступной host-памяти на Windows worker CUDA не смогла начать первый реальный batch. До просмотра результата был отдельно заморожен [CPU-протокол](protocol-cpu-own.json) с теми же моделью, loss, sampler, seed и 2+12 эпохами, но с **own-only** данными: 7248 строк организаторов, из них 6639 fit / 609 dev и 100 dev-ID исключены из fit. Первоначальный [CUDA-протокол](protocol.json) планировал 19967 fit строк, включая 13328 разрешённых внешних CARLA/Roundabout кропов. Таким образом, это реальная оценка **own-only CPU-кандидата**; разницу качества нельзя приписать одной архитектуре или переносить на неисполненный combined-вариант. Файлы fit/dev проверены по frozen split и SHA, val-метки при обучении и выборе веса/порога не открывались.

## Качество на одном протоколе

Многократно использованная val: 1110 query × 750 gallery, 832 query с допустимой парой, 278 без пары; исключаются совпадения одной камеры. mAP ниже — **full-gallery** для 832 имеющих пару, а F1/TNR отказа рассчитаны на 1110 запросах с порогами, выбранными только на dev. KR: `k1=6, k2=3, lambda=0.3`. Значения из [evaluation.json](results/evaluation.json) (SHA-256 `69edc6dd4d7c927a546dd16e1b3caf5c6a86988afbb9ce135b458d9cfd076941`). Baseline извлечён заново из тех же JPG/bbox и воспроизвёл замороженные mAP до сравнения с кандидатом.

| Модель | Режим | mAP | Rank-1 | Rank-5 | mINP | F1 отказа | TNR отказа | Разрыв по камере, mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `d1_j48` | cosine | 0.731609 | 0.685096 | 0.882212 | 0.692846 | 0.861329 | 0.079137 | 0.130752 |
| `d1_j48` | KR | **0.774092** | **0.730769** | 0.881010 | 0.748806 | 0.842038 | 0.723022 | 0.110273 |
| CrossViewCore | cosine | 0.276749 | 0.219952 | 0.433894 | 0.233376 | 0.856849 | **0.000000** | 0.296903 |
| CrossViewCore | KR | 0.311518 | 0.268029 | 0.438702 | 0.266552 | 0.856849 | **0.000000** | 0.279541 |
| Fusion 1024D, вес core 0.25 | cosine | 0.725141 | 0.669471 | 0.877404 | 0.686455 | 0.858355 | 0.122302 | 0.133879 |
| Fusion 1024D, вес core 0.25 | KR | 0.755009 | 0.700721 | 0.875000 | 0.732080 | 0.868111 | 0.705036 | 0.119999 |

При dev-выбранном пороге у одиночного ядра TNR=0 в обоих режимах: оно не отвергло ни один из 278 запросов без допустимой пары. Довольно высокий F1 сам по себе этого не показывает. На dev выбран вес core `0.25` по cosine mAP из заранее заданной сетки 0.25/0.50/0.75. Там fusion выглядел лучше baseline: cosine mAP 0.869438 против 0.859114, KR 0.898521 против 0.890097. На val преимущество не подтвердилось. Это не новый holdout: val уже использовалась в прошлых экспериментах, а release whitening исторически видел dev100. Нельзя трактовать dev-результат как независимое подтверждение или подбирать по val ещё один вес.

Парный bootstrap по **284 vehicle_id**, 832 query с парой, 4000 повторов, seed `20260925`; Holm-поправка для заранее заданного семейства из четырёх сравнений:

| Кандидат к baseline | Δ mAP | 95% CI Δ | p с Holm |
|---|---:|---:|---:|
| Core cosine | −0.454860 | [−0.498949; −0.410655] | 0.0019995 |
| Core KR | **−0.462573** | **[−0.509323; −0.415570]** | **0.0019995** |
| Fusion cosine | −0.006468 | [−0.021248; +0.007725] | 0.352912 |
| Fusion KR | **−0.019082** | **[−0.035499; −0.003246]** | **0.041990** |

Fusion cosine не отличился от baseline в пределах этого CI; fusion KR показал отрицательную дельту и после поправки. Требование приёмки, замороженное **до обучения**: KR ΔmAP ≥+0.005 и нижняя граница парного CI >0, cosine ΔmAP ≥0, F1 отказа не хуже baseline более чем на 0.02 в обоих режимах, плюс все пределы производительности. Одиночное ядро и fusion нарушают первые два требования независимо от их F1; у fusion дополнительно не пройден p95. Нового подбора гиперпараметров или порога по val не было.

Экспорт ONNX opset 17: SHA-256 `15162a9b3b7bd97a851a7e2cc588a00e69001e5bae522fa02e33026d7dd35909`, размер **4,902,116 байт**. На восьми реальных dev-кропах максимум |Torch−ONNX| `4.84e-7`, на всей val `8.36e-7` при лимите `1e-5`; top-1 и решения об отказе совпали для **1110/1110** запросов в cosine и KR (также для fusion). Экспорт и численная совместимость `PASS`, но они не исправляют низкое качество дескриптора.

## Цена инференса

[Benchmark JSON](results/benchmark.json), SHA-256 `e4c0ac3882f936d5ee24924f5b7f16afc0e7016aa010044f6ad9dc13d530c6c4`: один и тот же Fedora CPU, ONNX Runtime 1.30.0 CPUExecutionProvider, 2 потока, batch 1, первые 5 из 45 фиксированных val-кропов на warmup, 40 парных чередующихся замеров. Время отсчитывается от **уже подготовленного** RGB float32 208×208 тензора до L2-дескриптора, без JPEG-декодирования, API, Qdrant и KR. Поэтому эти FPS и p95 не равны задержке HTTP-поиска.

| Вариант | p50, мс | p95, мс | FPS batch-1 | Cold start, мс | Peak RSS свежего процесса, MiB | Веса, байт | Индекс 750 × float32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d1_j48` | 46.392 | 71.642 | 19.56 | 146.754 | 118.586 | 18,630,636 | 1,536,000 (512D) |
| CrossViewCore | **1.659** | **2.190** | **586.04** | **21.817** | 118.250 | **4,902,116** | 1,536,000 (512D) |
| Fusion | 47.549 | 93.175 | 18.49 | 160.609 | 124.008 | 23,532,752 | 3,072,000 (1024D) |

Для p95 предел `≤1.1×` baseline равен **78.807 мс** в этом же прогоне. Core проходит все записанные performance-пределы (p95, FPS, cold start, RSS, веса); fusion проваливает p95 (`1.301×` baseline), хотя остальные измеренные пределы проходят. CPU обучение и CPU инференс не дают значения GPU VRAM: **NOT APPLICABLE**, а не ноль. Fusion требует версионирования API/индекса из-за 1024D и запускал бы обе image-сети; сам по себе быстрый core не делает fusion быстрым.

## Воспроизведение и следы

Исходный commit кода в момент запуска — `8693d02`; фактические SHA-256 файлов, записанные в `training.json`: `core.py` `8149052e6318262f4272e2688f93cbe5ef4c203e448b0637e9d600ed8f767a56`, `dataset.py` `9ea47a933d83d420e5a6b57f9ab79e60466e30c82999f753dd48e9e1a02c6fbe`, `train.py` `239ccc02331c7ad8c111e3b770a1c36e5388358ab58e7f8629465d4f6e4a3ada`. Frozen CPU-протокол SHA-256 `fea8a176d0caa32e01179bc674364b28f66f34121a43038be0b4318b50be3f27`; исходные train NPZ/raw crops/split/ImageNet weights SHA указаны в [README](README.md) и `training.json.input_sha256`. Архивный evaluator и его `scope_metrics.py` проверены до импорта (`79fba705…` и `6dd48998…`); val image manifest `ee57a58e…` и **1864** входных файла проверены заново. Python 3.13.13, Torch 2.14.0, NumPy 2.5.3 для обучения; ONNX 1.22.0/ORT 1.30.0 для экспорта и оценки.

| Шаг, фактический CLI | Артефакт / SHA-256 | Лог; exit code |
|---|---|---|
| `train.py --protocol protocol-cpu-own.json --evaluator reid_metrics.py --data-npz train_crops.npz --data-raw crops_208.npy --split split.json --imagenet mobilenet_v3_small-047dcff4.pth --out cpu-own-run-8693d02` | `training.json` `a70930835bad2e6afa0fd4194ff2aa24cc6bad4e5d8bbfdd698b22bf86a06695`; `checkpoint.pt` `22fc2fdd92519ae21f08eb6feef3812dc92bb5dd0191b2f6c7f1bbdc6a966e1a` | `outputs/owncore-20260925/cpu-own-run-8693d02.log`; 0 |
| `export.py --checkpoint checkpoint.pt --protocol protocol-cpu-own.json --raw-crops crops_208.npy --out core.onnx` (оба входных SHA проверены) | `core.onnx` `15162a9b3b7bd97a851a7e2cc588a00e69001e5bae522fa02e33026d7dd35909`; `export.json` `9a3472765d51b44badb5a22213385a5e6853d71d9e53871b9711db9696e6b79c` | `outputs/owncore-20260925/cpu-own-run-8693d02.export.log`; 0 |
| `dev_selection.py --protocol protocol-cpu-own.json --metadata train_crops.npz --raw-crops crops_208.npy --model core.onnx --out selection.json` (protocol/model SHA проверены) | `selection.json` `ca74676b74288024766852b74ddc9772418e1b8191ebe867c82cb545bca2bf13` | `outputs/owncore-20260925/cpu-own-run-8693d02.selection.log`; 0 |
| `evaluate.py --data-dir /home/artem/projects/hackathon-lct-vehicle-reid/data --protocol protocol-cpu-own.json --model core.onnx --checkpoint checkpoint.pt --dev-selection selection.json --out cpu-own-validation-8693d02` (все SHA проверены) | `evaluation.json` `69edc6dd4d7c927a546dd16e1b3caf5c6a86988afbb9ce135b458d9cfd076941` | `outputs/owncore-20260925/cpu-own-validation-8693d02.log`; 0 |
| `benchmark.py --data-dir /home/artem/projects/hackathon-lct-vehicle-reid/data --protocol protocol-cpu-own.json --model core.onnx --dev-selection selection.json --out cpu-own-benchmark-8693d02` (все SHA проверены) | `benchmark.json` `e4c0ac3882f936d5ee24924f5b7f16afc0e7016aa010044f6ad9dc13d530c6c4` | `outputs/owncore-20260925/cpu-own-benchmark-8693d02.log`; 0 |

В таблице кратко показаны фактические аргументы; **полные команды с абсолютными путями, `timeout`/`nice` и `/usr/bin/time -v` находятся в указанных логах**. Команды для нового каталога результата приведены в [README](README.md). `outputs/` игнорируется Git: checkpoint, ONNX, изображения и логи существуют локально и должны передаваться отдельным manifest при handoff; **точные копии шести JSON-результатов** зафиксированы в [results/](results/) с теми же SHA-256. Этот отчёт не превращает экспериментальный ONNX в релизный артефакт.
