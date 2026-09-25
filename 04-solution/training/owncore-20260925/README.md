# CrossViewCore: самостоятельный дескриптор из изображения

Это **экспериментальный**, отдельный от сдаваемой `d1_j48` модели путь `RGB crop → 512-мерный L2-нормированный признак`. `CrossViewCore` в [core.py](core.py) использует **публичный предобученный** `torchvision` MobileNetV3-Small (`IMAGENET1K_V1`) как начальную свёрточную часть. Наш механизм — три обучаемые проекции карты признаков: global 256 и две горизонтальные области по 128. На втором этапе обучается **весь backbone**, а не только голова. Вход ONNX — `float32 RGB NCHW`, 208×208, значения 0…255 после bbox-кропа; деление на 255 и ImageNet-нормировка находятся внутри модели. Классификатор идентификаторов нужен только при обучении и не входит в выход `forward`.

Это **не** прежний [student над замороженными признаками `combined_v1`](../distill-20260925/README.md): здесь модель получает пиксели, не использует OSNet, teacher-вектор, `combined_v1` или whitening при инференсе. При этом MobileNet и его ImageNet-веса созданы вне команды; собственной архитектурой и собственным обучением являются global/local голова, межкамерный sampler/loss и полный fine-tune. Источник весов и точный SHA-256 записаны в протоколе. Код TorchVision — BSD-3-Clause; условия распространения предобученных весов требуют отдельной проверки перед включением сырого checkpoint в публичный комплект.

## Зафиксированный опыт

До основного прогона записаны [CUDA-протокол](protocol.json) (SHA-256 `0cdee6ec07e733612e98c5b2b58b7e21b66f1bad44f7904a152f267ac542c00b`) и [CPU fallback-протокол](protocol-cpu-own.json) (SHA-256 `fea8a176d0caa32e01179bc674364b28f66f34121a43038be0b4318b50be3f27`). CPU fallback зафиксирован **до просмотра результата новой модели**: на Windows worker CUDA не выполнила даже первый реальный batch из-за нехватки доступной host-памяти. Протокол явно задаёт `device=cpu`, четыре потока и пик RSS <4096 MiB; silent CUDA→CPU fallback в коде запрещён. Обучение CPU ведётся только на 7248 собственных строках: 6639 fit, 609 dev, 100 dev-ID. Два этапа — 2 эпохи с замороженным backbone и 12 с полным fine-tune, по 600 P8K4 batch/эпоху, seed `20260925`, один основной прогон, CE + cross-camera batch-hard triplet. Пилот после первых двух эпох останавливает запуск при росте loss, превышении памяти или рассчитанного бюджета времени. Самый высокий **dev cosine mAP** выбирает checkpoint; val не участвует в выборе эпохи.

Критичные входы CPU-протокола: `train_crops.npz` — `a910e8a3a10986845fb83f3a3fcd401bfc8213ac697d50864e36b8c84fefff9d`, выровненный `crops_208.npy` — `ea77290d1ad1ae6c245bc33e2ba19bc7c676a8185eda69f9028fe4d1b9e05057`, [фиксированный dev100 split](../distill-20260925/results/split.json) — `0f35d494bde29b0bac95094862cdd67cfc8b06cd7715c0affae0ac1a669ec620`, MobileNetV3-Small — `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`. Тренер также сверяет SHA-256 архивных `reid_metrics.py` и соседнего `scope_metrics.py` до их импорта, а затем пишет SHA протокола, входов и исходников в `inputs.json`/checkpoint. Данные организаторов и pretrained `.pth` находятся **вне Git**; команда не скачивает их скрыто.

**Итог прогона 25.09:** обучение 14/14 эпох, экспорт, dev selection, полная val-оценка и парный CPU benchmark завершены. Одиночное ядро: KR mAP **0.311518** против **0.774092** у release baseline, зато batch-1 p95 **2.190 мс** против **71.642 мс** на том же CPU. Fusion дал KR mAP **0.755009** и p95 **93.175 мс**. Скорость одиночной сети подтверждена в измеренной границе тензор→дескриптор, качество и отказ не проходят заранее заданную приёмку; **релиз остаётся `d1_j48`**. Метрики, интервалы, SHA, команды и ограничения — в [отчёте](REPORT.md), точные числовые manifests сохранены в [results/](results/). HTTP-сервис использует cosine; KR mAP нельзя приписывать ему. Dev100 исторически видел release whitening, val использовалась прежде: ни dev, ни val не являются новым нетронутым holdout.

## Локальное воспроизведение

Команды ниже выполняются из корня этого worktree на Fedora в уже подготовленных окружениях; пути входов можно заменить на разрешённые копии **с теми же SHA-256**. Для обучения достаточно PyTorch, TorchVision и NumPy; экспорт/оценка дополнительно используют ONNX, ONNX Runtime, Pillow и зависимости релизного сервиса. Проверенное локальное окружение обучения: Python 3.13, Torch 2.14.0, TorchVision 0.29.0, NumPy 2.5.3. Оценочные команды получают ONNX/ORT из существующей проектной venv через `PYTHONPATH`. Изолированный Windows Python 3.12 с Torch 2.5.1/TorchVision 0.20.1 также поддерживается кодом, но CPU fallback ниже относится к зафиксированному Fedora-прогону.

```bash
cd /home/artem/projects/lct-owncore-20260925
export CODE=04-solution/training/owncore-20260925
export PROTO="$CODE/protocol-cpu-own.json"
export PROTO_SHA=fea8a176d0caa32e01179bc674364b28f66f34121a43038be0b4318b50be3f27
export TRAIN_PY=/home/artem/yolov8-rtsp-person-detector/venv/bin/python
export DATA_NPZ=outputs/owncore-20260925/own-data/train_crops.npz
export DATA_RAW=outputs/owncore-20260925/own-data/crops_208.npy
export SPLIT=04-solution/training/distill-20260925/results/split.json
export WEIGHTS=outputs/owncore-20260925/input/mobilenet_v3_small-047dcff4.pth
export EVALUATOR=04-solution/eval/reid_metrics.py
export VAL_DATA=/home/artem/projects/hackathon-lct-vehicle-reid/data
test "$(sha256sum "$PROTO" | cut -d' ' -f1)" = "$PROTO_SHA"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

Каждый `--out` ниже должен указывать на **новый** путь. `--synthetic-smoke` проверяет обе стадии и градиенты на искусственных тензорах; `--smoke-only` делает один настоящий P8K4 fit batch с forward/backward/optimizer step, проверяет ненулевые градиенты и RSS, пишет `smoke.json`, но не начинает эпохи. Реальный smoke и полный run имеют **разные** каталоги результата.

```bash
"$TRAIN_PY" -B "$CODE/train.py" --synthetic-smoke --protocol "$PROTO" --evaluator "$EVALUATOR"
"$TRAIN_PY" -B "$CODE/train.py" --smoke-only --protocol "$PROTO" \
  --evaluator "$EVALUATOR" --data-npz "$DATA_NPZ" --data-raw "$DATA_RAW" \
  --split "$SPLIT" --imagenet "$WEIGHTS" \
  --out outputs/owncore-20260925/my-new-smoke
"$TRAIN_PY" -B "$CODE/train.py" --protocol "$PROTO" \
  --evaluator "$EVALUATOR" --data-npz "$DATA_NPZ" --data-raw "$DATA_RAW" \
  --split "$SPLIT" --imagenet "$WEIGHTS" \
  --out outputs/owncore-20260925/my-new-run
```

Основной run фиксирует `epochs.jsonl`, `pilot.json`, лучший `checkpoint.pt`, `dev_embeddings.npz`, `training.json` и `checkpoint-status.json`. Фактический каталог прогона: `outputs/owncore-20260925/cpu-own-run-8693d02/`; большие бинарные артефакты и логи остаются вне Git, а точные JSON-копии [training](results/training.json), [checkpoint status](results/checkpoint-status.json), [export](results/export.json), [dev selection](results/selection.json), [evaluation](results/evaluation.json) и [benchmark](results/benchmark.json) сохранены в Git. Статус checkpoint `COMPLETE`, `training.json.status=PASS`. Статус `PARTIAL` у другого, прерванного прогона пригоден только для диагностики; сравнение релизного кандидата требует `COMPLETE`. Повтор **после 27.09.2026 00:00 МСК** остановится по зафиксированному deadline: для нового опыта понадобится отдельный заранее зафиксированный протокол и новый идентификатор прогона, а не правка уже выполненного результата.

После успешного обучения экспорт проверяет строгую загрузку checkpoint, 8 реальных dev-кропов, форму/норму выхода и расхождение Torch↔ONNX ≤1e-5. В `--checkpoint-sha256` подаётся значение из `training.json`, которое должно совпасть с вычисленным SHA файла. Для оценки нужен Python с Torch/TorchVision/NumPy/ONNX/ORT/Pillow (на этой машине так соединены существующие venv); `VAL_DATA/images/` содержит разрешённые JPG организаторов, каждый сверяется с manifest.

```bash
export RUN=outputs/owncore-20260925/my-new-run
export CHECKPOINT_SHA=$(sha256sum "$RUN/checkpoint.pt" | cut -d' ' -f1)
export MODEL=outputs/owncore-20260925/my-new-model.onnx
export PYTHONPATH=/home/artem/projects/hackathon-lct-vehicle-reid/.venv/lib/python3.13/site-packages
"$TRAIN_PY" -B "$CODE/export.py" --repo "$PWD" \
  --checkpoint "$RUN/checkpoint.pt" --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --protocol "$PROTO" --protocol-sha256 "$PROTO_SHA" --raw-crops "$DATA_RAW" \
  --out "$MODEL" --manifest outputs/owncore-20260925/my-new-export.json
export MODEL_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
"$TRAIN_PY" -B "$CODE/dev_selection.py" --repo "$PWD" \
  --protocol "$PROTO" --protocol-sha256 "$PROTO_SHA" \
  --metadata "$DATA_NPZ" --raw-crops "$DATA_RAW" \
  --model "$MODEL" --model-sha256 "$MODEL_SHA" \
  --out outputs/owncore-20260925/my-new-selection.json
export SELECTION=outputs/owncore-20260925/my-new-selection.json
export SELECTION_SHA=$(sha256sum "$SELECTION" | cut -d' ' -f1)
"$TRAIN_PY" -B "$CODE/evaluate.py" --repo "$PWD" --data-dir "$VAL_DATA" \
  --protocol "$PROTO" --protocol-sha256 "$PROTO_SHA" \
  --model "$MODEL" --model-sha256 "$MODEL_SHA" \
  --checkpoint "$RUN/checkpoint.pt" --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --dev-selection "$SELECTION" --dev-selection-sha256 "$SELECTION_SHA" \
  --out outputs/owncore-20260925/my-new-evaluation
"$TRAIN_PY" -B "$CODE/benchmark.py" --repo "$PWD" --data-dir "$VAL_DATA" \
  --protocol "$PROTO" --protocol-sha256 "$PROTO_SHA" \
  --model "$MODEL" --model-sha256 "$MODEL_SHA" \
  --dev-selection "$SELECTION" --dev-selection-sha256 "$SELECTION_SHA" \
  --out outputs/owncore-20260925/my-new-benchmark
```

`dev_selection.py` выбирает вес fusion из заранее указанной сетки и **отдельные** пороги отказа для baseline/core/fusion × cosine/KR только на dev. Фактически выбран вес ядра `0.25`. Для разных геометрий используется нормированная конкатенация вектора baseline и core до 1024 измерений с `sqrt`-весами; KR работает один раз над объединёнными признаками. Это исследовательский fusion, не drop-in замена 512-мерного индекса. `evaluate.py` сначала воспроизводит release baseline на тех же val JPG, затем проверяет Torch↔ONNX ранжирование/top-1/отказы и считает парные сравнения. `benchmark.py` измеряет baseline, core и fusion на одном CPU, 2 потока, batch 1, 5 warmup + 40 чередующихся замеров. Граница задержки: готовый 208×208 тензор → дескриптор, без JPEG/API/Qdrant/KR. Метрики, CI, поправка Holm и все пределы принятия находятся в замороженном протоколе; фактически оба кандидата отклонены по качеству, fusion дополнительно не прошёл p95.
