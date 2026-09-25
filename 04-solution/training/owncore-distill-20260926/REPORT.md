# Проверка самостоятельного CrossViewCore после дистилляции

Срез: 26.09.2026, исходный код `d94d3fd` до этого отчёта. Два протокола были
зафиксированы до обучения и просмотра новой val: `protocol-own.json` SHA-256
`2943648c367ba6594e4b6c9d779d9eff4df1ceb3e902fef5d79d05dff1739d4f`
и `protocol-combined.json` SHA-256
`648f60b5e3a2a0004bf33aab2d432eba0e751b26393ff83f66ff84f7239458ac`.
Второй опыт разрешён по локальному
`outputs/owncore-distill-20260926/combined-go-d94d3fd.json`
до просмотра val первого. Оба опыта используют **одну отдельную**
MobileNetV3-Small `image → 512D` с собственной global/local головой;
`d1_j48` служит frozen teacher только при обучении. В ONNX-кандидате нет
двух учительских сетей. Единственный seed — `20260926`.

## Решение

**Цель Артёма — сравняться с KR mAP 0.774092 — не достигнута.** Лучшее
одиночное ядро на тех же 1110 query × 750 gallery даёт 0.327450. Оно
значительно быстрее на измеренной границе, но при dev-выбранном пороге
принимает все 278 запросов без допустимой пары (TNR 0). Объединение с
релизным ансамблем достигает 0.762604 KR mAP и 0.727007 cosine mAP,
что ниже базовых 0.774092 и 0.731609. Для fusion нужны 1024D признак,
новая gallery и версия контракта. Заранее заданные критерии смены релиза
не выполнены; сдаваемой остаётся `d1_j48`, её пороги и 512D индекс.

## Качество на val

Это многократно использованная ранее val, **не новый holdout**. mAP, Rank и
mINP считаются по 832 query с допустимой межкамерной парой; F1/TNR отказа —
по всем 1110 query, включая 278 без пары. KR — пакетное k-reciprocal
`k1=6,k2=3,lambda=0.3`; HTTP-сервис использует cosine. Пороги каждого
экспериментального режима выбирались только на историческом dev.

| Кандидат | Режим | mAP | Rank-1 | Rank-5 | mINP | F1 отказа | TNR отказа | Разрыв по камерам mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| d1_j48 | cosine | .731609 | .685096 | .882212 | .692846 | .861329 | .079137 | .130752 |
| d1_j48 | KR | .774092 | .730769 | .881010 | .748806 | .842038 | .723022 | .110273 |
| own core | cosine | .288768 | .236779 | .447115 | .242581 | .856849 | 0 | .297057 |
| own core | KR | .323997 | .290865 | .460337 | .276303 | .855670 | 0 | .280310 |
| own fusion | cosine | .723857 | .668269 | .883413 | .685073 | .858065 | .172662 | .134745 |
| own fusion | KR | .754402 | .700721 | .873798 | .729086 | .887395 | .420863 | .120439 |
| combined core | cosine | .293534 | .231971 | .475962 | .247802 | .856849 | 0 | .294172 |
| combined core | KR | .327450 | .278846 | .487981 | .280044 | .856849 | 0 | .277107 |
| combined fusion | cosine | .727007 | .675481 | .882212 | .687318 | .857912 | .176259 | .132914 |
| combined fusion | KR | .762604 | .711538 | .879808 | .739162 | .887601 | .539568 | .116384 |

Fusion — L2 каждой ветви, затем `sqrt(1-w)` релизного признака и
`sqrt(w)` нового в конкатенации; `w=0.25` выбран по dev cosine до val.
KR выполняется один раз над общим 1024D пространством. Координаты разных
сетей не усреднялись.

Парный bootstrap: 4000 повторов по 284 `vehicle_id`, 832 query; семейство
восьми объявленных сравнений скорректировано Holm. Ниже `Δ mAP = candidate −
d1_j48`. Полный машиночитаемый расчёт и все восемь сравнений:
`outputs/owncore-distill-20260926/holm8-d94d3fd.json`, SHA-256
`37f797d3c428f3c0bb34a3048f72810ebf658483a24c6430425a7798901be280`.

| Сравнение | Δ mAP | 95% CI | p Holm/8 |
|---|---:|---|---:|
| own core, KR | -.450095 | [-.496072, -.403315] | .003999 |
| own fusion, KR | -.019690 | [-.036562, -.003901] | .065984 |
| combined core, KR | -.446641 | [-.492405, -.400641] | .003999 |
| combined fusion, KR | -.011487 | [-.026954, +.003198] | .340415 |

Интервал combined fusion включает ноль, но наблюдаемый прирост отсутствует;
самостоятельное ядро достоверно хуже по этой диагностической val. Локальный
`p_Holm_four_tests` внутри каждого `evaluation.json` относится лишь к одной
попытке и **не** является поправкой по всему семейству.

## Обучение и стоимость

| Попытка | Fit | Эпохи × batch | Лучший dev cosine mAP | Checkpoint SHA-256 | ONNX SHA-256 |
|---|---:|---:|---:|---|---|
| own | 6639 кадров | 12 × 600 | .462237, эпоха 9 | `7b5532784c1508547b0d8dd405df119d42f7e37f582242e06ca22af68d4ae67e` | `c316b1e4bde9c316097f2a0372d62d33992af8ae4d2285ed428260068cae26bd` |
| combined | 19967 кадров | 8 × 600 | .457651, эпоха 4 | `30d98c3f4b8c4a99421dc4feb4f835f8e467b00f306bbe78ba3ab5b8cfd51467` | `4cb47b31cf2abf3eabb9fdf30f8de5bc4f06cfe2306670a01fe1bc9e95961df6` |

Проверено 7200 и 4800 реальных P8K4 batch соответственно; в каждом были
межкамерные позитивные пары и ненулевые обновления весов. Сумма времени эпох
на локальном CPU: 3088 с и 1837 с. Пик RSS: 2.20 и 3.77 GiB,
в пределах зафиксированного лимита 4 GiB. `teacher-combined.npz` содержит
только fit-цели; его префикс собственных строк побайтно согласован с
`teacher-own.npz`. Эти teacher-файлы и сырые кадры вне Git.

Парный CPU batch-1 тест при двух ORT threads: один и тот же набор 40
проверенных кадров, пять warmup, граница **готовый RGB208 tensor → L2
дескриптор**. JPEG, API, Qdrant и KR не включены. Отдельный замер combined:

| Модель | p50 / p95, мс | FPS | Холодный старт, с | RSS, MiB | Веса, байт | Признак / сырой индекс 750 |
|---|---:|---:|---:|---:|---:|---:|
| d1_j48 | 39.470 / 43.861 | 24.94 | .132 | 119.32 | 18630636 | 512D / 1536000 B |
| combined core | 1.406 / 1.727 | 706.94 | .021 | 117.68 | 4902116 | 512D / 1536000 B |
| combined fusion | 41.094 / 43.888 | 24.11 | .151 | 122.56 | 23532752 | 1024D / 3072000 B |

Все performance limits протокола прошли; это не доказывает скорость полного
сервиса. VRAM — `NOT APPLICABLE`, измерялся CPU. Аналогичный парный
own-бенчмарк в локальном `assessment-own-818456a/benchmark/benchmark.json`
прошёл, но имеет иной уровень фонового CPU и не используется для прямого
сравнения абсолютной задержки между двумя попытками.

## Воспроизведение и границы доказательства

Исходники эксперимента — в этом каталоге; pinned-протоколы, `inputs.json`
каждого прогона, манифесты teacher, checkpoint, ONNX и результаты находятся
локально под `outputs/owncore-distill-20260926/`. Обе val-проверки заново
извлекли релизный признак из реальных JPG, сверили его mAP с опубликованными
значениями **до** оценки кандидата. Нет подмены gallery или cached feature.
Torch и ONNX дали max |Δ| `7.56e-7` (own) и `9.83e-7` (combined), по всем
1110 query совпали top-1 и решение отказа для cosine/KR. Проверка экспорта
на восьми dev-кропах отдельно тоже прошла (`4.28e-7`, `4.81e-7`).

Команды из корня этого checkout выполнялись с Python 3.13.13,
PyTorch 2.14.0+cu130 (CPU), NumPy 2.5.3, ONNX Runtime 1.30.0,
`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`, через
`/home/artem/yolov8-rtsp-person-detector/venv/bin/python` и
`PYTHONPATH=/home/artem/projects/hackathon-lct-vehicle-reid/.venv/lib/python3.13/site-packages`.
Точные привязки входов и выходов/их SHA есть в `inputs.json`, `training.json`,
`selection.json`, `evaluation.json` и `benchmark.json` соответствующей
попытки. Исполненные точки входа:

```text
04-solution/training/owncore-distill-20260926/finetune.py --protocol ... --source-checkpoint ... --data-npz ... --data-raw ... --split ... --teacher-targets ... --evaluator ... --out ...
04-solution/training/owncore-20260925/export.py --repo ... --checkpoint ... --protocol ... --raw-crops ... --out ... --manifest ...
04-solution/training/owncore-20260925/dev_selection.py --repo ... --protocol ... --metadata ... --raw-crops ... --model ... --out ...
04-solution/training/owncore-20260925/evaluate.py --repo ... --data-dir ... --protocol ... --model ... --checkpoint ... --dev-selection ... --out ... --batch-size 16
04-solution/training/owncore-20260925/benchmark.py --repo ... --data-dir ... --protocol ... --model ... --dev-selection ... --out ...
04-solution/training/owncore-distill-20260926/aggregate.py --own-evaluation ... --combined-evaluation ... --own-protocol ... --combined-protocol ... --own-selection ... --combined-selection ... --repo ... --data-dir ... --out ...
```

Все команды требовали соответствующие `--*-sha256` параметры и завершились
exit 0, **кроме терминального статуса полного combined обучения**: оболочка
вернула 143 после записи восьмой эпохи, `training.json: PASS` и
`checkpoint-status.json: COMPLETE`. Поэтому его процессный статус отмечен
как аномальный, а не замолчан. Checkpoint SHA совпал; независимые export,
dev selection, full val и benchmark завершились exit 0/PASS.
Логи: `outputs/owncore-distill-20260926/train-{own-818456a,combined-d94d3fd}.log`,
`assessment-{own-818456a,combined-d94d3fd}/{export,selection,evaluation,benchmark}.log`
и `holm8-d94d3fd.log`. В каждом `evaluation.json` — 1110 проверенных query,
750 gallery, 0 skipped; `export_parity` — 1110 query, 0 ошибок top-1/отказа.

Учитель и начальный checkpoint исторически были выбраны с использованием
dev100; val также уже применялась в исследованиях. Групповой split исключает
100 dev ID из fit нового обучения, но не превращает этот dev или val в
нетронутый holdout. Новых попыток после просмотра val не запускалось.
