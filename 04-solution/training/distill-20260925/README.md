# Дистилляция d1_j48: прототип обучен, в релиз не принят

Student и fusion не прошли заранее объявленный критерий качества. Сдаваемой остаётся **d1_j48**; release-код, модели, пороги и gallery не менялись.

## Что реально выполнено

Замороженная combined_v1 + residual MLP512→256→512; один seed20260925, main30 эпох и одна ablation30 без relational-loss. Обучение заняло 278.8с, peakRSS 639.4MiB. Main выбран на эпохе29 по dev embedding loss, ablation на эпохе30. Внешний validation не участвовал в выборе.

Fit6639 кадров/1071ID; исторический dev100 —609 кадров. Refusal dev: 427 query ×182 gallery,25 ID без gallery-пары. Teacher whitening обучен только на fit. Fusion: вес student0.25, выбран на dev; размерность1024 — исключительно исследовательский формат.

Dev baseline/fusion используют тот же канонический release whitening, что и итоговая оценка. Он исторически обучен на всех train_fit, включая dev100: унаследованная экспозиция раскрыта, dev **не untouched**. Fit-only teacher применяется только для обучения student; его checkpoint не менялся.

## Качество на многократно использованном validation

1110query×750gallery,832 query с межкамерной парой. Market/presence, full-gallery AP, KR(6,3,0.3). F1/TNR ниже используют отдельные **dev-пороги**, в том числе для baseline; официальные релизные пороги показаны отдельно.

| Модель | Режим | mAP | Rank-1 | Rank-5 | mINP | F1 dev-порог | TNR dev-порог | Camera gap mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | cosine | 0.731609 | 0.685096 | 0.882212 | 0.692846 | 0.861329 | 0.079137 | 0.130752 |
| baseline | KR | 0.774092 | 0.730769 | 0.881010 | 0.748806 | 0.842038 | 0.723022 | 0.110273 |
| student | cosine | 0.707940 | 0.652644 | 0.856971 | 0.667816 | 0.860441 | 0.086331 | 0.143439 |
| student | KR | 0.745935 | 0.700721 | 0.858173 | 0.716658 | 0.871581 | 0.579137 | 0.124902 |
| fusion | cosine | 0.731071 | 0.680288 | 0.882212 | 0.693580 | 0.860733 | 0.079137 | 0.130331 |
| fusion | KR | 0.766010 | 0.727163 | 0.873798 | 0.737446 | 0.867108 | 0.604317 | 0.113562 |
| ablation | cosine | 0.710866 | 0.652644 | 0.862981 | 0.672673 | 0.859658 | 0.032374 | 0.141802 |
| ablation | KR | 0.743569 | 0.695913 | 0.861779 | 0.715112 | 0.879300 | 0.535971 | 0.125693 |

Официальный baseline при сохранённых релизных порогах:

| Режим | Порог | F1 | TNR |
|---|---:|---:|---:|
| cosine | 0.5141976914190476 | 0.768500 | 0.730216 |
| KR | 0.5282812306342437 | 0.809079 | 0.784173 |

Максимизация F1 на маленьком dev дала слабый перенос отказа на cosine: TNR низкий. Эти экспериментальные пороги не пригодны для переноса в сервис. Пороги вычислены только по историческому dev. Replay после независимой критики исправляет геометрию baseline, не подбирает пороги по внешнему validation; внешние метки не используются в selection.

## Неопределённость и ошибки

Парный bootstrap4000 по vehicle_id, seed20260925; обычные CI95 и Holm по четырём сравнениям student/fusion×cosine/KR. Абляция описательная; выигрышный seed не выбирался.

| Сравнение с d1 | Режим | ΔmAP | CI95 | p Holm | Исправлено / испорчено top-1 |
|---|---|---:|---|---:|---:|
| student | cosine | -0.023670 | [-0.036547;-0.010885] | 0.003999 | 14 / 41 |
| student | KR | -0.028156 | [-0.043966;-0.013094] | 0.003999 | 17 / 42 |
| fusion | cosine | -0.000538 | [-0.004947;+0.004028] | 0.772807 | 4 / 8 |
| fusion | KR | -0.008082 | [-0.015103;-0.002237] | 0.010997 | 4 / 7 |
| ablation | cosine | -0.020744 | [-0.033260;-0.008296] | описательно | 13 / 40 |
| ablation | KR | -0.030523 | [-0.045532;-0.016877] | описательно | 8 / 37 |

Перечни исправленных/испорченных query, корреляции ошибок и разбиение по камерам: [evaluation.json](results/evaluation.json). Независимой view-разметки нет: view-strata **NOT MEASURED**. Нельзя приписывать падение исключительно MLP или relational-loss: teacher whitening fit-only отличается от релизного, а отдельная абляция этого фактора не предусматривалась.

## Измеренная стоимость

worker-vm CPU, ORT1.30.0,2threads,batch1,warmup5,40кадров; порядок трёх моделей чередуется. Измеряется готовый тензор→нормированный признак; decode/API/Qdrant/KR сюда не входят. Узел общий, фоновая нагрузка не устранялась.

| Модель | p50 мс | p95 мс | FPS batch1 | Cold start с | PeakRSS cold MiB | Веса МБ | Размерность / raw gallery750 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 504.36 | 619.39 | 1.94 | 1.079 | 149.3 | 18.631 | 512 / 1536000Б |
| student | 286.33 | 452.38 | 3.34 | 0.517 | 129.6 | 9.795 | 512 / 1536000Б |
| fusion | 514.52 | 791.37 | 1.87 | 1.237 | 150.3 | 19.683 | 1024 / 3072000Б |

Ускорение student по p95:27.0%; overhead fusion:+27.8%. Даже выигрыш скорости не разрешает потерю качества: критерий student — lowerCI≥0 и pointΔ≥0 в обеих метриках. Он не выполнен. Fusion также не достигΔmAP≥0.005/lowerCI>0. VRAM — неприменимо(CPU); размер индекса — только raw векторы, без накладных расходов Qdrant. PeakRSS по модели измерен в отдельном холодном процессе, не является отдельным продолжительным нагрузочным тестом.

## Экспорт, проверки и артефакты

Head ONNX на1860 признаках: maxabs 1.64e-07. Полный image→student ONNX на40 реальных cached-crop входах: maxabs 8.94e-08. Cosine/KR:0 изменений top-1 и0 изменений отказа на1110query каждого режима; mAP совпал. [Проверка](results/export_ranking_check.json).

Исходные5 behavioral tests и новый тест dev/release-геометрии PASS (6/6); исходные4 guard-removal mutations и воспроизведение P2 в одноразовой копии убиты (5/5); исходные hashes неизменны. 60 штатных тестов evaluator PASS на исходном запуске; evaluator не менялся. Первоначальный RED был отсутствующим модулем, он **не считается** убитой мутацией. [Полный manifest](results/verification_manifest.json).

| Артефакт | SHA-256 |
|---|---|
| `artifacts/run/main.npz` | `8e80d2ad34b75d4cd98fee17d6e091f3ce3b7a5ada7b81b2d46ee746f6b3213f` |
| `artifacts/run/export/head.onnx` | `b547c8bfb79a0df84f8623fd45d736b4a80386f573c6b59b646f6faf70ef4dca` |
| `artifacts/run/export/student_combined_v1.onnx` | `188284ff7a56ff915ea6143cca62dca0a381e3ca7fc784b996b876b641ad3d06` |

Артефакты не входят в Git; сохранены локально в delivery worktree и на `worker-vm:~/lct-reid/jobs/distill_20260925/results/`. Все 24 файла `artifacts/run/` в delivery сверены с [local_artifact_manifest.json](results/local_artifact_manifest.json). Перед удалением worktree перенести **весь** каталог вместе с manifest, не один ONNX. Исходники/config/метрики/логи входят в Git. Никаких новых внешних весов или датасетов не скачивалось; исходный provenance/licensing OSNet и combined_v1 остаётся прежним.

## Запуск готового ядра

`infer.py` принимает настоящий JPEG/PNG и bbox, проверяет SHA полного ONNX и выдаёт нормированный float32-вектор512. Проверен реальным validation JPEG и исходным bbox: результат совпал с независимым training-runtime в пределах1e-5. Точная выполненная команда и погрешность — [infer_smoke.json](results/infer_smoke.json). Имена модели и результата явно экспериментальные; endpoint релизного сервиса не подменяется.

```bash
/home/artem/projects/hackathon-lct-vehicle-reid/.venv/bin/python \
  04-solution/training/distill-20260925/infer.py \
  --image /path/to/frame.jpg --bbox 10 20 200 100 \
  --out ./student-vector.npy
```

Путь/рамка в этом примере заменяются своими. Модель по умолчанию берётся из `artifacts/run/export/student_combined_v1.onnx`; после интеграции копируется весь `artifacts/run/`. Повторную запись существующего результата CLI отклоняет. Для собственного окружения нужны NumPy, Pillow и ONNX Runtime; версии проверенного окружения сохранены в export/benchmark manifests.

## Воспроизведение обучения

Команды на разрешённом worker-vm; `--repo` должен указывать на checkout/snapshot с SHA из protocol.json. Входные кэши уже находятся по проверяемым абсолютным путям manifest. Для другой машины сначала подготовить те же входы и явный новый path mapping; скрытого download нет. Новый запуск — только в **новый** OUT, существующие main.npz/training.json защищены от перезаписи.

```bash
cd ~/lct-reid/jobs/distill_20260925
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/experiment.py train --repo repo --out fresh-run --smoke-only
# Export/check smoke locally with export_head.py before the training continuation.
systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/experiment.py train --repo repo --out fresh-run
systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/experiment.py evaluate --repo repo --out fresh-run
# Export main.npz + export_validation.npz using export_head.py --backbone ... --out fresh-run/export.
systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 python3 code/verify_export.py --repo repo --out fresh-run
systemd-run --user --scope -p MemoryMax=2G -p CPUQuota=200% nice -n 10 ../job_45/venv/bin/python code/benchmark.py --repo repo --out fresh-run
```

Окружение обучения: Python3.14.3/torch2.9.1/NumPy2.4.6. Export: ONNX1.22.0/ORT1.30.0 в существующей локальной venv. Benchmark: существующая job45 venv, ORT1.30.0. Архивная learn_lw вызывается из её точного AST без запуска жёстко заданных исторических entry point. Нынешний service KR скопирован в изолированный snapshot, старый worker repo не изменён.

Worker clock отстаёт от workstation примерно на2.6ч. Связь результатов подтверждается protocol/source SHA и monotonic durations; mtime не используется как доказательство порядка. Код обучения: `53f7261`, baseline:`8e7a205`, frozen protocol:`e1557af`.

Вывод ограничен данным рецептом и бюджетом: он не доказывает невозможности дистилляции вообще. Дополнительных обучений после отрицательной оценки не запускалось.

Независимая приёмка: **PASS** на ML SHA `c8b1fdc` (интеграция delivery `edd9a72`). Критик проверил split/protocol, replay исправления dev whitening, смысловую мутацию, 24/24 артефакта и отрицательное решение. Отчёт: `outputs/independent-ml-review-20260925/REPORT.md` в ML worktree. Открытых блокирующих замечаний нет; default d1_j48 остаётся.

## Исправление независимого замечания P2: dev whitening

Прежние selection/evaluation, таблица и отчёт сохранены как **SUPERSEDED** в `results/superseded-dev-teacher-whitening/`. В них dev baseline ошибочно использовал fit-only teacher whitening, тогда как evaluation использовал release whitening.

Исправление применяет release whitening и к dev baseline/fusion. Перевыбраны вес fusion и отдельные cosine/KR-пороги только на dev. Вес остался0.25, поэтому ranking, mAP, доверительные интервалы и отрицательное решение не изменились. Изменились dev-пороги и их внешние F1/TNR для baseline/fusion. Teacher, оба checkpoint, ONNX и frozen protocol/acceptance проверены по исходным хешам и неизменны. Нового обучения не было.

Release whitening ранее видел dev100 — это ограничивает независимость dev-результата и явно сохранено в selection metadata. Повторная оценка не становится новой untouched-проверкой. Команды, hashes, exit codes и проверки replay: [replay-p2.json](results/replay-p2.json). Benchmark не повторён: вес fusion и вычисления остались прежними.

Удалённый первоначальный запуск сохранён в `worker-vm:~/lct-reid/jobs/distill_20260925/results/`; исправленная оценка — в `results-replay-p2/`, её код — в `code-replay-p2/`. Локальный `artifacts/run/` содержит актуальную исправленную selection/evaluation.
