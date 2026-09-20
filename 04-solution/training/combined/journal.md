# journal — job_48 (дообучение на объединённом наборе, GPU thinkpad)

## 2026-09-19, старт
- Прочитан BRIEF.md. План: сверка хэшей → скрипт обучения (комбинированный набор) →
  один зафиксированный прогон на worker (GPU) → перенос .pt+.onnx на worker-vm →
  оценка контуром 04-solution/eval (cos + KR(6,3,0,3), бутстрэпы vs 0.7621/0.6937,
  ансамбли, разрыв по камере) → абляция номера (extract_variants с новым ONNX).

## Инфраструктура (проверено)
- worker = Windows, PowerShell по умолчанию; `[Console]::OutputEncoding=[Text.Encoding]::UTF8`
  первой строкой. GPU Quadro M2000M 4 ГБ свободна (0 MiB, нет чужих python-процессов).
- GPU.lock на D:\lct-reid\ уже содержит `job_48` (BOM+job_48) — замок НАШ, оставлен.
- Окружение: D:\lct-reid\py312\python.exe (torch 2.5.1+cu118). Код: reid_train4.py
  (полный рецепт MCNL/LP-FT, 3 этапа), osnet_ain.py, reid_metrics.py, export_onnx2.py.
- Прошлая попытка ain_v2 (только наши данные): stage1 LP 5 эп (freeze_fc) dev mAP
  неподвижна 0.71358; stage2 conv4+conv5 15 эп lr 3.5e-5 → dev mAP 0.75578, но на val
  0.700 vs эталон 0.6937 (KR) — прирост в пределах шума, R1 хуже. Этап 3 не делали.

## Хэши после переноса (worker, Get-FileHash SHA256)
- combined_train.npz: `C1EE49DEE0873A5748E7B799613FD1E14B4D3689BCD93B3CF51D24D08C42652C` ==
  эталон c1ee49de…42652c (job_44) ✓
- combined_crops_208.npy: `BABD228FBE649999A09E9B2D83F3701DE9C1B17424A72E23331940DCCA3521CC` ==
  эталон babd228f…3521cc (job_44) ✓
- Оба файла на месте в D:\lct-reid\data\. Обучать можно.

## Структура данных (из build_combined.py job_44, проверяется пробой на worker)
- combined_train.npz: data/offsets/vehicle_id/camera_id/image_id/source/size/quality.
  raw combined_crops_208.npy (N,208,208,3) uint8 выровнен с npz построчно, own первые.
- own 7248 кадров / 1171 ID / камеры как в train_crops.npz; roundabout: ID+100000,
  камеры +1000 (511 ID, 6068 кадров); carla: ID+200000, камеры +2000 (605 ID, 7260 кадров).
  Итого 20 576 кадров, 2287 ID, 114 камер; 91% ID с ≥2 камерами.

## Контур оценки (worker-vm)
- Векторы val: извлечение из кадров ~/lct-reid/data/images/ по split/files/
  val_query.csv (1110) / val_gallery.csv (750), препроцессинг baseline/scripts/
  extract_embeddings.py (bbox→RGB→208 BILINEAR→0..255 NCHW→ONNX→L2).
- Метрики: 04-solution/eval/reid_metrics.py evaluate (market, presence), KR(6,3,0,3)
  из 04-solution/postproc/scripts/common.rerank, whitening job_42/job_45 (learn_lw/apply_lw),
  парный бутстрэп lib45.paired_bootstrap (4000, seed 20260916).
- Абляция номера: 04-solution/plate-ablation/scripts/{extract,eval}_variants.py,
  REID_MODEL_PATH env подменяет модель; кэш боксов work/boxes.json готов.
- Эталоны: d1_w1.0 = 0.7621 mAP (mAP@KR, 832 known-запроса); чистый OSNet+KR = 0.6937.

## Параметры попытки (зафиксированы ДО старта, 2026-09-19 ~16:15)
- Скрипт: reid_train5.py (= reid_train4.py/ain_v2, два изменения: combined-данные,
  dev-пул только own-ID). Оркестратор: job48_chain.py, запущен WMI PID 18264,
  логи D:\lct-reid\runs\combined_v1\.
- Данные: combined 20 576 кадров / 2 287 ID / 114 камер; train 19 967 кадров /
  2 187 классов (own 1 071 + foreign 1 116); dev = 100 own-ID (372 запроса/237 галерея,
  те же что ain_v2 — baseline dev mAP 0.71358 / CH 3.1192 подтверждён на эпохе 0).
- Сэмплер: C=4 камеры × P'=4 ID × K=4 кадра = 64, own_k=2, camera-balanced, seed 20260916;
  311 батчей/эпоху; лосс MCNL m1=m2=0.1 (pre-BN) + CE ls=0.1 (post-BN), w_id=1.0.
- Аугментации: hflip + pad-crop 10px; MixStyle p=0.5 α=0.1 после stem и mid; REA снят.
- Adam wd 5e-4 foreach=False; warmup 2 эпохи, косинус внутри этапа.
- Этап 1 (LP): 5 эп, --freeze-fc (только classifier), lr 3.5e-4.
- Этап 2: conv4+conv5+fc, 15 эп, lr 3.5e-5, cls-lr 3.5e-4.
- Этап 3 (ворота: best dev mAP этапа 2 > baseline И CH не растёт 3 замера подряд):
  полная разморозка, 6 эп, lr 1.5e-5, cls-lr 3.5e-4.
- Выбор модели (до старта): этап с большей dev mAP на конец; ничья → меньший CH.
  На val идёт ТОЛЬКО выбранная модель + её ансамбли.
- Дымовой прогон пройден: конфиг 2187 классов/107 камер, GPU 2595 MiB, ~223 с/эпоху.

## Ход обучения (worker time, лог runs/combined_v1/)
- 16:19:52 — WMI PID 18264, chain запущен. GPU.lock = job_48 (наш).
- Этап 1 (LP, freeze backbone): 5 эпох, ~218 с/эп; loss 7.87→7.50, id 7.46→7.31,
  dev mAP 0.71358 / CH 3.1192 неподвижны (LP — классификатор, как и в ain_v2).
  lr 3.5e-4 с warmup(2)+косинусом → 8.75e-5 к эп. 5. Готово 16:42:57.
- Этап 2 (conv4+conv5+fc) начат 16:43:00, 15 эп, lr 3.5e-5/cls 3.5e-4.
  эп.1 16:47:56: loss 6.92, mcnl 0.1741, dev mAP 0.71358→(эп1) те же, CH 3.1192 —
  рост dev ожидается после первых эпох (backbone lr мал), как в ain_v2.
  эп.2 16:52:53: dev mAP 0.71358 → смена параметров conv4/conv5 медленно доходит;
  ain_v2 тоже держал 0.71358 первые эпохи и сдвинулся к эп. 10+.
- Монитор v2 (события только на переходах/тревогах/терминале) — буфер buehtyhph.
- Эп.3 16:57:50: dev mAP впервые сдвинулась: 0.71358 → 0.73925, R1 0.68817 → 0.73656,
  CH 3.1192 → 3.1847. Направление как в ain_v2 (через 3 эпохи stage2 dev ожила).
- Эп.5 17:07:47: mAP 0.73925, CH 3.1861 (эп.3-5: 3.1847/3.1861 — стабильна).
  Сверка с ain_v2 (stage2 dev mAP по эпохам): 0.71425/0.72581/0.73077/0.74304/0.74078…
  — наш эп.1-5: 0.71358/0.71358/0.73925/0.73925/0.73925. Похожий ход, сдвиг на ~1 эпоху.
- 16:39:18 — этап 1 завершён за ~3.5 мин на эпоху (быстрее оценки), chain стартовал
  этап 2 с зафиксированными аргументами: resume stage1.pt, lr 3.5e-5, cls-lr 3.5e-4.
  Эпоха 0 (baseline после resume): dev mAP 0.71358 / CH 3.1192 — контроль сходится.
  ETA этапа 2: ~17:35 worker time.
- Проверка готовности контура оценки (worker-vm, до окончания обучения):
  venv = ~/lct-reid/jobs/job_45/venv/bin/python; s01/s02/s03 компилируются;
  lib45/reid_metrics/mask_ops импортируются; boxes.json (job_45b/out) 1110/750 ==
  split val_query/val_gallery 1110/750. Ждём только .onnx.
- Эп.1 stage2 (16:44:43): dev mAP 0.6534, CH 3.2316 — просадка при разморозке conv4/conv5.
  Сверка с ain_v2: там dip 0.6441 был на эп.2, потом рост до 0.75578 (эп.15).
  Наш dip на эп.1, мельче — поведение рецепта воспроизводится. 293 с/эп.
- Эталон траектории ain_v2 stage2 (dev mAP): 0.71395/0.6441/0.69501/0.71086/0.72519/
  0.72974/0.73292/0.73592/0.7432/0.74574/0.74938/0.74726/0.75263/0.74169/0.75578.
- Эп.2 stage2 (16:50): mAP 0.70631, CH 3.314 — восстановление идёт на ~1 эпоху
  быстрее ain_v2 (там 0.69501 на эп.3). Траектория повторяет рецепт.
- Эп.3 stage2 (16:54:56): loss 4.9299, mAP 0.73242, CH 3.251 (пик CH пройден,
  как у ain_v2). Опережаем траекторию ain_v2 (там эп.3=0.69501).
- Эп.4 stage2 (16:59:28): loss 4.1824, mAP 0.74584, R1 0.68280, CH 3.2148.
  Уровень ain_v2 эп.10 достигнут на эп.4. dev-динамика уверенно выше прошлого прогона.
- Эп.5 stage2 (17:04): loss 3.6001, mAP 0.76045, R1 0.70430, CH 3.1693 —
  на эп.5 уже выше финального dev ain_v2 (0.75578). GPU 100%, 286 с/эп.
- Эп.6-12 stage2: mAP 0.74868/0.74894/0.75870/0.77264/0.77159/0.77319/0.77652,
  CH монотонно ↓ 3.1561→3.0719. Эп.13: mAP 0.77719 (максимум), R1 0.73118, CH 3.0706.
  Эп.14-15: 0.77199/0.77211, CH 3.0717/3.0731.

## Обучение завершено (worker 17:52:17)
- Этап 2 финиш за 72.8 мин; лучшая dev mAP 0.77719 @ эп.13 (ain_v2: 0.75578 — прирост +0.0214).
- Ворота этапа 3: mAP 0.77719 > 0.71358 = True, но CH rising tail (3.0706→3.0717→3.0731)
  = True → этап 3 ПРОПУЩЕН по зафиксированному правилу (камеро-специфичность растёт).
- Победитель: этап 2. Export rc=0 → D:\lct-reid\job48_combined_s2.onnx
  SHA256 b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2,
  8 742 779 байт, opset 13, вход 1×3×208×208, dim 512.
- GPU.lock удалён сразу после экспорта (Remove-Item, проверено отсутствует).
- Полные логи: runs/combined_v1/{chain.log, stage1_log.jsonl (6 строк), stage2_log.jsonl
  (16 строк), stage{1,2}_config.json, chain_s{1,2}.out.txt}.
- Траектория stage2 dev mAP по эпохам (0-15):
  0.71358/0.6534/0.70631/0.73242/0.74584/0.76045/0.74868/0.74894/0.75870/0.77264/
  0.77159/0.77319/0.77652/0.77719/0.77199/0.77211
  (эталон ain_v2: 0.71395/0.6441/0.69501/0.71086/0.72519/0.72974/0.73292/0.73592/
   0.7432/0.74574/0.74938/0.74726/0.75263/0.74169/0.75578 — итог 0.75578 на эп.15).
  Комбинированный набор дал выше dev на каждой эпохе начиная с эп.2.

## План пост-обучения
1. Перенести onnx+pt+логи на worker-vm:~/lct-reid/jobs/job_48/ (выполняется).
2. s01_extract.py (venv job_45, OMP_NUM_THREADS=1) → out/val_{query,gallery}_j48.npy,
   train_fit_j48.npy (~50 мин CPU).
3. s02_eval.py: одиночная + 4 ансамбля + repro-эталоны + парные бутстрэпы B=4000.
4. s03_plate.py extract+eval: 7 вариантов, контрасты plate vs shift.
5. REPORT.md §3-6, финальный отчёт.
- Перенос на worker-vm выполнен: onnx (sha256 b1ba5021…02efb2 совпал на обоих
  концах), runs/combined_v1/ полностью (все .pt/.jsonl/.log/.json), scripts/.
- Баг в s01/s03 (runtime, py_compile его не ловит): next(sorted(glob)) — sorted
  возвращает list. Исправлено на sorted(...)[0] во всех копиях (local+vm).
- s01_extract.py запущен worker-vm PID 110510 (nohup, OMP_NUM_THREADS=1,
  venv job_45), лог out/s01.log. Модель загружена, sha256 в логе совпал,
  старт val_query 0/1110. CPU ~46% (1 поток, хост общий), RSS 528 МБ.
  ETA ~50 мин. Дальше: s02_eval → s03_plate extract+eval.
