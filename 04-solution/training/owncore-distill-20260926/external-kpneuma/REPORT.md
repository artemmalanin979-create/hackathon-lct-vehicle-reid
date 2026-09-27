# CrossViewCore на внешнем KPNEUMA

Статус: **PASS** для исполнения оценки, **не подтверждён выигрыш над релизом**.
Срез 28.09.2026; основной source SHA
`ed028e82702ee7d80904b5a49cd5cc5c23b5505c`. Замороженный до инференса
[протокол](PROTOCOL.md) SHA-256
`8b2e4edeb4dbf569d99ee21e8ae916584727c9a2e4bce2442678313f4fe6bbae`.
Веса **combined CrossViewCore** проверены по ONNX SHA-256
`4cb47b31cf2abf3eabb9fdf30f8de5bc4f06cfe2306670a01fe1bc9e95961df6`;
это одна сеть `image → 512D`, учитель при инференсе не используется.

Все **318 query и 318 gallery** оценены, 318 допустимых междетекторных пар,
0 пропущенных запросов. Проверка имён/содержимого, уникальности ID, пар и
checkpoint завершилась до запуска ONNX; [preflight](preflight.json) SHA-256
`b2302b8a23e7518f4f72f4361234bf9a58e7cd5f5c8e63d36c19a9c58ccc1359`.
Идеальная выдача дала mAP 1,0000, фиксированные случайные признаки — 0,0258.

| Модель | Поиск | mAP | 95% bootstrap CI mAP | Rank-1 | Rank-5 | mINP |
|---|---|---:|---|---:|---:|---:|
| Релиз `d1_j48` | cosine | 0,490256 | [0,442709; 0,536601] | 0,383648 | 0,603774 | 0,490256 |
| CrossViewCore | cosine | **0,444908** | [0,400001; 0,487090] | 0,314465 | 0,600629 | 0,444908 |
| Релиз `d1_j48` | KR 6/3/0.3 | 0,406962 | [0,361534; 0,450829] | 0,286164 | 0,525157 | 0,406962 |
| CrossViewCore | KR 6/3/0.3 | **0,408970** | [0,364678; 0,454601] | 0,286164 | 0,534591 | 0,408970 |

Для межмодельного сравнения обе модели **повторно извлекли признаки из тех же
636 PNG**. Агрегаты обоих прогонов воспроизвели прежние значения с
допуском 1e-9. [Парный результат](paired-results.json) сохраняет AP каждого
из 318 запросов в порядке полных имён файлов и 2000 bootstrap-повторов по ID:

| Режим | Δ mAP ядро − релиз | Парный 95% CI | Лучше / хуже / равно по AP запросов |
|---|---:|---|---:|
| cosine | −0,045348 | [−0,091880; +0,003068] | 112 / 132 / 74 |
| KR 6/3/0.3 | +0,002007 | [−0,045308; +0,050057] | 138 / 123 / 57 |

**Оба межмодельных интервала включают ноль.** В частности, точечные +0,002
KR mAP не подтверждают превосходство ядра над релизом. Внутри CrossViewCore
KR хуже cosine на 0,035939 mAP; парный bootstrap 95% CI
[−0,058270; −0,012164].

Полный проход 636 PNG (декодирование, resize и ONNX) занял **2,775 с** на
локальном CPU с одним ORT-потоком; ранжирование и метрики — 0,283 с.
Релизный отдельный проход занял 91,94 с, но **сопоставлять эти два elapsed
как ускорение нельзя**: прогоны не были чередующимися, фоновая нагрузка могла
различаться. Парный повторный прогон ниже тоже служит проверке AP, не
контролируемому speed benchmark. Парный замер границы готовый
тензор → дескриптор на val уже записан в основном
[отчёте CrossViewCore](../REPORT.md): p95 1,727 мс против 43,861 мс.

Исполненные команды (из корня основного репозитория), все три **exit 0**:

```bash
MODEL=/home/artem/projects/lct-owncore-distill-20260926/outputs/owncore-distill-20260926/assessment-combined-d94d3fd/core.onnx
CHECKPOINT=/home/artem/projects/lct-owncore-distill-20260926/outputs/owncore-distill-20260926/train-combined-d94d3fd/checkpoint.pt
DATA=/home/artem/tmp/lct-external-kpneuma-20260928
RUN=04-solution/training/owncore-distill-20260926/external-kpneuma/run.py
OUT=04-solution/training/owncore-distill-20260926/external-kpneuma

nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python "$RUN" \
  --dataset-root "$DATA/extracted/data/KPNEUMA" --archive "$DATA/data.tar.gz" \
  --model "$MODEL" --checkpoint "$CHECKPOINT" --preflight-only \
  --out "$OUT/preflight.json" > "$OUT/preflight.log" 2>&1
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python "$RUN" \
  --dataset-root "$DATA/extracted/data/KPNEUMA" --archive "$DATA/data.tar.gz" \
  --model "$MODEL" --checkpoint "$CHECKPOINT" --preflight "$OUT/preflight.json" \
  --out "$OUT/results.json" > "$OUT/run.log" 2>&1
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python "$OUT/paired.py" \
  --dataset-root "$DATA/extracted/data/KPNEUMA" --archive "$DATA/data.tar.gz" \
  --model "$MODEL" --checkpoint "$CHECKPOINT" \
  --out "$OUT/paired-results.json" > "$OUT/paired.log" 2>&1
```

Python 3.13.13, NumPy 2.5.3, Pillow 12.3.0, ONNX Runtime 1.30.0,
batch 16, ORT 1 thread. Машиночитаемый [результат](results.json) SHA-256
`ab26bee61a9e8998931d2783353011b2fad2fff7d4761c81198697794f674594`.
Парный [результат](paired-results.json) SHA-256
`0599ad0b5af97bc4a814bc77c1849e6b0cc0114b1d12a0df2df1b113a45fc3e4`;
`paired.py` SHA-256
`d8627f7d694a648feef37ab1991e8b45ea13fc2017effe44adec5c45333e50b1`;
исполнение `paired.py` exit 0, 318/318 запросов, 0 skipped, 95,124 с полного
последовательного прохода. Логи скопированы побайтно из локальных `.log` в
публикуемые [preflight-output.txt](preflight-output.txt),
[run-output.txt](run-output.txt) и [paired-output.txt](paired-output.txt).
Данные и ONNX
находятся вне Git; изображения не публикуются.

Это аэросъёмка с виртуальными детекторами, а не закрытый тест ЛЦТ.
Здесь ровно одна релевантная пара у каждого query; запросов без пары нет,
поэтому F1/TNR отказа **NOT MEASURED**. Внешний результат не меняет
решение о релизе: на локальной val собственного ядра KR mAP 0,327450 против
0,774092 у `d1_j48`, и релизная сборка остаётся прежней.
