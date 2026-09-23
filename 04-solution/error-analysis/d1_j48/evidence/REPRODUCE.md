# Повторение расчёта

Исследовательские скрипты запускались вне репозитория, в отдельном каталоге
`worker-vm:/home/fedora/lct-reid/jobs/job_86_review_20260923`.
Их тексты здесь имеют суффикс `.py.txt`, чтобы отличать след исполнения от кода сервиса.
Сеть, сервис, веса, пороги и исходники проекта не изменялись.

## Полный повтор численного этапа без инференса

В отдельном рабочем каталоге восстановить `01_numbers.py` из одноимённого `.py.txt`.
Разложить входы:

- `inputs/repo/04-solution/eval/*.py` — неизменённые файлы репозитория e80c12b;
- `inputs/repo/04-solution/service/app/__init__.py` и `app/core/{__init__,config,ranking,rerank}.py`;
- `inputs/repo/04-solution/split/files/val_query.csv` и `val_gallery.csv`;
- `inputs/embeddings.npy` — validation d1_j48, SHA-256
  `7c0e35b4da3092e1fe394f709b502e1dce251b1fdfe7afc5be7409291ee0dad0`;
  оригинал: `/home/fedora/lct-reid/jobs/job_55/out/val_batch/embeddings.npy`;
- `inputs/rerank_scores.npy` — контрольная матрица из job_82, SHA-256
  `3602e1cab4e999317cf8c4afb4c6728d16e540045b010df2705021b31a0c4102`;
- `inputs/val_{query,gallery}.{npy,ids}` — исходные OSNet-векторы из `baseline/out/`.

Все размеры и хеши входов находятся в `../out/summary.json → inputs`.
Повторить на worker с одним потоком:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 01_numbers.py
```

Скрипт импортирует штатные evaluate и rerank_scores. Assert проверяет точное совпадение
mAP/Rank-1 всех трёх ветвей, порядок .ids и исходных CSV; контрольная матрица тоже сверяется.
Никакие альтернативные формулы AP/CMC/F1/TNR здесь не реализованы. Временная отметка
нового `sample_frozen.json` будет другой; `population` и `sample` должны остаться теми же
при той же NumPy/PCG64. Исходная заморозка в комплекте не перезаписывается.

## Проверка финальной матрицы из самого комплекта

`../out/scores_d1_j48.npz` содержит ключ `scores`, float64, 1110×750.
Для воспроизведения метрик достаточно этой матрицы, штатного evaluate и двух CSV:
query_ids/gallery_ids берутся из vehicle_id, камеры из camera_id,
known_absent = has_mate == "0", threshold=0.5282812306342437,
camera_policy="market", refusal_mode="presence", include_rankings=True.
Сравнить результат с `../out/metrics_check.json → d1_j48`.
Скрипт итоговой проверки — `verify_numeric.py.txt`; результат — `verification.json`.

Проверка seed выполняется только по сохранённым числовым популяциям, без изображений.
Главные причины намеренно не воспроизводятся программным классификатором:
это визуальная разметка. Для проверки доступны все листы, заметки до и после уточнения
контекста и таблица каждого конкретного решения.

## Восстановление листов и иллюстраций

`02_sheets.py.txt` читает исходные JPEG из `/home/fedora/lct-reid/data/images`,
использует исходный bbox и только пропорциональное масштабирование с полями.
`04_context.py.txt` добавляет исходную рамку на полном кадре для семи спорных случаев.
`05_examples.py.txt` читает `examples_spec.json` и делает десять PNG 1920×1080.
Все исходные image_id, bbox и SHA-256 — в `../out/image_sources.json`.
Исходные JPEG не включены вторым полным набором: изображения в листах/примерах доступны
в комплекте, оригиналы — в `data/images` проекта и на worker.

Команда для проверки целостности готового комплекта из его корня:

```bash
sha256sum -c SHA256SUMS
```
