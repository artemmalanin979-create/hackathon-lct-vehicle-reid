# journal.md — job_100, критик правки имён с точкой

Задание: проверить правки 36d08f9 + c459c26 (resolve_image_path / image_candidates)
в репо `/home/artem/projects/hackathon-lct-vehicle-reid` @ 56bb045.
Ничего в репо не менять. Тяжёлое — на worker-vm.

## 2026-09-25

### Подготовка
- worker-vm доступен (sespel-worker.local, 4 ядра, 31 ГБ). Контейнер
  localhost/job72-jury:6179b69, сеть none, работает от root.
- Эталон job_99: submission.csv bfd372b2…d83f, embeddings.npy 1d85f346…9dab0,
  candidates.csv d1364538…ba66, run_info.json b08c4da3…5f1c, metrics.json
  5cce7143…2940. Метрики: rerank mAP 0.7740915539438481 / R1 0.7307692307692307;
  cosine 0.7316093422310448 / 0.6850961538461539.
- Копия репо на worker: ~/lct-reid/jobs/job_100/repo (rsync, exit 0).
  Сверены SHA-256 7 ключевых файлов (включая ONNX модель 4aaad3e5…d91f =
  model_sha256 подтверждённого прогона) — совпали с локальным рабочим деревом @56bb045.
- Контейнер работает от root: chmod 000 внутри него не отклоняется → «файл без
  чтения» проверен локально не-рутом.

### Свой прогон (stage2, оркестратор worker_orchestrate.sh)
- reproduce/run.py --mode val, offline, 2cpu/4g, batch=32, threads=2.
- VAL_RC=0. Входы: OK, 1864 файла. Продолжительность 888.6 c (в параллель шли
  мутации; у job_99 было 852.6 c).
- Три сдаваемых файла ПОБАЙТОВО совпали с подтверждённым прогоном job_99
  (sha256 идентичны: bfd372b2…, 1d85f346…, d1364538…).
- metrics.json: base/reranked mAP, Rank-1, Rank-5 и оба порога перекалибровки
  ТОЧНО равны job_99 (сверка пословно, stage6-metrics-compare.log).
- REPRODUCED-строки идентичны job99-данные-val-run.out, включая F1/TNR
  (0.7684996605566871 / 0.7302158273381295 cosine; 0.8090787716955942 /
  0.7841726618705036 rerank).

### Тесты
- service/tests: unittest 52/52 OK (15 c). pytest в jury-образе нет.
- Прочие наборы: eval 60/60, reproduce 16/16, audit/docs-fixes 7/7,
  05-presentation 3/3 — все OK.
- 06-documentation: 5 прогонов, 3 прошли; 2 заблокированы отсутствием pdfinfo
  (poppler) в образе — средовая причина, к правке отношения не имеет.
- Заявленные «144» = все наборы вместе; в jury-образе воспроизводимо 141+2
  средово-заблокированных.

### Мутации — важная методическая поправка
- Первая версия харнесса копировала дерево service в /scratch и гоняла там:
  ЧИСТАЯ копия дала failures=1, errors=38 — артефакт (тестам нужен контекст
  всего репо). Все выводы первого прогона (включая «все 10 пойманы») отброшены.
- Переделано: мутированный файл накладывается bind-mount'ом поверх /repo
  (mut_overlay.sh). Контроль PRISTINE через тот же путь: 52/52 OK.

### Итог мутаций (overlay-харнесс, 10 мутаций)
- Пойманы: M1 (7F+12E), M2 (17F+16E), M3 (2F), M4 (1F), M6 (8F), M7 (16E),
  M9 (5F), M10 (9F+1E) — все через unittest.
- НЕ пойманы: **M5** (снят guard `"" in suffixes`) и **M8** (снят FIFO-guard) —
  оба прогона 52/52 OK.
- M5 поведенчески: research-контракт suffixes=(".jpg",) меняет смысл для ID с
  расширением изображения: baseline car.png → ['car.png.jpg'], M5 → ['car.png']
  (измерено). Blast radius — только research/loader пути, штатный batch не трогает.
- M8 поведенчески: batch с FIFO-query при снятом guard — preflight съедает поток,
  loader висит на повторном открытии FIFO: HANG >60 c (измерено; baseline rc=0
  за 5.6 c, выходы побайтово равны обычному CSV). Механизм: mut-M8-fifo.log
  (left_for_loader_bytes=0).

### Пробы имён (probes_names.py, 18 изолированных кейсов, baseline)
- Все корректны: точечные ID (frame.v1, cam.2026.09.24), верхний регистр
  (UPPER.PNG литерально, Name.V1 → .jpg), пробел, unicode, symlink, без
  расширения (stem → stem.jpg; valid.dat — точное имя), приоритет
  .jpg>.jpeg>.png>точное, повреждённый точный файл не падает на тень.
- Защита в обе стороны: named.png не берёт named.png.jpg (нет файла/битый);
  named.png.jpg при наличии только named.png → FileNotFoundError; frame.v1.png
  литерально защищён от frame.v1.png.jpg.
- Файл chmod 000 (не-рут): preflight OK (is_file), отказ чистый на этапе
  декодирования (CropInputError → batch exit 2) — ложного принятия нет.

### Пробы CLI/потоков (probes_streaming.py, реальная модель, 5 строк)
- batch=32 / batch=1000(>строк) / FIFO-query / FIFO-gallery: embeddings и оба
  CSV ПОБАЙТОВО одинаковы (emb_sha fd9bd2d0…).
- batch=1: embeddings отличаются от batch=32 max |Δ|=6.7e-08 (документированный
  float32-дрейф; нормы в допуске), выходные файлы на фикстуре побайтово те же.
- --no-rerank: шкала cosine, порог по умолчанию косинусный, embeddings те же.
- --batch 0 / −1: rc=2, чистая ошибка argparse, вычислений нет.
- M9 поведенчески (batch=0 допущен): rc=1 с traceback ValueError из embed_rows,
  каталог выхода не создан — мусор с rc=0 (баг job_96) не возвращается.

### Ошибки/артефакты, которые нашёл и учёл
- pgrep -f "rsync.*job_100" на worker всегда «RUNNING» — самоподбор паттерна
  в cmdline bash -c; rsync на самом деле завершался.
- M8 e2e первый раз: забыл -v /data → ложные «Нет изображения»; переделано.
- M8 e2e второй раз: забыл -i у podman run → stdin не дошёл; переделано.
- Вывод: baseline FIFO e2e rc=0 5.6 c; M8 HANG >60 c.
