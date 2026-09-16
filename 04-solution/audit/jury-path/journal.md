# Журнал прохождения пути жюри (job_32)

Старт: 2026-09-16 15:27 MSK
Окружение: Linux 6.19.14, podman 5.8.2, podman-compose 1.6.0, Docker отсутствует.
Роль: посторонний человек, единственный источник — SOLUTION.md + README в репозитории.

## Хронология

- 15:27 — прочитал BRIEF.md. Создал журнал. Проверил окружение: podman 5.8.2, podman-compose 1.6.0.
- 15:29 — прочитал SOLUTION.md целиком (518 строк). План: клон → §3 шаг 1 → §3 шаг 2 (веса) → §3 шаг 3 (образы, с сетью и без) → §4 batch → §4 сервис/UI → §7 команда R/T/V → сверка артефактов.
  Заявленные числа для проверки: mAP 0.6936584725, Rank-1 0.6634615385 (строка «OSNet, переранжирование» §6), F1 0.8056112224, TNR 0.7769784173 (строка «Переранжирование, t_rr» §6).
- 15:31 — §3 шаг 1: пути. `DATA_DIR="$REPO/data"` в чистом клоне не существует (data в .gitignore);
  по указанию брифа взял `/home/artem/projects/hackathon-lct-vehicle-reid/data` только на чтение. Все `test -f/-d` прошли.
  Отпечатки §9 сверены: train/test_query/test_gallery/README.md — SHA-256 и размеры совпали.
- 15:32 — §3 шаг 2: `sh model/fetch_model.sh` — OK, 2.4 с, 8 836 743 байта, sha256 сверен скриптом (ЦЕЛ).
- 15:33 — обнаружено расхождение документации: SOLUTION.md §3 шаг 3 и §11 утверждают «сборка требует интернет»,
  но Dockerfile по умолчанию ставит пакеты ОФЛАЙН из `wheels/` (ARG PIP_SOURCE=offline). Слова «wheels», «колёса»,
  «PIP_SOURCE» в SOLUTION.md отсутствуют. 04-solution/service/README.md (стр. 331-341) тоже устарел:
  «офлайн-сборка ... сейчас это не сделано». Актуально только wheels/README.md.
- 15:35 — §3 шаг 3, сборка с сетью: `podman build --no-cache -t vehicle-reid-service .` — OK, 1 м 49 с.
  Подмена: docker→podman. Фактически зависимости ставились ОФЛАЙН из wheels/ (см. запись 15:33).
- 15:39 — сборка без сети: `podman build --no-cache --network none -t vehicle-reid-offline .` — OK, 3 м 25 с.
  Оговорка: базовый образ python:3.13-slim уже лежал в локальном хранилище (подтянут первой сборкой).
  На машине без него сборка с --network none упадёт на FROM; wheels/README.md об этом не предупреждает.
- 15:43 — команда V: `podman run --rm --network none vehicle-reid-service python -m pip freeze --all`
  — 31 строка, побайтово совпадает с блоком §10 SOLUTION.md и с requirements-lock.txt (без шапки-комментария).
  Офлайн-образ даёт тот же freeze. `pip check` — без замечаний. Python 3.13.15 — как в §10.
  Базовый образ: ID 51cce855bb6e...c210f и digest sha256:9d2e5553...85 — совпали с §10.
- 15:44 — `podman-compose pull qdrant` (COMPOSE_FILE из шага 1) — OK, 4 с.
  Digest qdrant sha256:0fb88974...863 совпал с §10.
- 15:45→15:51 — §4 пакетный прогон, команда как в документации (docker→podman):
  ПЕРВАЯ ПОПЫТКА УПАЛА через 5 м 33 с: `PermissionError: [Errno 13] Permission denied: '/out/embeddings.npy'`.
  Это тот самый SELinux-случай, про который в §4 есть только сноска мелким шрифтом.
  Прогон отрабатывает инференс целиком и падает только на записи — потерянные 5,5 минут.
- 15:51→15:57 — повтор с `--security-opt label=disable` (подмена №2) — OK, 5 м 27 с.
  run_info: threshold 0.49937235233589916, rerank=true, (6,3,0.3), 1110/750, (1860,512),
  1078 с кандидатами, 32 отказа (0.0288). embed_elapsed 317.7 с (у эталона 89.7 с — машина медленнее/загружена).
- 15:58 — СВЕРКА АРТЕФАКТОВ (п.3 брифа): submission.csv, candidates.csv, embeddings.npy — все три
  ПОБАЙТОВО совпали с 04-solution/service/artifacts-final/ рабочего дерева. run_info.json отличается
  только полями времени (embed/rank/total_elapsed_s) — как и ожидается.
- 15:59 — §4 сервис: `podman-compose up -d --no-build --pull never` (подмена №3: docker compose→podman-compose)
  — OK. Контейнеры service_qdrant_1 (healthy), service_api_1, service_loader_1.
  ВАЖНО: том `service_qdrant_storage` уже существовал (создан 10:04 сегодня прошлым прогоном в рабочем дереве).
  Имя проекта compose выводится из имени каталога (`service`), поэтому клон в другом месте
  переиспользует контейнеры и том рабочего дерева. Мой `up` не проверил старт с пустым хранилищем.
  loader всё равно пересоздал коллекцию: `{"collection": "gallery", "points": 750, "rows": 750}`, exit 0.
- 16:00 — health: storage.reachable=true, gallery_points=750. version: порог API 0.5495953464415451,
  batch_rerank {enabled_by_default: true, (6,3,0.3), 0.49937235233589916} — совпадает с §2 SOLUTION.md.
- 16:01 — веб-интерфейс: GET / → 200, 10 315 байт; /docs → 200; /openapi.json → 200 (8 эндпоинтов).
  Статика /static/app.css, /static/app.js, vendor/* → 200. Отпечатки Swagger (bundle 1 510 312 / a646692b…,
  css 155 212 / bc5e8d5c…, favicon 628 / 3ed612f4…) совпали с §9.
  Скриншот страницы (google-chrome --headless) — ui-index.png: шапка «галерея: 750 объектов»,
  зона загрузки кадра, панель результата, подвал «порог отказа 0.549595».
- 16:02 — живой запрос: POST /api/search (кадр 486dd80d…, bbox 1202,270,588,474, top_k=5)
  → refusal=false, best_confidence 0.705665, 1 кандидат. POST /api/embed → dim 512.
  /api/gallery/{id}/crop → 200 image/jpeg; /api/explain → 200, карта 13×13.
- 16:03 — команда T: ПЕРВЫЙ ЗАПУСК УПАЛ — `ModuleNotFoundError: No module named 'test_protocols'`,
  под капотом `ls: cannot open directory '.': Permission denied` — SELinux на `-v "$REPO:/repo:ro"`.
  Повтор с `--security-opt label=disable` (подмена №4) — `Ran 60 tests ... OK`, 10 с. §8 подтверждён.
  Замечание: сноска про label=disable есть только в §4 (пакетный прогон), хотя нужна и для T, R и калибровки.
- 15:52 — запущена команда R (с той же подменой podman + label=disable), лог R.log.
- 16:11 — команда R завершилась (запущена 15:52, ~19 минут). Вывод:
  REPRODUCED {"base_mAP": 0.6565692724907637, "rerank_mAP": 0.6936584724586873,
  "F1": 0.7330567081604425, "TNR": 0.697841726618705, "threshold": 0.5495953464415451}
  Все числа §6 совпали (mAP/Rank-1/Rank-5/mINP/mAP@10 обеих строк, Δ mAP 0.03708919996792359),
  обе таблицы отказа совпали, оба порога заново выбраны и совпали (threshold_abs_delta = 0.0),
  row_order = 1.0 на всех четырёх границах, split 1110/750/832/278, модель 8836743 / 4aaad3e5…
  ВНИМАНИЕ: печатаемые R «F1»/«TNR» — это косинусная ветвь (0.733/0.698), а НЕ заявленные в §6
  0.806/0.777; последние лежат в metrics.json → rerank_refusal.
  Хеши файлов test-прогона внутри R совпали с artifacts-final — второе независимое подтверждение
  (причём при другом --batch: 32 против умолчания в §4).
- 16:17 — §8 «185 прямых сверок»: tools/calibrate_threshold.py (podman + label=disable) — 5 м 13 с,
  verification.json: checks=185, max_abs_error=0.0; headline.json повторяет t_rr/t_cos и обе точки.
- 16:20 — §9 манифест изображений: 11416 7503286826 8274bf0d792e3b4fee1009f6fae7649466c296cffcb359c670150a587473cc4a
  — совпал точно (2 м 32 с, Python 3.13.13 на хосте).
- 16:22 — §4 офлайн-поставка. `podman save` БЕЗ флага: 415 203 840 байт, в архиве 1 образ с двумя тегами
  (поведение как описано, размер не совпал: в §4 указано 355 952 640).
  С `--multi-image-archive`: 596 321 792 байта, 2 образа (в §4 указано 537 070 592). `podman load -i` — OK.
  Разница в обоих случаях ровно 59 251 200 байт = слой `COPY wheels/ /wheels/` (podman history: 59.2 MB),
  который остаётся в образе, потому что `rm -rf /wheels` выполняется следующим слоем.
- 16:25 — §4 «повторная загрузка на другом каталоге»: `DATA_DIR=... podman-compose run --rm loader`
  — OK, 3 м 26 с, {"collection": "gallery", "points": 750, "rows": 750}.
- 16:27 — §8, честность пометок «не воспроизводится»: s09_timing.py и eval_variants.py падают сразу
  (`ModuleNotFoundError: No module named 'onnxruntime'`), s04_grid.py — `ValueError` на argv.
  Пометки в §8 верны, но команды в таблице записаны как исполняемые.
- 16:28 — проверка ссылок: 28 локальных ссылок SOLUTION.md и все ссылки README.md разрешаются в клоне.
  Эндпоинты, которые дёргает app.js (/api/search, /api/explain, /api/ui/state), есть в OpenAPI.
- 16:29 — уборка: `podman-compose down` (том service_qdrant_storage оставлен — он был до меня),
  удалён только мой образ vehicle-reid-offline. Тег vehicle-reid-service:latest существовал до прогона
  (моя сборка его перезаписала) — оставлен на месте.
- 16:33 — клон удалён, временные файлы в /tmp убраны. Контейнеров моих нет, том оставлен,
  тег vehicle-reid-service:latest оставлен (существовал до прогона). Данные организатора не менялись.
- 16:35 — REPORT.md готов. Итог: сдаваемые файлы воспроизводятся побайтово, головные числа сошлись
  точно, 9 расхождений документации (Н-1…Н-9), 4 подмены docker→podman + 1 отклонение по пути данных.
  Полное время пути — 1 ч 08 мин, из них ~40 мин чистого машинного времени.
