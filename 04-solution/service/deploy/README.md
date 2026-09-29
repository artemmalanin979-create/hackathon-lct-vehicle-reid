# ЛЦТ №7 — развёрнутый сервис и воспроизведение релиза

Срез 29.09.2026. Публичный прототип **DEPLOYED** на выделенной Yandex Cloud VM `84.201.146.191` (Ubuntu 24.04, 2 vCPU, 4 GiB RAM, SSD 50 GiB). Проверенный неизменный bundle данных и моделей построен из `fcb038af681405b07219542833ce3b80395055b0`; текущий API запущен из отдельно загруженного CPU-образа с revision `e3cd2b0d0a538fa2e383ba43acb763e903a63aef`. Новый основной GPU-образ предназначен для стенда организаторов, не для этой VM без NVIDIA. Галерея, веса и порог HTTP-поиска не менялись. Лендинг и публичные документы размещаются отдельно. Решение отправлено через платформу организатора 28.09.2026; изменения в репозитории допускаются до **29.09.2026, 23:59 МСК**. Перед обновлением сервиса сверяйте фактический контейнер, `/api/version`, ссылки и SHA.

## Адреса и состав

| Назначение | Публичный адрес |
|---|---|
| Основной вход | <https://iamcp.ru/> (также `www.iamcp.ru` и `lct.iamcp.ru`) |
| Рабочий сервис | <https://demo.lct.iamcp.ru/>; путь на основном домене: <https://iamcp.ru/workspace/> |
| Полная презентация для поля платформы | <https://iamcp.ru/submission/presentation.pdf> (16 страниц); редактируемый [PPTX](https://iamcp.ru/submission/presentation.pptx) |
| Документация для поля платформы | <https://iamcp.ru/submission/documentation.pdf> |
| Исходный код | Открытый [репозиторий команды на GitHub](https://github.com/artemmalanin979-create/hackathon-lct-vehicle-reid); зеркало для организаторов — закрытый [SourceCraft](https://sourcecraft.dev/lct-hackaton-2026/case-17-vehicle-digital-signature-team-33), анонимный 404 ожидаем |

На VM Caddy завершает HTTPS. Проверенная конфигурация сохранена побайтно в [Caddyfile.iamcp-20260929](Caddyfile.iamcp-20260929) (SHA-256 `c90647a169fd52efac53d9375e1848c6c344513f62cdda9f3d37de2bc557bac0`). `/submission/` выдаёт только три явно разрешённых файла из `/srv/lct/portal-assets/submission/`; `/workspace/`, `/api/`, `/static/`, `/materials/`, `/docs` и `/openapi.json` направляются на API, остальные пути основного домена — на статический лендинг `/srv/lct/landing/`. Для локальных OTF-шрифтов Caddy задаёт один корректный `Content-Type: font/otf`; проверены лендинг и оба адреса сервиса. `demo.lct.iamcp.ru` направляется на тот же API. Полный PDF/PPTX содержат обязательный слайд с контактами и публикуются для сдачи; технический PDF в каталоге материалов — отдельные 11 страниц без контактов. Qdrant не имеет внешнего порта, API слушает на хосте только `127.0.0.1:18071`. Сетевые входы VM: HTTPS/HTTP для жюри и ограниченный SSH; порт 6333 не открыт наружу.

Релизная модель — **d1_j48**: OSNet-AIN + дообученная `combined_v1`, нормировка, усреднение, whitening `ρ=0,5`, 512 чисел. HTTP-сервис ищет по cosine с собственным порогом. Для `submission.csv` пакетный код отдельно переранжирует cosine top-50 **каждого запроса независимо** методом KR `(6, 3, 0,3)`; числа этого режима нельзя приписывать HTTP. `GET /api/version` показывает SHA обеих ONNX и whitening. В текущем серверном индексе **750** объектов. Immutable bundle содержит нужные 1860 test-изображений и CSV для этого сценария, а не весь исходный датасет. Публикация дополнительных изображений организатора и test-меток не входит в развёртывание.

Демо-кадры проходят обычный `/api/search`, результат не подставляется заранее. Первый прежний пример с почти тем же кадром в top-1 заменён после [визуальной проверки](../ui-checks/demo-example-20260928.md): новый положительный query `e03daa7c…` имеет полный bbox, а первые два кандидата сняты с других точек. Меток закрытого test нет, поэтому это демонстрация ранжирования, а не доказанная межкамерная пара. Второй пример показывает отказ. Изменение каталога демо не изменило canonical batch-метрики.

## Что проверено и что не измерено

Для источника `fcb038a` локальный ignored `outputs/continuation-20260925/final-rehearsal.json` фиксирует `prepare_release.py`/`verify_release.py` PASS: **1891/1891** файлов, **2 086 076 897 B**, image revision label равен SHA. Предыдущий функциональный прогон на `efd47c9` проверил seed 750, health, `/api/version`, положительный поиск, отказ, explain и разрешённые материалы. Между ним и `fcb038a` изменились только `SOLUTION.md` и PDF; runtime-код не менялся. Публичный IP-прогон `outputs/continuation-20260925/yc-deploy/yc-live-external-ip.json` от 27.09 проверил модель, 750 точек, три материала по SHA, поиск и отказ. После перевода DNS 28.09 проверены HTTPS 200 на основных доменах и трёх `/submission/` файлах, API `status=ok`, `model_loaded=true`, `gallery_points=750`, `DEPLOYMENT_STATUS=DEPLOYED`. Новый образ `e3cd2b0` 29.09 проверен по OCI revision, `/api/version`, `/api/health`, положительному поиску и отказу; API и Qdrant остались healthy. Пакетный test-артефакт и официальная оценка валидации приведены в [описании воспроизведения](../../reproduce/README.md).

На VM в момент первой проверки `free -m` показывал **3121 MiB available**, `df -h /` — **43 GiB свободно**. Это мгновенные значения, не пик нагрузки. Более поздний [ограниченный замер](evidence/vps-benchmark-20260928/README.md) на том же работающем API зафиксировал IP нового VPS: на самой VM `POST /api/search` дал p50/p95 **0,141/0,149 с** (10 последовательных), при двух одновременных запросах **0,281/0,282 с** (8 запросов); с dev-машины через HTTPS к закреплённому `84.201.146.191` — **0,833/1,422 с** и **0,696/1,362 с** соответственно. После обновления API 29.09 десять последовательных вызовов на самой VM завершились HTTP 200 за **0,145893–0,148234 с** каждый; это малая выборка полного HTTP-поиска, не измерение FPS инференса на GPU организатора. API/Qdrant после прогона здоровы. Для `n≤10` p95 равен максимуму выборки и не является надёжной оценкой хвоста. HTTPS-замер включает сеть, TLS и Caddy, а не только инференс. Прежний [не закреплённый по IP журнал](evidence/20260928-live-http.json) показал более медленные ответы, но на dev-машине в это время был устаревший DNS-ответ со старым VPS; журнал не записал фактический IP, поэтому его нельзя уверенно приписать новому серверу. Прежний локальный CPU журнал `outputs/continuation-20260925/http-benchmark.json` относится к другой машине и не используется как VPS-скорость. Холодный старт, поведение при большем числе пользователей, пик RAM, стоимость/остаток гранта и срок обязательной доступности жюри **NOT MEASURED/не подтверждены**. Масштаб на миллион автомобилей из 750 объектов не следует.

## Воспроизводимая локальная сборка

Команды выполняются из чистого checkout **после логического commit**. Это сборка CPU-образа для текущего VPS без NVIDIA; основной GPU-образ для стенда организаторов описан в [инструкции сервиса](../README.md). Нужны разрешённые данные организатора в локальном `data/`, Podman 5.8.2 или совместимый Docker, два базовых образа из `04-solution/service/offline/`, `pdfinfo`/`pdftotext` и уже проверенный технический PDF `05-presentation/checks/content_preview.pdf` (11 страниц). Этот PDF генерируется штатным `build_presentation.py`/`validate_presentation.py`; для их запуска нужен локальный игнорируемый Git `05-presentation/team-data.md` из разрешённого источника. Скрипт релиза проверяет, что контактные номера из этого файла отсутствуют в техническом PDF. Не добавлять `team-data.md`, полный deck или данные организатора в Git.

```bash
python3 06-documentation/build_pdf.py --check
RELEASE_SHA=$(git rev-parse HEAD)
LCT_DATA_DIR=/home/artem/projects/hackathon-lct-vehicle-reid/data
podman build --pull=never --network=none -f 04-solution/service/Dockerfile.cpu \
  --build-arg SOURCE_SHA="$RELEASE_SHA" \
  -t "localhost/lct-release:$RELEASE_SHA" 04-solution/service
python3 04-solution/service/deploy/prepare_release.py \
  --data "$LCT_DATA_DIR" \
  --public-slides 05-presentation/checks/content_preview.pdf \
  --image "localhost/lct-release:$RELEASE_SHA" \
  --out "outputs/continuation-20260925/release-$RELEASE_SHA"
python3 04-solution/service/deploy/verify_release.py \
  "outputs/continuation-20260925/release-$RELEASE_SHA"
```

`prepare_release.py` требует чистое отслеживаемое дерево и OCI label `org.opencontainers.image.revision` с тем же SHA; до копирования проверяет canonical manifest, входные CSV, все три файла модели, актуальность документации и технический PDF. `verify_release.py` повторно хеширует каждый файл и отклоняет лишние. Результат содержит `release.json`, образ сервиса, архив Qdrant, `compose.yaml`, `release.env`, source archive, разрешённые test-изображения, артефакты и материалы. Полный PPTX/PDF выкладываются в portal assets **отдельно**, с отдельной проверкой SHA. Изменившийся исходник или новый UI требует нового SHA, image label и bundle; нельзя редактировать уже проверенный `release-fcb038a`.

## Размещение и внешний контроль

Bundle данных расположен в `/srv/lct/releases/fcb038af681405b07219542833ce3b80395055b0`; его Compose-проект — `lctfcb038a`, Qdrant volume принадлежит этому проекту. Текущий образ API `e3cd2b0…` загружен отдельно и запущен через `SERVICE_IMAGE` в окружении Compose; исходный bundle и `release.json` не редактировались. Предыдущий рабочий образ `6e8c1c9…` сохранён для отката. Семь актуальных файлов каталога `/api/materials` лежат отдельно в `/srv/lct/materials-official-20260929` и подключены только к API через `/srv/lct/compose-materials-official-20260929.yaml`. Их SHA совпали с `artifacts-final`, PDF документации и 11-страничным preview; публичный `/materials/manifest` вернул SHA `670d737c68e0d361b8fd29637c34b94c911327b5e58ab5b8fdc86df9e1717664`. Публичные полные PDF/PPTX и PDF документации были скачаны обратно по HTTPS: SHA `99f00be5…`, `c3bfa282…`, `be03590f…` совпали с локальными на момент первой проверки. После GPU-обновления 29.09 заменены только `/submission/documentation.pdf` и `/materials/solution`: обе копии скачаны обратно с SHA-256 `1b0955838b8dc42abe63f5f463a112fecc33741ef6c95995e471ec4bfd04deb2`. После записи контрольного прогона GPU-образа 29.09 та же пара PDF обновлена повторно: обе копии скачаны обратно по HTTPS с SHA-256 `8c3285da7cfd1ac9f26ba7b2bb660a68befe38310757659ca728344432a519bc`, совпадающим с локальным `06-documentation/SOLUTION.pdf`; прежние файлы сохранены как `.previous-bdbe67b`. API остался на CPU-образе, health показывает 750 объектов. На VM установлены Docker Engine/Compose V2 и Caddy. Перенос следующего полного bundle делать по проверенному SSH в **новый** каталог без `rsync --delete`, затем сверить manifest до запуска. Отдельный `/srv/lct/deployment.env` содержит `DEPLOYMENT_STATUS=DEPLOYED`; он не правит `release.json`. Текущее состояние проверяется так:

```bash
ROOT=/srv/lct/releases/fcb038af681405b07219542833ce3b80395055b0
python3 "$ROOT/deploy/verify_release.py" "$ROOT"
sudo env SERVICE_IMAGE=localhost/lct-release:e3cd2b0d0a538fa2e383ba43acb763e903a63aef \
  DEPLOYMENT_STATUS=DEPLOYED docker compose -p lctfcb038a \
  --env-file "$ROOT/release.env" -f "$ROOT/compose.yaml" \
  -f /srv/lct/compose-materials-official-20260929.yaml ps
sudo docker inspect lctfcb038a-api-1 \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
curl -fsS http://127.0.0.1:18071/api/health
curl -fsS http://127.0.0.1:18071/api/version
curl -fsS http://127.0.0.1:18071/api/materials
```

Из внешней сети отдельно проверить TLS, `GET /api/health`, фактический поиск по JPEG+bbox, отказ, crop кандидата, JSON/CSV экспорт, семь файлов `/materials/` и три файла `/submission/`. При локально устаревшем DNS использовать, например, `curl -fsS --resolve demo.lct.iamcp.ru:443:84.201.146.191 https://demo.lct.iamcp.ru/api/health`; это проверяет конкретную VM, но **не** доказывает распространение DNS для жюри. Browser-проверка обеих тем и мобильной ширины остаётся отдельной от `curl`. После обновления публичных документов сравнить их SHA с локальными файлами; PDF в старом immutable bundle при этом не становится новым сам собой.

Новый полный runtime нужно поднять под отдельным Compose-проектом, коллекцией и loopback-портом, проверить health/750 точек и пользовательский путь, затем переключить upstream Caddy. Если на VM не хватает памяти для двух API одновременно, остановить старый API в коротком согласованном окне, оставив bundle и volume для отката. Перед reload сохранить `/etc/caddy/Caddyfile` в отдельную копию, выполнить `sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile`; при неудаче вернуть старый файл и reload. Текущая рабочая копия на VM — `/etc/caddy/Caddyfile.lct-known-good-20260929`; копия до добавления маршрута `/materials/*` — `/etc/caddy/Caddyfile.lct-before-materials-20260929`. При дефекте текущего образа вернуть `SERVICE_IMAGE=localhost/lct-release:6e8c1c9d7be015073252f1d30da181737894067c` и убрать Compose override в том же проекте `lctfcb038a`; индекс и volume при этом остаются. Не выполнять `docker compose down -v`, `rsync --delete`, удаление VM/IP или volume как часть отката.

Репозиторий сдачи находится в организации хакатона SourceCraft; он закрыт для
анонимного просмотра. Сервис, презентация и документация опубликованы по
адресам выше.
