# Vehicle ReID: сборка, запуск и воспроизведение результатов

Документация решения по разделам 7–9 и 12 ТЗ. Проверка выполнена 15–16 сентября 2026 года на Linux x86_64 с Podman 5.8.2 и podman-compose 1.6.0; **20 сентября 2026 года** конфигурация признака сменилась на d1_j48 (две модели + whitening, §2) и вся цепочка перепроверена из отдельного чистого дерева на Podman 5.8.1 (офлайн-сборка, пакетный прогон, побайтовая сверка артефактов, 60 тестов контура — §8). Ниже приведены команды **Docker** для жюри; непосредственно Docker на машине проверки отсутствует.

**Подтверждены сборка образа без обращения к PyPI, инференс без сети, работа API с Qdrant и пересчёт основных метрик из изображений. Полное воспроизведение всех исследовательских отчётов из одного git-репозитория пока не обеспечено.** Зависимости едут колёсами в составе решения (раздел 3), из внешнего на сборке остаются только базовые образы. Недостающие материалы и непроверенные числа перечислены в разделах 8–11.

## 1. Назначение и архитектура

Сервис принимает изображение автомобиля и предоставленный bbox `(x, y, w, h)`, извлекает признак, ищет похожие объекты в галерее и возвращает кандидатов либо пустой ответ. Сдаваемый признак — двумодельный ансамбль **d1_j48**: публичный OSNet-AIN (OMZ 2022.1) плюс наш дообученный OSNet-AIN **combined_v1** (LP-FT на RoundaboutHD, MIT + CARLA, Apache-2.0 + train организатора; журнал обучения — [training/combined/](04-solution/training/combined/)), с общим whitening. OCR, распознавание номера и детекция автомобиля в вычислительный путь не входят; влияние области пластины исследовалось отдельно для обеих моделей ([training/combined/REPORT.md §5](04-solution/training/combined/REPORT.md), [plate-ablation/](04-solution/plate-ablation/)).

| Компонент | Код | Ответственность |
|---|---|---|
| Подготовка изображения | [preprocess.py](04-solution/service/app/core/preprocess.py) | Чтение JPEG/PNG, bbox, RGB, изменение размера |
| Модель | [model.py](04-solution/service/app/core/model.py) | Проверка SHA-256 трёх файлов весов (две ONNX-модели + матрица whitening), ONNX Runtime CPU, двумодельный признак со whitening |
| Пакетная обработка | [batch.py](04-solution/service/app/batch.py) | Последовательное чтение CSV, извлечение векторов, ранжирование, три сдаваемых файла |
| Переранжирование | [rerank.py](04-solution/service/app/core/rerank.py) | Совместная обработка всех запросов и галереи по k-взаимным соседям |
| HTTP API | [main.py](04-solution/service/app/api/main.py) | Получение файла/bbox или вектора, валидация, поиск, OpenAPI |
| Галерея | [store.py](04-solution/service/app/api/store.py), Qdrant | Векторы и метаданные, точный поиск по косинусу |
| Загрузчик галереи | [load_gallery.py](04-solution/service/app/load_gallery.py) | Извлечение признаков галереи и пересоздание коллекции Qdrant |
| Клиент | [index.html](04-solution/service/app/static/index.html) | Страница браузера, выдаваемая API; отдельного frontend-контейнера нет |
| Измерения | [reid_metrics.py](04-solution/eval/reid_metrics.py), [scope_metrics.py](04-solution/eval/scope_metrics.py) | Ранговые метрики, отказ, явные варианты протокола |

API и Qdrant запускаются отдельными контейнерами. Пакетный режим использует образ API и работает с файлами, без БД. Это позволяет проверить сдачу без предварительной загрузки галереи в сервер. Общий модуль извлечения признаков сохраняет одинаковый препроцессинг в обоих режимах.

## 2. Методы и рабочая конфигурация

### Признак

Путь обработки: bbox в пикселях исходного кадра → RGB → PIL bilinear, `208×208` → `float32`, диапазон `0…255`, NCHW → **две модели на одном препроцессированном батче**: OSNet-AIN (OMZ 2022.1) и combined_v1 (наш дообученный OSNet-AIN) → L2-нормировка каждого сырого вектора `512` → покомпонентное среднее → whitening `y = (x − m) @ P.T` в `float32` (матрица `P` и вектор `m` обучены на `train_fit` по дельтам ансамбля, ρ=0,5; обучение и проверка — [training/combined/](04-solution/training/combined/)) → L2-нормировка с накоплением в `float64`, сохранение в `float32`. Порядок важен: среднее нормируется до whitening, потому что `m` обучена на нормированных входах. Внешняя ImageNet-нормировка не применяется: обе модели уже содержат нормировку входа. Инференс выполняется через `CPUExecutionProvider`.

Происхождение **combined_v1**: LP-FT дообучение OSNet-AIN (заморозка классификатора → частичная → полная разморозка по заранее зафиксированным воротам) на объединённом наборе RoundaboutHD (лицензия MIT) + CARLA (лицензия Apache-2.0) + train организатора. Полный журнал: [training/combined/journal.md](04-solution/training/combined/journal.md), отчёт с метриками и бутстрэпом — [training/combined/REPORT.md](04-solution/training/combined/REPORT.md). Веса поставляются файлом `model/osnet_ain_combined_v1.onnx` (SHA-256 — §9); в git по `*.onnx`-исключению не входят и приложены к поставке.

Дельта от хранения whitening в `float32` вместо `float64`-оригинала измерена отдельно: метрики валидации не меняются ни в одной из 16 печатных цифр, максимальное отличие эмбеддинга `1.43e-08` (раздел 8).

### Ранжирование и отказ

Пакетный режим по умолчанию применяет k-reciprocal re-ranking на объединении **всех запросов данного прогона и всей галереи**. Он смешивает жаккардову дистанцию между взаимными окрестностями и нормированную косинусную дистанцию. Внешняя оценка `s = 1 − d`: больше означает ближе. Сохраняемые эмбеддинги от этой операции не меняются.

**Рабочая строка конфигурации (с 20.09.2026):** `batch: k1=6, k2=3, λ=0.3, t_rr=0.5282812306342437; API / batch --no-rerank: t_cos=0.5141976914190476`. До 20.09 сдавалась одномодельная конфигурация OSNet+KR с `t_rr=0.49937235233589916`, `t_cos=0.5495953464415451` (её числа сохранены в истории раздела 6). Пороги перекалиброваны на эмбеддингах d1_j48 тем же правилом, что и раньше (раздел 6).

Исполняемый источник значений — [config.py](04-solution/service/app/core/config.py). После изменения калибровки обновляется строка выше; команды ниже читают значения из кода и заново рассчитывают порог. Фактическая конфигурация batch записывается в `run_info.json`, конфигурация API доступна в `/api/version`.

HTTP API выполняет **точный косинусный поиск** Qdrant (`exact=True`). Его ответы нельзя сравнивать с пакетным переранжированием как с одним алгоритмом: состав остальных запросов влияет на пакетный результат. `--no-rerank` включает косинусный batch. В обоих режимах принимаются оценки `s >= t`; при равных оценках в batch сохраняется порядок строк gallery-CSV. Оценка близости не является вероятностью.

Порог задаётся `--threshold` для batch, полем `threshold` для API, либо переменными `REID_THRESHOLD` / `REID_THRESHOLD_RERANK` внутри процесса. Переменная, заданная только в shell хоста, не передаётся контейнеру автоматически: для `docker run` нужен `-e`, для Compose — настройка `environment`/override. API ограничивает выдачу параметром `top_k` (по умолчанию 10, максимум 100); `candidates.csv` содержит все прошедшие порог пары.

## 3. Подготовка на машине с интернетом

Нужны Linux x86_64, Docker Engine с Compose V2, shell и `curl`. GPU не требуется. Нужен полный набор организатора с изображениями; CSV из git недостаточно. Указанные команды запускаются из корня полученного репозитория.

### Шаг 1. Пути и данные

```bash
export REPO="$(pwd -P)"
export DATA_DIR="$REPO/data"
export OUT_DIR="$REPO/outputs/reproduce"
export COMPOSE_FILE="$REPO/04-solution/service/docker-compose.yml"
mkdir -p "$OUT_DIR"
test -d "$DATA_DIR/images"
test -f "$DATA_DIR/train.csv"
test -f "$DATA_DIR/test_query.csv"
test -f "$DATA_DIR/test_gallery.csv"
```

Архив данных получают через личный кабинет [задачи № 7](https://i.moscow/cabinet/hackaton/lct/contest/4e03d57b5c1d4ef987d8966258c03bd8). После распаковки непосредственно в `DATA_DIR` должны находиться `images/`, `train.csv`, `test_query.csv`, `test_gallery.csv`, `README.md`. Проверенного прямого URL архива, его SHA-256 и отдельной лицензии в поставке нет. Отпечатки распакованного набора приведены в разделе 9.

### Шаг 2. Веса

```bash
(cd "$REPO/04-solution/service" && sh model/fetch_model.sh)
```

Скрипт проверяет по полному SHA-256 все три сдаваемых файла весов и скачивает по прямой ссылке OMZ только то, чего нет рядом (OSNet). После него файлы находятся в `04-solution/service/model/`. При несовпадении любого хеша сборку не продолжать. Наличие файла в рабочем каталоге не означает, что он включён в git: `*.onnx` исключён `.gitignore`, оба ONNX должны приехать в архиве поставки.

### Шаг 3. Образы

```bash
docker pull python:3.13-slim
docker compose pull qdrant
docker build --no-cache -t vehicle-reid-service "$REPO/04-solution/service"
```

**Пакеты из PyPI на этом шаге не скачиваются: сборка идёт офлайн, из репозитория.** В решении лежит каталог [wheels/](04-solution/service/wheels/) — 30 файлов `.whl`, 59 146 237 байт: ровно те пакеты, что перечислены в разделе 10 (семь прямых зависимостей и их транзитивные). Колёс 30, а строк в разделе 10 — 31: `pip` приходит из базового образа и отдельным колесом не везётся. Колёса получены `pip download -r requirements.txt -c requirements-lock.txt` **в том же базовом образе** `python:3.13-slim`, поэтому платформа и версия Python у них те же, что у сборки; команда пересборки каталога — в [wheels/README.md](04-solution/service/wheels/README.md). [Dockerfile](04-solution/service/Dockerfile) по умолчанию собирается с `ARG PIP_SOURCE=offline`, то есть `pip install --no-index --find-links=/wheels -r requirements.txt -c requirements-lock.txt`. Это наш ответ на разд. 9 ТЗ: образ собирается в изолированной среде, без выхода наружу.

**Что на этом шаге всё-таки нужно из сети — два базовых образа:** `python:3.13-slim` для сборки и `qdrant/qdrant:v1.15.5` для хранилища. Поэтому `docker pull python:3.13-slim` стоит первой строкой: **без заранее вытянутого базового образа изолированная сборка падает на первой инструкции `FROM`, ещё до `pip`.** Если сборка идёт на машине с сетью, отдельный `pull` не нужен — `docker build` вытянет образ сам.

**`--network none` не делает сборку изолированной.** Проверено на пустом хранилище образов: этот флаг отрезает сеть только у шагов `RUN`, а базовый образ процесс сборки тянет из реестра поверх сети хоста (в логе видно `Trying to pull docker.io/library/python:3.13-slim`). Настоящая проверка изолированности — запретить и обращение к реестру:

```bash
docker pull python:3.13-slim                   # один раз, на машине с сетью
docker build --no-cache --pull=never --network none \
  -t vehicle-reid-service "$REPO/04-solution/service"
```

Без базового образа в локальном хранилище эта команда останавливается сразу: `python:3.13-slim: image not known`. С ним — проходит целиком (проверено; в логе `Looking in links: /wheels`), и `pip freeze --all` собранного образа построчно совпадает с разделом 10.

**Запасной путь — ставить из сети:** `docker build --build-arg PIP_SOURCE=network -t vehicle-reid-service "$REPO/04-solution/service"`. Он нужен, если колёса разошлись с `requirements.txt` или требуется другая платформа. Сам каталог `wheels/` нужен в любом случае: `COPY wheels/` в Dockerfile общий для обеих ветвей, без каталога сборка не начнётся.

В сдаваемый образ колёса **не попадают**: зависимости ставятся в отдельной стадии сборки, наружу уходит только каталог установленных пакетов. Веса копируются внутрь образа и проверяются при сборке, затем повторно при загрузке модели. Транзитивные зависимости закреплены: `pip install` идёт с constraints-файлом [requirements-lock.txt](04-solution/service/requirements-lock.txt) — полным `pip freeze --all` проверенной сборки (раздел 10). Не закреплены хеши wheel и тег базового образа `python:3.13-slim` (digest проверенной среды указан в разделе 10).

### Замечание для хостов с SELinux (Fedora, RHEL, CentOS Stream и подобные)

Это касается **четырёх** команд ниже — всех, которые монтируют каталоги хоста: пакетный прогон (раздел 4), команды **T** и **R** (раздел 7), калибровка порога (раздел 8). Docker на таком хосте обычно расставляет метки сам; Podman — нет, и контейнер получает отказ в доступе к смонтированным каталогам. Проверенное решение — добавить в каждую такую команду `--security-opt label=disable`; альтернатива — суффикс `:z`/`:Z` у каждого `-v`.

**Добавьте флаг до первого запуска, а не после отказа.** `app.batch` пишет все три файла **в самом конце**, поэтому `PermissionError: [Errno 13] Permission denied: '/out/embeddings.npy'` приходит через 5,5 минут уже выполненного инференса — весь расчёт придётся повторить. Команда **T** падает быстро, но с сообщением не по делу: `ModuleNotFoundError: No module named 'test_protocols'` — файл на месте, просто каталог `/repo` не читается.

## 4. Запуск и офлайн-поставка

### Сервис с загруженной галереей — одна команда

После подготовки из раздела 3:

```bash
docker compose up -d --no-build --pull never
```

Интерфейс: <http://localhost:8000/>. Swagger: <http://localhost:8000/docs>. OpenAPI: <http://localhost:8000/openapi.json>. Статические файлы Swagger поставляются локально. Загрузчик читает `$DATA_DIR/test_gallery.csv` и **пересоздаёт** коллекцию; повторный запуск заменяет её прежнее содержимое. Дождитесь завершения loader и проверьте:

```bash
curl --fail http://localhost:8000/api/health
curl --fail http://localhost:8000/api/version
```

В health должны быть `storage.reachable=true` и непустое `storage.gallery_points`. Отдельное поле `status="ok"` само по себе не означает готовность галереи. До загрузки поиск возвращает HTTP 409 (`{"detail":"галерея не загружена — выполните app.load_gallery"}`), при недоступной БД — 503. Оба ответа проверены.

**`gallery_points` не доказывает, что отработала именно ваша загрузка.** Том Qdrant именованный (`qdrant_storage`), а имя проекта Compose берётся из имени каталога с `docker-compose.yml` — это всегда `service`, одинаково у рабочего дерева и у любого клона. Поэтому `up` на машине, где сервис уже когда-то поднимали, видит **прежний** том: health отдаёт `gallery_points=750` ещё до того, как ваш загрузчик дошёл до первого кадра. Так и случилось при сквозной проверке из чистого клона.

Достоверный признак — **завершение самого загрузчика**, а не содержимое хранилища:

```bash
docker compose logs --no-log-prefix loader | tail -1
# {"collection": "gallery", "points": 750, "rows": 750}
```

Эту строку `app.load_gallery` печатает последним действием — после `upsert` и проверки `points == rows`; до неё процесс не завершается, а при несовпадении падает. Нет строки — загрузка не дошла до конца, чем бы ни отвечал health. Проверено и под podman-compose: вывод тот же. Код выхода контейнера виден в `docker compose ps -a loader` (столбец `State` — `exited (0)`).

**Запуск с заведомо пустым хранилищем** (нужен, если прежний том мог остаться от другого прогона):

```bash
docker compose down -v          # удаляет контейнеры И именованный том проекта
docker compose up -d --no-build --pull never
```

Проверено на чистом томе: `gallery_points` равен `null` всё время работы загрузчика и становится `750` сразу после его выхода; загрузка 750 объектов заняла 64,5 с. Если удалять чужой том нежелательно, дайте прогону собственное имя проекта — `docker compose -p reid-check up -d …`, тогда создаётся отдельный том `reid-check_qdrant_storage`, а `docker compose -p reid-check down -v` убирает за собой только его.

Загрузка галереи входит в `up`: `loader` — обычный разовый сервис Compose (профиля `tools` больше нет), он стартует после healthcheck Qdrant, отрабатывает и выходит. Готовность Qdrant проверяется healthcheck-запросом `/readyz` (в образе нет curl, поэтому проверка идёт через `/dev/tcp` bash), а `api` и `loader` объявлены зависимыми с `condition: service_healthy`. Дополнительно сам загрузчик ждёт доступности хранилища до извлечения векторов (`--wait`, по умолчанию 120 с) — это нужно для сред, где ожидание healthcheck не выполняется (например, podman-compose). Повторная загрузка на другом каталоге данных: `DATA_DIR=/путь/к/data docker compose run --rm loader` — эта форма блокирует терминал до конца загрузки и печатает ту же итоговую строку, то есть тоже годится как признак готовности.

`docker compose up -d` возвращает управление, пока загрузчик ещё работает: около минуты на 750 объектов галереи. До конца загрузки поиск отвечает 409 — это ожидаемо, а не отказ сервиса.

### Пакетный инференс — одна команда, сеть отключена

```bash
docker run --rm --network none \
  -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -m app.batch \
  --images-dir /data/images --query /data/test_query.csv \
  --gallery /data/test_gallery.csv --out-dir /out --threads 2
```

Для закрытого теста заменяются только входной каталог и CSV. Метки `vehicle_id` и `camera_id` инференсу не нужны. Выходной каталог должен быть доступен на запись. **На хосте с SELinux добавьте `--security-opt label=disable`** — см. замечание в конце раздела 3; там же сказано, почему отказ приходит только через 5,5 минут.

### Передача на машину без интернета

На машине сборки сохраните **оба** образа:

```bash
docker save -o "$OUT_DIR/runtime-images.tar" vehicle-reid-service docker.io/qdrant/qdrant:v1.15.5
sha256sum "$OUT_DIR/runtime-images.tar" > "$OUT_DIR/runtime-images.tar.sha256"
```

**Podman вместо Docker:** `podman save -o файл образ1 образ2` молча кладёт в архив только один образ (оба тега при этом навешиваются на него) — нужен `podman save --multi-image-archive -o ...`; проверено на podman 5.8.2 для `vehicle-reid-service` + `qdrant/qdrant:v1.15.5`: с флагом в архиве два образа и **529 025 536** байт, без флага — один образ и **347 906 048**. Побайтово эти размеры не воспроизводимы и сверять их не нужно: слои образа содержат отметки времени файлов, поэтому две независимые сборки одного и того же Dockerfile дают чуть разные архивы. Контрольный повтор на другой сборке из чистого клона: **528 972 288** и **347 852 800** — те же величины с точностью 53 248 байт (0,01 %). Проверяемое утверждение здесь другое и оно выполняется точно: **без флага в архиве один образ с двумя тегами, с флагом — два образа**. Передайте архив образов, код, ONNX-файл из `model/` и данные. На целевой машине выполните `docker load -i /путь/к/runtime-images.tar`, задайте пути из шага 1, затем используйте команду запуска выше без `--build`. Для пакетного режима достаточно образа сервиса и тестовых данных. **Чистый git-клон не является полной офлайн-поставкой**: образы, данные и веса должны быть приложены отдельно. Веса отдельно нужны для воспроизведения сборки; работающий образ уже содержит их.

## 5. Формат сдаваемых файлов

Схема основана на `README.md` архива организатора, а не только на общих формулировках ТЗ.

| Файл | Содержимое |
|---|---|
| `submission.csv` | Заголовок `query_id,gallery_id_1,…,gallery_id_10`; строки query в порядке входного CSV, кандидаты по убыванию оценки |
| `embeddings.npy` | `float32`, `(N_query + N_gallery, 512)`; сначала все query, затем все gallery, порядок внутри блоков как в CSV |
| `candidates.csv` | `query_id,gallery_id,confidence`; все пары выше порога; уверенность записана с шестью знаками после точки; отсутствие строк для query означает отказ |
| `run_info.json` | Служебный протокол: модель, хеш, версии, порог, режим и шкала, размеры, счётчики, время |

Переранжирование меняет `submission.csv` и `candidates.csv`. Оно не меняет `embeddings.npy`. Старые файлы [baseline/artifacts](04-solution/baseline/artifacts/) сняты в другом режиме/при другом пороге и не служат эталоном новой выдачи. Каталог [service/artifacts-final/](04-solution/service/artifacts-final/) обновлён 20.09.2026: прогон текущей конфигурации d1_j48 с рабочими порогами раздела 2, сверен побайтово с пересчётом из чистого дерева (раздел 8). У проверенного выданного теста 1110 query и 750 gallery: ожидается матрица `(1860, 512)`.

## 6. Протокол метрик и достигнутые значения

Валидация находится в [split/files](04-solution/split/files/): 1110 запросов, 750 объектов галереи. У 832 запросов есть кросс-камерная пара, у 278 пары нет. Идентичности `train_fit` отделены от валидации. SHA-256 CSV закреплены в [manifest.json](04-solution/split/files/manifest.json).

Ранжирование оценивается с `camera_policy="market"`: из галереи запроса удаляются объекты **той же идентичности и той же камеры**. Другие автомобили с той же камеры остаются отрицательными кандидатами. ТЗ допускает и трактовку «удалить всю камеру»; она реализована как `all_same_camera`, но приведённые числа относятся к `market`.

AP — среднее precision в позициях всех верных допустимых совпадений; mAP — среднее AP по **832** запросам с парой. Rank-k — доля таких запросов с верным объектом среди первых k. mINP — среднее отношения числа верных объектов к рангу последнего верного объекта. Для mAP@10 сначала отбираются десять исходных кандидатов, потом применяется фильтр камеры; знаменатель AP — все верные в допустимой галерее. Если жюри усреднит AP по всем запросам с нулями для отсутствующих, результат будет другим; обе ветви записывает контур.

| Метод | mAP, вся галерея | Rank-1 | Rank-5 | mINP | mAP@10 |
|---|---:|---:|---:|---:|---:|
| **d1_j48, косинус** | 0.7316093422 | 0.6850961538 | 0.8822115385 | 0.6928460434 | 0.7242262382 |
| **d1_j48, переранжирование** | **0.7740915539** | **0.7307692308** | 0.8810096154 | 0.7488060310 | **0.7684546226** |
| Прежняя сдаваемая конфигурация OSNet+KR, переранжирование (до 20.09) | 0.6936584725 | 0.6634615385 | 0.7872596154 | 0.6608640365 | 0.6834008859 |

Прирост mAP от переранжирования у d1_j48 — **0.0424822117** (косинус → переранжирование). Значения воспроизведены сервисным конвейером из изображений и совпали с измерительным контуром до всех 16 цифр (валидационный батч сервиса, прогнанный через `tools/eval_split.py`, даёт ровно `0.7740915539438481 / 0.7307692307692307 / 0.7684546226343101` и косинус `0.7316093422310448`); источник чисел и методика — [training/combined/s02_metrics.json](04-solution/training/combined/s02_metrics.json) и [training/combined/REPORT.md](04-solution/training/combined/REPORT.md). Числа полного ранжирования нельзя объявлять метрикой усечённого `submission.csv`. Метрики закрытого теста организатора неизвестны: в доступных test-CSV нет идентичностей и камер.

> **Пометка честности (переключение 20.09.2026).** Сдаваемая конфигурация сменилась: OSNet+KR (0.6937 / 0.6635) → d1_j48 (0.7741 / 0.7308). Отрыв d1_j48 от **прежнего чемпиона измерительного контура** d1 (0.7621 / 0.7175) составляет **+0.0120 — меньше заранее зафиксированной приёмочной σ(mAP) = 0.0134**, то есть по приёмочному правилу прирост против чемпиона **не засчитывается**. Парный бутстрэп при этом значим (Δ = +0.0120, CI95 [+0.0040, +0.0205], p = 0.003, B = 4000); переключение выполнено **по явному решению Артёма** при этом значимом бутстрэпе. Полное сравнение кандидатов и критерий приёмки — [training/combined/REPORT.md §4–§6](04-solution/training/combined/REPORT.md). Разрыв по камере (KR mAP: без исключения same-camera минус market) у сдаваемой конфигурации **+0.1103** — лучше, чем у прежнего чемпиона (+0.1144) и у исходного OSNet (+0.1417); абляция зоны номерной пластины для combined_v1 пройдена — снижение при её маскировании статистически неотличимо от контрольной зоны той же площади (все контрасты накрывают 0, REPORT.md §5).

### Обоснование порога

Единица отказа в основной ветви — **запрос**, `refusal_mode="presence"`. Положительный класс означает наличие пары в допустимой галерее. Принятие определяется максимумом оценки после камерной фильтрации. Это не precision/F1 всех строк `candidates.csv` и не проверка правильности идентичности top-1.

Правило: выбрать порог, максимизирующий `min(TNR, F1_0.10, F1_0.25, F1_0.40)`. Индексы — предполагаемые доли запросов без пары. Для каждого порога вычисляются recall `r` и доля ложных принятий `f`; при доле отсутствующих `p` используется `F1_p = 2(1−p)r / ((1−p)(1+r)+pf)`, `TNR=1−f`. При изменении только доли классов TNR и recall сохраняются. Это обоснование выбора компромисса, а не модель распределения скрытого теста.

**Доопределение при равенстве — часть правила.** Кандидаты — все различные значения уверенности top-1 на валидации. Максимум не всегда единственный: когда минимум даёт `TNR`, целевая функция постоянна на участках сетки, где не сдвигается ни один запрос **без** пары. При равенстве берётся больший `F1`, затем больший `TNR`. На косинусе это не формальность: плато состоит из двух порогов — `0.5141976914190476` (TP 566, F1 0.7684996606) и `0.5145892426611729` (TP 565, F1 0.7676630435), у обоих `TNR` 0.7302158273; без доопределения выбор между ними произволен, а расходятся они на 3.9e-04 — в 400 раз больше допуска сверки в команде R. Правило записано так во всех реализациях: [`calibrate_threshold.py`](04-solution/service/tools/calibrate_threshold.py) (им посчитаны рабочие пороги), [`refusal/scripts/analyze.py`](04-solution/refusal/scripts/analyze.py), [`s06_refusal.py`](04-solution/training/ensemble/scripts/s06_refusal.py) и команда R (разд. 7).

| Режим | TP / FP / FN / TN отсутствующих | F1 | TNR | Recall | AUC-PR, трапеции |
|---|---|---:|---:|---:|---:|
| Максимум F1 на косинусе d1_j48 (не рабочая точка) | 785 / 194 / 47 / 84 | 0.8669243512 | 0.3021582734 | 0.9435096154 | 0.9094743821 |
| Косинус d1_j48, `t_cos` | 566 / 75 / 266 / 203 | 0.7684996606 | 0.7302158273 | 0.6802884615 | 0.9094743821 |
| Переранжирование d1_j48, `t_rr` | 606 / 60 / 226 / 218 | 0.8090787717 | 0.7841726619 | 0.7283653846 | 0.9182837159 |

Значения перекалиброваны 20.09.2026 на эмбеддингах d1_j48 прежним правилом; инструмент — `tools/calibrate_threshold.py`, 187 прямых сверок объявленных точек с неизменённым контуром (max |ошибка| < 2e-12). До 20.09 (прежняя конфигурация OSNet+KR, калибровка [refusal/](04-solution/refusal/)): `t_cos` 0.5495953464415451 → F1 0.7330567082 / TNR 0.6978417266; `t_rr` 0.49937235233589916 → F1 0.8056112224 / TNR 0.7769784173.

**Какая строка относится к сдаваемому режиму.** По умолчанию batch работает с переранжированием, поэтому заявленные в задании **F1 0.8090787717 и TNR 0.7841726619 — это строка «Переранжирование d1_j48, `t_rr`»**, и в `metrics.json` она лежит в `rerank_refusal`. Пара 0.7684996606 / 0.7302158273 из строки «Косинус d1_j48, `t_cos`» относится к косинусной ветви — это режим HTTP API и `batch --no-rerank`, в `metrics.json` это `base.refusal`. Числа одинаковые по смыслу и разные по конфигурации; команда R (раздел 7) печатает **обе пары с подписями**, чтобы их нельзя было перепутать при сверке.

Косинусная калибровка прежней конфигурации описана в [refusal/REPORT.md](04-solution/refusal/REPORT.md); текущие обе точки пересчитываются командой R и сверяются `calibrate_threshold.py`. Порог нельзя округлять перед сравнением или переносить между шкалами.

Параметры переранжирования выбраны на отдельном протоколе из `train_fit`, но рабочие пороги выбраны и показаны на одной валидации. Новая калибровка не имеет независимого holdout-замера. В сервисе нет скрытых идентичностей для камерной фильтрации: число непустых ответов сырого batch на validation может отличаться от TP+FP в таблице. На выданном тесте число отказов проверяемо, их правильность — нет.

## 7. Воспроизведение основных чисел одной командой

**Команда R** после раздела 3 пересчитывает векторы из изображений, создаёт сдаваемые файлы отдельно для validation/test, вызывает существующий контур метрик, повторяет выбор обоих порогов, проверяет границы порядка векторов и записывает `$OUT_DIR/metrics.json`. Готовые `.npy` из чужих рабочих каталогов ей не нужны. Сохраняются обе ветви порядка фильтрации top-10. Интерпретатор берётся из того же контейнера, что и инференс.

**R, T и калибровка монтируют каталоги хоста.** На хосте с SELinux в каждую из них нужно добавить `--security-opt label=disable` — см. замечание в конце раздела 3.

```bash
docker run --rm -i --network none --cpus 2 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B - /repo /data /out <<'PY'
"""Recompute published validation numbers from images, without cached embeddings."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

repo, data, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(repo / "04-solution/eval"))
sys.path.insert(0, str(repo / "04-solution/postproc/scripts"))
from reid_metrics import evaluate, scores_from_embeddings
from common import rerank
from app.core import config
from app.core.model import Embedder
from app.core.preprocess import read_rows, load_crop

split = repo / "04-solution/split/files"
def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

qm, gm = rows(split / "val_query.csv"), rows(split / "val_gallery.csv")
manifest = json.loads((split / "manifest.json").read_text())
for name, digest in manifest["sha256"].items():
    assert hashlib.sha256((split / name).read_bytes()).hexdigest() == digest, name
assert not {r["vehicle_id"] for r in rows(split / "train_fit.csv")} & {r["vehicle_id"] for r in qm + gm}

# Якоря калибровки 20.09.2026 (конфигурация d1_j48): рабочие пороги и черновой
# argmax-F1 из сохранённого прогона tools/calibrate_threshold.py.
calib_dir = repo / "04-solution/service/calib-d1_j48"
headline = json.loads((calib_dir / "headline.json").read_text())
summary = json.loads((calib_dir / "summary.json").read_text())
tau = float(headline["cosine_rule_reproduced"]["threshold"])   # рабочий t_cos
rr_tau = float(headline["rerank_threshold"])                   # рабочий t_rr
old_tau = float(summary["cosine_market_presence"]["selected"]["best_f1"]["threshold"])
for name, query, gallery in (("val", split / "val_query.csv", split / "val_gallery.csv"),
                             ("test", data / "test_query.csv", data / "test_gallery.csv")):
    cmd = [sys.executable, "-B", "-m", "app.batch", "--images-dir", str(data / "images"),
           "--query", str(query), "--gallery", str(gallery), "--out-dir", str(out / name),
           "--threads", "2", "--batch", "32"]
    subprocess.run(cmd, check=True)

emb = np.load(out / "val/embeddings.npy", allow_pickle=False)
assert emb.shape == (len(qm) + len(gm), 512) and emb.dtype == np.float32
assert np.isfinite(emb).all()
q, g = emb[:len(qm)], emb[len(qm):]
assert np.max(np.abs(np.linalg.norm(emb.astype(np.float64), axis=1) - 1)) < 1e-6
scores = scores_from_embeddings(q, g, metric="cosine")
metadata = dict(query_ids=[r["vehicle_id"] for r in qm], gallery_ids=[r["vehicle_id"] for r in gm],
    query_cameras=[r["camera_id"] for r in qm], gallery_cameras=[r["camera_id"] for r in gm],
    known_absent=np.array([r["has_mate"] == "0" for r in qm]), camera_policy="market", refusal_mode="presence")

def assess(matrix, threshold):
    r = evaluate(matrix, threshold=threshold, **metadata)
    return {"ranking": r["ranking_full_gallery"], "top10": r["ranking_top_k"],
            "top10_by_filter_order": r["ranking_top_k_by_filter_order"],
            "refusal": {k: v for k, v in r["refusal"].items() if k != "pr_curve"}, "counts": r["counts"]}

base = assess(scores, tau)
distance, seconds = rerank(q, g, 6, 3, 0.3)
reranked = assess(1 - distance, 0.)
report = {
    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                    "onnxruntime": __import__("onnxruntime").__version__, "threads": 2, "batch": 32},
    "model": {"bytes": config.MODEL_PATH.stat().st_size,
              "sha256": hashlib.sha256(config.MODEL_PATH.read_bytes()).hexdigest()},
    "model2": {"bytes": config.MODEL2_PATH.stat().st_size,
               "sha256": hashlib.sha256(config.MODEL2_PATH.read_bytes()).hexdigest()},
    "whitening": {"bytes": config.WHITENING_PATH.stat().st_size,
                  "sha256": hashlib.sha256(config.WHITENING_PATH.read_bytes()).hexdigest()},
    "split": {"query": len(qm), "gallery": len(gm), "known": int(sum(r["has_mate"] == "1" for r in qm)),
              "unknown": int(sum(r["has_mate"] == "0" for r in qm))},
    "base": base, "old_threshold": assess(scores, old_tau), "reranked": reranked,
    "rerank_seconds": seconds,
    "delta_mAP": reranked["ranking"]["mAP"] - base["ranking"]["mAP"],
    "service_config": {k: getattr(config, k) for k in dir(config) if k.startswith(("DEFAULT_THRESHOLD", "RERANK"))},
    "batch": {name: json.loads((out / name / "run_info.json").read_text()) for name in ("val", "test")},
}
# Перекалибровка обоих порогов — тем же правилом и в той же записи, что у
# инструмента, которым посчитаны рабочие пороги (service/tools/calibrate_threshold.py,
# он же в refusal/scripts/analyze.py и training/ensemble/scripts/s06_refusal.py):
# максимум от min(TNR, F1 при долях отказных 0.10/0.25/0.40), при равенстве —
# больший F1, затем больший TNR. Тай-брейк здесь — часть правила, а не оформление:
# на косинусе минимум даёт именно TNR, а TNR постоянен на участках сетки без
# «неизвестных» событий, поэтому максимум достигается на плато; его концы
# (0.5141977 и 0.5145892, tp 566 против 565) расходятся на 3.9e-04 — в 400 раз
# больше допуска сверки ниже.
def recalibrate(matrix):
    """Операционные точки контура -> порог. Сетка кандидатов — каждое различное
    значение уверенности top-1; сентинел «ничего не принято» (threshold None)
    пропускается: recall 0 даёт F1 0, то есть цель 0, и выиграть он не может."""
    known, unknown = report["split"]["known"], report["split"]["unknown"]
    curve = evaluate(matrix, threshold=0., **metadata)["refusal"]["pr_curve"]
    rows = []
    for t, tp, fp in zip(curve["thresholds"], curve["tp"], curve["fp"]):
        if t is None:
            continue
        recall, fpr = tp / known, fp / unknown
        f1s = [2*(1-p)*recall / ((1-p)*(1+recall)+p*fpr) for p in (.10, .25, .40)]
        rows.append({"threshold": float(t), "objective": min(1-fpr, *f1s),
                     "f1_at_priors_0.10_0.25_0.40": f1s, "f1": 2*tp / (tp + known + fp),
                     "tnr": 1-fpr, "tp": tp, "fp": fp})
    return max(rows, key=lambda r: (r["objective"], r["f1"], r["tnr"], r["threshold"]))

best = recalibrate(scores)
report["calibration"] = dict(best, historical_threshold=tau,
                             threshold_abs_delta=abs(best["threshold"] - tau))
rr_best = recalibrate(1 - distance)
report["rerank_calibration"] = dict(rr_best, historical_threshold=rr_tau,
                                    threshold_abs_delta=abs(rr_best["threshold"] - rr_tau))
report["rerank_refusal"] = assess(1-distance, rr_best["threshold"])["refusal"]
report["rerank_at_service_threshold"] = assess(
    1-distance, getattr(config, "DEFAULT_THRESHOLD_RERANK", rr_best["threshold"]))["refusal"]
# Check model-to-row order by independently inferring boundary rows at batch 1.
model = Embedder(threads=2)
meta_rows = read_rows(split / "val_query.csv") + read_rows(split / "val_gallery.csv")
report["row_order"] = {}
for index in (0, len(qm)-1, len(qm), len(meta_rows)-1):
    vector = model.embed_one(load_crop(data / "images", meta_rows[index]))
    own = float(scores_from_embeddings(vector[None], emb[index:index+1])[0, 0])
    assert own > 1 - 1e-10
    report["row_order"][str(index)] = own
# Informational speed, warm model, complete decode/crop/inference path.
timings = []
for _ in range(2):
    start = time.perf_counter()
    model.embed_rows(data / "images", meta_rows[:32], batch_size=1)
    timings.append((time.perf_counter()-start)*1000/32)
report["current_batch1_ms_per_object"] = timings
report["files"] = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                   for name in ("val", "test") for p in (out / name).iterdir() if p.is_file()}
# Контрольная сверка с эталоном контура (конфигурация d1_j48, f64-конвейер
# job_48). Допуск 1e-6: это в 3 порядка жёстче различимого, при этом честно для
# пересчёта на другом CPU/потоках (последние биты float32-инференса плавают).
reference = json.loads((repo / "04-solution/training/combined/s02_metrics.json").read_text())["d1_j48"]
for name, block, key in (("base", base, "cos"), ("rerank_tuned", reranked, "kr")):
    for metric in ("mAP", "Rank-1", "Rank-5"):
        assert abs(block["ranking"][metric] - reference[key][metric]) < 1e-6, (name, metric)
# И перекалибровка тем же правилом сходится к сохранённым порогам 20.09. Допуск
# тот же 1e-6 и по той же причине: порог — это значение уверенности, оно едет
# вместе с последними битами float32-инференса. Измерено: векторы этого прогона
# расходятся с векторами прогона калибровки (job_55, другой хост и 4 потока) до
# 1.2e-07, что даёт |Δ| порога 1.0e-08 на косинусе и 2.7e-09 на переранжировании.
for scale, got, want in (("cos", best["threshold"], tau), ("rr", rr_best["threshold"], rr_tau)):
    assert abs(got - want) < 1e-6, (scale, got, want, abs(got - want))
(out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
# Обе конфигурации печатаются рядом и подписанными: одни и те же имена метрик
# у косинуса и у переранжирования значат разные числа, и §6 приводит обе строки.
for label, ranking, threshold, refusal in (
        ("cosine  (t_cos; HTTP API и batch --no-rerank; metrics.json -> base)",
         base["ranking"], best["threshold"], base["refusal"]),
        ("rerank  (t_rr;  batch по умолчанию — числа §6; metrics.json -> rerank_refusal)",
         reranked["ranking"], rr_best["threshold"], report["rerank_refusal"])):
    print("REPRODUCED", label, json.dumps({
        "mAP": ranking["mAP"], "Rank-1": ranking["Rank-1"], "threshold": threshold,
        "F1": refusal["f1"], "TNR": refusal["tnr"]}))
PY
```

Ожидаемый вывод последних двух строк — обе конфигурации подписаны, сверять §6 нужно с той, что относится к нужному режиму (значения соответствуют перекалибровке 20.09; при пересчёте на другой машине допуск сравнения 1e-6, см. assert внутри R)). Поле `threshold` — порог, выбранный **заново** внутри R по правилу из §6, а не прочитанный из `config.py`: на другой машине его последние цифры поедут вместе с последними битами float32-инференса. Ниже — фактический вывод прогона 21.09.2026 (2 CPU, 2 потока); отклонение от рабочих порогов `0.5141976914190476` и `0.5282812306342437` составило 1.0e-08 и 2.7e-09, остальные поля совпали до всех печатных цифр:

```text
REPRODUCED cosine  (t_cos; HTTP API и batch --no-rerank; metrics.json -> base) {"mAP": 0.7316093422310448, "Rank-1": 0.6850961538461539, "threshold": 0.5141977018197194, "F1": 0.7684996605566871, "TNR": 0.7302158273381295}
REPRODUCED rerank  (t_rr;  batch по умолчанию — числа §6; metrics.json -> rerank_refusal) {"mAP": 0.7740915539438481, "Rank-1": 0.7307692307692307, "threshold": 0.5282812332956895, "F1": 0.8090787716955942, "TNR": 0.7841726618705036}
```

Для проверки самого контура — **команда T**:

```bash
docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -w /repo/04-solution/eval \
  vehicle-reid-service python -B -m unittest -q test_metrics test_protocols
```

Для фиксации среды — **команда V**:

```bash
docker run --rm --network none vehicle-reid-service python -m pip freeze --all
```

Дополнительно в сервисе есть [tools/eval_split.py](04-solution/service/tools/eval_split.py) и [tools/calibrate_threshold.py](04-solution/service/tools/calibrate_threshold.py). Они проверены на векторах, заново полученных командой R: ранжирование совпало до всех 16 цифр, калибровка прошла 187 прямых сверок с контуром. Эти скрипты не копируются в образ Dockerfile и требуют доступа к исходному репозиторию; для калибровки нужно явно передать выходной каталог.

## 8. Таблица «число → команда» и границы воспроизведения

| Числа/утверждение | Как получить | Результат проверки |
|---|---|---|
| 1110 query, 750 gallery, 832 с парой, 278 без пары; размерность 512 | R → `split`, `batch`, `row_order` | Пересчитано; SHA-256 сплита и разделение идентичностей проверены |
| Все mAP, Rank-1/5, mINP, mAP@10 из §6 и прирост 0.0424822117 | R → `base`, `reranked`, `delta_mAP` | Совпали с [s02_metrics.json](04-solution/training/combined/s02_metrics.json); допуск проверки `1e-6`. На референс-окружении VM сервисный батч совпал с контуром точно, до всех 16 цифр (`tools/eval_split.py`) |
| Оба рабочих порога, F1/TNR/Recall, TP/FP/FN/TN, AUC-PR из §6 | R → `calibration`, `rerank_calibration`, `base.refusal`, `rerank_refusal`, `old_threshold` | Пересчитаны через настоящий `evaluate()`; сверены с калибровкой 20.09 ([service/calib-d1_j48/](04-solution/service/calib-d1_j48/)) |
| Параметры `(6,3,0.3)` как оптимум подбора | **Не команда, а историческая справка.** Сетку считал `s04_grid.py <label> <query.npy> <gallery.npy> <query_meta.csv> <gallery_meta.csv>` — пять обязательных аргументов, без них `ValueError` на разборе `argv`. Векторов, которые он ждёт (`04-solution/postproc/out/*.npy` из `s03_extract.py`), в git нет: `.gitignore` исключает `*.npy`. Сохранён результат: [grid_tune_base.json](04-solution/postproc/out/grid_tune_base.json) | Применение параметров проверено командой R; исходная сетка заново не воспроизведена из чистого git |
| 8 836 743 байта и SHA-256 сдаваемого OSNet | R → `model`; `sha256sum 04-solution/service/model/*.onnx` | Скачано заново, размер и хеш совпали |
| Второй ONNX combined_v1: 8 742 779 байт, `b1ba5021…02efb2`; матрица whitening f32: 1 051 114 байт, `eb4433ff…4b3c5b5`; f64-оригинал матрицы: 2 101 738 байт, `4b02f158…267eb625` | `sha256sum 04-solution/service/model/osnet_ain_combined_v1.onnx 04-solution/service/model/lw_ens_j48_rho0.5.npz 04-solution/training/combined/lw_ens_j48_rho0.5_f64.npz` | Совпали с `config.py` и проверкой в Dockerfile; f64-оригинал нужен только для пересчёта дельты f32 |
| Дельта метрик от f32-хранения whitening | скрипт прогона job_55 (сравнение A f64 / B f32-хранение / C f32-сервис на одних векторах валидации) | `d_kr_mAP = 0.0`, `d_cos_mAP = 0.0` (все 16 печатных цифр совпали); max |Δэмбеддинг| = 1.4282077284710759e-08 |
| Порядок `(query; gallery)` в `embeddings.npy` | R → `row_order`, независимый batch=1 на границах | Проверено переизвлечением, а не чтением отчёта |
| 60 тестов метрик | T | Пройдены (15–16.09); повторно пройдены 20.09.2026 из чистого дерева после переключения на d1_j48 |
| 187 прямых сверок новой калибровки (20.09) | `docker run --rm --network none -v "$REPO:/repo:ro" -v "$OUT_DIR:/out" vehicle-reid-service python -B /repo/04-solution/service/tools/calibrate_threshold.py /out/val/embeddings.npy /out/calibration` | Пройдены на независимо извлечённых векторах конфигурации d1_j48; результат — [headline.json](04-solution/service/calib-d1_j48/headline.json) и `verification.json` |
| Точные версии всех пакетов образа | V | Полный freeze проверенной сборки — §10 |
| Исторические 43.8 мс/объект и 0.55 с на rerank | **Не команда, а историческая справка.** Замер делал `s09_timing.py`. Ему нужны исследовательское окружение (раздел 10), каталог изображений (`REID_DATA_DIR`) и промежуточные векторы `04-solution/postproc/out/*.npy` из шага `s03_extract.py`; `.npy` в git не входят. Из чистого клона скрипт падает трейсбеком `FileNotFoundError` на первом же недостающем входе — запускать его не нужно. Сохранён результат: [s09_timing.json](04-solution/postproc/out/s09_timing.json) | Точные исторические тайминги не воспроизведены. R выполняет новый ограниченный замер, с другим объёмом выборки и лимитом CPU |
| Эффект зоны пластины около −0.003 mAP, 95% CI [−0.011; +0.004]; расширенная маска: верхняя граница +0.016 | **Не команда, а историческая справка.** Считал `eval_variants.py` по векторам 13 вариантов из `plate-ablation/work/emb/`; их в git нет. Запущенный в исследовательском окружении (раздел 10), скрипт это и сообщает — понятным сообщением, а не трейсбеком: `нет векторов вариантов в …/work/emb (13 из 13, например base)` | **Не воспроизведено из git:** нет `work/emb`, детекций и исходной ручной разметки. Числа только из [ablation.json](04-solution/plate-ablation/out/ablation.json) и [отчёта](04-solution/plate-ablation/REPORT.md) |
| Качество локализации пластины, bootstrap/crossfit и расширенные EDA-числа | Команды соответствующих [plate](04-solution/plate-ablation/REPORT.md), [refusal](04-solution/refusal/REPORT.md), [EDA](04-solution/eda/REPORT.md) отчётов | Полная цепочка не восстановлена; не включать в перечень независимо воспроизведённых результатов |
| Размеры, SHA-256 и версии внешних источников | `sha256sum FILE`, `wc -c < FILE`, V; manifest внешних eval-исходников | Измерены для локальных файлов; неизвестные версии и хеши датасетов явно отмечены в §9 |

**Про строки «историческая справка».** Скрипты `04-solution/postproc/scripts/` и `04-solution/plate-ablation/scripts/` — рабочие записи исследования, а не часть сдаваемого пути; **запускать их для проверки решения не нужно.** Из чистого клона они и не отработают: в образ сервиса они не копируются, системный `python3` даёт `ModuleNotFoundError: No module named 'onnxruntime'` (им нужно исследовательское окружение из раздела 10), а в этом окружении они упираются в промежуточные входы, которых в git нет (`*.npy`, `work/`, каталог изображений). Проверено: `eval_variants.py` сообщает об этом понятной строкой, `s09_timing.py` и `s04_grid.py` — трейсбеком `FileNotFoundError` и `ValueError` соответственно. Всё, что из этих цепочек вошло в заявленные числа, сохранено рядом в `out/*.json`, а сдаваемые метрики пересчитывает команда R — без единого из этих скриптов.

Скорость зависит от оборудования и его загрузки; значения R — справочный замер, не FPS стандартизованного скрипта жюри. Ранжирование работает квадратично по памяти: одна матрица `float64` занимает `8(Q+G)²` байт, одновременно используется несколько. Заявления о галерее на миллион объектов или стабильном real-time-потоке не проверены.

## 9. Все использованные внешние модели, данные и исходники

### Модели и веса

| Назначение | Версия, источник и лицензия | Размер и SHA-256 |
|---|---|---|
| Сдаваемый OSNet-AIN (модель 1 конвейера) | Open Model Zoo **2022.1**, `vehicle-reid-0001`. [Прямой ONNX](https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx), [карточка релиза 2022.1.0](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/models/public/vehicle-reid-0001/model.yml). MIT для оригинальной модели, [текст лицензии](https://raw.githubusercontent.com/sovrasov/deep-person-reid/ea27fd23c962addbd24d8586c5aaf8afe60db0db/LICENSE) | **8 836 743 байта** (8.427 MiB; 8.837 MB). `4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f` |
| Сдаваемый combined_v1 (модель 2 конвейера) | Наш дообученный OSNet-AIN: LP-FT на RoundaboutHD (MIT) + CARLA (Apache-2.0) + train организатора. Внешнего URL нет; журнал — [training/combined/](04-solution/training/combined/). Поставляется в архиве решения (`*.onnx` в git исключён) | **8 742 779 байт**. `b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2` |
| Сдаваемая матрица whitening | Наш артефакт: `P`, `m` (ρ=0,5), обучены на `train_fit` по дельтам ансамбля; поставляется в `float32`, f64-оригинал — [training/combined/lw_ens_j48_rho0.5_f64.npz](04-solution/training/combined/lw_ens_j48_rho0.5_f64.npz) | **1 051 114 байт**. `eb4433ffd5e38d3751d5cb04e234090060a83be6d1720b47bcf2fa1274b3c5b5` (f64: 2 101 738 байт, `4b02f158fa54d89a54f9ad30d7e471f1baa14d7019ea9ea26b42ea7c267eb625`) |
| Сравнение второй модели, в сервис не включено | FastReID SBS ResNet50-IBN, веса релиза **v0.1.1**. [Прямая загрузка](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth), [Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d/MODEL_ZOO.md). Код Apache-2.0; отдельная лицензия checkpoint в исследовательском пакете не найдена | **198 261 759 байт**. `57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd` |
| Локальная производная предыдущего checkpoint | `veri_sbs_R50-ibn.model_only.pth`, сохранение `ckpt["model"]` скриптом [s07_fastreid_extract.py](04-solution/postproc/scripts/s07_fastreid_extract.py); отдельного внешнего URL нет | **99 273 627 байт**. `8595fa79eeb09f35c565729be1e8826e3b1de7951c86bff0aff048c78529076f` |

Весовой лимит ТЗ (≤ 2 ГБ) проверяется по сумме трёх сдаваемых файлов: 8 836 743 + 8 742 779 + 1 051 114 = **18 630 636 байт**. Два ONNX поставляются вне git (`*.onnx` в `.gitignore`) и проверяются по SHA-256 при сборке образа и при загрузке модели; матрица whitening входит в git. FastReID использовался при исследовании, поэтому раскрыт, хотя в runtime не нужна. [Отчёт постобработки](04-solution/postproc/REPORT.md) объясняет отказ от ансамбля и TTA на том этапе; итоговый ансамбль d1_j48 — [training/combined/REPORT.md](04-solution/training/combined/REPORT.md).

### Обучающие источники публичных весов

**OSNet обучен не только на VeRi.** Upstream [README.rst, commit `ea27fd23c962addbd24d8586c5aaf8afe60db0db`](https://github.com/sovrasov/deep-person-reid/blob/ea27fd23c962addbd24d8586c5aaf8afe60db0db/README.rst) перечисляет VeRi, VERI-Wild, CompCars и VMMRdb. В [папке автора](https://drive.google.com/drive/folders/1C-yTiJrvStMkwgHm9GvqwIG7EKjhSEjl) находится тот же ONNX с ID, указанным в `original_source` карточки OMZ, и [конфигурация обучения](https://drive.google.com/uc?export=download&id=1z2ODsTcbCJr3e1h9vGmiulpIn2X6kz4J): `sources=[['veri','veriwild'],['universemodels']]`. Конфигурация имеет 2019 байт, SHA-256 `d939a7b35268e41420b4520b6763bf437bc417e971d8560bf1a847c31b2b231a`. Имена и версия экспортированного графа проверяются по самому ONNX; эта конфигурация не является полным журналом экспорта.

| Датасет | Как использован | Версия, доступ и условия |
|---|---|---|
| Набор организатора LCT 2026, задача №7 | Подбор, калибровка, абляции; вход дообучения combined_v1 (train-часть) | Именованной версии нет; идентификация по SHA-256 ниже. Доступ через кабинет организатора. Отдельная лицензия/прямой URL архива не приложены |
| [RoundaboutHD](https://huggingface.co/datasets/yl4300/RoundaboutHD) | Дообучение combined_v1: 511 ID / 6 068 кадров ReID-сабсета | MIT по [карточке репозитория данных Bath](https://researchdata.bath.ac.uk/1574/) (с оговоркой о формулировке — [datasets-open/REPORT.md](02-research/datasets-open/REPORT.md) §6); снапшот зафиксирован в [training/combined/journal.md](04-solution/training/combined/journal.md) |
| [CARLA-ReID](https://github.com/sekilab/VehicleReIdentificationDataset) | Дообучение combined_v1: 605 ID / 7 260 кадров | Apache-2.0 на репозиторий (подтверждена GitHub API); данные лежат на Dropbox без отдельного файла лицензии — та же оговорка; снапшот — [training/combined/journal.md](04-solution/training/combined/journal.md) |
| [VeRi-776](https://vehiclereid.github.io/VeRi/) | Предобучение OSNet и сравнивавшегося FastReID | Семейство VeRi-776; точный архив автора весов, размер и SHA-256 **неизвестны**. Официальный доступ по запросу; некоммерческое использование |
| [VERI-Wild](https://github.com/PKU-IMRE/VERI-Wild) | Предобучение OSNet | В upstream не зафиксировано, какой выпуск/ревизия архива; нельзя автоматически подставлять 2.0. Размер и SHA-256 неизвестны. Доступ по запросу, некоммерческое использование |
| [CompCars](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) | Предобучение OSNet, объединение с VMMRdb | Датасет 2015 года; точные части и ревизия набора автора весов неизвестны. [Инструкция загрузки](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt). Только некоммерческие исследования, ограничено дальнейшее распространение; размер/SHA-256 использованных архивов неизвестны |
| [VMMRdb](https://github.com/faezetta/VMMRdb) | Предобучение OSNet, объединение с CompCars | Датасет из публикации 2017 года; полный набор или подмножество автора весов не указаны. [Архив автора](https://www.dropbox.com/s/uwa7c5uz7cac7cw/VMMRdb.zip?dl=0). В репозитории MIT, отдельное подтверждение условий для всех изображений отсутствует; размер/SHA-256 использованного архива неизвестны |
| [ImageNet](https://www.image-net.org/download.php) | FastReID Model Zoo указывает ImageNet-предобучение backbone | Точная ревизия и исходный checkpoint не указаны в переданных материалах. Изображения нами не скачивались. OSNet YAML также включает `pretrained=True`, но происхождение его исходной инициализации полностью не установлено |

Изображения RoundaboutHD и CARLA-ReID нашей командой скачаны и использованы для дообучения combined_v1 (состав и хэши собранного набора — [training/combined/journal.md](04-solution/training/combined/journal.md), [datasets-open/REPORT.md](02-research/datasets-open/REPORT.md) §5); VeRi, VERI-Wild, CompCars, VMMRdb — только готовые веса, сами изображения нами не скачивались. Публичный checkpoint с MIT/Apache-лицензией не заполняет отсутствующие сведения о версиях обучающих данных. Полная цепочка их происхождения остаётся документированным пробелом по разделам 7 и 12 ТЗ; нельзя писать, что все использованные данные безусловно открыты и воспроизводимы.

### Отпечатки фактически выданного набора

```text
train.csv         9556 строк  536677 байт  bd1df45b052ae9aabb7fd356898e244875f8cad2f111a3f76977c1b90bce9268
test_query.csv    1110 строк   54288 байт  97e1ed21942bae9c95b1ce2e5d339d9f635bf49fa484367e6b19349789bb9b4c
test_gallery.csv   750 строк   36693 байт  a64ed21fa39bbfd8c7a172468d043415800500b6ed695cdb8162188cf9005496
```

Изображения: **11 416 файлов, 7 503 286 826 байт**. SHA-256 текстового манифеста по всем изображениям — `8274bf0d792e3b4fee1009f6fae7649466c296cffcb359c670150a587473cc4a`. Чтобы получить тот же манифест независимо от локали, выполните из `DATA_DIR`:

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path
manifest = hashlib.sha256(); count = total = 0
for p in sorted(Path('images').iterdir()):
    if not p.is_file():
        continue
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    manifest.update(f'{digest}  images/{p.name}\n'.encode())
    count += 1; total += p.stat().st_size
print(count, total, manifest.hexdigest())
PY
```

Эта дополнительная команда требует Python 3.11+ и читает все изображения. Сам `data/README.md` имеет SHA-256 `b517186daf5186f8ace0b3d2feb9fbce328cd09e620f22b84ff9a2a292d4575c`.

### Внешний код и статические ресурсы

| Источник | Закреплённая версия / лицензия / проверка |
|---|---|
| FastReID для сравнения модели | commit `c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d`, Apache-2.0; внешний клон не включён в git решения |
| FastReID / Torchreid для сверки метрик | commits `7ed6240e2cb5e56e5ccd61744a3045f24ea7e62d` / `6b8fe56638d81c36b7872e8e41efe18232335f2f`; Apache-2.0 / MIT. [Manifest с прямыми URL, размерами и полными SHA-256](04-solution/eval/sources/manifest.json), исходники и LICENSE включены в `eval/sources/` |
| MATLAB `compute_AP.m`, `evaluation.m` для сверки соглашения | person-re-ranking commit `ca27f37dd88b27ee1f9dbc5668a9d0f45989ca71`; те же URL/хеши в manifest. Отдельная лицензия этих файлов не установлена |
| Алгоритм k-reciprocal | Zhong et al., CVPR 2017; порт в проекте, NumPy float64. Проверяемый [Python-исходник](https://github.com/zhunzhong07/person-re-ranking/blob/278f2c704e7f033b767f4158dff4febc03a0b78a/python-version/re_ranking_ranklist.py), commit `278f2c704e7f033b767f4158dff4febc03a0b78a`, 4250 байт, SHA-256 `04bf1b5886ef06275d88be4d68d8f8cfd3ebc892e026f41e848ce294ab192b8e`. У исходного репозитория нет корневого LICENSE; конкретная ревизия первоначального портирования не зафиксирована. Лицензия FastReID автоматически к этому файлу не приписывается |
| Swagger UI | **5.29.5**, Apache-2.0, [релиз](https://github.com/swagger-api/swagger-ui/tree/v5.29.5), [LICENSE](https://github.com/swagger-api/swagger-ui/blob/v5.29.5/LICENSE). Локальные файлы — `service/app/static/vendor/`; отдельный LICENSE рядом с ними не приложен |
| Qdrant | **v1.15.5**, [исходники](https://github.com/qdrant/qdrant/tree/v1.15.5), Apache-2.0; Docker-образ `docker.io/qdrant/qdrant:v1.15.5` |

Отпечатки Swagger: `swagger-ui-bundle.js` — 1 510 312 байт, `a646692ba5c95a74f99bb2c15ac879dec9a0001a72aed133ad65068da9e90c97`; `swagger-ui.css` — 155 212 байт, `bc5e8d5c013477cf1f35e2fb8ba1dff66be0f72f24e669a509635657145e1acb`; `favicon.png` — 628 байт, `3ed612f41e050ca5e7000cad6f1cbe7e7da39f65fca99c02e99e6591056e5837`. Проверка: `sha256sum 04-solution/service/app/static/vendor/*`.

## 10. Полный список библиотек и версии среды

### Сдаваемый контейнер

Python **3.13.15**, Linux x86_64; корневая исследовательская `.venv` — Python **3.13.13**. Исходный образ сборки: `python:3.13-slim`, проверенный image ID `51cce855bb6e44a8ff6ed0f46ded8850f246bfc7549801459f83fc34b80c210f`, registry digest `sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285`. Проверенный Qdrant digest: `sha256:0fb8897412abc81d1c0430a899b9a81eb8328aa634e7242d1bc804c1fe8fe863`. Dockerfile/Compose сейчас используют теги, поэтому эти digest описывают проверенную среду, но не принуждают будущую сборку к ней.

Прямые зависимости: ONNX Runtime 1.30.0 (MIT), NumPy 2.5.3 (BSD и лицензии включённых компонентов), Pillow 12.3.0 (MIT-CMU), FastAPI 0.141.1 (MIT), Uvicorn 0.53.0 (BSD-3-Clause), python-multipart 0.0.32 (Apache-2.0), qdrant-client 1.19.0 (Apache-2.0). Версии из [requirements.txt](04-solution/service/requirements.txt) установлены и проверены `pip check`.

Полный результат `pip freeze --all`, включая транзитивные зависимости и установщик; те же 31 строка лежат в репозитории как [requirements-lock.txt](04-solution/service/requirements-lock.txt) и подставляются в сборку constraints-файлом:

```text
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
certifi==2026.7.22
click==8.5.0
fastapi==0.141.1
flatbuffers==25.12.19
grpcio==1.84.0
h11==0.16.0
h2==4.4.1
hpack==4.2.0
httpcore==1.0.9
httpx==0.28.1
hyperframe==6.1.0
idna==3.19
numpy==2.5.3
onnxruntime==1.30.0
packaging==26.3
pillow==12.3.0
pip==26.2.1
portalocker==3.2.0
protobuf==7.36.1
pydantic==2.13.5
pydantic_core==2.46.5
python-multipart==0.0.32
qdrant-client==1.19.0
starlette==1.6.0
typing-inspection==0.4.4
typing_extensions==4.16.0
urllib3==2.7.0
uvicorn==0.53.0
```

Этот список фиксирует фактическую проверенную сборку и подключён к Dockerfile как constraints — в обеих ветвях: офлайн `pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt -c requirements-lock.txt` и сетевой `pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt`. Проверено пересборкой (офлайн, `--no-cache --pull=never --network none`) — `pip freeze --all` нового образа совпал с файлом построчно, `pip check` без замечаний. Полным lock-файлом с хешами wheel он при этом не является. Все имена пакетов соответствуют их публикациям на PyPI, например [onnxruntime 1.30.0](https://pypi.org/project/onnxruntime/1.30.0/), [NumPy 2.5.3](https://pypi.org/project/numpy/2.5.3/), [qdrant-client 1.19.0](https://pypi.org/project/qdrant-client/1.19.0/). Хеши wheel-файлов в текущем файле зависимостей отсутствуют. Системные библиотеки фиксируются сохранением образа; отдельный список Debian-пакетов можно получить `docker run --rm --network none vehicle-reid-service dpkg-query -W`.

### Исследовательские окружения

Это окружение **не нужно** ни для сборки образа, ни для запуска сервиса, ни для команд R/T/V: они целиком живут в контейнере. Оно нужно только тем, кто хочет открыть исследовательские скрипты `04-solution/postproc/`, `04-solution/plate-ablation/`, `04-solution/eval/` вне контейнера. Само по себе оно их не «чинит»: промежуточные `.npy` и каталоги `work/` в git не входят (раздел 11), так что скрипты всё равно упрутся в недостающие входы — часть понятным сообщением, часть трейсбеком (см. раздел 8). Окружение убирает только одну помеху — `ModuleNotFoundError: No module named 'onnxruntime'`.

Корневая `.venv` в git не входит; создаётся с нуля так (нужна сеть):

```bash
python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install onnxruntime==1.30.0 numpy==2.5.3 pillow==12.3.0 onnx==1.22.0
```

Проверено: `pip freeze --all` такого окружения даёт ровно список ниже (строка `pip` — версия, с которой поставляется `venv` используемого Python; в проверенной среде это 26.2.1). Дальше скрипты запускаются своим интерпретатором, например `"$REPO/.venv/bin/python" 04-solution/plate-ablation/scripts/eval_variants.py`.

Полный freeze корневой `.venv` (инспекция графа, baseline, метрики, постобработка, абляции):

```text
flatbuffers==25.12.19
ml_dtypes==0.6.0
numpy==2.5.3
onnx==1.22.0
onnxruntime==1.30.0
packaging==26.3
pillow==12.3.0
pip==26.2.1
protobuf==7.36.1
typing_extensions==4.16.0
```

Зависимость `onnx==1.22.0` нужна для инспекции ONNX и не включена в runtime. Файл [eval/requirements.txt](04-solution/eval/requirements.txt) задаёт диапазон `numpy>=1.24,<3`, а не точную версию; опубликованные численные сравнения проверены с NumPy 2.5.3.

Полный freeze отдельного окружения сравнения FastReID:

```text
filelock==3.32.3
fsspec==2026.7.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.5.3
pillow==12.3.0
pip==24.3.1
PyYAML==6.0.3
setuptools==78.1.0
sympy==1.14.0
tabulate==0.10.0
termcolor==3.3.0
torch==2.14.0+cpu
typing_extensions==4.16.0
yacs==0.1.8
```

Построение исследовательских графиков также использует **matplotlib**, отсутствующий в корневой `.venv` и runtime. При аудите в системном Python обнаружены: Matplotlib 3.10.9, NumPy 2.2.6, Pillow 12.3.0, contourpy 1.3.3, cycler 0.11.0, fonttools 4.61.0, kiwisolver 1.5.0, packaging 24.2, pyparsing 3.1.2, python-dateutil 2.8.2, six 1.17.0. Это снимок доступной вспомогательной среды; единый lock исходного построения всех рисунков не сохранён.

Обход Python-импортов и файлов зависимостей проведён по всем каталогам `04-solution/`, включая review и проверочные исходники. Неиспользованные модели/датасеты из обзорных списков `02-research/` не являются зависимостями результата. Заготовки последующего обучения в `.worker-payload/` в перечисленные проверенные числа не входят.

## 11. Известные ограничения и недостающие материалы

- **Полная исследовательская воспроизводимость пока не достигнута.** Часть скриптов содержит `/home/artem/projects/...` и ссылки на временные `work/`, `metadata/`, `evaluator_snapshot/`, `.ids`, `.npy`, которых нет в git. Команда R обходит эти зависимости для основных метрик, но не восстанавливает все старые эксперименты.
- **Абляция пластины — не доказательство полного отсутствия использования ГРЗ, но проверена для обеих сдаваемых моделей.** Для combined_v1 маскирование зоны номера статистически неотличимо от контрольной зоны той же площади (все контрасты накрывают 0, [training/combined/REPORT.md §5](04-solution/training/combined/REPORT.md)); для исходного OSNet — [plate-ablation/](04-solution/plate-ablation/). Исходной ручной разметки/детекций/эмбеддингов вариантов в git нет — числа берутся из сохранённых json. Верхнюю границу `+0.016 mAP` нельзя называть «меньше процента метрики».
- **Не полностью известны версии внешних обучающих архивов и часть условий их распространения.** RoundaboutHD и CARLA-ReID зафиксированы по снапшоту в журнале обучения; цепочка происхождения предобучения OSNet (VeRi, VERI-Wild, CompCars, VMMRdb) остаётся документированным пробелом. Все обнаруженные источники раскрыты в §9; отсутствующие сведения названы отсутствующими.
- **Офлайн-сборка закрыта не полностью: остаются базовые образы.** Пакеты ставятся из `wheels/` в составе решения, PyPI на сборке не нужен, инференс и запуск сети не требуют. Но `python:3.13-slim` и `qdrant/qdrant:v1.15.5` в репозиторий не кладутся — их нужно либо вытянуть заранее на машине с сетью, либо привезти архивом образов (раздел 4). Флаг `--network none` эту дыру не закрывает и не проверяет: он отрезает сеть только у шагов `RUN`, а базовый образ сборка тянет из реестра — проверка изолированности требует `--pull=never` (раздел 3). Версии всех пакетов закреплены constraints-файлом, но хешей wheel в нём нет, колёса лежат без подписей и отдельной сверки хешей, а базовые образы в Dockerfile/Compose заданы тегами, а не digest; необходимые уведомления о сторонних лицензиях приложены не ко всем vendored-файлам.
- **API и batch используют разные алгоритмы поиска.** API — cosine, batch — совместное переранжирование. Порог batch зависит от шкалы; качество пакетного ранжирования не является измерением качества одиночного API-запроса.
- **Параметры протокола жюри не полностью определены.** Это касается усечения top-10, порядка камерной фильтрации, знаменателя mAP и единицы F1. В документации выбрана и названа одна ветвь, альтернативы доступны в контуре.
- **Обучение проводилось (LP-FT combined_v1), но нет GPU-замера жюри, ANN на миллион объектов и испытания длительного потока.** Одновременно хранятся матрицы попарных оценок; ограничение по памяти относится ко всему поиску, а не только к одному батчу изображений.
- **Валидация ввода неодинакова:** API проверяет границы bbox и тип изображения; batch в основном доверяет CSV и допускает PIL-padding при выходе bbox за кадр. Формат сдачи рассчитан на уникальные `image_id` и галерею, достаточную для десяти кандидатов.
- **Переключение конфигурации 20.09 по честной пометке §6:** против прежнего чемпиона контура прирост по приёмочному правилу не засчитан (+0.0120 < σ 0.0134), хотя бутстрэп значим (p=0.003); решение о переключении — человеческое, зафиксировано в [training/combined/REPORT.md §6](04-solution/training/combined/REPORT.md).

## 12. Что подготовить к передаче жюри

Код вместе с этим `SOLUTION.md`, три файла из test-прогона, **три файла весов** с полными SHA-256 (OSNet — публичный, скачивается `fetch_model.sh`; combined_v1 и матрица whitening — наши, поставляются в архиве/git), оба контейнерных образа для офлайн-работы, инструкция с путями к данным и версиями. Каталог `04-solution/service/wheels/` входит в код и выгружается вместе с ним — это то, чем закрыт разд. 9 ТЗ (сборка в изолированной среде), вырезать его из поставки нельзя. Требование ТЗ буквально называет `Readme.md`; здесь документ называется `SOLUTION.md`, поскольку главный README занят описанием проекта. При оформлении сдачи нужно явно указать ссылку на этот файл как документацию решения; наличие нужного содержания само по себе не подтверждает соблюдение буквального имени файла.

Недостающие исходники исследовательских протоколов, лицензии и provenance нужно приложить до заявления о полном воспроизведении **всех** результатов. Презентация, публикация прототипа и загрузка на платформу этой документацией не выполняются.
