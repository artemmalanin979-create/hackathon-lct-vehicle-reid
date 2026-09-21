# job_75 — Docker и полная галерея

Проверяемая ревизия: `07abc5babce0f6c95fcb037cbe743262e04577cc`, `main`. Дата: 22.09.2026. Проверки, сборки и инференс выполнены только на `worker-vm`. Исходный репозиторий не исправлялся; коммитов и публикаций нет.

**Итог: офлайн-путь Docker Engine / Compose V2 и пользовательские HTTP-сценарии на двух полных галереях по 750 объектов пройдены в описанном ниже стенде.** Найдены уточнения Docker-документации: изоляция daemon, отображение image ID/размеров и SELinux. **Ограничение rerank-порога нужно сохранить:** ложное принятие автомобиля при отказе cosine воспроизводится и с 750 gallery, если меняется состав query. Конкретный старый ложный приём серого кадра на полных галереях не воспроизвёлся.

## Docker: подтверждённые результаты

Fedora 44: **Moby/Docker Engine 29.7.2**, Buildx **0.37.1**, встроенный BuildKit **0.32.2**, builder driver **docker**, containerd **2.3.5**. Проверяется настоящий daemon Docker, не совместимый CLI Podman. [Версии и исходное состояние Podman](evidence/worker/tools-and-podman-before.log).

Штатная установка:

```bash
sudo dnf -y install moby-engine docker-cli docker-compose docker-buildx
```

В текущих Fedora-репозиториях Compose уже **5.5.1**. Для буквальной проверки запрошенной ветки **V2** отдельно установлен официальный **2.40.3** в каталог задания `docker-config-v2/cli-plugins/docker-compose`, с проверкой опубликованного SHA-256. Системный пакет не заменялся. [Лог установки](evidence/worker/dnf-install.log), [проверка V2](evidence/worker/compose-v2-sha256.log). Это явное дополнение к штатной установке, поскольку Fedora 44 не предоставляет пакет 2.x. Различие версий описано в [официальной истории Compose](https://docs.docker.com/compose/intro/history/).

Новый клон находится на узле в `/home/fedora/lct-reid/jobs/job_75/repo`. Транспорт — копия Git-объектов прежнего аудита плюс bundle с диапазоном `8374816..main`, затем **новый `git clone --no-hardlinks --branch main`**. Рабочие файлы прежнего клона, его готовый runtime-образ и Podman-хранилище в сборку не подставлялись. Alternates нет; 1140 tracked-файлов. Прямое получение исходников с GitHub этой проверкой не подтверждается.

| Проверка | Результат | Свидетельство |
|---|---|---|
| Отдельное пустое Docker-хранилище | Образы и контейнеры отсутствовали до импорта | [Образы](evidence/worker/images-before.log), [контейнеры](evidence/worker/containers-before.log) |
| SHA двух архивов и импорт | Оба архива прошли проверку и `docker load` | [SHA](evidence/worker/base-sha256.log), [Python](evidence/worker/python-load.log), [Qdrant](evidence/worker/qdrant-load.log) |
| Веса из клона | Три SHA совпали; fetch-script выполнен без внешней сети | [Лог](evidence/worker/fetch-model.log) |
| Сборка Dockerfile без кэша | Успех, 01:35:20–01:37:23 MSK | [Полный build log](evidence/worker/build.log), [builder](evidence/worker/buildx-ls.log) |
| Runtime-зависимости | `pip check` прошёл, все 31 версии совпали с lock | [pip check](evidence/worker/pip-check.log), [сверка](evidence/worker/runtime-lock.json) |
| Registry при отсутствующем base | Docker попытался скачать manifest, сеть daemon заблокировала попытку | [Отрицательный контроль](evidence/worker/missing-base-blocked.log) |
| `docker build --pull=never` | Парсер отвергает строку `never` как неверный boolean | [Лог](evidence/worker/docker-pull-never.log) |
| Внешняя сеть контейнера Compose | TCP к внешнему адресу: errno 101, сеть недоступна | [Контроль](evidence/worker/compose-egress-control.log) |

Сборка выполнена **точной командой из SOLUTION §3**:

```bash
docker image inspect python:3.13-slim >/dev/null
docker build --no-cache --pull=false --network none \
  -t vehicle-reid-service "$REPO/04-solution/service"
```

Внешняя сеть предварительно отключена у **самого daemon и его собственного containerd/BuildKit**. В SOLUTION эта предпосылка названа, но конкретная команда отсутствует. Проверенный способ на Fedora/systemd, без отключения SSH или сети всего узла:

```bash
TASK=/home/fedora/lct-reid/jobs/job_75
mkdir -p "$TASK/docker-config"
printf '{}\n' > "$TASK/docker-daemon.json"
sudo systemd-run --unit=job75-docker --slice=job75.slice \
  --property=PrivateNetwork=yes --property=Delegate=yes \
  --property=TasksMax=infinity --property=Restart=no \
  /usr/bin/dockerd --config-file="$TASK/docker-daemon.json" \
  --data-root="$TASK/docker-data" --exec-root=/run/job75-docker \
  --pidfile="$TASK/dockerd.pid" --host="unix://$TASK/docker.sock" \
  --cgroup-parent=job75.slice --debug
sudo systemctl set-property --runtime job75.slice CPUQuota=200% MemoryMax=6G
```

Клиент запускается с `DOCKER_HOST=unix://$TASK/docker.sock`; команды импорта/build/Compose выполнялись в root-shell через `sudo bash` над приложенными сценариями, с экспортом переменных **внутри** этого shell. Пользователь в группу docker не добавлялся. Использован штатный builder с driver `docker`. Фактический root containerd — **`$TASK/docker-data/containerd/daemon`**, отдельный от системного. В namespace daemon есть loopback и собственные Docker bridge, но **нет внешнего интерфейса и default route**. Поэтому контейнеры общаются друг с другом, а registry/PyPI недоступны. [Состояние namespace](evidence/worker/daemon-isolation.log), [сеть во время инференса](evidence/worker/daemon-network-during-inference.log), [запуск daemon](scripts/start-docker.sh), [импорт и build](scripts/import-build.sh).

Импорт и вся дальнейшая работа идут через этот же сокет. Для HTTP-проверки опубликованного порта в данном стенде нужен вход в namespace:

```bash
PID=$(sudo systemctl show job75-docker -p MainPID --value)
sudo nsenter -t "$PID" -n curl --fail http://127.0.0.1:8000/api/health
```

Это инфраструктура аудитора, а не изменение приложения. Порт внутри изолированного namespace не становится обычным `localhost:8000` внешнего хоста. В обычной офлайн-машине жюри с daemon в её основном namespace документированный `localhost` сохраняет смысл.

**Отличие от штатного Fedora unit:** полный прогон использовал отдельный daemon с настройками Docker по умолчанию, без `--selinux-enabled`; системный SELinux оставался Enforcing. Штатный Fedora `docker.service` этот флаг включает и использует системный containerd. Это различие дополнительно проверено отдельными SELinux-пробами ниже, а не скрыто. [Штатный unit](evidence/worker/fedora-docker-unit.txt), [настройки основного daemon](evidence/worker/docker-info.log).

## Что в Docker-инструкции верно, а что требует уточнения

1. **Docker-команда `--pull=false --network none` синтаксически верна и сборку проходит.** Она требует локального base и реальной изоляции daemon. В актуальном SOLUTION уже правильно сказано, что `--pull=false` само по себе не запрещает скачивание отсутствующего base. Отрицательный контроль это подтвердил. `--pull=never` допустим у `podman build`, но не у `docker build`; у `docker compose up --pull never` это, напротив, допустимая Compose-опция.
2. **Фраза offline/README про «отключение сети у демона либо предварительную загрузку образов» неверна.** Предварительная загрузка и отключение сети — разные условия. Только импорт не создаёт изоляцию. Исправленная по смыслу последовательность: импорт в выбранное хранилище → отключение внешней сети daemon/BuildKit → build с указанными флагами → Compose с запретом pull. Конкретный работающий способ приведён выше; файлы проекта не правились.
3. **Обещание «должно получиться ровно это» для ID и размеров `docker images` не переносится с Podman на Docker 29.** Этот Docker использует containerd image store — штатный выбор для новых установок 29+, что подтверждает [документация Docker](https://docs.docker.com/engine/storage/containerd/). Архивы исправны: config-byte sequence и все rootfs diff IDs совпали. Но image ID в Docker здесь — digest импортированного manifest, а ожидаемые числа README — SHA config.

| Образ | Config SHA, обещанный как ID | ID Docker 29 после load | `docker images` / `image inspect.Size` |
|---|---|---|---|
| Python | `51cce855bb6e…` | `cfd07ed85b05…` | 270 MB / 130 389 265 байт |
| Qdrant | `0ad2e23181e5…` | `da16f065092a…` | 369 MB / 181 085 859 байт |

Полные хеши, manifests и сравнение байтов — [image-identity.json](evidence/worker/image-identity.json), [список Docker](evidence/worker/images-loaded.log), [проверяющий сценарий](scripts/image-inventory.py). Надёжная офлайн-проверка для этой версии — SHA архивов, правильные теги, config SHA внутри импортированного manifest и совпадение слоёв. Registry-digest, указанный в таблице README, отдельным обращением к реестру не сверялся.

**Ещё одно неверное обобщение — SOLUTION §3, замечание о SELinux:** фраза «Docker … обычно расставляет метки сам» не делает обычный bind-mount `:/data:ro` безопасно воспроизводимым на штатном Fedora Docker. Найденные данные job_73 **уже имели `container_file_t`**, поэтому они читались и при включённом SELinux. Чтобы отделить это наследство от поведения Docker, сделана отдельная копия только gallery-CSV с явно заданной обычной домашней меткой `user_home_t`:

| Docker с `--selinux-enabled`, SELinux Enforcing | Результат |
|---|---|
| Обычный `-v "$DATA_DIR:/data:ro"`, метка `user_home_t` | Exit 1, `PermissionError: [Errno 13]` на CSV; автоматической переметки нет |
| Тот же mount с `--security-opt label=disable` | Успех, прочитаны 750 строк |
| `chcon -Rt container_file_t "$DATA_DIR"`, затем исходный mount `:ro` | Успех, прочитаны 750 строк |

[Первый отказ](evidence/worker/selinux-fresh-unlabelled.log), [label=disable](evidence/worker/selinux-fresh-label-disable.log), [chcon](evidence/worker/selinux-fresh-chcon.log), [метки до](evidence/worker/selinux-fresh-labels-before.log) и [после](evidence/worker/selinux-fresh-labels-after.log). Речь о локальной аудиторской копии, исходные данные job_73 не перемечались. Для **неизменённого Compose** точная подготовка данных на таком Fedora:

```bash
sudo chcon -Rt container_file_t "$DATA_DIR"
docker compose up -d --no-build --pull never
```

Команда `chcon` уже есть в service/README под замечанием о Podman; она нужна и Docker с включённой SELinux-изоляцией. Суффикс `:z` также проверен отдельно. Полный повтор сборки/инференса с SELinux-enabled daemon не делался: здесь проверены именно доступ к данным и способы исправить метки, а весь численный прогон относится к явно описанному основному daemon.

## Данные и границы численной проверки

По журналам job_72/job_73 найдены реальные данные на узле. Исходные каталоги не менялись; сделаны отдельные копии:

- `job_73-OQZJggyo/data-test-only` → `job_75/data/test`: 1110 test-query, **750 test-gallery**, 1860 JPEG.
- `job_73-OQZJggyo/data-val-only` → `job_75/data/val`: 1860 validation-JPEG, CSV — из `split/files` нового клона. Для неизменённого Compose-loader те же CSV скопированы под именами `test_query.csv`/`test_gallery.csv`; это явно обозначенная validation-ветвь, не подмена test.
- `/var/tmp/job72-jury-dMALXs33/diagnostic/fixtures` → `job_75/data/old-fixtures`: точные прежние 4 query / 10 gallery. Это монтажные кропы с графикой из репозиторной иллюстрации; они не выдаются за исходные кадры организатора.

Штатный preflight проверил **1862 test-входа и 1864 validation-входа**: размеры и SHA-256 совпали. [Test](evidence/worker/inputs-test.json), [validation](evidence/worker/inputs-val.json).

**Важное условие интерпретации:** прежние `q_foreign_10` и `q_foreign_11` были чужими только для старой галереи из десяти кропов. Они происходят из validation-строк **989/1039**, vehicle_id **795/1319**, у которых в полной validation-галерее есть пары. Их принятие на полной галерее нельзя автоматически назвать false positive. Для отрицательного контроля заранее выбраны реальные validation-строки **0, 316, 332** (`has_mate=0`), для положительного межкамерного — **61**. Серый кадр взят ровно из прежнего контроля.

## Что получилось на полной галерее

**Штатный test-750 поднят исходным Compose V2 одной командой** после предварительной сборки:

```bash
export DOCKER_HOST=unix://$TASK/docker.sock
export DOCKER_CONFIG=$TASK/docker-config-v2
export COMPOSE_FILE=$REPO/04-solution/service/docker-compose.yml
export DATA_DIR=$TASK/data/test
docker compose -p job75-test up -d --no-build --pull never
```

Qdrant прошёл автоматический healthcheck; loader завершился с **exit 0**, `{"collection":"gallery","points":750,"rows":750}`. Проверен не только счётчик: выгруженные из настоящего Qdrant **все 750 image_id и bbox** совпали с CSV, векторы имеют размерность 512. [Код выхода](evidence/worker/test-loader-exit.txt), [полный Compose log](evidence/worker/test-compose-all.log), [HTTP-сценарий](evidence/worker/test-http/summary.json).

Самосопоставление первого объекта даёт 1.0 и правильный image_id. Первый штатный test-query возвращает кандидата с 0.744258; **его правильность по test-ground-truth не утверждается**, поскольку меток test нет. Серый кадр отвергнут с 0.306337. `/`, `/docs`, `/openapi.json`, `/api/ui/state`, embed и показ кропа отвечают. Передача собственного порога работает: 0.306335 принимает серый запрос, 0.306339 отказывает. Порог по умолчанию не менялся. [Проверка границы в API](evidence/worker/test-http/threshold-around-grey-score.json).

Отдельно **`docker compose up --build -d`** из service/README прошёл в том же изолированном daemon для нового validation-проекта: сборка и зависимость от healthcheck не потребовали правок. [Лог команды](evidence/worker/val-compose-up.log). Таким образом проверены обе текущие документированные формы запуска.

**Validation-750 также реально загружена штатным loader в отдельный настоящий Qdrant:** exit 0, `points=rows=750`, все payloads совпали с CSV. Затем исходные кадры отправлены в работающий HTTP API:

| Запрос | Результат HTTP | Лучшая cosine-оценка |
|---|---|---:|
| Межкамерная пара, vehicle_id 1067, query-строка 61 | Принят; правильный top-1 `9562bbdaffa14ee78764f9cf90ada568` | 0.708452 |
| Чужой vehicle_id 397, строка 0 | Отказ, candidates пуст | 0.466739 |
| Чужой vehicle_id 1482, строка 316 | Отказ, candidates пуст | 0.428037 |
| Чужой vehicle_id 925, строка 332 | Отказ, candidates пуст | 0.336360 |
| Тот же серый кадр | Отказ, candidates пуст | 0.293167 |

Все ответы HTTP 200, штатный `t_cos=0.5141976914190476`. [Loader](evidence/worker/val-loader.log), [exit](evidence/worker/val-loader-exit.txt), [полный HTTP-сценарий с разметкой](evidence/worker/val-http/summary.json). Пороговые overrides по обе стороны best_score прошли также здесь; это отдельные проверочные запросы, настройки сервиса не менялись. [Граница API](evidence/worker/val-http/threshold-around-grey-score.json).

**Полный validation-batch 1110×750 заново извлёк признаки из JPEG в новом Docker-образе**, сеть отключена. Использованы неизменённые `app.batch`, `config`, scoring и измерительный контур из клона. `embeddings.npy` — `(1860,512)`, float32; все значения конечны. CSV повторного расчёта теми же сервисными scoring-функциями побайтово совпал со штатным `app.batch`. [Run info](evidence/worker/val-batch/run_info.json), [проверка CSV](evidence/worker/rerank/official-batch-candidates-check.json).

| Режим | mAP | Rank-1 | F1 | TNR | Ложные принятия среди 278 запросов без пары |
|---|---:|---:|---:|---:|---:|
| Cosine, штатный `t_cos` | 0.7316093422 | 0.6850961538 | 0.7684996606 | 0.7302158273 | 75 |
| Rerank, штатный `t_rr` | 0.7740915539 | 0.7307692308 | 0.8090787717 | 0.7841726619 | 60 |

Числа совпали с опубликованными; метрики посчитаны по документированным `market` + `presence`, mAP/R1 — по 832 запросам с парой. Это метрики **того же calibration-сплита**, не новый независимый holdout. [Полные новые метрики](evidence/worker/rerank/validation-metrics.json).

Штатный `service/tools/calibrate_threshold.py` тоже выполнен на этих новых векторах: **187 сверок, max_abs_error=0.0**. Он заново выбирает `t_cos=0.5141977018197194` и `t_rr=0.5282812332956895`; абсолютные отличия от config — **1.0401×10⁻⁸** и **2.6614×10⁻⁹**, F1/TNR прежние. [Новые пороги](evidence/worker/calibration/headline.json), [сверки](evidence/worker/calibration/verification.json). Проверено и граничное правило `score >= threshold`: при пороге, равном лучшему скору, кандидат принят; при следующем большем float — отказ. [Проверка на реальных score-строках](evidence/worker/rerank/score-boundary-check.json).

## Воспроизводится ли проблема rerank-порога при 750 объектах

**Да, ложное принятие при отказе cosine возможно и на полной галерее из 750. Но точный старый результат с серым кадром на полной галерее не повторился. Решает также состав query, а не только количество gallery.**

Для начала новым Docker-образом воспроизведён прежний контроль **4×10**, включая прежние значения 0.541908 / 0.768959 / 0.801990. Это исключает объяснение различий другой моделью или порогами. [Новый контроль 4×10](evidence/worker/rerank/legacy-4x10.json).

| Тот же контрольный запрос | Cosine, 4×10 | RR, 4×10 | Cosine, test-750 | RR, test-750 | Интерпретация на test |
|---|---:|---:|---:|---:|---|
| `q_foreign_10` | 0.199984, отказ | 0.541908, принят | 0.592590, принят | 0.899181, принят | Без vehicle_id правильность неизвестна |
| `q_foreign_11` | 0.131052, отказ | 0.768959, принят | 0.712536, принят | 0.902025, принят | Без vehicle_id правильность неизвестна |
| Серый кадр | 0.251708, отказ | 0.801990, принят | 0.306337, отказ | 0.473609, отказ | Старый ложный приём серого здесь исчез |

[Новые результаты 4×750 test](evidence/worker/rerank/legacy-4x750-test.json). Для серого запрос **1×750** дал тот же отказ. На validation-750 у двух названных автомобилей найдены правильные vehicle_id: RR 0.993234 / 0.996133. Их принятие там корректно; прежнее слово «чужие» описывало состав старой маленькой галереи. [4×750 validation с разметкой](evidence/worker/rerank/legacy-4x750-val.json).

**Подтверждённый контрпример с настоящим чужим автомобилем:** исходный кадр `64a7b342e5654ac38e16ab1cc2d58037`, vehicle_id **397**. Ни одного такого ID среди всех 750 gallery нет. Векторы запроса и галереи во всех следующих строках **одни и те же**, из нового полного Docker-batch; меняется только состав других query:

| Контекст запроса | Cosine / решение | Rerank / решение |
|---|---|---|
| Один запрос × те же 750 | 0.4667395182 / отказ | **0.8639154832 / ложное принятие** |
| Панель из 5 запросов × те же 750 | 0.4667395182 / отказ | **0.8639154832 / ложное принятие** |
| Все штатные 1110 запросов × те же 750 | 0.4667395182 / отказ | 0.3039154832 / отказ |

Ошибочный кандидат — `b0bffbecdb9b4b96a884f11f01b0f9a8`. [Одиночный запрос](evidence/worker/rerank/panel-1-1x750-val.json), [панель с размеченными положительным/отрицательными кадрами](evidence/worker/rerank/labelled-5x750-val.json), [полный контекст](evidence/worker/rerank/full-1110x750-val.json).

У положительного межкамерного запроса из строки 61 правильный top-1: cosine **0.7084523949**; RR **0.5788630424** в панели 5×750 и **0.9305437139** в полном 1110×750. Два других заранее выбранных чужих автомобиля, ID 1482/925, получают отказ в обоих режимах в панели и полном контексте.

Серый кадр на validation-750:

- Cosine **0.2931673323**, отказ.
- RR при 1×750, 4×750 и 5×750: **0.4706614910**, отказ.
- RR при добавлении серого к полным 1110 query: **0.4703873551**, отказ. Все скоры исходных 1110 запросов при этом остались прежними, max delta = 0. [Проверка 1111×750](evidence/worker/rerank/grey-in-full-query-context.json).

**Практический вывод:** заявленная калибровка работает на полном 1110×750 с ожидаемыми F1/TNR, но **условия «в галерее 750» недостаточно** для переноса `t_rr` на отдельный запрос или малый пакет. Ограничение в текущем SOLUTION §2 уже правильно относится ко всему составу query/gallery; его нужно сохранить. Оснований снимать его или задавать универсальный минимальный размер галереи нет. Для одиночного пользовательского поиска текущий API обоснованно остаётся cosine. Это не обещает нулевых false positives и не превращает similarity в вероятность.

Новые пороги в config не записывались, автоматический fallback не добавлялся. Штатные значения во всех контрольных решениях: `t_cos=0.5141976914190476`, `t_rr=0.5282812306342437`, KR(6,3,0.3).

## Отступления и оставшиеся границы

- Git получен локальным транспортом; данные взяты из прежних заданий по прямому указанию BRIEF, не скачаны заново через кабинет организатора. До нового клона предварительно читались tracked-документы исходного чистого дерева. Исполнялось только дерево нового клона.
- Изоляция Docker конкретизирована собственным systemd unit; выбран встроенный builder `docker`. Полный цикл проверен на Compose **2.40.3**. Системный Compose **5.5.1** дополнительно прочитал конфигурацию и состояние готового стенда; отдельная полная сборка/загрузка через него не повторялась.
- Во время двух параллельных извлечений общий лимит памяти повышен с 4 до 6 GiB, CPU оставался 2; собственные loader/batch получали индивидуальные квоты. OOM не было. Время под квотами и параллельной нагрузкой не является проверкой обещания «около минуты» или заявленного бенчмарка.
- Две ошибки вспомогательных сценариев аудитора: первоначально неверно указан путь к CSV без `split/files`; затем перезапись собственного shell-скрипта, пока его первый экземпляр ожидал loader, вызвала EOF **после успешного loader**. Решение не исправлялось и не перезапускалось ради сокрытия ошибки; код выхода loader, полный лог Compose, HTTP-результаты и все 750 payloads сохранила другая ветвь сбора. Подробности — [journal.md](journal.md).

**Осталось непроверенным:**

- Интерактивные действия мышью в браузере; проверен пользовательский HTTP-сценарий и отдача статики/Swagger/кропов.
- Качество поиска на закрытом размеченном test организатора: имеющиеся test-CSV не содержат vehicle_id. Также нет новой независимой holdout-калибровки или доказанного минимального безопасного размера query/gallery.
- Полный цикл приложения со штатным SELinux-enabled Fedora unit, другие версии Engine/Compose, rootless Docker, отдельный/удалённый BuildKit, другие архитектуры. SELinux-доступ к bind-mount отдельно проверен, его условия явно изложены выше.
- Получение исходных данных посторонним членом жюри через внешний кабинет и остальные вопросы предыдущей общей приёмки, не входившие в два пункта job_75.

Оба рабочих Compose-проекта и собственные Docker-daemon остановлены после проверки; образы, тома и доказательства оставлены в каталоге задания на worker-vm. Podman сохранился и прошёл отдельный офлайн-запуск Python-контейнера. [Проверка сосуществования](evidence/worker/podman-coexistence.log). Исходный репозиторий и проверяемый клон остались на том же коммите с чистым tracked-состоянием.

Полный рабочий каталог на узле: `/home/fedora/lct-reid/jobs/job_75`. Копия доказательств приложена в `evidence/worker/`; [машинный итог](evidence/worker/acceptance-summary.json), [финальный пустой список контейнеров](evidence/worker/docker-containers-final.log), [чистый status клона](evidence/worker/clone-status-final.txt). Сценарии проверки — `scripts/`, последовательность и ошибки аудитора — [journal.md](journal.md), контрольные суммы — [SHA256SUMS](SHA256SUMS).
