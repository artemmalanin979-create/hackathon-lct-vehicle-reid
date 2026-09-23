# Колёса зависимостей

30 файлов, 59 146 237 байт: зависимости из `requirements.txt` и
`requirements-lock.txt`. Они входят в Git; установка пакетов по умолчанию не
обращается к PyPI. Для офлайн-сборки нужен правильный локальный Python base.
Базовые образы лежат в [offline/](../offline/README.md) — по отдельному архиву на образ,
с отпечатками в `SHA256SUMS`.

## Что проверено

[Аудит 21.09](../../audit/jury-path-2/REPORT.md) собрал неизменённый Dockerfile
на Podman 5.8.1 с заранее имевшимся `python:3.13-slim`: без кэша, без pull,
с сетью `RUN` в режиме `none`, а сам процесс сборки — в отдельном network
namespace через `unshare --net`. Установка из `/wheels` и `pip check` прошли;
31 версия совпала с lock (30 колёс плюс pip из базового образа).

Из каталога `04-solution/service`, после импорта Python base:

```bash
podman image exists docker.io/library/python:3.13-slim
podman build --no-cache --pull=never --network none -t vehicle-reid-offline .
```

Эта команда запрещает pull в Podman и сеть у `RUN`. Проверку полной изоляции
проводят также при отключённой внешней сети процесса сборки. Один
`--network none` изолирует только `RUN` и не запрещает загрузку базового образа.
У Docker `--pull` — булев флаг: `--pull=never` к `docker build` неприменим.
Для Docker с локальным base используется `--pull=false`, а внешний доступ
отключают у daemon/BuildKit. Этот путь проверен на Docker Engine 29.7.2 /
Compose 2.40.3 — [отчёт](../../audit/docker-path/README.md).
Семантика флагов: [Docker build reference](https://docs.docker.com/reference/cli/docker/buildx/build/).

Этот аудит не воспроизвёл test-артефакты из кадров: исходных изображений в его
клоне не было. Условия прежнего побайтового сравнения и manifest выпуска
21.09 приведены в [artifacts-final/README.md](../artifacts-final/README.md).

## Как пересобрать колёса

На машине с интернетом, под Linux x86_64, в том же Python base:

```bash
cd 04-solution/service
podman run --rm -v "$PWD:/src:z" -w /src python:3.13-slim \
  pip download --no-cache-dir -r requirements.txt -c requirements-lock.txt -d /src/wheels
```

Тег base не закрепляет digest. Версии, проверенные digest и границы офлайн-поставки
описаны в [SOLUTION.md §3, §10](../../../SOLUTION.md).

## Запасной путь

При несовместимости колёс с платформой можно явно разрешить установку из сети:

```bash
mkdir -p wheels
docker build --build-arg PIP_SOURCE=network -t vehicle-reid-service .
```

Каталог `wheels/` обязателен в обеих ветвях: Dockerfile безусловно выполняет
`COPY wheels/ /wheels/`. В сетевой ветви он может быть пустым; отсутствие каталога
не исправляется выбором `PIP_SOURCE=network`. Сетевой путь не является
офлайн-сборкой. Колёса не входят в финальный слой runtime: копируются только
установленные пакеты из отдельной стадии.
