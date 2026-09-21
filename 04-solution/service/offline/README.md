# Базовые образы для сборки без интернета

ТЗ п. 8 требует, чтобы инференс работал без доступа в сеть, а все необходимое
поставлялось в составе решения. Колёса Python лежат в `../wheels/`, веса — в
`../model/`. Здесь — два базовых образа, которые иначе пришлось бы тянуть из реестра:
на них ссылаются `Dockerfile` и `docker-compose.yml`.

| Образ | Назначение | Digest |
|---|---|---|
| `python:3.13-slim` | базовый образ сервиса | `sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285` |
| `qdrant/qdrant:v1.15.5` | векторная база | `sha256:0fb8897412abc81d1c0430a899b9a81eb8328aa634e7242d1bc804c1fe8fe863` |

## Загрузка

```bash
sha256sum -c base-images.tar.gz.sha256
docker load -i base-images.tar.gz     # или: podman load -i base-images.tar.gz
```

Проверено `podman load`: оба образа встают, код возврата 0.

После загрузки собирать и запускать с запретом обращения к реестру:

```bash
docker build --pull=never -t lct-reid-service ..
```

Без `--pull=never` сборка попытается сходить в реестр за базовым образом, даже если
`--network none` отрезает сеть у шагов `RUN`: базовый образ тянется до них.

## Как собран архив

```bash
docker save python:3.13-slim qdrant/qdrant:v1.15.5 | gzip -9 > base-images.tar.gz
```

44 МБ. Оба образа распространяются свободно: `python:3.13-slim` — официальный образ Docker
на Debian с Python под лицензией PSF, `qdrant/qdrant` — Apache 2.0.

Архив и его отпечаток обновляются вместе с версиями в `Dockerfile` и
`docker-compose.yml`; при расхождении верным считается `Dockerfile`, а архив надо
пересобрать приведённой выше командой.
