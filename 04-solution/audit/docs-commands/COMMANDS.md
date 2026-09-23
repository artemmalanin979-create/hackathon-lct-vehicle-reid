# Полный реестр команд

Исходные тексты сохранены с переносами; ID совпадают с REPORT.md.

## C001

[SOLUTION.md:79](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:79)

```bash
export REPO="$(pwd -P)"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.045 с. [лог](evidence/C001-a2.log).

Команда завершилась успешно.

## C002

[SOLUTION.md:80](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:80)

```bash
export DATA_DIR="$REPO/data"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.048 с. [лог](evidence/C002-a2.log).

Команда завершилась успешно.

## C003

[SOLUTION.md:81](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:81)

```bash
export OUT_DIR="$REPO/outputs/reproduce"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.043 с. [лог](evidence/C003-a2.log).

Команда завершилась успешно.

## C004

[SOLUTION.md:82](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:82)

```bash
export COMPOSE_FILE="$REPO/04-solution/service/docker-compose.yml"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.052 с. [лог](evidence/C004-a2.log).

Команда завершилась успешно.

## C005

[SOLUTION.md:83](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:83)

```bash
mkdir -p "$OUT_DIR"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.045 с. [лог](evidence/C005-a2.log).

Команда завершилась успешно.

## C006

[SOLUTION.md:84](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:84)

```bash
python3 04-solution/reproduce/check_inputs.py --mode val --data-dir "$DATA_DIR" \
  --report "$OUT_DIR/inputs-val.json"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.077 с. [лог](evidence/C006-a2.log).

Ожидаемый exit 2: нет 1860 JPG; перечислены имена, каталог назначения и источник данных.

## C007

[SOLUTION.md:93](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:93)

```bash
(cd "$REPO/04-solution/service" && sh model/fetch_model.sh)
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.316 с. [лог](evidence/C007.log).

Команда завершилась успешно.

## C008

[SOLUTION.md:104](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:104)

```bash
(cd 04-solution/service/offline && sha256sum -c SHA256SUMS)
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.106 с. [лог](evidence/C008.log).

Команда завершилась успешно.

## C009

[SOLUTION.md:105](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:105)

```bash
docker load -i 04-solution/service/offline/python-3.13-slim.tar.gz
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.488 с. [лог](evidence/C009.log).

Команда завершилась успешно.

## C010

[SOLUTION.md:106](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:106)

```bash
docker load -i 04-solution/service/offline/qdrant-v1.15.5.tar.gz
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 2.371 с. [лог](evidence/C010.log).

Команда завершилась успешно.

## C011

[SOLUTION.md:124](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:124)

```bash
podman image exists docker.io/library/python:3.13-slim
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.113 с. [лог](evidence/C011.log).

Команда завершилась успешно.

## C012

[SOLUTION.md:125](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:125)

```bash
podman build --no-cache --pull=never --network none \
  -t vehicle-reid-service "$REPO/04-solution/service"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 16.426 с. [лог](evidence/C012.log).

Команда завершилась успешно.

## C013

[SOLUTION.md:136](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:136)

```bash
docker image inspect python:3.13-slim >/dev/null
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.052 с. [лог](evidence/C013.log).

Команда завершилась успешно.

## C014

[SOLUTION.md:137](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:137)

```bash
docker build --no-cache --pull=false --network none \
  -t vehicle-reid-service "$REPO/04-solution/service"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 11.503 с. [лог](evidence/C014.log).

Команда завершилась успешно.

## C015

[SOLUTION.md:166](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:166)

```bash
docker compose -f 04-solution/service/docker-compose.yml up --build -d --pull never
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 17.446 с. [лог](evidence/C015.log).

Docker Compose собрал образ и запустил сервисы; без данных loader затем вышел 1, что проверено отдельно.

## C016

[SOLUTION.md:193](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:193)

```bash
docker compose up -d --no-build --pull never
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.671 с. [лог](evidence/C016.log).

Сервисы запускаются без build/pull; полнота галереи ограничена отсутствующими данными.

## C017

[SOLUTION.md:199](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:199)

```bash
curl --fail http://localhost:8000/api/health
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.066 с. [лог](evidence/C017.log).

HTTP 200 после инициализации API; gallery_points=null без входов loader.

## C018

[SOLUTION.md:200](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:200)

```bash
curl --fail http://localhost:8000/api/version
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.04 с. [лог](evidence/C018.log).

HTTP 200; API сообщает текущую d1_j48, версии и настройки.

## C019

[SOLUTION.md:210](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:210)

```bash
docker compose logs --no-log-prefix loader | tail -1
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.092 с. [лог](evidence/C019.log).

Вместо финального JSON последняя строка лога — FileNotFoundError; это корректно показывает незавершённый loader. Код pipeline 0 сам по себе не означает загрузку галереи.

## C020

[SOLUTION.md:219](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:219)

```bash
docker compose down -v          # удаляет контейнеры И именованный том проекта
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.289 с. [лог](evidence/C020-a2.log).

Команда завершилась успешно.

## C021

[SOLUTION.md:220](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:220)

```bash
docker compose up -d --no-build --pull never
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 6.751 с. [лог](evidence/C021.log).

Команда завершилась успешно.

## C022

[SOLUTION.md:232](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:232)

```bash
docker run --rm --network none \
  -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -m app.batch \
  --images-dir /data/images --query /data/test_query.csv \
  --gallery /data/test_gallery.csv --out-dir /out --threads 2
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 1.676 с. [лог](evidence/C022-a2.log).

Без test CSV — FileNotFoundError (1). Отдельный позитивный контроль на 2+10 синтетических кадрах прошёл (0), mount /out и четыре файла подтверждены.

## C023

[SOLUTION.md:246](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:246)

```bash
docker save -o "$OUT_DIR/runtime-images.tar" vehicle-reid-service docker.io/qdrant/qdrant:v1.15.5
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 6.304 с. [лог](evidence/C023.log).

Команда завершилась успешно.

## C024

[SOLUTION.md:247](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:247)

```bash
sha256sum "$OUT_DIR/runtime-images.tar" > "$OUT_DIR/runtime-images.tar.sha256"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.361 с. [лог](evidence/C024.log).

Команда завершилась успешно.

## C025

[SOLUTION.md:390](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:390)

```bash
python3 04-solution/reproduce/check_inputs.py --mode val --data-dir "$DATA_DIR" \
  --report "$OUT_DIR/inputs-val.json"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.089 с. [лог](evidence/C025-a2.log).

Ожидаемый exit 2: нет 1860 JPG; понятная диагностика.

## C026

[SOLUTION.md:405](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:405)

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B /repo/04-solution/reproduce/run.py \
  --mode val --data-dir /data --out-dir /out
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.666 с. [лог](evidence/C026.log).

Ожидаемый exit 2, preflight внутри runtime без сети; инференс не начинался.

## C027

[SOLUTION.md:437](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:437)

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B /repo/04-solution/reproduce/run.py \
  --mode test --data-dir /data --out-dir /out
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.599 с. [лог](evidence/C027.log).

Ожидаемый exit 2: нет test CSV и изображений; понятная диагностика.

## C028

[SOLUTION.md:457](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:457)

```bash
docker run --rm --network none -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -w /repo/04-solution/eval \
  vehicle-reid-service python -B -m unittest -q test_metrics test_protocols
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.615 с. [лог](evidence/C028.log).

60 контейнерных тестов прошли без сети.

## C029

[SOLUTION.md:460](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:460)

```bash
docker run --rm --network none vehicle-reid-service python -m pip freeze --all
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.514 с. [лог](evidence/C029.log).

31 строка freeze; точное совпадение requirements-lock.txt.

## C030

[SOLUTION.md:557](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:557)

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

Cwd: `/home/artem/tmp/job_88-audit/clean/data`. Exit: `1`. Время: 0.059 с. [лог](evidence/C030.log).

FileNotFoundError: images/ отсутствует; отдельного пояснения скрипт не выдаёт.

## C031

[SOLUTION.md:664](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:664)

```bash
python3 -m venv "$REPO/.venv"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.45 с. [лог](evidence/C031.log).

Команда завершилась успешно.

## C032

[SOLUTION.md:665](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:665)

```bash
"$REPO/.venv/bin/pip" install onnxruntime==1.30.0 numpy==2.5.3 pillow==12.3.0 onnx==1.22.0
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 5.696 с. [лог](evidence/C032.log).

Установка прошла. Утверждение о точном freeze не воспроизводится: protobuf 7.36.2 против 7.36.1; pip зависит от venv, как оговорено.

## C033

[04-solution/audit/docker-path/README.md:27](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/audit/docker-path/README.md:27)

```bash
sudo systemd-run --unit=job75-docker --slice=job75.slice \
  --property=PrivateNetwork=yes --property=Delegate=yes \
  --property=TasksMax=infinity --property=Restart=no \
  /usr/bin/dockerd --config-file="$TASK/docker-daemon.json" \
  --data-root="$TASK/docker-data" --exec-root=/run/job75-docker \
  --pidfile="$TASK/dockerd.pid" --host="unix://$TASK/docker.sock" \
  --cgroup-parent=job75.slice --debug
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 0.058 с. [лог](evidence/C033.log).

Не проверен работоспособный daemon: нет системного /usr/bin/dockerd. TASK и docker-daemon.json в рецепте не определены; portable Docker основного стенда работает по другому пути.

## C034

[04-solution/baseline/README.md:141](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:141)

```bash
(cd 04-solution/service && sh model/fetch_model.sh)
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.057 с. [лог](evidence/C034-a2.log).

Команда завершилась успешно.

## C035

[04-solution/baseline/README.md:149](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:149)

```bash
PY=../../.venv/bin/python
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `0`. Время: 0.002 с. [лог](evidence/C035.log).

Команда завершилась успешно.

## C036

[04-solution/baseline/README.md:150](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:150)

```bash
M=../service/model/osnet_ain_x1_0_vehicle_reid.onnx
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `0`. Время: 0.002 с. [лог](evidence/C036.log).

Команда завершилась успешно.

## C037

[04-solution/baseline/README.md:151](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:151)

```bash
IMG=../../data/images
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `0`. Время: 0.002 с. [лог](evidence/C037.log).

Команда завершилась успешно.

## C038

[04-solution/baseline/README.md:154](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:154)

```bash
for part in query gallery; do
  $PY scripts/extract_embeddings.py --csv ../split/files/val_$part.csv --images-dir $IMG \
      --model $M --out out/val_$part.npy --ids-out out/val_$part.ids
  $PY scripts/extract_embeddings.py --csv ../split/files/val_$part.csv --images-dir $IMG \
      --model $M --mask-bottom 0.30 --out out/val_${part}_mask30.npy
  $PY scripts/extract_embeddings.py --csv ../split/files/val_$part.csv --images-dir $IMG \
      --model $M --grayscale --out out/val_${part}_gray.npy
  $PY scripts/extract_embeddings.py --csv ../split/files/val_$part.csv --images-dir $IMG \
      --model $M --mask-bottom 0.30 --grayscale --out out/val_${part}_mask30gray.npy
done
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `1`. Время: 1.447 с. [лог](evidence/C038.log).

Все 8 вызовов цикла выполнены: FileNotFoundError на первом недостающем JPG каждого варианта.

## C039

[04-solution/baseline/README.md:166](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:166)

```bash
$PY scripts/run_eval.py
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `1`. Время: 0.07 с. [лог](evidence/C039.log).

FileNotFoundError: baseline/out/val_query.ids (производный вход предыдущего шага).

## C040

[04-solution/baseline/README.md:169](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:169)

```bash
for part in query gallery; do
  $PY scripts/extract_embeddings.py --csv ../../data/test_$part.csv --images-dir $IMG \
      --model $M --out out/test_$part.npy --ids-out out/test_$part.ids
done
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `1`. Время: 0.25 с. [лог](evidence/C040.log).

Оба вызова цикла выполнены: FileNotFoundError для test_query.csv/test_gallery.csv.

## C041

[04-solution/baseline/README.md:173](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:173)

```bash
$PY scripts/make_submission.py --threshold 0.34921352213815304
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `1`. Время: 0.065 с. [лог](evidence/C041.log).

FileNotFoundError: data/test_query.csv.

## C042

[04-solution/baseline/README.md:174](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/baseline/README.md:174)

```bash
$PY scripts/check_embeddings_order.py
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/baseline`. Exit: `1`. Время: 0.135 с. [лог](evidence/C042.log).

FileNotFoundError: baseline/artifacts/embeddings.npy; файл не входит в клон.

## C043

[04-solution/cloud/README.md:31](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/cloud/README.md:31)

```bash
./up.sh              # машина lct-gpu, зона d, образ ubuntu-2204-lts-cuda-12-2, печатает IP
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/cloud`. Exit: `1`. Время: 0.044 с. [лог](evidence/C043.log).

Понятный отказ: yc не найден; требование профиль/ключ явно указано. Облачные ресурсы не создавались.

## C044

[04-solution/cloud/README.md:32](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/cloud/README.md:32)

```bash
PREEMPTIBLE=1 ./up.sh   # то же, прерываемая — в четыре раза дешевле
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/cloud`. Exit: `1`. Время: 0.061 с. [лог](evidence/C044.log).

Понятный отказ: yc не найден; облачные ресурсы не создавались.

## C045

[04-solution/cloud/README.md:33](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/cloud/README.md:33)

```bash
./status.sh          # что существует в каталоге и сколько стоит в час — запускать после каждого прогона
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/cloud`. Exit: `1`. Время: 0.054 с. [лог](evidence/C045.log).

Понятный отказ: yc не найден; профиль/ключ отсутствуют у постороннего.

## C046

[04-solution/cloud/README.md:34](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/cloud/README.md:34)

```bash
./down.sh            # снести всё: машина с диском, адреса, подсеть, сеть; идемпотентно
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/cloud`. Exit: `1`. Время: 0.049 с. [лог](evidence/C046.log).

Понятный отказ: yc не найден; удалений облачных ресурсов не выполнялось.

## C047

[04-solution/eval/README.md:78](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval/README.md:78)

```bash
python -B -m unittest -v test_metrics   # тесты
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/eval`. Exit: `0`. Время: 0.263 с. [лог](evidence/C047-a2.log).

47 unit tests прошли; повторно без сети.

## C048

[04-solution/eval/README.md:79](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval/README.md:79)

```bash
python -B verify.py                     # полная проверка, включая мутации
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/eval`. Exit: `0`. Время: 15.509 с. [лог](evidence/C048-a2.log).

PASS: 60 tests; 25/25 mutants detected, без сети.

## C049

[04-solution/eval/review/README.md:69](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/eval/review/README.md:69)

```bash
cd review && python3 -B fuzz.py && python3 -B mutants.py && python3 -B demos.py && python3 -B tz_checks.py
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/eval`. Exit: `1`. Время: 0.056 с. [лог](evidence/C049.log).

Первый fuzz.py падает с ModuleNotFoundError: reid_metrics. Оставшиеся три команды отдельно дают тот же сбой (см. evidence/eval-review-subcommands.json).

## C050

[04-solution/postproc/README.md:77](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/postproc/README.md:77)

```bash
PY=../../.venv/bin/python
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/postproc`. Exit: `0`. Время: 0.003 с. [лог](evidence/C050.log).

Команда завершилась успешно.

## C051

[04-solution/postproc/README.md:81](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/postproc/README.md:81)

```bash
cp ../baseline/out/val_query.npy ../baseline/out/val_gallery.npy out/
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/postproc`. Exit: `1`. Время: 0.003 с. [лог](evidence/C051.log).

cp понятно сообщает об отсутствующих baseline .npy; README прямо задаёт их как результат предварительного шага.

## C052

[04-solution/postproc/README.md:82](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/postproc/README.md:82)

```bash
cp ../baseline/out/val_query.ids ../baseline/out/val_gallery.ids out/
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/postproc`. Exit: `1`. Время: 0.003 с. [лог](evidence/C052.log).

cp понятно сообщает об отсутствующих baseline .ids; явная зависимость от предварительного шага.

## C053

[04-solution/postproc/README.md:85](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/postproc/README.md:85)

```bash
$PY scripts/s01_verify_baseline.py
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/postproc`. Exit: `1`. Время: 0.067 с. [лог](evidence/C053.log).

FileNotFoundError: postproc/out/val_query.ids.

## C054

[04-solution/postproc/README.md:89](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/postproc/README.md:89)

```bash
$PY scripts/s03_extract.py --sets val --variants 208,208f,256
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/postproc`. Exit: `1`. Время: 0.196 с. [лог](evidence/C054.log).

Понятный отказ: сначала получить базовые векторы скриптом baseline.

## C055

[04-solution/postproc/README.md:92](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/postproc/README.md:92)

```bash
$PY scripts/s05b_make_tta_vectors.py --sets val
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/postproc`. Exit: `1`. Время: 0.068 с. [лог](evidence/C055.log).

Понятный отказ: сначала шаг 3, перечислены недостающие TTA-векторы.

## C056

[04-solution/postproc/README.md:95](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/postproc/README.md:95)

```bash
$PY scripts/s06_final_val.py
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/postproc`. Exit: `1`. Время: 0.07 с. [лог](evidence/C056.log).

FileNotFoundError: postproc/out/val_query_208.npy.

## C057

[04-solution/reproduce/README.md:66](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:66)

```bash
export REPO="$(pwd -P)"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.003 с. [лог](evidence/C057.log).

Команда завершилась успешно.

## C058

[04-solution/reproduce/README.md:67](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:67)

```bash
export DATA_DIR="$REPO/data"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.002 с. [лог](evidence/C058.log).

Команда завершилась успешно.

## C059

[04-solution/reproduce/README.md:68](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:68)

```bash
export OUT_DIR="$REPO/outputs/reproduce"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.002 с. [лог](evidence/C059.log).

Команда завершилась успешно.

## C060

[04-solution/reproduce/README.md:69](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:69)

```bash
mkdir -p "$OUT_DIR"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.003 с. [лог](evidence/C060.log).

Команда завершилась успешно.

## C061

[04-solution/reproduce/README.md:70](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:70)

```bash
python3 04-solution/reproduce/check_inputs.py --mode val --data-dir "$DATA_DIR" \
  --report "$OUT_DIR/inputs-val.json"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.04 с. [лог](evidence/C061.log).

Ожидаемый exit 2 с понятной диагностикой нехватки validation JPG.

## C062

[04-solution/reproduce/README.md:89](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:89)

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B /repo/04-solution/reproduce/run.py \
  --mode val --data-dir /data --out-dir /out
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.572 с. [лог](evidence/C062.log).

Ожидаемый exit 2 внутри runtime без сети.

## C063

[04-solution/reproduce/README.md:137](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:137)

```bash
cd 04-solution/reproduce
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.002 с. [лог](evidence/C063.log).

Команда завершилась успешно.

## C064

[04-solution/reproduce/README.md:138](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:138)

```bash
python3 -B -m unittest -v test_check_inputs
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/reproduce`. Exit: `0`. Время: 0.14 с. [лог](evidence/C064-a2.log).

5 тестов preflight/разделения val и test прошли без сети.

## C065

[04-solution/service/README.md:32](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:32)

```bash
cd 04-solution/service
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.04 с. [лог](evidence/C065.log).

Команда завершилась успешно.

## C066

[04-solution/service/README.md:33](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:33)

```bash
docker compose up --build -d
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 3.101 с. [лог](evidence/C066.log).

Команда завершилась успешно.

## C067

[04-solution/service/README.md:40](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:40)

```bash
docker compose -f 04-solution/service/docker-compose.yml up --build -d
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 3.677 с. [лог](evidence/C067.log).

Команда завершилась успешно.

## C068

[04-solution/service/README.md:51](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:51)

```bash
curl --fail http://localhost:8000/api/health   # storage.gallery_points > 0
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.297 с. [лог](evidence/C068.log).

HTTP 200; API доступен, но gallery_points=null без данных. Комментарий >0 требует успешного loader, чего здесь нет.

## C069

[04-solution/service/README.md:113](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:113)

```bash
DATA_DIR=/абсолютный/путь/к/data docker compose run --rm loader
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `1`. Время: 2.564 с. [лог](evidence/C069.log).

Loader завершается 1 с FileNotFoundError: /data/test_gallery.csv; нет пользовательского сообщения о получении данных.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
DATA_DIR="$REPO/data" docker compose run --rm loader
```

## C070

[04-solution/service/README.md:132](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:132)

```bash
docker build -t vehicle-reid-service .
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 1.778 с. [лог](evidence/C070.log).

Команда завершилась успешно.

## C071

[04-solution/service/README.md:133](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:133)

```bash
docker run --rm --network none \
  -v /путь/к/тестовому/набору:/data:ro \
  -v /путь/к/выводу:/out \
  vehicle-reid-service \
  python -m app.batch --images-dir /data/images \
      --query /data/test_query.csv --gallery /data/test_gallery.csv \
      --out-dir /out
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `1`. Время: 1.073 с. [лог](evidence/C071.log).

FileNotFoundError при отсутствии test CSV; контейнер запускается успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" vehicle-reid-service python -m app.batch --images-dir /data/images --query /data/test_query.csv --gallery /data/test_gallery.csv --out-dir /out
```

## C072

[04-solution/service/README.md:197](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:197)

```bash
python -m app.batch ... --no-rerank        # флаг
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `1`. Время: 0.249 с. [лог](evidence/C072.log).

Флаг --no-rerank распознан; далее FileNotFoundError на отсутствующем test CSV.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
python -m app.batch --images-dir ../../data/images --query ../../data/test_query.csv --gallery ../../data/test_gallery.csv --out-dir ../../outputs/batch-local --no-rerank
```

## C073

[04-solution/service/README.md:198](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:198)

```bash
REID_RERANK=0 python -m app.batch ...      # переменная окружения
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `1`. Время: 0.131 с. [лог](evidence/C073.log).

REID_RERANK=0 принят; далее FileNotFoundError на отсутствующем test CSV.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
REID_RERANK=0 python -m app.batch --images-dir ../../data/images --query ../../data/test_query.csv --gallery ../../data/test_gallery.csv --out-dir ../../outputs/batch-local
```

## C074

[04-solution/service/README.md:322](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:322)

```bash
python 04-solution/reproduce/run.py --mode val --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.045 с. [лог](evidence/C074.log).

Ожидаемый exit 2: validation preflight сообщает, что именно отсутствует.

## C075

[04-solution/service/README.md:323](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:323)

```bash
python 04-solution/reproduce/run.py --mode test --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.043 с. [лог](evidence/C075.log).

Ожидаемый exit 2: test preflight сообщает, что именно отсутствует.

## C076

[04-solution/service/artifacts-final/README.md:54](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/artifacts-final/README.md:54)

```bash
cd 04-solution/service
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.044 с. [лог](evidence/C076.log).

Команда завершилась успешно.

## C077

[04-solution/service/artifacts-final/README.md:55](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/artifacts-final/README.md:55)

```bash
python -m app.batch \
  --images-dir ../../data/images \
  --query ../../data/test_query.csv \
  --gallery ../../data/test_gallery.csv \
  --out-dir <каталог>
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `1`. Время: 0.163 с. [лог](evidence/C077.log).

FileNotFoundError: test CSV; placeholder <каталог> заменён явным новым выходным каталогом.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
python -m app.batch --images-dir ../../data/images --query ../../data/test_query.csv --gallery ../../data/test_gallery.csv --out-dir ../../outputs/artifacts-check
```

## C078

[04-solution/service/offline/README.md:19](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:19)

```bash
sha256sum -c SHA256SUMS
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 0.105 с. [лог](evidence/C078.log).

Команда завершилась успешно.

## C079

[04-solution/service/offline/README.md:20](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:20)

```bash
docker load -i python-3.13-slim.tar.gz     # или podman load -i …
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 0.381 с. [лог](evidence/C079.log).

Команда завершилась успешно.

## C080

[04-solution/service/offline/README.md:21](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:21)

```bash
docker load -i qdrant-v1.15.5.tar.gz
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 0.53 с. [лог](evidence/C080.log).

Команда завершилась успешно.

## C081

[04-solution/service/offline/README.md:22](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:22)

```bash
docker images | grep -E 'python|qdrant'    # сверить идентификаторы и размеры
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 0.042 с. [лог](evidence/C081.log).

ID обоих образов совпали; Docker показывает 126 MB и 178 MB, а README обещает ровно 130/181 MB. Podman для тех же ID считает 130/181 MB: размер зависит от engine.

## C082

[04-solution/service/offline/README.md:39](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:39)

```bash
podman build --pull=never --network none -t vehicle-reid-service ..
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 4.595 с. [лог](evidence/C082.log).

Команда завершилась успешно.

## C083

[04-solution/service/offline/README.md:51](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:51)

```bash
docker save python:3.13-slim   | gzip -9 > python-3.13-slim.tar.gz
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 15.704 с. [лог](evidence/C083.log).

Команда завершилась успешно.

## C084

[04-solution/service/offline/README.md:52](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:52)

```bash
docker save qdrant/qdrant:v1.15.5 | gzip -9 > qdrant-v1.15.5.tar.gz
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 20.589 с. [лог](evidence/C084.log).

Команда завершилась успешно.

## C085

[04-solution/service/offline/README.md:53](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:53)

```bash
sha256sum python-3.13-slim.tar.gz qdrant-v1.15.5.tar.gz > SHA256SUMS
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 0.14 с. [лог](evidence/C085.log).

Команда завершилась успешно.

## C086

[04-solution/service/wheels/README.md:20](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/wheels/README.md:20)

```bash
podman image exists docker.io/library/python:3.13-slim
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 0.078 с. [лог](evidence/C086.log).

Команда завершилась успешно.

## C087

[04-solution/service/wheels/README.md:21](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/wheels/README.md:21)

```bash
podman build --no-cache --pull=never --network none -t vehicle-reid-offline .
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 14.802 с. [лог](evidence/C087.log).

Команда завершилась успешно.

## C088

[04-solution/service/wheels/README.md:41](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/wheels/README.md:41)

```bash
cd 04-solution/service
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.033 с. [лог](evidence/C088.log).

Команда завершилась успешно.

## C089

[04-solution/service/wheels/README.md:42](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/wheels/README.md:42)

```bash
podman run --rm -v "$PWD:/src:z" -w /src python:3.13-slim \
  pip download --no-cache-dir -r requirements.txt -c requirements-lock.txt -d /src/wheels
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 8.209 с. [лог](evidence/C089-a2.log).

Все 30 wheels проверены/получены, exit 0. Первоначальная DNS-ошибка принадлежала стенду: после выделения сетевого Podman storage исходная команда прошла без изменений.

## C090

[04-solution/service/wheels/README.md:54](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/wheels/README.md:54)

```bash
mkdir -p wheels
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 0.042 с. [лог](evidence/C090.log).

Команда завершилась успешно.

## C091

[04-solution/service/wheels/README.md:55](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/wheels/README.md:55)

```bash
docker build --build-arg PIP_SOURCE=network -t vehicle-reid-service .
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 465.943 с. [лог](evidence/C091-a2.log).

Команда завершилась успешно.

## C092

[04-solution/split/README.md:73](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/README.md:73)

```bash
python3 make_split.py [--out DIR] [--seed N] [--p-refusal F]
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/split`. Exit: `1`. Время: 0.119 с. [лог](evidence/C092-a2.log).

После скрытия авторского repo — FileNotFoundError по hardcoded /home/artem/projects/hackathon-lct-vehicle-reid/data/train.csv. Первый случайный успех с чужими данными не засчитан.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
python3 make_split.py
```

## C093

[04-solution/split/README.md:74](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/README.md:74)

```bash
python3 check_split.py files/ [--full-scan]
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/split`. Exit: `1`. Время: 0.114 с. [лог](evidence/C093-a2.log).

FileNotFoundError по тому же абсолютному авторскому пути.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
python3 check_split.py files/
```

## C094

[04-solution/split/README.md:75](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/split/README.md:75)

```bash
python3 eval_degenerate.py     # вырожденные прогоны
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/split`. Exit: `1`. Время: 0.114 с. [лог](evidence/C094-a2.log).

ModuleNotFoundError: reid_metrics; EVAL_DIR содержит абсолютный авторский путь.

## C095

[04-solution/training/src/README.md:89](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:89)

```bash
export WORK="$PWD/outputs/combined-rebuild"  # новый каталог, не авторские runs
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.002 с. [лог](evidence/C095.log).

Команда завершилась успешно.

## C096

[04-solution/training/src/README.md:90](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:90)

```bash
export DATA_DIR="$PWD/data"                # полный набор организатора
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.002 с. [лог](evidence/C096.log).

Команда завершилась успешно.

## C097

[04-solution/training/src/README.md:91](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:91)

```bash
mkdir -p "$WORK/datasets/carla" "$WORK/datasets/roundabout"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.003 с. [лог](evidence/C097.log).

Команда завершилась успешно.

## C098

[04-solution/training/src/README.md:92](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:92)

```bash
curl -fL -o "$WORK/datasets/RoundaboutHD.zip" \
  https://huggingface.co/datasets/yl4300/RoundaboutHD/resolve/main/RoundaboutHD.zip
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1875.83 с. [лог](evidence/C098.log).

Полный архив 8 075 314 494 байта скачан; опубликованный SHA совпал (C100).

## C099

[04-solution/training/src/README.md:94](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:94)

```bash
curl -fL -o "$WORK/datasets/VeRi_CARLA_dataset.zip" \
  'https://www.dropbox.com/s/cg1etrs22y2xb62/VeRi_CARLA_dataset.zip?dl=1'
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 324.911 с. [лог](evidence/C099.log).

Полный архив 2 529 020 889 байт скачан; опубликованный SHA совпал (C100).

## C100

[04-solution/training/src/README.md:97](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:97)

```bash
sha256sum "$WORK/datasets/"*.zip
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 8.321 с. [лог](evidence/C100.log).

Оба напечатанных SHA и размера точно совпали с training/src/README.md:76–77.

## C101

[04-solution/training/src/README.md:98](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:98)

```bash
unzip -q "$WORK/datasets/VeRi_CARLA_dataset.zip" -d "$WORK/datasets/carla"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 11.503 с. [лог](evidence/C101.log).

Архив CARLA распакован в документированный каталог, exit 0.

## C102

[04-solution/training/src/README.md:99](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:99)

```bash
unzip -q "$WORK/datasets/RoundaboutHD.zip" 'RoundaboutHD/RoundaboutHD_Reid_subset/*' \
  -d "$WORK/datasets/roundabout"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 7.127 с. [лог](evidence/C102.log).

Указанная glob-маска RoundaboutHD_Reid_subset найдена и распакована, exit 0.

## C103

[04-solution/training/src/README.md:138](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:138)

```bash
python 04-solution/training/src/prepare_own.py \
  --images-dir "$DATA_DIR/images" --out-dir "$WORK/own"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 0.137 с. [лог](evidence/C103.log).

Понятный отказ: отсутствуют 7248 training images; README предупреждает об отсутствии train_fit JPG.

## C104

[04-solution/training/src/README.md:140](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:140)

```bash
python 04-solution/training/src/datasets/build_combined.py \
  --own-npz "$WORK/own/train_crops.npz" --own-raw "$WORK/own/crops_208.npy" \
  --add "carla:$WORK/datasets/carla/**/*.jpg:carla:200000:2000" \
  --add "roundabout:$WORK/datasets/roundabout/**/*.jpg:roundabout:100000:1000" \
  --out-npz "$WORK/combined_train.npz" --out-raw "$WORK/combined_crops_208.npy"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 0.069 с. [лог](evidence/C104.log).

FileNotFoundError: own/train_crops.npz после неподготовленного own-набора.

## C105

[04-solution/training/src/README.md:177](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:177)

```bash
   python 04-solution/training/src/initialize_from_onnx.py \
     --onnx 04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx \
     --out "$WORK/osnet_ain_init.npz"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.141 с. [лог](evidence/C105.log).

Инициализация из Git ONNX успешна; SHA initial NPZ совпал с документом.

## C106

[04-solution/training/src/README.md:187](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:187)

```powershell
   .\py312\python.exe .\build_from_onnx.py --npz .\osnet_ain_init.npz --save .\osnet_ain_start.pt
```

Cwd: `D:\lct-reid (isolated Windows logon session)`. Exit: `0`. Время: 4.426 с. [лог](evidence/windows/C106-native-retry.log).

Windows: 559 тензоров совпали, создан PT 8 958 188 байт с точным историческим SHA; повтор после исправления канонического пути стенда, исходный код не менялся.

## C107

[04-solution/training/src/README.md:200](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:200)

```powershell
   .\py312\python.exe -c "import numpy as np; np.savez('probe64.npz', x=np.zeros((2,3,208,208),dtype=np.float32))"
```

Cwd: `D:\lct-reid (isolated Windows logon session)`. Exit: `0`. Время: 0.681 с. [лог](evidence/windows/C107-native.log).

Windows: создан новый синтетический probe, shape (2,3,208,208), float32.

## C108

[04-solution/training/src/README.md:201](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:201)

```powershell
   .\py312\python.exe .\job48_chain.py
```

Cwd: `D:\lct-reid (isolated Windows logon session)`. Exit: `1`. Время: 4.409 с. [лог](evidence/windows/C108-native.log).

Windows: chain завершилась 1/RuntimeError stage1; [дочерний traceback](evidence/windows/chain-chain_s1.out.txt) — FileNotFoundError на combined_train.npz. Входы в README оговорены.

## C109

[04-solution/training/src/README.md:207](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:207)

```powershell
   .\py312\python.exe .\reid_train5.py --run combined_v1 --stage 1 --epochs 5 --freeze-fc --lr 3.5e-4
```

Cwd: `D:\lct-reid (isolated Windows logon session)`. Exit: `1`. Время: 4.313 с. [лог](evidence/windows/C109-native.log).

Windows: FileNotFoundError с traceback на data/combined_train.npz; понятного preflight нет. Структура и необходимые массивы в README оговорены.

## C110

[04-solution/training/src/README.md:208](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:208)

```powershell
   .\py312\python.exe .\reid_train5.py --run combined_v1 --stage 2 --epochs 15 --resume-from stage1.pt --lr 3.5e-5 --cls-lr 3.5e-4
```

Cwd: `D:\lct-reid (isolated Windows logon session)`. Exit: `1`. Время: 4.195 с. [лог](evidence/windows/C110-native.log).

Windows: FileNotFoundError с traceback на data/combined_train.npz; до загрузки checkpoint не дошло. Входы в README оговорены.

## C111

[04-solution/training/src/README.md:209](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:209)

```powershell
   .\py312\python.exe .\export_onnx2.py --weights .\runs\combined_v1\stage2.pt --out .\job48_combined_s2.onnx
```

Cwd: `D:\lct-reid (isolated Windows logon session)`. Exit: `1`. Время: 4.181 с. [лог](evidence/windows/C111-native.log).

Windows: FileNotFoundError с traceback на runs/combined_v1/stage2.pt после отсутствующих данных обучения. Входы в README оговорены.

## C112

[04-solution/training/src/README.md:241](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:241)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python 04-solution/training/src/rebuild_whitening.py \
  --crops "$WORK/own/crops_208.npy" --out-dir "$WORK/whitening"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 0.135 с. [лог](evidence/C112.log).

FileNotFoundError: own/crops_208.npy; понятного preflight нет.

## I001

[SOLUTION.md:110](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:110)

```bash
docker pull python:3.13-slim
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 25.78 с. [лог](evidence/I001-a1.log).

Команда завершилась успешно.

## I002

[SOLUTION.md:114](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:114)

```bash
pip download -r requirements.txt -c requirements-lock.txt
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `обёртка 124; pip 0`. Время: 60.428 с. [лог](evidence/I002-container-completion.log).

420-секундный timeout прервал Docker CLI обёртки; исходный pip продолжил работать. Здесь записан его реальный код, полученный docker wait, и полный контейнерный лог. Команда не подменялась.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm -v "$REPO/04-solution/service:/src" -w /src python:3.13-slim pip download -r requirements.txt -c requirements-lock.txt
```

## I003

[SOLUTION.md:114](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:114)

```bash
pip install --no-index --find-links=/wheels -r requirements.txt -c requirements-lock.txt
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 7.605 с. [лог](evidence/I003.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none -v "$REPO/04-solution/service:/src:ro" -v "$REPO/04-solution/service/wheels:/wheels:ro" -w /src python:3.13-slim pip install --no-index --find-links=/wheels -r requirements.txt -c requirements-lock.txt
```

## I004

[SOLUTION.md:146](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:146)

```bash
mkdir -p "$REPO/04-solution/service/wheels"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.056 с. [лог](evidence/I004.log).

Команда завершилась успешно.

## I005

[SOLUTION.md:146](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:146)

```bash
docker build --build-arg PIP_SOURCE=network -t vehicle-reid-service "$REPO/04-solution/service"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.769 с. [лог](evidence/I005-a2.log).

Команда завершилась успешно.

## I006

[SOLUTION.md:148](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:148)

```bash
pip freeze --all
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.616 с. [лог](evidence/I006.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip freeze --all
```

## I007

[SOLUTION.md:214](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:214)

```bash
docker compose ps -a loader
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.084 с. [лог](evidence/I007.log).

Команда завершилась успешно.

## I008

[SOLUTION.md:223](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:223)

```bash
docker compose -p reid-check up -d …
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 6.787 с. [лог](evidence/I008-a1.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker compose -p reid-check up -d --no-build --pull never
```

## I009

[SOLUTION.md:223](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:223)

```bash
docker compose -p reid-check down -v
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 11.181 с. [лог](evidence/I009-a1.log).

Команда завершилась успешно.

## I010

[SOLUTION.md:225](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:225)

```bash
DATA_DIR=/путь/к/data docker compose run --rm loader
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 3.036 с. [лог](evidence/I010.log).

/путь/к/data заменён на каталог чистого клона.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
DATA_DIR="$REPO/data" docker compose run --rm loader
```

## I011

[SOLUTION.md:227](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:227)

```bash
docker compose up -d
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.861 с. [лог](evidence/I011.log).

Команда завершилась успешно.

## I012

[SOLUTION.md:250](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:250)

```bash
podman save -o файл образ1 образ2
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.674 с. [лог](evidence/I012-a1.log).

Успешно подтверждена описанная ловушка Podman: в архиве 1 manifest entry с двумя tags.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
podman save -o "$OUT_DIR/podman-single.tar" vehicle-reid-service docker.io/qdrant/qdrant:v1.15.5
```

## I013

[SOLUTION.md:250](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:250)

```bash
podman save --multi-image-archive -o ...
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 3.57 с. [лог](evidence/I013-a1.log).

В архиве 2 manifest entries; оба образа сохранены, как обещано.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
podman save --multi-image-archive -o "$OUT_DIR/podman-multi.tar" vehicle-reid-service docker.io/qdrant/qdrant:v1.15.5
```

## I014

[SOLUTION.md:250](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:250)

```bash
docker load -i /путь/к/runtime-images.tar
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.583 с. [лог](evidence/I014.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker load -i "$OUT_DIR/runtime-images.tar"
```

## I015

[SOLUTION.md:272](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:272)

```bash
sha256sum -c SHA256SUMS
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/artifacts-final`. Exit: `0`. Время: 0.076 с. [лог](evidence/I015.log).

Команда завершилась успешно.

## I016

[SOLUTION.md:489](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:489)

```bash
sha256sum 04-solution/service/model/*.onnx
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.058 с. [лог](evidence/I016.log).

Команда завершилась успешно.

## I017

[SOLUTION.md:490](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:490)

```bash
sha256sum 04-solution/service/model/osnet_ain_combined_v1.onnx 04-solution/service/model/lw_ens_j48_rho0.5.npz 04-solution/training/combined/lw_ens_j48_rho0.5_f64.npz
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.056 с. [лог](evidence/I017.log).

Команда завершилась успешно.

## I018

[SOLUTION.md:494](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:494)

```bash
docker run --rm --network none -v "$REPO:/repo:ro" -v "$OUT_DIR:/out" vehicle-reid-service python -B /repo/04-solution/service/tools/calibrate_threshold.py /out/val/embeddings.npy /out/calibration
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 0.793 с. [лог](evidence/I018.log).

FileNotFoundError: /out/val/embeddings.npy; validation сначала требует данных организатора.

## I019

[SOLUTION.md:499](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:499)

```bash
sha256sum FILE
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.047 с. [лог](evidence/I019.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
sha256sum 04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx
```

## I020

[SOLUTION.md:499](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:499)

```bash
wc -c < FILE
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.038 с. [лог](evidence/I020.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
wc -c < 04-solution/service/model/osnet_ain_x1_0_vehicle_reid.onnx
```

## I021

[SOLUTION.md:585](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:585)

```bash
sha256sum 04-solution/service/app/static/vendor/*
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.066 с. [лог](evidence/I021.log).

Команда завершилась успешно.

## I022

[SOLUTION.md:595](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:595)

```bash
pip check
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.384 с. [лог](evidence/I022.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip check
```

## I023

[SOLUTION.md:597](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:597)

```bash
pip freeze --all
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.202 с. [лог](evidence/I023.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip freeze --all
```

## I024

[SOLUTION.md:633](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:633)

```bash
pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt -c requirements-lock.txt
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 6.277 с. [лог](evidence/I024.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none -v "$REPO/04-solution/service:/src:ro" -v "$REPO/04-solution/service/wheels:/wheels:ro" -w /src python:3.13-slim pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt -c requirements-lock.txt
```

## I025

[SOLUTION.md:633](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:633)

```bash
pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 368.72 с. [лог](evidence/I025-a1.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm -v "$REPO/04-solution/service:/src" -w /src python:3.13-slim pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt
```

## I026

[SOLUTION.md:633](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:633)

```bash
pip freeze --all
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.224 с. [лог](evidence/I026.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip freeze --all
```

## I027

[SOLUTION.md:633](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:633)

```bash
pip check
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.19 с. [лог](evidence/I027.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip check
```

## I028

[SOLUTION.md:633](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:633)

```bash
docker run --rm --network none vehicle-reid-service dpkg-query -W
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.403 с. [лог](evidence/I028.log).

Команда завершилась успешно.

## I029

[SOLUTION.md:668](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:668)

```bash
pip freeze --all
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 0.203 с. [лог](evidence/I029.log).

Новый research freeze: protobuf 7.36.2; остальные 8 библиотек совпали, pip 24.3.1 обусловлен системным venv.

## I030

[SOLUTION.md:668](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:668)

```bash
"$REPO/.venv/bin/python" 04-solution/plate-ablation/scripts/eval_variants.py
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 0.176 с. [лог](evidence/I030.log).

Понятный отказ: отсутствуют все 13 вариантов эмбеддингов, как прямо предупреждает SOLUTION.md.

## I031

[04-solution/audit/docker-path/README.md:15](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/audit/docker-path/README.md:15)

```bash
pip check
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.247 с. [лог](evidence/I031.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip check
```

## I032

[04-solution/audit/jury-path/README.md:12](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/audit/jury-path/README.md:12)

```bash
pip freeze
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.211 с. [лог](evidence/I032.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip freeze
```

## I033

[04-solution/audit/jury-path-2/README.md:12](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/audit/jury-path-2/README.md:12)

```bash
pip check
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.143 с. [лог](evidence/I033.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip check
```

## I034

[04-solution/reproduce/README.md:99](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:99)

```bash
python 04-solution/reproduce/run.py --mode val --data-dir "$DATA_DIR" --out-dir "$OUT_DIR"
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.081 с. [лог](evidence/I034.log).

Ожидаемый exit 2, понятная диагностика validation.

## I035

[04-solution/service/README.md:120](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:120)

```bash
podman-compose up -d
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `0`. Время: 0.758 с. [лог](evidence/I035-a3.log).

Podman Compose стартовал после настройки healthcheck unit на собственное isolated storage. Первоначальный timeout — ограничение стенда, не дефект Compose проекта. Без данных loader падает.

## I036

[04-solution/service/README.md:154](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:154)

```bash
python -m app.batch ...
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service`. Exit: `1`. Время: 0.17 с. [лог](evidence/I036.log).

FileNotFoundError при отсутствии test CSV, как в прямом batch.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
python -m app.batch --images-dir ../../data/images --query ../../data/test_query.csv --gallery ../../data/test_gallery.csv --out-dir ../../outputs/inline-batch
```

## I037

[04-solution/service/README.md:338](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/README.md:338)

```bash
pip freeze --all
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.205 с. [лог](evidence/I037.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip freeze --all
```

## I038

[04-solution/service/artifacts-final/README.md:40](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/artifacts-final/README.md:40)

```bash
sha256sum -c SHA256SUMS
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/artifacts-final`. Exit: `0`. Время: 0.058 с. [лог](evidence/I038.log).

Команда завершилась успешно.

## I039

[04-solution/service/offline/README.md:61](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:61)

```bash
podman tag 0ad2e23181e5 docker.io/qdrant/qdrant:v1.15.5
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 3.009 с. [лог](evidence/I039-a1.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
podman load -i "$REPO/04-solution/service/offline/qdrant-v1.15.5.tar.gz" && podman tag 0ad2e23181e5 docker.io/qdrant/qdrant:v1.15.5
```

## I040

[04-solution/service/wheels/README.md:14](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/wheels/README.md:14)

```bash
pip check
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.228 с. [лог](evidence/I040.log).

Команда завершилась успешно.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
docker run --rm --network none vehicle-reid-service python -m pip check
```

## I041

[04-solution/training/src/README.md:13](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/training/src/README.md:13)

```bash
sha256sum -c SHA256SUMS
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/training/src`. Exit: `0`. Время: 0.045 с. [лог](evidence/I041.log).

Все строки SHA256SUMS архивных training-исходников совпали.

## I042

[06-documentation/build_pdf.py:2](/home/artem/projects/hackathon-lct-vehicle-reid/06-documentation/build_pdf.py:2)

```bash
python3 06-documentation/build_pdf.py
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `0`. Время: 1.748 с. [лог](evidence/I042.log).

Генератор PDF отработал офлайн; source hash совпадает с текущим SOLUTION.md. Исходный PDF в Git старый; результат записан лишь в расходуемый клон.

## I043

[04-solution/service/offline/README.md:20](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/offline/README.md:20)

```bash
podman load -i python-3.13-slim.tar.gz
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/service/offline`. Exit: `0`. Время: 4.663 с. [лог](evidence/I043.log).

Команда завершилась успешно.

## I044

[SOLUTION.md:225](/home/artem/projects/hackathon-lct-vehicle-reid/SOLUTION.md:225)

```bash
docker compose run --rm loader
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `1`. Время: 7.105 с. [лог](evidence/I044.log).

FileNotFoundError: /data/test_gallery.csv.

## I045

[04-solution/reproduce/README.md:98](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/reproduce/README.md:98)

```bash
docker run --rm --network none --cpus 2 --memory 4g \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/repo:ro" -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" \
  vehicle-reid-service python -B /repo/04-solution/reproduce/run.py \
  --mode all --data-dir /data --out-dir /out
```

Cwd: `/home/artem/tmp/job_88-audit/clean`. Exit: `2`. Время: 0.68 с. [лог](evidence/I045-a2.log).

Ожидаемый exit 2: ранний отказ до инференса в Docker --mode all, согласно явно описанной замене режима.

## I046

[04-solution/cloud/bigres/README.md:45](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/cloud/bigres/README.md:45)

```bash
./run_bigres.sh
```

Cwd: `/home/artem/tmp/job_88-audit/clean/04-solution/cloud/bigres`. Exit: `1`. Время: 0.191 с. [лог](evidence/I046.log).

FileNotFoundError: cloud/bigres/payload/manifest.json; launch README не предупреждает о недостающем payload.

## F07

[04-solution/service/model/fetch_model.sh:7](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:7)

```sh
set -eu
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F08

[04-solution/service/model/fetch_model.sh:8](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:8)

```sh
cd "$(dirname "$0")"
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F11

[04-solution/service/model/fetch_model.sh:11](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:11)

```sh
FILE=osnet_ain_x1_0_vehicle_reid.onnx
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F12

[04-solution/service/model/fetch_model.sh:12](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:12)

```sh
URL=https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/$FILE
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F13

[04-solution/service/model/fetch_model.sh:13](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:13)

```sh
SHA=4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F14

[04-solution/service/model/fetch_model.sh:14](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:14)

```sh
[ -f "$FILE" ] || curl -sSL -o "$FILE" "$URL"
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F15

[04-solution/service/model/fetch_model.sh:15](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:15)

```sh
echo "$SHA  $FILE" | sha256sum -c -
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F18

[04-solution/service/model/fetch_model.sh:18](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:18)

```sh
FILE2=osnet_ain_combined_v1.onnx
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F19

[04-solution/service/model/fetch_model.sh:19](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:19)

```sh
SHA2=b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F20

[04-solution/service/model/fetch_model.sh:20](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:20)

```sh
if [ ! -f "$FILE2" ]; then
    echo "нет $FILE2 — дообученные веса поставляются в архиве решения; скачивать неоткуда" >&2
    exit 1
fi
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F24

[04-solution/service/model/fetch_model.sh:24](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:24)

```sh
echo "$SHA2  $FILE2" | sha256sum -c -
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F27

[04-solution/service/model/fetch_model.sh:27](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:27)

```sh
W=lw_ens_j48_rho0.5.npz
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F28

[04-solution/service/model/fetch_model.sh:28](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:28)

```sh
SHA_W=eb4433ffd5e38d3751d5cb04e234090060a83be6d1720b47bcf2fa1274b3c5b5
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F29

[04-solution/service/model/fetch_model.sh:29](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:29)

```sh
if [ ! -f "$W" ]; then
    echo "нет $W — возьмите из репозитория решения (training/combined/ хранит f64-оригинал)" >&2
    exit 1
fi
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```

## F33

[04-solution/service/model/fetch_model.sh:33](/home/artem/projects/hackathon-lct-vehicle-reid/04-solution/service/model/fetch_model.sh:33)

```sh
echo "$SHA_W  $W" | sha256sum -c -
```

Cwd: `/home/artem/tmp/job_88-audit/fetch-controls/complete`. Exit: `0`. Время: 0.064 с. [лог](evidence/fetch-complete.log).

В контексте fetch_model.sh; success и отсутствующие/испорченные файлы проверены отдельными контролями.

Выполненная форма (контекст/подстановки отмечены в логе):

```bash
bash -x fetch_model.sh
```
