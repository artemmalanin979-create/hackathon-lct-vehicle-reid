# Образ сервиса инференса Vehicle ReID.
# Веса модели попадают в образ на этапе сборки (COPY model/) — на запуске
# доступ в интернет не нужен; sha256 весов проверяется при сборке.

# ── Стадия 1: зависимости ───────────────────────────────────────────────────
# Отдельная стадия нужна ровно для того, чтобы каталог колёс НЕ попал в
# итоговый образ. COPY всегда создаёт свой слой, и `rm -rf /wheels` в следующем
# слое вес не возвращает: раньше образ и офлайн-архив были тяжелее на 59 251 200
# байт. Здесь колёса живут только в этой стадии, наружу уходит /install.
#
# requirements.txt — семь прямых зависимостей, requirements-lock.txt — полный
# снимок проверенной сборки (`pip freeze --all` из образа): как constraints он
# закрепляет и транзитивные версии. Хешей wheel в нём нет.
#
# По умолчанию сборка идёт ОФЛАЙН, из каталога wheels/ в составе решения: 30 пакетов,
# 57 МБ, собраны в этом же базовом образе командой из wheels/README.md. Так сборка не
# зависит от доступности PyPI — жюри может собирать образ в изолированной среде.
#
# Запасной путь — ставить из сети: --build-arg PIP_SOURCE=network. Сам каталог
# wheels/ нужен в любом случае: COPY ниже общий для обеих ветвей, без каталога
# сборка не начнётся.
FROM python:3.13-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

ARG PIP_SOURCE=offline
COPY requirements.txt requirements-lock.txt ./
COPY wheels/ /wheels/
RUN if [ "$PIP_SOURCE" = "offline" ]; then \
        pip install --prefix=/install --no-cache-dir --no-index --find-links=/wheels \
            -r requirements.txt -c requirements-lock.txt; \
    else \
        pip install --prefix=/install --no-cache-dir \
            -r requirements.txt -c requirements-lock.txt; \
    fi

# ── Стадия 2: сдаваемый образ ───────────────────────────────────────────────
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# /install собран стадией deps в схеме posix_prefix, поэтому ложится в
# /usr/local один в один: lib/python3.13/site-packages и bin/uvicorn.
COPY --from=deps /install /usr/local

# Веса — в составе решения. Если файла нет (например, клон без бинарников),
# model/fetch_model.sh скачает и проверит его ДО сборки образа.
COPY model/ model/
RUN python - <<'EOF'
import hashlib, pathlib, sys
expected = "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f"
p = pathlib.Path("model/osnet_ain_x1_0_vehicle_reid.onnx")
if not p.is_file():
    sys.exit("нет файла весов model/osnet_ain_x1_0_vehicle_reid.onnx — "
             "запустите model/fetch_model.sh перед сборкой")
digest = hashlib.sha256(p.read_bytes()).hexdigest()
assert digest == expected, f"sha256 весов не совпал: {digest}"
print("веса модели в образе, sha256 OK")
EOF

COPY app/ app/

ENV MODEL_PATH=/srv/model/osnet_ain_x1_0_vehicle_reid.onnx

EXPOSE 8000
# По умолчанию — HTTP-API; пакетный прогон запускается тем же образом:
#   docker run --rm --network none -v ...:/data:ro -v ...:/out <image> \
#     python -m app.batch --images-dir /data/images --query ... --gallery ... --out-dir /out
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
