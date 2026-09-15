# Образ сервиса инференса Vehicle ReID.
# Веса модели попадают в образ на этапе сборки (COPY model/) — на запуске
# доступ в интернет не нужен; sha256 весов проверяется при сборке.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# requirements.txt — семь прямых зависимостей, requirements-lock.txt — полный
# снимок проверенной сборки (`pip freeze --all` из образа): как constraints он
# закрепляет и транзитивные версии. Хешей wheel в нём нет.
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt

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
