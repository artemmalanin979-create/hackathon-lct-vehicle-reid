#!/bin/sh
# Скачивание весов модели с проверкой sha256 — нужно только если бинарник
# не приехал с репозиторием. Сборка и запуск сервиса сеть не используют.
set -eu
cd "$(dirname "$0")"
FILE=osnet_ain_x1_0_vehicle_reid.onnx
URL=https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/$FILE
SHA=4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f

[ -f "$FILE" ] || curl -sSL -o "$FILE" "$URL"
echo "$SHA  $FILE" | sha256sum -c -
