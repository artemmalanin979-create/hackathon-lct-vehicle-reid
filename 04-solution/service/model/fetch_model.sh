#!/bin/sh
# Проверка весов с sha256 — нужна перед сборкой образа. OSNet скачивается,
# только если бинарник не приехал с репозиторием; combined_v1 и матрица
# whitening — наши артефакты, поставляются в репозитории решения
# (combined_v1 в git по *.onnx-исключению — взять из архива поставки).
# Сборка и запуск сервиса сеть не используют.
set -eu
cd "$(dirname "$0")"

# 1. OSNet-AIN (OMZ 2022.1) — единственный внешний файл, к нему есть URL.
FILE=osnet_ain_x1_0_vehicle_reid.onnx
URL=https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/$FILE
SHA=4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f
[ -f "$FILE" ] || curl -sSL -o "$FILE" "$URL"
echo "$SHA  $FILE" | sha256sum -c -

# 2. combined_v1 — дообученный нами OSNet-AIN; URL нет, файл обязан лежать рядом.
FILE2=osnet_ain_combined_v1.onnx
SHA2=b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2
if [ ! -f "$FILE2" ]; then
    echo "нет $FILE2 — дообученные веса поставляются в архиве решения; скачивать неоткуда" >&2
    exit 1
fi
echo "$SHA2  $FILE2" | sha256sum -c -

# 3. Матрица whitening (f32-поставка; f64-оригинал — ../../training/combined/).
W=lw_ens_j48_rho0.5.npz
SHA_W=eb4433ffd5e38d3751d5cb04e234090060a83be6d1720b47bcf2fa1274b3c5b5
if [ ! -f "$W" ]; then
    echo "нет $W — возьмите из репозитория решения (training/combined/ хранит f64-оригинал)" >&2
    exit 1
fi
echo "$SHA_W  $W" | sha256sum -c -
