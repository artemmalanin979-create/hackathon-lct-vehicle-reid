# GPU-first inference image for the organizer's CUDA 12.2 / RTX A5000 host.
# Build may use PyPI; all packages, CUDA user-space libraries, and model weights
# are installed into the image before network-isolated inference starts.

# Separate dependency stage keeps downloaded wheels and pip cache out of runtime.
FROM python:3.13-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

COPY requirements-gpu.txt requirements-gpu-lock.txt ./
RUN pip install --prefix=/install --no-cache-dir \
        -r requirements-gpu.txt -c requirements-gpu-lock.txt

# Runtime image; the NVIDIA host driver enters through --gpus all, never pip.
FROM python:3.13-slim

ARG SOURCE_SHA=unspecified
LABEL org.opencontainers.image.revision="${SOURCE_SHA}"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LCT_DEVICE=cuda \
    LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.13/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.13/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.13/site-packages/nvidia/curand/lib:/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib

WORKDIR /srv

# /install maps to /usr/local: site-packages and bin/uvicorn.
COPY --from=deps /install /usr/local

# Check the GPU wheel and CUDA user-space linkage without initializing the CUDA
# provider. The provider needs ORT's plugin host symbols and a real NVIDIA driver
# at session creation, neither of which is present during an ordinary build.
RUN python - <<'EOF'
from pathlib import Path
import subprocess
import onnxruntime as ort

assert ort.__version__ == "1.20.2", ort.__version__
assert "CUDAExecutionProvider" in ort.get_available_providers(), ort.get_available_providers()
provider = Path(ort.__file__).parent / "capi/libonnxruntime_providers_cuda.so"
result = subprocess.run(["ldd", str(provider)], check=True, capture_output=True, text=True)
assert "not found" not in result.stdout, result.stdout
print("CUDA provider user-space linkage OK:", provider)
EOF

# Веса — в составе решения. Если файла нет (например, клон без бинарников),
# model/fetch_model.sh скачает и проверит его ДО сборки образа. combined_v1 и
# матрица whitening поставляются в git-репозитории решения.
COPY model/ model/
RUN python - <<'EOF'
import hashlib, pathlib, sys
expected = {
    "model/osnet_ain_x1_0_vehicle_reid.onnx":
        "4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f",
    "model/osnet_ain_combined_v1.onnx":
        "b1ba5021275b34079a1653608bdfd215fc9404306dc909852e4cdfa03402efb2",
    "model/lw_ens_j48_rho0.5.npz":
        "eb4433ffd5e38d3751d5cb04e234090060a83be6d1720b47bcf2fa1274b3c5b5",
}
for name, want in expected.items():
    p = pathlib.Path(name)
    if not p.is_file():
        sys.exit(f"нет файла весов {name} — для OSNet запустите model/fetch_model.sh "
                 "перед сборкой, остальные файлы поставляются в репозитории")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == want, f"sha256 {name} не совпал: {digest}"
print("все три файла весов в образе, sha256 OK")
EOF

COPY app/ app/

ENV MODEL_PATH=/srv/model/osnet_ain_x1_0_vehicle_reid.onnx \
    MODEL2_PATH=/srv/model/osnet_ain_combined_v1.onnx \
    WHITENING_PATH=/srv/model/lw_ens_j48_rho0.5.npz

EXPOSE 8000
# По умолчанию — HTTP-API; пакетный прогон запускается тем же образом:
#   docker run --rm --network none -v ...:/data:ro -v ...:/out <image> \
#     python -m app.batch --images-dir /data/images --query ... --gallery ... --out-dir /out
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
