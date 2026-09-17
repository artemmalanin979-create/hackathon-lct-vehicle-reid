#!/usr/bin/env bash
# Локальная проверка lct-provision.sh в контейнере ubuntu:22.04 (без GPU, без torch — только apt/uv/python3.12/Blender/ORT).
# Хранилище podman — внутри папки задания (tmpfs), после теста удаляется.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD/podman-root"; RUN="$PWD/podman-run"
P=(podman --root "$ROOT" --runroot "$RUN")
mkdir -p tests/ci-out
# shellcheck disable=SC2016  # скрипт внутри контейнера намеренно в одинарных кавычках
"${P[@]}" run --rm --name lct-ci -v "$PWD/tests/ci-extract:/mnt/ci:ro,Z" -v "$PWD/tests/ci-out:/out:Z" docker.io/library/ubuntu:22.04 bash -c '
set -Eeuo pipefail
cp /mnt/ci/lct-provision.sh /usr/local/sbin/lct-provision.sh; chmod 755 /usr/local/sbin/lct-provision.sh
mkdir -p /opt/lct; cp /mnt/ci/smoke.py /mnt/ci/blender-smoke.py /opt/lct/; cp /mnt/ci/lct.sh /etc/profile.d/lct.sh
useradd -m -s /bin/bash artem
# source всех функций скрипта (без финального блока выполнения)
source <(sed "/^log \"lct-provision start/,\$d" /usr/local/sbin/lct-provision.sh)
step hold_driver s_hold_driver
step apt_upgrade s_apt_upgrade
step apt_deps    s_apt_deps
step python      s_python
echo "=== dry-run резолва torch (индекс cu126)"; uv pip install --dry-run --python "$VENV/bin/python" --index-url "$TORCH_INDEX" torch torchvision 2>&1 | tail -25
echo "=== установка ORT/onnx/numpy/pillow (без torch)"; uv pip install --python "$VENV/bin/python" "$ORT_SPEC" onnx numpy pillow
step blender     s_blender
step dirs        s_dirs
echo "=== /usr/bin/time:"; /usr/bin/time -f "time ok %e s" true
echo "=== blender:"; blender -b --version | head -1
echo "=== blender-smoke (ожидаем FAIL [] — GPU нет):"; blender -b --factory-startup --python /opt/lct/blender-smoke.py 2>&1 | grep -E "SMOKE_BLENDER|rror" || true
echo "=== python/ORT:"; "$VENV/bin/python" - <<PY
import sys, numpy, PIL, onnx, onnxruntime as ort
print("python", sys.version.split()[0], "numpy", numpy.__version__, "Pillow", PIL.__version__, "onnx", onnx.__version__, "ort", ort.__version__, ort.get_available_providers())
PY
echo "=== повторный запуск шагов (идемпотентность):"; step apt_deps s_apt_deps; step blender s_blender
ls -la /usr/local/bin/blender /opt/blender/.lct-version; cat /opt/blender/.lct-version
df -h / | tail -1
echo CONTAINER_TEST_OK
' 2>&1 | tee tests/ci-out/container-test.log
echo "== очистка хранилища podman =="
"${P[@]}" rmi -f docker.io/library/ubuntu:22.04 >/dev/null 2>&1 || true
"${P[@]}" system reset --force >/dev/null 2>&1 || true
rm -rf "$ROOT" "$RUN"
