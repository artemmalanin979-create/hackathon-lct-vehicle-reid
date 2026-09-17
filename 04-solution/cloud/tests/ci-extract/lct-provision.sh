#!/usr/bin/env bash
# Первичная настройка lct-gpu. Идемпотентно (шаги помечаются в /var/lib/lct), без интерактива.
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export UV_PYTHON_INSTALL_DIR=/opt/uv/python UV_CACHE_DIR=/opt/uv/cache UV_NO_PROGRESS=1
STAMP=/var/lib/lct
VENV=/opt/reid/venv
PY=3.12
VM_USER=artem
BLENDER_VER=4.5.14
BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-${BLENDER_VER}-linux-x64.tar.xz"
BLENDER_SHA256="9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3"   # из https://download.blender.org/release/Blender4.5/blender-4.5.14.sha256
TORCH_INDEX="https://download.pytorch.org/whl/cu126"   # CUDA 12.6 runtime в wheel; драйвер 535 (CUDA 12.2) совместим (minor version compatibility)
ORT_SPEC="onnxruntime-gpu==1.26.0"                      # последняя сборка PyPI под CUDA 12 (с 1.27 — CUDA 13, требует драйвер >=580)

mkdir -p "$STAMP"
log()  { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step() {  # step <имя> <команда...> — выполняется один раз, при успехе ставится метка
  local name=$1; shift
  if [[ -f "$STAMP/$name.done" ]]; then log "skip $name (уже сделано)"; return 0; fi
  log "=== $name"; "$@"; touch "$STAMP/$name.done"; log "=== $name ok"
}
apt_retry() { local i; for i in 1 2 3 4 5; do apt-get "$@" && return 0; log "apt-get $* не удалось, попытка $i"; sleep 15; done; return 1; }

s_hold_driver() {  # не трогаем драйвер NVIDIA/CUDA и ядро: их апгрейд без перезагрузки ломает nvidia-smi
  local pkgs=()
  mapfile -t pkgs < <(dpkg-query -W -f='${binary:Package}\n' 'nvidia*' 'libnvidia*' 'cuda*' 'linux-image*' 'linux-headers*' 'linux-modules*' 2>/dev/null | sort -u)
  if ((${#pkgs[@]})); then apt-mark hold "${pkgs[@]}"; fi
  log "на hold: ${#pkgs[@]} пакетов"
}
s_apt_upgrade() {
  apt_retry update
  apt_retry -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold upgrade
  apt_retry -y autoremove
}
s_apt_deps() {
  apt_retry install -y --no-install-recommends \
    time xz-utils curl ca-certificates git htop jq unzip rsync \
    python3 python3-venv python3-pip \
    libxi6 libxxf86vm1 libxfixes3 libxrender1 libxinerama1 libxcursor1 libxrandr2 \
    libgl1 libglu1-mesa libegl1 libxkbcommon0 libsm6 libice6 libdbus-1-3 libgomp1
}
s_python() {
  python3 -m pip install -q --upgrade uv
  uv python install "$PY"
  [[ -x "$VENV/bin/python" ]] || uv venv --python "$PY" --seed "$VENV"
  "$VENV/bin/python" -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
}
s_torch() {
  uv pip install --python "$VENV/bin/python" --index-url "$TORCH_INDEX" torch torchvision
}
s_reid_pkgs() {
  uv pip install --python "$VENV/bin/python" \
    "$ORT_SPEC" onnx timm transformers open_clip_torch numpy pillow safetensors huggingface_hub
  chown -R "$VM_USER:$VM_USER" /opt/reid
}
s_blender() {
  local dl=/opt/blender-dl f
  mkdir -p "$dl"; f="$dl/blender-${BLENDER_VER}-linux-x64.tar.xz"
  if [[ "$(cat /opt/blender/.lct-version 2>/dev/null || true)" == "$BLENDER_VER" ]]; then log "Blender $BLENDER_VER уже установлен"; return 0; fi
  if [[ -f "$f" ]] && echo "$BLENDER_SHA256  $f" | sha256sum -c - >/dev/null 2>&1; then
    log "архив Blender уже скачан и проверен"
  else
    rm -f "$f"
    curl -fL --retry 5 --retry-delay 10 -o "$f" "$BLENDER_URL"
  fi
  echo "$BLENDER_SHA256  $f" | sha256sum -c -
  rm -rf /opt/blender; mkdir -p /opt/blender
  tar -xJf "$f" -C /opt/blender --strip-components=1
  ln -sfn /opt/blender/blender /usr/local/bin/blender
  echo "$BLENDER_VER" > /opt/blender/.lct-version
  rm -f "$f"
}
s_dirs() {
  mkdir -p /data/reid /data/render /data/hf
  chown -R "$VM_USER:$VM_USER" /data
}
smoke() {  # самопроверка — выполняется при каждом запуске, результат в /var/log/lct-smoke.log
  {
    echo "### $(date '+%F %T') lct smoke"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "nvidia-smi FAIL"
    "$VENV/bin/python" /opt/lct/smoke.py || echo "SMOKE_REID FAIL"
    blender --version | head -1
    blender -b --factory-startup --python /opt/lct/blender-smoke.py 2>&1 | grep -E 'SMOKE_BLENDER|Error|error' || echo "SMOKE_BLENDER FAIL"
    df -h / | tail -1
  } > /var/log/lct-smoke.log 2>&1 || true
  cat /var/log/lct-smoke.log
}

log "lct-provision start (uptime $(cut -d' ' -f1 /proc/uptime) с)"
step hold_driver s_hold_driver
step apt_upgrade s_apt_upgrade
step apt_deps    s_apt_deps
step python      s_python
step torch       s_torch
step reid_pkgs   s_reid_pkgs
step blender     s_blender
step dirs        s_dirs
smoke
log "lct-provision done (uptime $(cut -d' ' -f1 /proc/uptime) с)"
