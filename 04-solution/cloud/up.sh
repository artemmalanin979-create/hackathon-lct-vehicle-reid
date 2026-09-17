#!/usr/bin/env bash
# up.sh — поднять GPU-машину lct-gpu в Yandex Cloud. Идемпотентно: что уже есть — пропускает.
#
# Создаёт (только если нет): сеть lct-net, подсеть lct-subnet (ru-central1-d, 10.10.0.0/24),
# ВМ lct-gpu: standard-v3-t4, 4 vCPU / 16 ГБ / 1×T4, загрузочный network-ssd 60 ГБ из образа
# ubuntu-2204-lts-cuda-12-2, публичный динамический IPv4, cloud-init из cloud-init.yaml
# (ssh-ключ подставляется из ~/.ssh/id_ed25519.pub). В конце печатает IP и пишет его в lct-gpu.ip.
#
# Использование:
#   ./up.sh                           обычная ВМ
#   PREEMPTIBLE=1 ./up.sh             прерываемая ВМ (дешевле; живёт ≤24 ч, могут выключить)
#   PLATFORM=standard-v3-t4i ./up.sh  T4i (24 ГБ VRAM)
#   ./up.sh --dry-run                 показать команды создания, ничего не создавать (get/list выполняются)
#
# Стоимость (RUB с НДС, prices.json от 2026-09-17): T4 4/16 + диск 60 ГБ + IP ≈ 81,25 ₽/ч; остановленная ≈ 1,19 ₽/ч (диск).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=lct.env
source ./lct.env
# shellcheck source=lct-lib.sh
source ./lct-lib.sh

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) die "неизвестный аргумент: $arg (см. --help)" ;;
  esac
done

run() {  # run <cmd...> — выполнить изменяющую команду (или напечатать её в --dry-run)
  if (( DRY_RUN )); then printf '[dry-run] +'; printf ' %q' "$@"; printf '\n'; return 0; fi
  log "+ $*"
  "$@"
}

# ---------- 0. предпроверки (только чтение) ----------
require_tools
[[ -f "$SSH_PUBKEY_FILE" ]] || die "нет ssh-ключа $SSH_PUBKEY_FILE"
SSH_PUBKEY="$(tr -d '\r\n' < "$SSH_PUBKEY_FILE")"
[[ "$SSH_PUBKEY" == ssh-* ]] || die "$SSH_PUBKEY_FILE не похож на открытый ключ OpenSSH"
[[ -f "$CLOUD_INIT_TEMPLATE" ]] || die "нет шаблона $CLOUD_INIT_TEMPLATE"
grep -q '__SSH_PUBKEY__' "$CLOUD_INIT_TEMPLATE" || die "в $CLOUD_INIT_TEMPLATE нет плейсхолдера __SSH_PUBKEY__"
[[ -f "$PRICES_JSON" ]] || die "нет $PRICES_JSON"

img="$("$YC" compute image get --id "$IMAGE_ID" --format json 2>&1)" || die "образ $IMAGE_ID недоступен: $img"
log "образ: $(jq -r '"\(.name) (family \(.family), создан \(.created_at))"' <<<"$img")"
latest="$("$YC" compute image get-latest-from-family "$IMAGE_FAMILY" --folder-id standard-images --format json 2>/dev/null | jq -r .id || true)"
if [[ -n "$latest" && "$latest" != "$IMAGE_ID" ]]; then
  log "ВНИМАНИЕ: в семействе $IMAGE_FAMILY уже есть более новый образ $latest (используем закреплённый $IMAGE_ID)"
fi

# ---------- 1. сеть и подсеть (создать только если нет) ----------
if resource_exists vpc network get --name "$NET_NAME"; then
  log "сеть $NET_NAME уже есть"
else
  run "$YC" vpc network create --name "$NET_NAME" --description "LCT GPU (up.sh)" --folder-id "$FOLDER_ID"
fi

if resource_exists vpc subnet get --name "$SUBNET_NAME"; then
  log "подсеть $SUBNET_NAME уже есть"
else
  run "$YC" vpc subnet create --name "$SUBNET_NAME" --network-name "$NET_NAME" --zone "$ZONE" \
      --range "$SUBNET_RANGE" --description "LCT GPU (up.sh)" --folder-id "$FOLDER_ID"
fi

# ---------- 2. cloud-init: подставить ssh-ключ ----------
umask 077
: > "$CLOUD_INIT_RENDERED"
while IFS= read -r line || [[ -n "$line" ]]; do
  printf '%s\n' "${line//__SSH_PUBKEY__/$SSH_PUBKEY}"
done < "$CLOUD_INIT_TEMPLATE" > "$CLOUD_INIT_RENDERED"
umask 022
grep -q '__SSH_PUBKEY__' "$CLOUD_INIT_RENDERED" && die "плейсхолдер не подставился"
log "cloud-init отрендерен: $CLOUD_INIT_RENDERED ($(wc -c < "$CLOUD_INIT_RENDERED") байт)"

# ---------- 3. ВМ ----------
preempt_args=()
(( PREEMPTIBLE )) && preempt_args=(--preemptible)

if resource_exists compute instance get --name "$VM_NAME"; then
  status="$(yc_json compute instance get --name "$VM_NAME" | jq -r .status)"
  log "ВМ $VM_NAME уже есть, статус $status"
  if [[ "$status" == "STOPPED" ]]; then
    run "$YC" compute instance start --name "$VM_NAME" --folder-id "$FOLDER_ID"
  fi
else
  run "$YC" compute instance create \
    --name "$VM_NAME" --hostname "$VM_NAME" --zone "$ZONE" --folder-id "$FOLDER_ID" \
    --platform "$PLATFORM" --cores "$CORES" --memory "$MEMORY_GB" --gpus "$GPUS" "${preempt_args[@]}" \
    --create-boot-disk "name=$BOOT_DISK_NAME,type=$DISK_TYPE,size=${DISK_SIZE_GB},image-id=$IMAGE_ID,auto-delete=true" \
    --network-interface "subnet-name=$SUBNET_NAME,nat-ip-version=ipv4" \
    --metadata-from-file "user-data=$CLOUD_INIT_RENDERED" \
    --metadata serial-port-enable=1 \
    --labels "project=lct,owner=artem,created-by=up-sh" \
    --description "LCT GPU: Re-ID + Blender (up.sh)"
fi

if (( DRY_RUN )); then
  log "dry-run: ничего не создано. Оценка стоимости работы: $(vm_hourly_cost "$PLATFORM" "$CORES" "$MEMORY_GB" "$GPUS" "$PREEMPTIBLE" 1 "$DISK_TYPE" "$DISK_SIZE_GB") ₽/ч"
  exit 0
fi

# ---------- 4. IP, файл, подсказки ----------
inst="$(yc_json compute instance get --name "$VM_NAME")"
IP="$(jq -r '.network_interfaces[0].primary_v4_address.one_to_one_nat.address // empty' <<<"$inst")"
[[ -n "$IP" ]] || die "у ВМ нет публичного IP (см. yc compute instance get --name $VM_NAME)"
printf '%s\n' "$IP" > "$IP_FILE"
log "ВМ $VM_NAME ($(jq -r .status <<<"$inst")), публичный IP: $IP  -> записан в $IP_FILE"

cost="$(vm_hourly_cost "$PLATFORM" "$CORES" "$MEMORY_GB" "$GPUS" "$PREEMPTIBLE" 1 "$DISK_TYPE" "$DISK_SIZE_GB")"
log "стоимость работы: ≈ $cost ₽/ч (RUB с НДС); остановленная: только диск ≈ $(jq -r --arg t "$DISK_TYPE" --argjson g "$DISK_SIZE_GB" '(.disk[$t]*$g*10000|round)/10000' "$PRICES_JSON") ₽/ч"

log "жду ssh на $IP:22 (до 5 мин)…"
for _ in $(seq 1 60); do
  if timeout 3 bash -c "exec 3<>/dev/tcp/$IP/22" 2>/dev/null; then log "ssh порт открыт"; break; fi
  sleep 5
done

echo "$IP"
cat <<MSG
Дальше:
  ssh $VM_USER@$IP                                     # ключ $SSH_PUBKEY_FILE
  ssh $VM_USER@$IP sudo tail -f /var/log/lct-provision.log   # cloud-init ставит стек ещё ~10–15 мин
  ssh $VM_USER@$IP cat /var/log/lct-smoke.log         # SMOKE_REID OK / SMOKE_BLENDER OK — можно работать
  ./status.sh                                          # что существует и сколько стоит в час
  ./down.sh                                            # снести всё
MSG
