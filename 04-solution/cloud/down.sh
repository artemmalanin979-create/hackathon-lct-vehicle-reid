#!/usr/bin/env bash
# down.sh — снести ВСЁ, что создал up.sh: ВМ lct-gpu вместе с загрузочным диском, публичный IP,
# подсеть lct-subnet, сеть lct-net. Идемпотентно: чего нет — пропускает, не падает.
# Чужие ресурсы (сеть default, её подсети, любые объекты без имён lct-*) НЕ трогает — только предупреждает.
#
# Использование:
#   ./down.sh             снести
#   ./down.sh --dry-run   показать, что было бы удалено
#   ./down.sh --yes       без вопроса подтверждения
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=lct.env
source ./lct.env
# shellcheck source=lct-lib.sh
source ./lct-lib.sh

DRY_RUN=0; YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die "неизвестный аргумент: $arg" ;;
  esac
done

run() {
  if (( DRY_RUN )); then printf '[dry-run] +'; printf ' %q' "$@"; printf '\n'; return 0; fi
  log "+ $*"
  "$@"
}

require_tools

# ---------- что есть сейчас ----------
instances="$(yc_json compute instance list)"
disks="$(yc_json compute disk list)"
addresses="$(yc_json vpc address list)"
subnets="$(yc_json vpc subnet list)"
networks="$(yc_json vpc network list)"

echo "Сейчас в каталоге $FOLDER_ID:"
echo "  ВМ:      $(jq -r 'if length == 0 then "-" else map("\(.name) [\(.status)]") | join(", ") end' <<<"$instances")"
echo "  диски:   $(jq -r 'if length == 0 then "-" else map("\(.name // .id) (\(.size|tonumber/1073741824|floor) ГБ)") | join(", ") end' <<<"$disks")"
echo "  адреса:  $(jq -r 'if length == 0 then "-" else map("\(.name // .id) \(.external_ipv4_address.address // "")") | join(", ") end' <<<"$addresses")"
echo "  подсети: $(jq -r 'if length == 0 then "-" else map(.name) | join(", ") end' <<<"$subnets")"
echo "  сети:    $(jq -r 'if length == 0 then "-" else map(.name) | join(", ") end' <<<"$networks")"

if (( ! DRY_RUN && ! YES )); then
  read -r -p "Удалить $VM_NAME (+диск $BOOT_DISK_NAME), адреса lct-*, $SUBNET_NAME, $NET_NAME? [y/N] " ans
  [[ "$ans" == [yY] ]] || { log "отменено"; exit 0; }
fi

# ---------- 1. ВМ (диск auto-delete=true удаляется вместе с ней) ----------
if jq -e --arg n "$VM_NAME" 'any(.[]; .name == $n)' <<<"$instances" >/dev/null; then
  run "$YC" compute instance delete --name "$VM_NAME" --folder-id "$FOLDER_ID"
else
  log "ВМ $VM_NAME нет — пропускаю"
fi

# ---------- 2. загрузочный диск, если остался (auto-delete не сработал / диск создавали отдельно) ----------
if (( ! DRY_RUN )); then disks="$(yc_json compute disk list)"; fi
mapfile -t leftover_disks < <(jq -r --arg n "$BOOT_DISK_NAME" '.[] | select(.name == $n) | .id' <<<"$disks")
if ((${#leftover_disks[@]})); then
  for id in "${leftover_disks[@]}"; do run "$YC" compute disk delete --id "$id" --folder-id "$FOLDER_ID"; done
else
  log "диска $BOOT_DISK_NAME нет — пропускаю"
fi
other_disks="$(jq -r --arg n "$BOOT_DISK_NAME" '[.[] | select(.name != $n)] | length' <<<"$disks")"
if (( other_disks > 0 )); then
  log "ВНИМАНИЕ: в каталоге есть ещё $other_disks дисков не от up.sh — они платные (network-ssd ≈1,19 ₽/ч за 60 ГБ), см. yc compute disk list"
fi

# ---------- 3. публичные адреса: динамический освобождается вместе с ВМ; статические lct-* удаляем ----------
mapfile -t lct_addrs < <(jq -r '.[] | select((.name // "") | startswith("lct")) | .id' <<<"$addresses")
if ((${#lct_addrs[@]})); then
  for id in "${lct_addrs[@]}"; do run "$YC" vpc address delete --id "$id" --folder-id "$FOLDER_ID"; done
else
  log "зарезервированных адресов lct-* нет — пропускаю"
fi
other_addrs="$(jq -r '[.[] | select((.name // "") | startswith("lct") | not)] | length' <<<"$addresses")"
if (( other_addrs > 0 )); then
  log "ВНИМАНИЕ: есть $other_addrs зарезервированных адресов не lct-* — они платные (≈0,60 ₽/ч каждый), удалите вручную: yc vpc address list"
fi

# ---------- 4. подсеть и сеть ----------
if jq -e --arg n "$SUBNET_NAME" 'any(.[]; .name == $n)' <<<"$subnets" >/dev/null; then
  run "$YC" vpc subnet delete --name "$SUBNET_NAME" --folder-id "$FOLDER_ID"
else
  log "подсети $SUBNET_NAME нет — пропускаю"
fi
if jq -e --arg n "$NET_NAME" 'any(.[]; .name == $n)' <<<"$networks" >/dev/null; then
  run "$YC" vpc network delete --name "$NET_NAME" --folder-id "$FOLDER_ID"
else
  log "сети $NET_NAME нет — пропускаю"
fi

# ---------- 5. локальные артефакты ----------
run rm -f "$IP_FILE" "$CLOUD_INIT_RENDERED"

# ---------- 6. контроль ----------
if (( DRY_RUN )); then log "dry-run: ничего не удалено"; exit 0; fi
echo; echo "Осталось в каталоге:"
"$YC" compute instance list --folder-id "$FOLDER_ID"
"$YC" compute disk list --folder-id "$FOLDER_ID"
"$YC" vpc address list --folder-id "$FOLDER_ID"
"$YC" vpc network list --folder-id "$FOLDER_ID"
left="$(yc_json compute instance list | jq length)"; leftd="$(yc_json compute disk list | jq length)"
if (( left == 0 && leftd == 0 )); then log "ВМ и дисков нет — платных ресурсов от up.sh не осталось"; else log "ВНИМАНИЕ: остались ВМ: $left, дисков: $leftd"; fi
