#!/usr/bin/env bash
# status.sh — что сейчас существует в каталоге и сколько это стоит в час (по prices.json, RUB с НДС).
# Только чтение (yc ... list). Запускать часто — это контроль расходов.
#
# Считает: ВМ (GPU+vCPU+RAM по платформе, только RUNNING; прерываемые — по preemptible-тарифу),
# публичные IP (у RUNNING ВМ и зарезервированные), диски (по типу), снимки, образы, файловые хранилища.
# Сети/подсети бесплатны — показываются справочно. Неизвестная платформа/тип диска -> "?" (не считается).
#
# Использование: ./status.sh [--json]
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=lct.env
source ./lct.env
# shellcheck source=lct-lib.sh
source ./lct-lib.sh

JSON_OUT=0
[[ "${1:-}" == "--json" ]] && JSON_OUT=1

require_tools
[[ -f "$PRICES_JSON" ]] || die "нет $PRICES_JSON"

instances="$(yc_json compute instance list)"
disks="$(yc_json compute disk list)"
addresses="$(yc_json vpc address list)"
snapshots="$(yc_json compute snapshot list)"
images="$(yc_json compute image list)"
filesystems="$(yc_json compute filesystem list)"
networks="$(yc_json vpc network list)"
subnets="$(yc_json vpc subnet list)"

report="$(jq -n \
  --argjson prices "$(cat "$PRICES_JSON")" \
  --argjson instances "$instances" --argjson disks "$disks" --argjson addresses "$addresses" \
  --argjson snapshots "$snapshots" --argjson images "$images" --argjson filesystems "$filesystems" \
  --argjson networks "$networks" --argjson subnets "$subnets" '
  def gb: (tonumber / 1073741824);
  def r2: (. * 100 | round) / 100;

  ($instances | map(
      . as $i
      | ($prices.platform[$i.platform_id]) as $pl
      | ($i.scheduling_policy.preemptible // false) as $pre
      | ($i.resources.cores | tonumber) as $cores
      | ($i.resources.memory | gb) as $mem
      | ($i.resources.gpus // "0" | tonumber) as $gpus
      | ($i.resources.core_fraction // "100" | tonumber) as $frac
      | ([$i.network_interfaces[]? | .primary_v4_address.one_to_one_nat.address // empty] | first // "") as $ip
      | (if $i.status != "RUNNING" then 0
         elif ($pl == null or $frac != 100) then null
         else ($pl[(if $pre then "gpu_preemptible" else "gpu" end)] * $gpus
             + $pl[(if $pre then "cpu_preemptible" else "cpu" end)] * $cores
             + $pl[(if $pre then "ram_preemptible" else "ram" end)] * $mem
             + (if $ip != "" then $prices.public_ip.active else 0 end)) end) as $cost
      | {kind: "vm", name: $i.name, id: $i.id, status: $i.status, zone: $i.zone_id, platform: $i.platform_id,
         preemptible: $pre, spec: "\($gpus) GPU / \($cores) vCPU@\($frac)% / \($mem|floor) ГБ", ip: $ip,
         cost_h: $cost}
  )) as $vms |

  ($disks | map(
      . as $d | ($prices.disk[$d.type_id]) as $p | ($d.size | gb) as $g
      | {kind: "disk", name: ($d.name // $d.id), id: $d.id, status: $d.status, zone: $d.zone_id,
         spec: "\($d.type_id) \($g|floor) ГБ", attached: (($d.instance_ids // []) | join(",")),
         cost_h: (if $p == null then null else $p * $g end)}
  )) as $dsk |

  ($addresses | map(
      . as $a
      | {kind: "address", name: ($a.name // $a.id), id: $a.id, spec: ($a.external_ipv4_address.address // "?"),
         status: (if $a.used then "used" else "inactive" end),
         cost_h: (if $a.used then $prices.public_ip.active
                  else ($prices.public_ip.active + $prices.public_ip.inactive_reserved_extra) end)}
  )) as $adr |

  ($snapshots | map({kind: "snapshot", name: (.name // .id), id: .id, status: .status,
                     spec: "\((.storage_size|gb)|floor) ГБ хранение", cost_h: ((.storage_size|gb) * $prices.snapshot_gb)})) as $snp |
  ($images    | map({kind: "image", name: (.name // .id), id: .id, status: .status,
                     spec: "\((.storage_size|gb)|floor) ГБ хранение", cost_h: ((.storage_size|gb) * $prices.image_gb)})) as $img |
  ($filesystems | map({kind: "filesystem", name: (.name // .id), id: .id, status: .status,
                     spec: "\(.type_id) \((.size|gb)|floor) ГБ",
                     cost_h: (if .type_id == "network-ssd" then (.size|gb) * $prices.filesystem_ssd_gb else null end)})) as $fs |

  ($vms + $dsk + $adr + $snp + $img + $fs) as $items
  | ($items | map(.cost_h // 0) | add // 0) as $total
  | ($items | map(select(.cost_h == null)) | length) as $unknown
  | {items: $items,
     networks: ($networks | map(.name)), subnets: ($subnets | map("\(.name) [\(.zone_id)]")),
     total_h: ($total | r2), total_day: (($total * 24) | r2), total_month30: (($total * 720) | r2),
     unknown_priced: $unknown}
')"

if (( JSON_OUT )); then printf '%s\n' "$report"; exit 0; fi

echo "Каталог $FOLDER_ID, $(date '+%F %T'), цены prices.json (RUB с НДС, за час)"
echo "----------------------------------------------------------------------------------------------"
if [[ "$(jq '.items | length' <<<"$report")" == "0" ]]; then
  echo "  (платных ресурсов нет: ни ВМ, ни дисков, ни адресов, ни снимков, ни образов, ни файловых хранилищ)"
else
  jq -r '.items[] | [.kind, .name, (.status // ""), (.spec // ""), (.ip // .attached // ""),
                     (if .cost_h == null then "?" else ((.cost_h*10000|round)/10000|tostring) end)] | @tsv' <<<"$report" \
    | awk -F'\t' '{printf "  %-10s %-22s %-10s %-34s %-16s %8s ₽/ч\n", $1, $2, $3, $4, $5, $6}'
fi
echo "----------------------------------------------------------------------------------------------"
jq -r '"  ИТОГО: \(.total_h) ₽/ч  =  \(.total_day) ₽/сутки  =  \(.total_month30) ₽/30 дней" +
       (if .unknown_priced > 0 then "   (⚠ \(.unknown_priced) объектов с неизвестным тарифом не посчитаны)" else "" end)' <<<"$report"
jq -r '"  сети (бесплатно): \(.networks | join(", "))\n  подсети (бесплатно): \(.subnets | join(", "))"' <<<"$report"
