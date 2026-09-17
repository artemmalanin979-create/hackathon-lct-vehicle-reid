# lct-lib.sh — общие функции для up.sh / down.sh / status.sh (source-ится после lct.env).
# shellcheck shell=bash

log()  { printf '%s [lct] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die()  { printf '%s [lct] ОШИБКА: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 1; }

require_tools() {
  [[ -x "$YC" ]] || die "yc не найден: $YC"
  command -v jq >/dev/null || die "нужен jq"
  local ver
  ver="$("$YC" version 2>/dev/null | sed -n 's/^Yandex Cloud CLI \([0-9.]*\).*/\1/p')"
  [[ -n "$ver" ]] || die "yc version не отвечает"
  log "yc $ver, профиль: $("$YC" config profile list 2>/dev/null | awk '/ACTIVE/{print $1}')"
  local fid
  fid="$("$YC" config get folder-id 2>/dev/null || true)"
  [[ "$fid" == "$FOLDER_ID" ]] || die "в профиле folder-id='$fid', ожидался $FOLDER_ID — не тот профиль/каталог"
}

# yc_json <yc args...> — команда с --format json, вывод в stdout; при ошибке — die.
yc_json() {
  local out
  out="$("$YC" "$@" --folder-id "$FOLDER_ID" --format json 2>&1)" \
    || die "yc $* : $out"
  printf '%s' "$out"
}

# resource_exists <yc get-команда...> — 0: есть, 1: нет («not found»), любая другая ошибка — die.
# Пример: resource_exists compute instance get --name lct-gpu
resource_exists() {
  local out rc
  out="$("$YC" "$@" --folder-id "$FOLDER_ID" --format json 2>&1)"; rc=$?
  if (( rc == 0 )); then return 0; fi
  if grep -q -i -E 'not found|NotFound|не найден' <<<"$out"; then return 1; fi
  die "yc $* : $out"
}

# vm_hourly_cost <platform> <cores> <memory_gb> <gpus> <preemptible 0|1> <has_public_ip 0|1> <disk_type> <disk_gb>
# печатает стоимость ₽/час (RUB с НДС) по prices.json; неизвестная платформа/диск -> "?"
vm_hourly_cost() {
  jq -r --arg p "$1" --argjson c "$2" --argjson m "$3" --argjson g "$4" --argjson pre "$5" \
        --argjson ip "$6" --arg dt "$7" --argjson dg "$8" '
    def r4: (. * 10000 | round) / 10000;
    (.platform[$p]) as $pl | (.disk[$dt]) as $dp |
    if ($pl == null or $dp == null) then "?" else
      ( ($pl[(if $pre==1 then "gpu_preemptible" else "gpu" end)] * $g)
      + ($pl[(if $pre==1 then "cpu_preemptible" else "cpu" end)] * $c)
      + ($pl[(if $pre==1 then "ram_preemptible" else "ram" end)] * $m)
      + ($dp * $dg)
      + (if $ip==1 then .public_ip.active else 0 end) ) | r4 | tostring
    end' "$PRICES_JSON"
}
