#!/bin/bash
set -u
cd ~/lct-reid/datasets
. ~/lct-reid/jobs/job_44/.venv/bin/activate
LOG=~/lct-reid/jobs/job_44/download.log
mark(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
hashit(){ [ -f "$1" ] && { sz=$(stat -c%s "$1"); sh=$(sha256sum "$1" | cut -d" " -f1); mark "HASH $1 bytes=$sz sha256=$sh"; }; }
df -h ~ | tail -1 | tee -a "$LOG"

# 1) RoundaboutHD full zip (HF, confirmed 200, Range OK) -> extract ReID subset later
mark "START RoundaboutHD.zip (HF, ~8.08GB)"
curl -sL -C - -A "Mozilla/5.0" -o RoundaboutHD.zip \
  "https://huggingface.co/datasets/yl4300/RoundaboutHD/resolve/main/RoundaboutHD.zip"
mark "END RoundaboutHD.zip rc=$?"; hashit RoundaboutHD.zip

# 2) VeRi-776 via gdown (GDrive large-file confirm handled by gdown)
mark "START VeRi.zip (gdown)"
gdown --id 0B0o1ZxGs_oVZWmtFdXpqTGl3WUU -O VeRi.zip 2>>"$LOG"
mark "END VeRi.zip rc=$?"; hashit VeRi.zip

# 3) CityFlowV2-ReID via gdown
mark "START AICity21_Track2_ReID.zip (gdown)"
gdown --id 18LJ92tGKMThsHJI0BIxZCo8VUytsXWTc -O AICity21_Track2_ReID.zip 2>>"$LOG"
mark "END CityFlow rc=$?"; hashit AICity21_Track2_ReID.zip

# 4) CARLA-ReID via Dropbox dl=1
mark "START VeRi_CARLA_dataset.zip (Dropbox)"
curl -sL -C - -A "Mozilla/5.0" -o VeRi_CARLA_dataset.zip \
  "https://www.dropbox.com/s/cg1etrs22y2xb62/VeRi_CARLA_dataset.zip?dl=1"
mark "END CARLA rc=$?"; hashit VeRi_CARLA_dataset.zip

mark "ALL DONE"
ls -la ~/lct-reid/datasets | tee -a "$LOG"
df -h ~ | tail -1 | tee -a "$LOG"
