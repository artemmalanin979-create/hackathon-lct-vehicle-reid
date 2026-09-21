#!/bin/bash
set -u
cd ~/lct-reid/datasets
. ~/lct-reid/jobs/job_44/.venv/bin/activate
LOG=~/lct-reid/jobs/job_44/download2.log
mark(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
hashit(){ [ -f "$1" ] && { sz=$(stat -c%s "$1"); sh=$(sha256sum "$1"|cut -d" " -f1); mark "HASH $1 bytes=$sz sha256=$sh"; }; }
mark "START VeRi via gdown fuzzy"
gdown --fuzzy "https://drive.google.com/uc?id=0B0o1ZxGs_oVZWmtFdXpqTGl3WUU" -O VeRi.zip 2>>"$LOG"
mark "END VeRi rc=$?"; hashit VeRi.zip
mark "START CityFlowV2-ReID via gdown fuzzy"
gdown --fuzzy "https://drive.google.com/uc?id=18LJ92tGKMThsHJI0BIxZCo8VUytsXWTc" -O AICity21_Track2_ReID.zip 2>>"$LOG"
mark "END CityFlow rc=$?"; hashit AICity21_Track2_ReID.zip
mark "ALL DONE"; ls -la ~/lct-reid/datasets | tee -a "$LOG"
