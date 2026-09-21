#!/bin/bash
set -u
cd ~/lct-reid/datasets
. ~/lct-reid/jobs/job_44/.venv/bin/activate
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
LOG=~/lct-reid/jobs/job_44/download3.log
mark(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
hashit(){ [ -f "$1" ] && { sz=$(stat -c%s "$1"); sh=$(sha256sum "$1"|cut -d" " -f1); mark "HASH $1 bytes=$sz sha256=$sh"; }; }

mark "START RoundaboutHD.zip (curl, retry+resume)"
curl -sSL --retry 5 --retry-delay 8 --retry-all-errors -C - -A "Mozilla/5.0" \
  -o RoundaboutHD.zip "https://huggingface.co/datasets/yl4300/RoundaboutHD/resolve/main/RoundaboutHD.zip" 2>>"$LOG"
mark "END RoundaboutHD rc=$?"; hashit RoundaboutHD.zip

mark "START VeRi.zip (gdown)"
gdown "https://drive.google.com/uc?id=0B0o1ZxGs_oVZWmtFdXpqTGl3WUU" -O VeRi.zip 2>>"$LOG"
mark "END VeRi rc=$?"; hashit VeRi.zip

mark "START CityFlowV2-ReID (gdown)"
gdown "https://drive.google.com/uc?id=18LJ92tGKMThsHJI0BIxZCo8VUytsXWTc" -O AICity21_Track2_ReID.zip 2>>"$LOG"
mark "END CityFlow rc=$?"; hashit AICity21_Track2_ReID.zip

mark "ALL DONE"; ls -la ~/lct-reid/datasets | tee -a "$LOG"; df -h ~ | tail -1 | tee -a "$LOG"
