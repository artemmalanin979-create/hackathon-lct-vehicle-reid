#!/bin/bash
cd ~/lct-reid/datasets
LOG=~/lct-reid/jobs/job_44/dl_roundabout.log
echo "[$(date +%H:%M:%S)] resume RoundaboutHD full (for full-file sha256)" >> $LOG
curl -sSL --retry 20 --retry-delay 15 --retry-all-errors -C - -A "Mozilla/5.0" \
  -o RoundaboutHD.zip "https://huggingface.co/datasets/yl4300/RoundaboutHD/resolve/main/RoundaboutHD.zip" 2>>$LOG
echo "[$(date +%H:%M:%S)] curl rc=$? size=$(stat -c%s RoundaboutHD.zip)" >> $LOG
if [ "$(stat -c%s RoundaboutHD.zip)" = "8075314494" ]; then
  echo "[$(date +%H:%M:%S)] HASH RoundaboutHD.zip sha256=$(sha256sum RoundaboutHD.zip|cut -d\  -f1)" >> $LOG
else
  echo "[$(date +%H:%M:%S)] size mismatch, not final" >> $LOG
fi
