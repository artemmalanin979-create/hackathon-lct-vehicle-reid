#!/bin/bash
set -u
J=~/lct-reid/jobs/job_44; D=~/lct-reid/datasets; LOG=$J/finish_carla.log
. $J/.venv/bin/activate
m(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
: > "$LOG"
# 1. wait for CARLA curl to finish (pid 17520) and size stable
m "waiting for CARLA download to finish"
while kill -0 17520 2>/dev/null; do sleep 10; done
sleep 3
sz=$(stat -c%s $D/VeRi_CARLA_dataset.zip); m "CARLA final bytes=$sz (expected 2529020889)"
sh=$(sha256sum $D/VeRi_CARLA_dataset.zip | cut -d" " -f1); m "CARLA sha256=$sh"
# 2. verify zip integrity + list
if ! unzip -t $D/VeRi_CARLA_dataset.zip >/dev/null 2>&1; then m "ZIP TEST FAILED (partial?)"; fi
m "top-level entries:"; unzip -Z1 $D/VeRi_CARLA_dataset.zip 2>/dev/null | sed "s#/.*##" | sort -u | head | tee -a "$LOG"
m "total jpg in zip:"; unzip -Z1 $D/VeRi_CARLA_dataset.zip 2>/dev/null | grep -icE "\.jpg$" | tee -a "$LOG"
m "sample names:"; unzip -Z1 $D/VeRi_CARLA_dataset.zip 2>/dev/null | grep -iE "\.jpg$" | head -4 | tee -a "$LOG"
# 3. extract jpgs
mkdir -p $D/carla; m "unzipping..."; unzip -o -q $D/VeRi_CARLA_dataset.zip -d $D/carla 2>>"$LOG"; m "unzip rc=$?"
GL=$(python3 -c "import glob,os; c=glob.glob(os.path.expanduser(\"~/lct-reid/datasets/carla/**/*.jpg\"),recursive=True); print(len(c)); print(c[0] if c else \"NONE\")")
m "found jpgs: $GL"
# 4. build combined (CARLA only for now; RoundaboutHD added when it lands)
m "building combined set (own + CARLA)..."
OMP_NUM_THREADS=4 taskset -c 0-3 python $J/build_combined.py \
  --add "carla:$HOME/lct-reid/datasets/carla/**/*.jpg:carla:200000:2000" \
  --out-npz $J/combined_train.npz --out-raw $J/combined_crops_208.npy 2>&1 | tee -a "$LOG"
m "sha256 combined:"; sha256sum $J/combined_train.npz $J/combined_crops_208.npy 2>/dev/null | tee -a "$LOG"
ls -la $J/combined_train.npz $J/combined_crops_208.npy 2>/dev/null | tee -a "$LOG"
df -h ~ | tail -1 | tee -a "$LOG"; m "FINISH_CARLA DONE"
