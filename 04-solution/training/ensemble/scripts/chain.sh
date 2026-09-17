#!/bin/bash
# ждёт окончания прохода osnet в s02, останавливает s02 (свой процесс), запускает s02b на подмножестве
cd ~/lct-reid/jobs/job_43
until grep -q "ainv1\] 0/" logs/s02.log || grep -q "^DONE\|Traceback" logs/s02.log; do sleep 15; done
if ! grep -q "^DONE" logs/s02.log; then
  pkill -f "s02_extract_crops.py" ; sleep 3
  echo "[chain] s02 остановлен после прохода osnet: $(date +%T)" >> logs/s02.log
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
taskset -c 0-3 venv/bin/python s02b_extract_subset.py > logs/s02b.log 2>&1
echo "[chain] s02b завершён: $(date +%T)" >> logs/s02b.log
