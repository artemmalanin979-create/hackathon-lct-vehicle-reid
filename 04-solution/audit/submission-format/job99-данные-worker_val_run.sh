#!/bin/bash
# Полный R-val прогон исправленного репозитория на worker-vm (offline podman, 2cpu/4g).
set -u
ROOT=~/lct-reid/jobs/job_99
mkdir -p "$ROOT"/{out,scratch}
podman run --rm --network none --cpus 2 --memory 4g \
  --security-opt label=disable \
  -v "$ROOT/repo:/repo:ro" \
  -v ~/lct-reid/data:/data:ro \
  -v "$ROOT/out:/out" \
  -v "$ROOT/scratch:/scratch" \
  -e TMPDIR=/scratch -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -w /repo/04-solution/service localhost/job72-jury:6179b69 \
  python -B /repo/04-solution/reproduce/run.py --mode val --data-dir /data --out-dir /out
code=$?
echo "VAL_RUN_EXIT=$code"
exit $code
