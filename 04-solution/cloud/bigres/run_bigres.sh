#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS="${BIGRES_BLAS_THREADS:-2}"
export OMP_NUM_THREADS="${BIGRES_BLAS_THREADS:-2}"
export MKL_NUM_THREADS="${BIGRES_BLAS_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${PYTHON:-}" ]]; then
    BIGRES_PYTHON="$PYTHON"
elif [[ -x "$TASK_DIR/venv-gpu/bin/python" ]]; then
    BIGRES_PYTHON="$TASK_DIR/venv-gpu/bin/python"
elif [[ -x "$TASK_DIR/venv/bin/python" ]]; then
    BIGRES_PYTHON="$TASK_DIR/venv/bin/python"
else
    BIGRES_PYTHON=python3
fi
if [[ $# -eq 0 || "$1" == --* ]]; then set -- run "$@"; fi
exec "$BIGRES_PYTHON" -u "$TASK_DIR/bigres.py" "$@"
