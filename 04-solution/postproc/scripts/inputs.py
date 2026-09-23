"""Input locations and stdlib preflight for documented postprocessing steps."""
import os
import sys
from pathlib import Path

JOB = Path(__file__).resolve().parents[1]
REPO = JOB.parents[1]
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT = REPO / "04-solution/split/files"
sys.path.insert(0, str(REPO / "04-solution/service"))
from app.input_checks import require_dataset, require_files

BASE_HINT = (
    "Сначала извлеките векторы baseline и скопируйте .npy/.ids в postproc/out/ "
    "(postproc/README.md, шаг 1). CSV сплита восстанавливаются из Git."
)
TTA_HINT = (
    "Сначала s03_extract.py --sets val --variants 208,208f,256, затем "
    "s05b_make_tta_vectors.py --sets val (postproc/README.md, шаги 3–4). "
    "Для tune выберите --sets tune. CSV сплита восстанавливаются из Git."
)
