"""Input locations and stdlib preflight for the baseline command line tools."""
import os
import sys
from pathlib import Path

JOB = Path(__file__).resolve().parents[1]
REPO = JOB.parents[1]
DATA = Path(os.environ.get("REID_DATA_DIR", REPO / "data"))
SPLIT = REPO / "04-solution/split/files"
sys.path.insert(0, str(REPO / "04-solution/service"))
from app.input_checks import DATA_HINT, require_dataset, require_files, require_images

VECTOR_HINT = (
    "Сначала извлеките векторы и .ids командами baseline/README.md (шаги 1 и 3). "
    "CSV сплита восстанавливаются из Git. " + DATA_HINT
)
