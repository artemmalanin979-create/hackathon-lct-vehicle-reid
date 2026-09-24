"""Lightweight input checks shared by batch and research CLIs (stdlib only)."""
from __future__ import annotations

import csv
import stat
import sys
from pathlib import Path


DATA_HINT = (
    "Положите CSV и images/ из набора организатора задачи №7 в каталог данных. "
    "Набор доступен участнику в личном кабинете, жюри — у организатора. "
    "Проверьте пути --query/--gallery/--csv и --images-dir; в Compose — DATA_DIR. "
    "Инструкция: 04-solution/reproduce/README.md."
)


def fail_inputs(problems, hint):
    print(f"Входы не готовы: {len(problems)} проблем.", file=sys.stderr)
    for problem in problems[:12]:
        print(f"  {problem}", file=sys.stderr)
    if len(problems) > 12:
        print(f"  ... ещё {len(problems) - 12}", file=sys.stderr)
    print(hint + "\nВычисления не запущены.", file=sys.stderr)
    raise SystemExit(2)


def require_files(paths, *, hint):
    problems = []
    for path in dict.fromkeys(map(Path, paths)):
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            problems.append(f"Нет файла: {path}")
            continue
        except OSError as exc:
            problems.append(f"Не читается: {path}: {exc}")
            continue
        if stat.S_ISDIR(mode):
            problems.append(f"Ожидался файл, найден каталог: {path}")
        elif stat.S_ISREG(mode):
            try:
                with path.open("rb"):
                    pass
            except OSError as exc:
                problems.append(f"Не читается: {path}: {exc}")
        # Do not open streams here: even opening a FIFO can block or consume its
        # only writer. The actual loader checks readability on its single read.
    if problems:
        fail_inputs(problems, hint)


def image_candidates(images_dir, image_id, suffixes=(".jpg", ".jpeg", ".png", "")):
    """An explicit filename is literal; extensionless IDs use the caller's order.

    Callers using only ('.jpg',) retain their JPEG-only research contract.
    """
    if "" in suffixes and Path(image_id).suffix:
        suffixes = ("",)
    return [Path(images_dir) / f"{image_id}{suffix}" for suffix in suffixes]


def require_images(images_dir, image_ids, *, suffixes=(".jpg", ".jpeg", ".png", "")):
    images_dir = Path(images_dir)
    problems = []
    for image_id in dict.fromkeys(image_ids):
        # Match the caller's actual resolver; do not impose snapshot hashes or counts.
        if not any(path.is_file() for path in image_candidates(images_dir, image_id, suffixes)):
            problems.append(f"Нет изображения: image_id={image_id}, каталог {images_dir}, "
                            f"расширения {suffixes}")
    if problems:
        fail_inputs(problems, DATA_HINT)


def require_dataset(csv_paths, images_dir, *, suffixes=(".jpg", ".jpeg", ".png", ""), limit=0):
    csv_paths = list(csv_paths)
    require_files(csv_paths, hint=DATA_HINT)
    image_ids = []
    for path in csv_paths:
        # A pipe/FIFO cannot be replayed. Leave it intact for the actual loader;
        # early image diagnostics are deliberately limited to regular CSV files.
        if not Path(path).is_file():
            continue
        try:
            with Path(path).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if limit:
                rows = rows[:limit]
            image_ids.extend(row["image_id"] for row in rows)
        except (OSError, UnicodeError, csv.Error, KeyError) as exc:
            fail_inputs([f"Не удаётся прочитать image_id из {path}: {exc}"], DATA_HINT)
    require_images(images_dir, image_ids, suffixes=suffixes)
