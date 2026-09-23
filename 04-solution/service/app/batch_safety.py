"""Atomic publication of a complete batch output; standard library only."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import os
from pathlib import Path
import shutil
import tempfile

OUTPUT_FILES = {"embeddings.npy", "submission.csv", "candidates.csv", "run_info.json"}


def _check_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise SystemExit(f"Выходной каталог {destination} — symlink; укажите обычный каталог")
    if destination.exists():
        if not destination.is_dir():
            raise SystemExit(f"Выходной путь {destination} не является каталогом")
        children = list(destination.iterdir())
        if any(p.name not in OUTPUT_FILES or not p.is_file() or p.is_symlink() for p in children):
            raise SystemExit(f"В {destination} есть посторонние файлы; укажите отдельный --out-dir")


def _publish(staging: Path, destination: Path) -> None:
    _check_destination(destination)
    if destination.exists():
        # A single exchange also handles a previous non-empty result atomically.
        # Never remove the old result before the new one is visible.
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise OSError(errno.ENOSYS, "renameat2 недоступен; нужен новый --out-dir") from exc
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(staging), -100, os.fsencode(destination), 2) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
    else:
        staging.rename(destination)


@contextmanager
def atomic_output(destination: Path):
    """Sibling directory: SIGKILL can leave .*.incomplete-*, never a partial result.

    Publication protects against process death, not power loss/fsync durability.
    An old complete result remains visible until the exchange succeeds.
    """
    destination = destination.absolute()
    _check_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.incomplete-",
                                    dir=destination.parent))
    try:
        yield staging
        if {p.name for p in staging.iterdir()} != OUTPUT_FILES:
            raise RuntimeError("Неполный выходной комплект; публикация отменена")
        try:
            _publish(staging, destination)
        except OSError as exc:
            raise SystemExit(
                f"Не удалось опубликовать комплект в {destination}: {exc}. "
                "Укажите --out-dir в доступном для записи родительском каталоге "
                "на той же файловой системе (не корень bind mount)."
            ) from exc
    finally:
        # After EXCHANGE this is the old result, otherwise only our private temp.
        if staging.exists():
            shutil.rmtree(staging)
