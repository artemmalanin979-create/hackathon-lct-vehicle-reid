#!/usr/bin/env python3
"""Check the published validation/test snapshot before importing ML dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inputs(repo: Path, data: Path, mode: str, *, manifest_path: Path | None = None) -> dict:
    manifest_path = manifest_path or HERE / "inputs-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modes = ("val", "test") if mode == "all" else (mode,)
    required = {}
    if "val" in modes:
        for entry in manifest["repo_files"]:
            required[repo / entry["path"]] = entry
    for branch in modes:
        for entry in manifest["data"][branch]:
            required[data / entry["path"]] = entry
    missing, mismatched, unreadable = [], [], []
    for path, expected in required.items():
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            actual_size = path.stat().st_size
            if actual_size != expected["bytes"]:
                mismatched.append({"path": str(path), "expected_bytes": expected["bytes"], "actual_bytes": actual_size})
            else:
                actual_sha = sha256(path)
                if actual_sha != expected["sha256"]:
                    mismatched.append({"path": str(path), "expected_sha256": expected["sha256"], "actual_sha256": actual_sha})
        except OSError as exc:
            unreadable.append({"path": str(path), "error": str(exc)})
    return {"ok": not (missing or mismatched or unreadable), "mode": mode,
            "repo": str(repo), "data_dir": str(data), "required_files": len(required),
            "manifest_sha256": sha256(manifest_path),
            "missing": missing, "mismatched": mismatched, "unreadable": unreadable}


def print_result(result: dict) -> None:
    if result["ok"]:
        print(f"Входы {result['mode']}: OK, {result['required_files']} файлов, размеры и SHA-256 совпали.", flush=True)
        return
    print(f"Входы {result['mode']} не готовы: нет файлов — {len(result['missing'])}; "
          f"не совпали отпечатки — {len(result['mismatched'])}; "
          f"не читаются — {len(result['unreadable'])}.", file=sys.stderr)
    for path in result["missing"][:12]:
        print(f"  Нет: {path}", file=sys.stderr)
    for entry in (result["mismatched"] + result["unreadable"])[:12]:
        print(f"  Не прошёл проверку: {entry['path']}", file=sys.stderr)
    print(f"Положите исходные файлы набора организатора в {result['data_dir']}, "
          f"изображения — в {Path(result['data_dir']) / 'images'}. "
          "Сохраняйте имена файлов из архива. CSV сплита восстанавливаются из Git.\n"
          "Участник получает набор из личного кабинета задачи №7; жюри — из того же "
          "выданного набора у организатора. Инструкция: 04-solution/reproduce/README.md.\n"
          "Инференс не запущен. Полный список доступен в JSON-отчёте (--report).",
          file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("val", "test", "all"), default="val")
    parser.add_argument("--repo-dir", type=Path, default=REPO)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Write every missing/mismatched path as JSON")
    args = parser.parse_args()
    result = check_inputs(args.repo_dir.resolve(), args.data_dir.resolve(), args.mode)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_result(result)
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
