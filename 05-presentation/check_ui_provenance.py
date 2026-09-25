"""Check that the presentation's real API screenshot depicts current UI sources.

The PNG and source hashes were recorded when the browser capture was made. A
later source edit requires a fresh capture or an explicit review of the delta;
rewriting the provenance alone cannot make an old screenshot current.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ui_source_repo(deck_repo: Path) -> Path:
    """Use an explicit integrated checkout while authoring in a deck worktree."""
    return Path(os.environ.get("LCT_PRESENTATION_UI_REPO", str(deck_repo))).resolve()


def check(repo: Path, provenance: Path) -> dict:
    recorded = json.loads(provenance.read_text(encoding="utf-8"))
    deck_repo = provenance.resolve().parent.parent.parent
    expected = [
        ("05-presentation/assets/ui-search.png", deck_repo,
         recorded["source_sha256"]),
        *((name, repo, digest) for name, digest in recorded["source_files_sha256"].items()),
        (recorded["capture_script"], repo, recorded["capture_script_sha256"]),
    ]
    if "capture_proxy" in recorded:
        expected.append((recorded["capture_proxy"], deck_repo,
                         recorded["capture_proxy_sha256"]))
    items = []
    for name, checkout, digest in expected:
        source = checkout / name
        actual = sha256(source) if source.is_file() else None
        items.append({"path": name, "expected_sha256": digest,
                      "actual_sha256": actual, "match": actual == digest})
    return {"status": "PASS" if all(item["match"] for item in items) else "STALE",
            "capture_source_git_sha": recorded["source_git_sha"], "items": items}


def require_current_capture(repo: Path, provenance: Path) -> None:
    """Stop ordinary deck build/validation before an old UI can be delivered."""
    report = check(repo, provenance)
    if report["status"] != "PASS":
        changed = ", ".join(item["path"] for item in report["items"] if not item["match"])
        raise RuntimeError(f"Presentation UI screenshot is STALE: {changed}. "
                           "Capture the current service through the real API and update provenance.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=ui_source_repo(root.parent))
    parser.add_argument("--provenance", type=Path, default=root / "assets/ui-provenance.json")
    args = parser.parse_args()
    result = check(args.repo, args.provenance)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
