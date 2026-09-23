#!/usr/bin/env python3
"""Check every PDF URI against tracked files, including clean-clone portability."""
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parents[3]
PDF = REPO / "06-documentation/SOLUTION.pdf"


def main():
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0"))
    urls = subprocess.check_output(["pdfinfo", "-url", str(PDF)], text=True)
    checks = []
    for line in urls.splitlines():
        match = re.match(r"\s*(\d+)\s+Annotation\s+(.*)$", line)
        if not match:
            continue
        page, url = int(match[1]), match[2]
        uri = urlsplit(url)
        if uri.scheme in ("http", "https", "mailto"):
            checks.append({"page": page, "url": url, "kind": "external", "ok": True})
            continue
        target = (PDF.parent / unquote(uri.path)).resolve()
        relative = str(target.relative_to(REPO)) if target.is_relative_to(REPO) else None
        in_git = relative in tracked or relative is not None and any(p.startswith(relative.rstrip("/") + "/") for p in tracked)
        checks.append({"page": page, "url": url, "kind": "local", "target": relative,
                       "ok": not uri.scheme and target.exists() and in_git})
    broken = [c for c in checks if not c["ok"]]
    print(json.dumps({"annotations": len(checks), "local": sum(c["kind"] == "local" for c in checks),
                      "external": sum(c["kind"] == "external" for c in checks),
                      "external_network_checked": False, "broken": broken, "links": checks},
                     ensure_ascii=False, indent=2))
    return bool(broken) or not checks


if __name__ == "__main__":
    sys.exit(main())
