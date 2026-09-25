"""Positive and negative controls for the browser screenshot provenance gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from check_ui_provenance import check, require_current_capture


class UiProvenanceTest(unittest.TestCase):
    def test_capture_sources_then_stale_script(self) -> None:
        with TemporaryDirectory() as tmp:
            deck_repo = Path(tmp) / "deck"
            ui_repo = Path(tmp) / "ui"
            files = {
                "05-presentation/assets/ui-search.png": b"actual browser capture",
                "04-solution/service/app/static/index.html": b"<main>Search</main>",
                "04-solution/service/app/static/app.js": b"performSearch()",
                "04-solution/service/ui-checks/presentation-shots.cjs": b"capture()",
                "05-presentation/capture_proxy.py": b"loopback gateway",
            }
            digests = {}
            for name, content in files.items():
                path = (deck_repo if name.startswith("05-presentation/") else ui_repo) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                digests[name] = hashlib.sha256(content).hexdigest()
            provenance = deck_repo / "05-presentation/assets/ui-provenance.json"
            provenance.write_text(json.dumps({
                "source_sha256": digests["05-presentation/assets/ui-search.png"],
                "source_files_sha256": {
                    name: digests[name] for name in files if "/static/" in name
                },
                "capture_script": "04-solution/service/ui-checks/presentation-shots.cjs",
                "capture_script_sha256": digests["04-solution/service/ui-checks/presentation-shots.cjs"],
                "capture_proxy": "05-presentation/capture_proxy.py",
                "capture_proxy_sha256": digests["05-presentation/capture_proxy.py"],
                "source_git_sha": "capture-commit",
            }), encoding="utf-8")
            self.assertEqual(check(ui_repo, provenance)["status"], "PASS")
            require_current_capture(ui_repo, provenance)

            script = ui_repo / "04-solution/service/app/static/app.js"
            script.write_bytes(b"performSearchWithChangedState()")
            report = check(ui_repo, provenance)
            self.assertEqual(report["status"], "STALE")
            changed = [item["path"] for item in report["items"] if not item["match"]]
            self.assertEqual(changed, ["04-solution/service/app/static/app.js"])
            with self.assertRaisesRegex(RuntimeError, "app.js"):
                require_current_capture(ui_repo, provenance)


if __name__ == "__main__":
    unittest.main()
