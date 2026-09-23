"""Freshness regressions using the actual shipped PDF, without rendering.

Also run `python -m unittest -v check_build_pdf_render` from this directory in
the documented rendering environment to verify source changes during rendering.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class PdfFreshnessTests(unittest.TestCase):
    def check(self, source, pdf):
        return subprocess.run([sys.executable, "-B", "-S", str(HERE / "build_pdf.py"),
                               "--source", str(source), "--output", str(pdf), "--check"],
                              capture_output=True, text=True)

    def test_shipped_pdf_is_current(self):
        p = self.check(HERE.parent / "SOLUTION.md", HERE / "SOLUTION.pdf")
        self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        self.assertTrue(json.loads(p.stdout)["ok"])

    def test_changed_source_is_rejected_without_rewriting_pdf(self):
        with tempfile.TemporaryDirectory() as td:
            source, pdf = Path(td) / "SOLUTION.md", Path(td) / "SOLUTION.pdf"
            source.write_bytes((HERE.parent / "SOLUTION.md").read_bytes() + b"\nChanged.\n")
            before = (HERE / "SOLUTION.pdf").read_bytes()
            pdf.write_bytes(before)
            p = self.check(source, pdf)
            self.assertEqual(p.returncode, 2, p.stderr + p.stdout)
            self.assertFalse(json.loads(p.stdout)["ok"])
            self.assertIn("PDF устарел", p.stderr)
            self.assertEqual(pdf.read_bytes(), before)

    def test_missing_and_invalid_pdf_are_diagnostic(self):
        with tempfile.TemporaryDirectory() as td:
            pdf = Path(td) / "missing.pdf"
            for contents in (None, b"not a PDF"):
                if contents is not None:
                    pdf.write_bytes(contents)
                p = self.check(HERE.parent / "SOLUTION.md", pdf)
                self.assertEqual(p.returncode, 2)
                self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
