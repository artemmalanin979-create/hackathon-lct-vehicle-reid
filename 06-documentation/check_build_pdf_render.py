"""Post-render regressions using real Markdown, WeasyPrint and pdfinfo.

Run explicitly in the documented rendering environment:
    python -m unittest -v check_build_pdf_render
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import build_pdf
from weasyprint import HTML


class PdfRenderTests(unittest.TestCase):
    def test_source_change_during_render_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            source, output = Path(td) / "source.md", Path(td) / "output.pdf"
            original_render = HTML.render
            for change_source in (False, True):
                with self.subTest(change_source=change_source):
                    source.write_text("# Test\n\nBefore rendering.\n")

                    def render_then_change(html, *args, **kwargs):
                        document = original_render(html, *args, **kwargs)
                        if change_source:
                            source.write_text("# Test\n\nChanged during rendering.\n")
                        return document

                    stderr = io.StringIO()
                    with patch.object(HTML, "render", render_then_change), \
                         patch.object(sys, "argv", ["build_pdf.py", "--source", str(source),
                                                   "--output", str(output)]), \
                         contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(stderr):
                        try:
                            code = build_pdf.main()
                        except SystemExit as exc:
                            code = exc.code
                    self.assertEqual(build_pdf.check_freshness(source, output)["ok"],
                                     not change_source)
                    self.assertEqual(code, 2 if change_source else 0, stderr.getvalue())
                    if change_source:
                        self.assertIn("Source changed during generation", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
