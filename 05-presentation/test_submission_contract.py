"""Regression checks for the builder AND the delivered technical slide text."""
import ast
from pathlib import Path
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "submission.csv": ("галерее ≥ 10", "независимо от отказа"),
    "candidates.csv": ("исходным score ≥ порога", "без лимита 10", "успешном batch", "уникальных query ID"),
}


class SubmissionContractTests(unittest.TestCase):
    def assert_contract(self, rows):
        for artifact, clauses in EXPECTED.items():
            with self.subTest(artifact=artifact):
                self.assertIn(artifact, rows)
                for clause in clauses:
                    self.assertIn(clause, rows[artifact])
        self.assertNotIn("выше порога", " ".join(rows.values()))

    def test_builder_contract_in_table_rows(self):
        tree = ast.parse((ROOT / "build_presentation.py").read_text())
        rows = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.List) and len(node.elts) == 2:
                if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
                    rows[node.elts[0].value] = node.elts[1].value
        self.assert_contract(rows)

    def check_deck(self, path):
        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        rows = {}
        with ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            # The preview retains the original slide part names after removing
            # five participant slides; inspect actual cells, not notes or XML bytes.
            for part in z.namelist():
                if part.startswith("ppt/slides/slide") and part.endswith(".xml"):
                    for tr in ET.fromstring(z.read(part)).findall(".//a:tr", ns):
                        cells = [" ".join(t.text or "" for t in cell.findall(".//a:t", ns))
                                 for cell in tr.findall("a:tc", ns)]
                        if len(cells) == 2:
                            rows[cells[0]] = cells[1]
        self.assert_contract(rows)

    def test_main_pptx_contract(self):
        self.check_deck(ROOT / "ЛЦТ2026-задача7-ПРОСВЕТ.pptx")

    def test_preview_pptx_contract(self):
        self.check_deck(ROOT / "checks/content_preview.pptx")


if __name__ == "__main__":
    unittest.main()
