"""Keep the measured F1/F7 claims attached to their reproduction conditions."""
from pathlib import Path
import unittest


class BoundaryDocumentationTests(unittest.TestCase):
    def setUp(self):
        text = (Path(__file__).resolve().parents[1] / "SOLUTION.md").read_text()
        self.section = text.split('id="known-behavior-limits"', 1)[1].split('id="validation-metrics"', 1)[0]

    def test_f1_reports_numeric_profile_order_and_changed_decisions(self):
        for required in ("37 query × 37 gallery", "синтетическом", "порядком",
                         "--batch 32 --threads 2", "(6, 3, 0.3)",
                         "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                         "Python 3.13.15", "NumPy 2.5.3", "Pillow 12.3.0", "ONNX Runtime 1.30.0",
                         "0.43999999999999984", "0.5282812306342437", "4/37", "0/37",
                         "с 1 на 2", "перестановок", "уже вычисленных векторах"):
            with self.subTest(condition=required):
                self.assertIn(required, self.section)

    def test_f7_count_has_explicit_zero_threshold_and_gallery_size(self):
        f7 = self.section.split("**F7", 1)[1].split("**Остаток", 1)[0]
        self.assertIn("--threshold 0", f7)
        self.assertIn("15 объектов", f7)
        self.assertIn("3 и 0", f7)


if __name__ == "__main__":
    unittest.main()
