"""Numeric audit coverage must not depend on tables or known wrong digits."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_numbers as audit


class NumberAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(audit, 'REPO', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        audit.CHECKS.clear()
        source = self.root / '04-solution/raw.json'
        source.parent.mkdir()
        source.write_text('{"left": 0.7000200442705277, "right": 0.6936584724586875, '
                          '"delta": 0.006361571811840161}')
        self.doc = self.root / 'README.md'

    def check_prose(self):
        audit.prose('README.md', [r'прирост\s+(?P<claim>' + audit.NUMBER + ')'],
                    audit.ref('raw.json', 'delta'))

    def test_all_repeated_prose_and_bullet_claims_are_checked(self):
        self.doc.write_text('Измеренный прирост 0,0063.\n\n- Повтор: прирост +0,0063.\n'
                            'Итоговый прирост 0,0064.\n')
        self.check_prose()
        self.assertEqual([c['ok'] for c in audit.CHECKS], [False, False, True])
        self.assertEqual([c['line'] for c in audit.CHECKS], [1, 3, 4])

    def test_selector_does_not_depend_on_known_wrong_value(self):
        self.doc.write_text('Измеренный прирост 0,1234.\n')
        self.check_prose()
        self.assertEqual(len(audit.CHECKS), 1)
        self.assertFalse(audit.CHECKS[0]['ok'])
        self.assertEqual(audit.CHECKS[0]['expected'], '0.0064')

    def test_correct_repeated_claims_pass(self):
        self.doc.write_text('прирост 0,0064; повторный прирост +0.0064.\n')
        self.check_prose()
        self.assertEqual(len(audit.CHECKS), 2)
        self.assertTrue(all(c['ok'] for c in audit.CHECKS))
        inventory = audit.numeric_inventory(['README.md'], audit.CHECKS)
        self.assertEqual([c['status'] for c in inventory], ['checked', 'checked'])
        self.assertEqual(audit.coverage_status(inventory, True), 0)

    def test_missing_semantic_binding_cannot_pass_silently(self):
        self.doc.write_text('Переформулированный текст без чисел.\n')
        with self.assertRaises(AssertionError):
            self.check_prose()

    def test_deltas_use_unrounded_json_values(self):
        value, source = audit.difference(audit.ref('raw.json', 'left'), audit.ref('raw.json', 'right'))
        self.assertIsInstance(value, Decimal)
        self.assertEqual(value, Decimal('0.0063615718118402'))
        self.doc.write_text('Разница 0,0064.\n')
        audit.check_tokens('README.md', 1, ['0,0064'], [(value, source)])
        self.assertTrue(audit.CHECKS[0]['ok'])

    def test_inventory_includes_nonzero_signed_and_scientific_prose(self):
        self.doc.write_text('Прирост −0,0063; время 12.75; погрешность +1,43e-8.\n'
                            '- Порог 0.0064.\n| mAP | 0.7741 |\n'
                            '```\nvalue = 1e-8\n```\n')
        inventory = audit.numeric_inventory(['README.md'], [])
        self.assertEqual([c['claim'] for c in inventory],
                         ['−0,0063', '12.75', '+1,43e-8', '0.0064', '0.7741', '1e-8'])
        self.assertEqual([c['kind'] for c in inventory], ['prose'] * 4 + ['table', 'code'])

    def test_unbound_new_prose_prevents_complete_audit(self):
        self.doc.write_text('прирост 0,0064\nНеизвестная метрика 0,1234.\n')
        self.check_prose()
        inventory = audit.numeric_inventory(['README.md'], audit.CHECKS)
        self.assertEqual([c['status'] for c in inventory], ['checked', 'unbound'])
        self.assertEqual(audit.coverage_status(inventory, True), 2)
        self.assertEqual(audit.coverage_status(inventory, False), 0)


if __name__ == '__main__':
    unittest.main()
