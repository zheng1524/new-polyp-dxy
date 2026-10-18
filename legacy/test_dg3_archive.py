from pathlib import Path
import unittest

from dxy_dg3.archive import EXPECTED_B0, verify


class DG3ArchiveTest(unittest.TestCase):
    def test_committed_b0_metrics(self):
        root = Path(__file__).resolve().parents[1]
        actual = verify(root)
        for name, expected in EXPECTED_B0.items():
            if isinstance(expected, int):
                self.assertEqual(actual[name], expected)
            else:
                self.assertAlmostEqual(actual[name], expected, places=10)
