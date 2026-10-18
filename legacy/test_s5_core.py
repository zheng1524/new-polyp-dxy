import math
import unittest

import numpy as np

from dxy_s5cpag.aggregation import b1_capture_prediction
from dxy_s5cpag.area_support import area_consistency
from dxy_s5cpag.centerline import reconstruct_inner_from_outer_mid
from dxy_s5cpag.scale import current_rectified_scale


class CoreSmokeTests(unittest.TestCase):
    def test_centerline_inner_reconstruction(self):
        outer = {"cx": 100, "cy": 80, "major_radius": 50, "minor_radius": 40, "angle_deg": 15}
        mid = {"cx": 101, "cy": 81, "major_radius": 45, "minor_radius": 35, "angle_deg": 15}
        inner = reconstruct_inner_from_outer_mid(outer, mid)
        self.assertEqual(inner["cx"], 102)
        self.assertEqual(inner["cy"], 82)
        self.assertEqual(inner["major_radius"], 40)
        self.assertEqual(inner["minor_radius"], 30)

    def test_b1_formula_and_dynamic_gate(self):
        prediction, dynamic, eligible, reason = b1_capture_prediction([100, 110, 120, 130], [50, 55, 60, 65], 10)
        self.assertTrue(eligible, reason)
        self.assertGreater(dynamic, .05)
        self.assertAlmostEqual(prediction, 10 * 115 / 57.5)

    def test_observed_area_excludes_unknown(self):
        body = np.array([[1, 1], [0, 0]], dtype=bool)
        ellipse = np.array([[1, 1], [1, 1]], dtype=bool)
        unknown = np.array([[0, 0], [1, 1]], dtype=bool)
        c, p, body_n, ellipse_n, overlap_n = area_consistency(body, ellipse, unknown)
        self.assertEqual((body_n, ellipse_n, overlap_n), (2, 2, 2))
        self.assertEqual((c, p), (1, 1))

    def test_rectified_scale_is_finite(self):
        payload = {"metadata": {"ring_inner_diameters_mm": [10.]}, "rings": [{
            "inner_ellipse": {"cx": 100., "cy": 100., "major_radius": 40., "minor_radius": 35., "angle_deg": 10.},
            "outer_ellipse": {"cx": 100., "cy": 100., "major_radius": 45., "minor_radius": 40., "angle_deg": 10.},
            "quality": {},
        }]}
        scale, diameter, _, _ = current_rectified_scale(payload)
        self.assertTrue(math.isfinite(scale) and scale > 0)
        self.assertTrue(math.isfinite(diameter) and diameter > 0)


if __name__ == "__main__":
    unittest.main()
