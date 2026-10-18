"""Public real data exercise the actual pipeline, not cached prediction playback."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from dxy_mid import _engine as engine
from dxy_mid.batch import load_manifest
from dxy_mid.diagnostics import observe_files, export_observation

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT/'examples/reproduction_cases'


class PublicDataIntegrity(unittest.TestCase):
    def test_real_failure_cli_compares_legitimate_missing_axis_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT/'scripts/reproduce_public.py'),
                            '--case', 'failure', '--output', tmp, '--workers', '1'],
                           check=True, stdout=subprocess.DEVNULL)
            scope = json.loads((Path(tmp)/'reproduction_scope.json').read_text())
            self.assertTrue(scope['passed'])
            self.assertEqual(scope['subset_numeric_groups'], 0)
            frame = pd.read_csv(Path(tmp)/'frames.csv')
            self.assertTrue(frame.D_mm.isna().all())
            self.assertTrue((frame.ring_reason == 'ring_recovery_unreliable').all())

    def test_public_assets_hashes_mask_ids_and_scope(self):
        sums = pd.read_csv(CASES/'SHA256SUMS.csv')
        self.assertEqual(len(sums), 54)
        for r in sums.itertuples():
            raw = (CASES/r.path).read_bytes()
            self.assertEqual(len(raw), r.bytes)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), r.sha256)
        rows = load_manifest(CASES/'frames.csv')
        self.assertEqual(len(rows), 27)
        for r in rows:
            image = cv2.imread(r['image_path'])
            mask = cv2.imread(r['mask_path'], cv2.IMREAD_UNCHANGED)
            self.assertEqual(image.shape[:2], mask.shape)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)) <= {0, 1, 2})
            self.assertNotIn('GT_mm', r)
        scope = pd.read_csv(CASES/'scope.csv')
        self.assertEqual(int(scope.complete_frozen_group.sum()), 1)
        complete = pd.read_csv(CASES/'multivideo.csv')
        self.assertEqual(len(complete), 20)
        self.assertEqual(complete.video.nunique(), 2)
        self.assertEqual(sorted(complete.groupby('video').size()), [10, 10])
        frozen = pd.read_csv(ROOT/'validation/frozen39/inputs.csv')
        self.assertEqual(set(complete.source_id), set(frozen.loc[frozen.group.eq(complete.group.iloc[0]), 'source_id']))

    def test_original_numerical_files_remain_frozen(self):
        evidence = ROOT/'validation/evidence/public_cases_20261018/numerical_freeze_unchanged.json'
        frozen = json.loads(evidence.read_text())
        for name, expected in frozen['original_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)


class PublicActualGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = {r['source_id']: r for r in load_manifest(CASES/'mechanisms.csv')}
        specs = json.loads((CASES/'visualization_selection.json').read_text())
        cls.observed = {}
        cls.expected = pd.read_csv(CASES/'expected_mechanisms.csv').set_index('source_id')
        cls.expected_ring = json.loads((CASES/'expected_ring.json').read_text())
        cls.previous_profiler = sys.getprofile()
        for name in ['typical_Ip_stages', 'q_recovery', 'zero_P_occlusion', 'ring_failure']:
            r = rows[specs[name]['source_id']]
            cls.observed[name] = observe_files(r['image_path'], r['mask_path'], inner_diameter_mm=r['ring_mm'], source_id=r['source_id'], stem=r['stem'])

    def test_actual_geometry_and_quality_match_frozen(self):
        numeric = ['L_px', 'L_mid_px', 'scale_mm_per_px', 'D_frame_mm', 'P_count', 'Q_count']
        flags = dict(ring_valid='ring_valid', normal_Gate_pass='default_Gate_pass',
                     Q_conflict_recovery='Q_distance_recovery', strict_stable='stability_review_pass',
                     measurement_eligible='measurement_eligible')
        for name, observed in self.observed.items():
            a = observed.measurement.audit
            expected = self.expected.loc[a['source_id']]
            for key in numeric:
                target = expected['D_mm' if key == 'D_frame_mm' else key]
                np.testing.assert_allclose(a[key], target, atol=1e-7, rtol=1e-10, equal_nan=True, err_msg=name+' '+key)
            for key, old in flags.items():
                self.assertEqual(a[key], bool(expected[old]), name+' '+key)
            self.assertEqual(a['ring_reason'], expected['ring_reason'])
            ring = observed.measurement.ring
            old_ring = self.expected_ring[a['source_id']]
            self.assertEqual(ring.valid, old_ring['valid'])
            self.assertEqual(ring.geometry.get('branch', ''), old_ring['branch'])
            for kind, ellipse in old_ring['ellipses'].items():
                for key, value in ellipse.items():
                    np.testing.assert_allclose(ring.geometry[kind][key], value, atol=1e-7, rtol=1e-10, equal_nan=True)
            self.assertTrue(all(observed.checks.values()))
        self.assertIs(sys.getprofile(), self.previous_profiler)

    def test_Q_constraints_traceable_and_recovery_remains_unstable(self):
        normal = self.observed['typical_Ip_stages']
        recovery = self.observed['q_recovery']
        zero = self.observed['zero_P_occlusion']
        self.assertTrue(normal.measurement.audit['normal_Gate_pass'])
        self.assertTrue(recovery.measurement.audit['Q_conflict_recovery'])
        self.assertFalse(recovery.measurement.audit['strict_stable'])
        self.assertFalse(recovery.measurement.audit['normal_Gate_pass'])
        self.assertGreater(recovery.measurement.audit['Q_removed_short_arc_count'], 0)
        self.assertEqual(zero.measurement.audit['P_count'], 0)
        self.assertGreater(zero.measurement.audit['Q_count'], 0)
        self.assertTrue(zero.measurement.audit['P_zero_occlusion_inference'])
        with tempfile.TemporaryDirectory() as tmp:
            export_observation(recovery, tmp)
            q = pd.read_csv(Path(tmp)/'boundary_solver_points.csv')
            self.assertEqual(int(q.solver_Q.sum()), recovery.measurement.audit['Q_count'])
            self.assertTrue(q.loc[q.solver_Q, ['normal_x', 'normal_y', 'lo_px', 'hi_px']].notna().all().all())
            self.assertTrue((q.loc[q.solver_Q, 'classification'] == 'Q').all())

    def test_real_ring_failure_is_not_a_measurement(self):
        a = self.observed['ring_failure'].measurement.audit
        self.assertFalse(a['ring_valid'])
        self.assertFalse(a['measurement_eligible'])
        self.assertTrue(np.isnan(a['D_frame_mm']))
        self.assertEqual(a['ring_reason'], 'ring_recovery_unreliable')

    def test_real_two_video_arithmetic_and_equal_weight_contract(self):
        # Expected data used only to test Fusion arithmetic, not geometry output.
        d = pd.read_csv(CASES/'expected_multivideo.csv')
        g, v = engine.fusion(d.to_dict('records'))
        self.assertEqual(len(v), 2)
        medians = d.groupby('video').D_mm.median()
        self.assertAlmostEqual(g.prediction_mm.iloc[0], float(np.median(medians)))
        # Duplication changes frame counts, not either video's weight.
        augmented = pd.concat([d, d[d.video.eq(d.video.iloc[0])]])
        g2, _ = engine.fusion(augmented.to_dict('records'))
        self.assertAlmostEqual(g2.prediction_mm.iloc[0], g.prediction_mm.iloc[0])


if __name__ == '__main__':
    unittest.main()
