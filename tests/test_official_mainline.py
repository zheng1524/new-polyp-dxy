"""The machine pointer, complete official tables and real result gallery agree."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest
from urllib.parse import urlsplit,unquote

import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class OfficialMainline(unittest.TestCase):
    def test_machine_pointer_and_single_package(self):
        m=json.loads((ROOT/'mainline/MAINLINE_MANIFEST.json').read_text())
        self.assertEqual(m['mainline_id'],'DXY-MID');self.assertEqual(m['status'],'OFFICIAL_MAINLINE')
        self.assertFalse(m['historical_prediction_fallback'])
        self.assertIn('DXY-MID',(ROOT/'mainline/CURRENT_POINTER.md').read_text())
        for retired in ['base38','ip88_extension','dg3','legacy','src/dxy_s5cpag','src/dxy_dg3']:
            self.assertFalse((ROOT/retired).exists(),retired)
        for package in ['dxy_s5cpag','dxy_dg3','polypseg']:
            self.assertIsNone(importlib.util.find_spec(package),package)
        for path,expected in m['artifacts_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),expected,path)

    def test_complete_CSV_Excel_and_frozen_arithmetic(self):
        spec=importlib.util.spec_from_file_location('official_table_export',ROOT/'mainline/scripts/build_mainline.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        groups,videos,metrics=module.calculate(ROOT/'validation/results',ROOT/'validation/frozen39/GT_mapping.csv',ROOT/'validation/frozen39/inputs.csv')
        self.assertTrue(module.verify_files(groups,videos,metrics,ROOT/'mainline')['passed'])
        self.assertEqual(len(groups),39);self.assertEqual(len(videos),79)
        self.assertEqual(int(groups.crosses_10mm.sum()),2)
        self.assertAlmostEqual(metrics['median_absolute_relative_error_pct'],4.4113269936186095,places=9)

    def test_gallery_points_axes_values_and_failure_are_real(self):
        directory=ROOT/'docs/results_gallery';index=pd.read_csv(directory/'index.csv')
        self.assertEqual(len(index),11)
        self.assertEqual(set(index.error_band),{'0–2%','2–5%','5–10%','10–15%','>15%'})
        frames=pd.read_csv(ROOT/'validation/results/frame_measurements.csv').set_index('source_id')
        inputs=pd.read_csv(ROOT/'validation/frozen39/inputs.csv').set_index('source_id')
        groups=pd.read_csv(ROOT/'mainline/tables/group_results.csv').set_index('group_id')
        public=pd.read_csv(ROOT/'examples/reproduction_cases/frames.csv').set_index('source_id');public_Q_checks=0
        for row in index.itertuples():
            e=json.loads((directory/row.evidence).read_text());f=frames.loc[row.source_id];g=groups.loc[row.group_id]
            self.assertFalse(e['geometry_refit_for_plot']);self.assertTrue(e['checks_passed']);self.assertTrue(e['Q_original_contour_indices_verified'])
            self.assertEqual(e['figure_sha256'],hashlib.sha256((directory/row.figure).read_bytes()).hexdigest())
            for key in ['image_sha256','mask_sha256']:self.assertEqual(e[key],inputs.loc[row.source_id,key])
            self.assertAlmostEqual(e['frame_Dxy_mm'],f.D_mm,places=8)
            self.assertAlmostEqual(e['group_prediction_mm'],g.DXY_MID_mm,places=10)
            self.assertAlmostEqual(e['GT_mm'],g.GT_mm,places=10)
            self.assertAlmostEqual(np.linalg.norm(np.diff(e['endpoints'],axis=0)),f.L_px,places=7)
            self.assertAlmostEqual(2*np.linalg.svd(e['chosen']['basis'],compute_uv=False)[0],f.L_px,places=7)
            self.assertAlmostEqual(2*e['ring']['centerline']['major_radius'],f.L_mid_px,places=7)
            self.assertAlmostEqual(f.L_px*f.scale_mm_per_px,f.D_mm,places=9)
            if row.source_id in public.index:
                mask=cv2.imread(str(ROOT/'examples/reproduction_cases'/public.loc[row.source_id,'mask_path']),0)
                contours,_=cv2.findContours((mask==1).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                for q in e['Q_indices']:
                    np.testing.assert_array_equal(q['q'],contours[q['contour_id']].reshape(-1,2)[q['contour_index']]);public_Q_checks+=1
        self.assertGreater(public_Q_checks,0)
        bad=pd.read_csv(directory/'failure_index.csv')
        self.assertEqual(len(bad),1)
        for row in bad.itertuples():
            f=frames.loc[row.source_id];self.assertFalse(f.ring_valid);self.assertTrue(np.isnan(f.D_mm));self.assertTrue((directory/row.figure).is_file())

    def test_current_README_links(self):
        for path in [ROOT/'README.md',ROOT/'mainline/README.md',ROOT/'mainline/CURRENT_POINTER.md',ROOT/'mainline/ERROR_RANKING.md',ROOT/'docs/results_gallery/README.md']:
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                u=urlsplit(target)
                if u.scheme or not u.path:continue
                self.assertTrue((path.parent/unquote(u.path)).exists(),f'{path}: {target}')


if __name__=='__main__':unittest.main()
