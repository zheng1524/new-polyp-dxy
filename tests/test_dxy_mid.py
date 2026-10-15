import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from dxy_mid import MeasurementConfig, measure_frame
from dxy_mid import _engine as e
from dxy_mid.batch import load_manifest
from dxy_mid._engine.q_arc_filter import analyze

class FrozenInterfaceTests(unittest.TestCase):
    def test_native_mid_same_frame_and_radius_diameter(self):
        f=e.FrameInput(np.zeros((40,40,3),np.uint8),np.zeros((40,40),np.uint8),5.,'id','i','m')
        ellipse=dict(cx=20,cy=20,major_radius=20.,minor_radius=7.,angle_deg=45.)
        ring=e.RingResult(True,'ok',dict(centerline=ellipse),{})
        s=e.scale(f,ring,'mid_major')
        self.assertEqual(s.reference_diameter_px,40.);self.assertEqual(s.reference_diameter_mm,6.2)
        self.assertAlmostEqual(s.mm_per_px,6.2/40);np.testing.assert_array_equal(s.matrix,np.eye(3))
        self.assertEqual(s.coordinate_system,'native_image_px')
    def test_equal_video_weight_and_unstable_frames(self):
        frames=[dict(group='g',video='a',D_mm=2.,measurement_eligible=True,stability_review_pass=False,Q_distance_recovery=True) for _ in range(10)]
        frames+=[dict(group='g',video='b',D_mm=8.,measurement_eligible=True,stability_review_pass=True,Q_distance_recovery=False),
                 dict(group='g',video='c',D_mm=np.nan,measurement_eligible=False,stability_review_pass=False,Q_distance_recovery=False)]
        g,v=e.fusion(frames);self.assertEqual(g.prediction_mm.iloc[0],5.)
        self.assertEqual(g.n_valid_videos.iloc[0],2)
        g,v=e.fusion(frames,True);self.assertEqual(g.prediction_mm.iloc[0],8.)
    def test_small_closed_hole_and_true_center(self):
        m=np.zeros((180,180),np.uint8);cv2.circle(m,(90,90),65,2,-1);cv2.circle(m,(90,90),43,0,-1);m[87:90,145:148]=0
        z,fill,a=e.holes(m);self.assertEqual(fill.sum(),9);self.assertEqual(z[90,90],0)
        z2,fill2,_=e.holes(z);self.assertFalse(fill2.any());np.testing.assert_array_equal(z,z2)
        np.testing.assert_array_equal(z==1,m==1)
    def test_short_Q_wrap_and_full_contour(self):
        c=np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[4,1],[4,2],[4,3],[4,4],[3,4],[2,4],[1,4],[0,4],[0,3],[0,2],[0,1]])
        def qs(ids):return [dict(contour_id=0,contour_index=i,global_index=i,q=c[i]) for i in ids]
        removed,arcs=analyze([c],qs([0,15]));self.assertEqual(removed,{0,15});self.assertTrue(arcs[0]['wraps'])
        removed,arcs=analyze([c],qs(range(16)));self.assertFalse(removed);self.assertEqual(arcs[0]['arc_length_px'],16.)
    def test_hard_ring_failure_is_NaN(self):
        image=np.zeros((100,100,3),np.uint8);labels=np.zeros((100,100),np.uint8)
        result=measure_frame(image,labels,inner_diameter_mm=5.,source_id='no-ring')
        self.assertFalse(result.audit['ring_valid']);self.assertFalse(result.audit['measurement_eligible'])
        self.assertEqual(result.audit['stability_flags'],'ring_failure')
        self.assertTrue(np.isnan(result.audit['D_frame_mm']));self.assertEqual(result.audit['history_prediction_fallback'],0)
    def test_invalid_coordinates_and_config_rejected(self):
        with self.assertRaises(ValueError):MeasurementConfig(scale_method='affine').validate()
        with self.assertRaises(ValueError):measure_frame(np.zeros((10,10,3),np.uint8),np.zeros((5,5),np.uint8),inner_diameter_mm=5.,source_id='x')
    def test_manifest_hash_and_duplicate_prevention(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);cv2.imwrite(str(t/'im.png'),np.zeros((20,20,3),np.uint8));cv2.imwrite(str(t/'m.png'),np.zeros((20,20),np.uint8))
            row=dict(source_id='x',group='label',video='v',stem='seed',ring_mm=5.,image_path='im.png',mask_path='m.png',image_sha256='bad')
            p=t/'frames.csv';pd.DataFrame([row]).to_csv(p,index=False)
            with self.assertRaises(ValueError):load_manifest(p)
            row.pop('image_sha256');pd.DataFrame([row,row]).to_csv(p,index=False)
            with self.assertRaises(ValueError):load_manifest(p)

if __name__=='__main__':unittest.main()
