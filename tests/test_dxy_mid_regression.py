"""Real example and optional full39 regression. No saved prediction generation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
class DataRegression(unittest.TestCase):
    def test_real_public_mask_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable,'-m','dxy_mid','measure','--manifest',str(ROOT/'examples/mask_demo/frames.csv'),'--output',tmp,'--workers','1'],check=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/verify_regression.py'),'--actual',str(Path(tmp)/'frames.csv'),'--expected',str(ROOT/'examples/mask_demo/expected.csv'),'--output',str(Path(tmp)/'regression.json')],check=True,stdout=subprocess.DEVNULL)
    @unittest.skipUnless(os.environ.get('DXY_REGRESSION_DATA'),'Full39 dataset not configured; public example remains mandatory')
    def test_frozen39_from_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable,str(ROOT/'scripts/reproduce39.py'),'--data-root',os.environ['DXY_REGRESSION_DATA'],'--output',tmp,'--workers','4'],check=True)

if __name__=='__main__':unittest.main()
