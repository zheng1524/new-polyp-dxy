"""Fresh mask-level frozen39 regression; expected predictions are evaluation only."""
import argparse
from pathlib import Path
from dxy_mid.batch import run_manifest
from verify_regression import compare
from verify_ring_regression import compare_ring

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True,help='Unpacked current image/mask bundle, with images/ and masks/')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--render',action='store_true');a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];baseline=repo/'validation/frozen39'
    run_manifest(baseline/'inputs.csv',a.output,data_root=a.data_root,workers=a.workers,render=a.render)
    # No frozen predictions or GT are opened until independent computation ends.
    result=compare(a.output/'frames.csv',baseline/'expected_frames.csv',a.output/'regression.json')
    ring_result=compare_ring(a.output,baseline/'expected_ring.json',a.output/'ring_regression.json')
    result['passed'] &= ring_result['passed']
    print('Frozen39 regression:', 'PASS' if result['passed'] else 'FAIL')
    raise SystemExit(0 if result['passed'] else 1)
