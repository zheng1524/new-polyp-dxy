"""Independently measure public JPEG/masks, then compare frozen expectations.

Examples:
  python scripts/reproduce_public.py --case all --output outputs/public --render
  python scripts/reproduce_public.py --case multivideo --output outputs/videos
  python scripts/reproduce_public.py --case failure --output outputs/failure
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from dxy_mid.batch import run_manifest
from verify_regression import compare
from verify_ring_regression import compare_ring

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(all='frames.csv', mechanisms='mechanisms.csv', multivideo='multivideo.csv', failure='failure.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=NAMES, default='all')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--render', action='store_true')
    args = ap.parse_args()
    cases = ROOT/'examples/reproduction_cases'
    # Prediction finishes before any expected CSV or GT is read.
    frames, groups, videos = run_manifest(cases/NAMES[args.case], args.output,
                                         workers=args.workers, render=args.render)
    regression = compare(args.output/'frames.csv', cases/f'expected_{args.case}.csv', args.output/'regression.json')
    ring_reference = json.loads((cases/'expected_ring.json').read_text())
    ring_subset = {sid: ring_reference[sid] for sid in frames.source_id}
    (args.output/'expected_ring_subset.json').write_text(json.dumps(ring_subset))
    ring_regression = compare_ring(args.output, args.output/'expected_ring_subset.json', args.output/'ring_regression.json')
    scopes = pd.read_csv(cases/'scope.csv')
    scopes = scopes[scopes.group.isin(frames.group)]
    status = dict(case=args.case, published_subset_frames=len(frames),
                  published_subset_groups=len(groups), numeric_subset_frames=int(frames.measurement_eligible.sum()),
                  subset_numeric_groups=int(groups.valid.sum()),
                  complete_frozen_groups=[g for g in scopes.loc[scopes.complete_frozen_group, 'group']
                                          if len(frames[frames.group == g]) == int(scopes.loc[scopes.group == g, 'original_frozen_frames'].iloc[0])],
                  caution='Only complete_frozen_groups reproduce full frozen group aggregation; other groups are mechanism subsets.',
                  passed=regression['passed'] and ring_regression['passed'])
    (args.output/'reproduction_scope.json').write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(json.dumps(status, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status['passed'] else 1)


if __name__ == '__main__':
    main()
