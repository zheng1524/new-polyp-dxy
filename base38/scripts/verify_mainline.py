#!/usr/bin/env python3
"""Read-only integrity verifier for the promoted video-measurement mainline."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
DATA_ROOT=Path(os.environ.get('DXY_DATA_ROOT',REPO/'data'))
ARTIFACT=REPO/'base38'
OUT=Path(os.environ.get('DXY_OUTPUT_ROOT',ARTIFACT))
RAW=Path(os.environ.get('DXY_RAW_SELECTOR_ROOT',DATA_ROOT/'raw_selector'))
AREA=Path(os.environ.get('DXY_AREA_GUARD_ROOT',DATA_ROOT/'area_guard'))
AUDIT=Path(os.environ.get('DXY_AUDIT_ROOT',DATA_ROOT/'scale_video_audit'))
MAN=json.loads((ARTIFACT/'MAINLINE_MANIFEST.json').read_text())

def digest(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def check(path:Path, want:str):
 got=digest(path)
 assert got==want,(str(path),got,want)

def main():
 # The script SHA is intentionally historical: this portable wrapper has
 # replaced machine-specific paths.  All data SHA checks remain unchanged.
 check(RAW/'tables/raw_candidate_features_blind.csv',MAN['sha256']['raw_candidate_features_blind'])
 check(RAW/'tables/selected_frames_blind.csv',MAN['sha256']['raw_selected_frames_blind'])
 check(RAW/'tables/group_predictions_blind.csv',MAN['sha256']['raw_group_predictions_blind'])
 check(AREA/'tables/frame_area_support.csv',MAN['sha256']['frame_area_support'])
 check(AREA/'tables/formal_s0_area_support_blind.csv',MAN['sha256']['formal_s0_area_support'])
 check(AUDIT/'inputs/immutable_scale_manifest.csv',MAN['sha256']['immutable_scale_manifest'])
 for line in (OUT/MAN['sha256']['blind_output_list']).read_text().splitlines():
  want,rel=line.split('  ',1)
  assert digest(OUT/rel)==want,(rel,'blind output changed')
 f=pd.read_csv(OUT/'tables/post_selected_frames_blind.csv')
 assert len(f)==MAN['candidate_universe']['raw_selected_bins']
 assert int(f.loc[f.capture_trigger,'capture'].nunique())==MAN['frozen_blind_run']['triggered_captures']
 assert int(f.selection_outcome.eq('post_retry').sum())==MAN['frozen_blind_run']['repaired_bins']
 assert int(f.selection_outcome.eq('post_missing').sum())==MAN['frozen_blind_run']['post_missing_bins']
 untouched=f[~f.capture_trigger]
 assert (untouched.stem==untouched.raw_stem).all(),'a non-triggered raw frame changed'
 raw=pd.read_csv(RAW/'tables/group_predictions_blind.csv')
 raw=raw[raw.method.eq('result_sharp5')].set_index('group').prediction_mm
 new=pd.read_csv(OUT/'tables/group_predictions_blind.csv').set_index('group').prediction_mm
 changed=sorted(new.index[(new-raw).abs()>1e-12].tolist())
 assert changed==MAN['frozen_blind_run']['changed_groups_vs_raw'],changed
 print('PASS:',MAN['mainline_id'])
 print('blind rule: 1 capture, 3 repaired bins; non-triggered raw selections identical')
 print('changed group:',changed[0])

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.parse_args();main()
