#!/usr/bin/env python3
"""Read-only integrity verifier for the promoted video-measurement mainline."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path('/home/liu/polyp_research')
OUT=ROOT/'experiments/dxy_post_capture_relaxed_area_guard_20260922'
MAN=json.loads((OUT/'MAINLINE_MANIFEST.json').read_text())

def digest(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def check(rel:str, want:str):
 got=digest(ROOT/rel)
 assert got==want,(rel,got,want)

def main():
 check('experiments/dxy_post_capture_relaxed_area_guard_20260922/scripts/run_post_capture_guard.py',MAN['sha256']['mainline_script'])
 check('experiments/dxy_raw_video_frame_selection_20260921/tables/raw_candidate_features_blind.csv',MAN['sha256']['raw_candidate_features_blind'])
 check('experiments/dxy_raw_video_frame_selection_20260921/tables/selected_frames_blind.csv',MAN['sha256']['raw_selected_frames_blind'])
 check('experiments/dxy_raw_video_frame_selection_20260921/tables/group_predictions_blind.csv',MAN['sha256']['raw_group_predictions_blind'])
 check('experiments/dxy_area_guarded_sharp5_20260922/tables/frame_area_support.csv',MAN['sha256']['frame_area_support'])
 check('experiments/dxy_area_guarded_sharp5_20260922/tables/formal_s0_area_support_blind.csv',MAN['sha256']['formal_s0_area_support'])
 check('experiments/dxy_scale_video_audit_20260919/inputs/immutable_scale_manifest.csv',MAN['sha256']['immutable_scale_manifest'])
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
 raw=pd.read_csv(ROOT/'experiments/dxy_raw_video_frame_selection_20260921/tables/group_predictions_blind.csv')
 raw=raw[raw.method.eq('result_sharp5')].set_index('group').prediction_mm
 new=pd.read_csv(OUT/'tables/group_predictions_blind.csv').set_index('group').prediction_mm
 changed=sorted(new.index[(new-raw).abs()>1e-12].tolist())
 assert changed==MAN['frozen_blind_run']['changed_groups_vs_raw'],changed
 print('PASS:',MAN['mainline_id'])
 print('blind rule: 1 capture, 3 repaired bins; non-triggered raw selections identical')
 print('changed group:',changed[0])

if __name__=='__main__':main()
