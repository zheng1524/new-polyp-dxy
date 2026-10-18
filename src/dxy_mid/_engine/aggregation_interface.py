"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import math as math
import numpy as np
import pandas as pd


def aggregate(frames,strict_only=False):
 """All valid D_i -> per-video median -> median of videos (equal weights)."""
 d=pd.DataFrame(frames);videos=[];groups=[]
 for (group,video),part in d.groupby(['group','video'],sort=True):
  valid=part.measurement_eligible.astype(bool)&np.isfinite(part.D_mm)&(part.D_mm>0)
  if strict_only:valid&=part.stability_review_pass.astype(bool)
  good=part[valid];x=good.D_mm.to_numpy(float)
  videos.append(dict(group=group,video=video,n_input_frames=len(part),n_valid_frames=len(x),median_mm=float(np.median(x)) if len(x) else math.nan,IQR_mm=float(np.quantile(x,.75)-np.quantile(x,.25)) if len(x) else math.nan,strict_frames=int(good.stability_review_pass.sum()),unstable_frames=int((~good.stability_review_pass.astype(bool)).sum()),Q_conflict_recovery_frames=int(good.Q_distance_recovery.sum()),reason='' if len(x) else 'no_valid_same_frame_physical_measurements',strict_only=strict_only))
 v=pd.DataFrame(videos)
 for group,part in v.groupby('group',sort=True):
  x=part.loc[np.isfinite(part.median_mm),'median_mm'].to_numpy(float)
  groups.append(dict(group=group,n_input_videos=len(part),n_valid_videos=len(x),prediction_mm=float(np.median(x)) if len(x) else math.nan,valid=bool(len(x)),video_equal_weight=True,strict_only=strict_only,reason='' if len(x) else 'scale_not_attached_or_no_eligible_frame',history_prediction_fallback=0))
 return pd.DataFrame(groups),v


# Bind relocated dependencies after defining functions (including override hooks).
