"""Unchanged B1 capture aggregation and formal group median fusion."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def b1_capture_prediction(length_px, ring_px, ring_mm: float) -> tuple[float, float, bool, str]:
    """Frozen B1: R_mm * median(L) / median(R), with original eligibility."""
    length = np.asarray(length_px, dtype=float)
    ring = np.asarray(ring_px, dtype=float)
    keep = np.isfinite(length) & np.isfinite(ring) & (ring > 0)
    length, ring = length[keep], ring[keep]
    dynamic = float(np.percentile(ring, 95) / np.percentile(ring, 5) - 1) if len(ring) >= 2 else math.nan
    if len(ring) < 4:
        return math.nan, dynamic, False, "fewer_than_4_valid_bins"
    if not math.isfinite(dynamic) or dynamic < 0.05:
        return math.nan, dynamic, False, "ring_range_below_5pct"
    return float(ring_mm * np.median(length) / np.median(ring)), dynamic, True, ""


def fuse_formal_rows(formal: pd.DataFrame, capture_predictions: pd.DataFrame) -> pd.DataFrame:
    """Replace only eligible video captures, then use the frozen group median."""
    view = formal.copy()
    view["capture"] = view.capture_id.str.replace("video:", "", regex=False)
    eligible = capture_predictions[capture_predictions.eligible.astype(bool)].set_index("capture").prediction_mm.to_dict()
    view = view[view.current_valid.astype(bool) & pd.to_numeric(view.current_Dxy_mm, errors="coerce").notna()].copy()
    view["prediction_mm"] = pd.to_numeric(view.current_Dxy_mm)
    view["replaced"] = view.source_kind.eq("video_keyframe") & view.capture.isin(eligible)
    view.loc[view.replaced, "prediction_mm"] = view.loc[view.replaced, "capture"].map(eligible)
    return view
