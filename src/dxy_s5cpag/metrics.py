"""Final group-level metric implementation used by the S5-CPAG reports."""
from __future__ import annotations
import pandas as pd


def summary(predictions: pd.DataFrame) -> dict:
    data = predictions.dropna(subset=["prediction_mm", "reference_mm"]).copy()
    signed = 100 * (data.prediction_mm - data.reference_mm) / data.reference_mm
    absolute = signed.abs()
    return {
        "coverage": len(data), "median_abs_error_pct": absolute.median(),
        "MAE_mm": (data.prediction_mm - data.reference_mm).abs().mean(),
        "mean_abs_relative_error_pct": absolute.mean(), "median_signed_error_pct": signed.median(),
        "within5_pct": 100 * (absolute <= 5).mean(), "within10_pct": 100 * (absolute <= 10).mean(),
        "p90_pct": absolute.quantile(.90), "p95_pct": absolute.quantile(.95), "max_pct": absolute.max(),
    }
