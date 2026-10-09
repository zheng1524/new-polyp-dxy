"""Offline verification for the committed D-G3 frozen-result package."""
from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd


EXPECTED_B0 = {
    "coverage": 39,
    "median_abs_error_pct": 4.980738446606654,
    "MAE_mm": 0.6310721590553092,
    "p95_pct": 20.219854095804195,
    "max_pct": 23.685498698929642,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def b0_metrics(repo_root: Path) -> dict[str, float | int]:
    """Recalculate B0/D-G3 final metrics from the committed evaluated table."""
    path = repo_root / "dg3" / "tables" / "group_predictions_evaluated.csv"
    rows = pd.read_csv(path)
    rows = rows[rows["method"].eq("B0")].copy()
    assert len(rows) == 39, f"expected 39 B0 groups, found {len(rows)}"
    absolute = pd.to_numeric(rows["abs_error_pct"], errors="raise")
    delta_mm = pd.to_numeric(rows["prediction_mm"], errors="raise") - pd.to_numeric(rows["reference_mm"], errors="raise")
    return {
        "coverage": int(len(rows)),
        "median_abs_error_pct": float(absolute.median()),
        "MAE_mm": float(delta_mm.abs().mean()),
        "p95_pct": float(absolute.quantile(0.95)),
        "max_pct": float(absolute.max()),
    }


def verify(repo_root: Path) -> dict[str, float | int]:
    actual = b0_metrics(repo_root)
    for key, expected in EXPECTED_B0.items():
        got = actual[key]
        if isinstance(expected, int):
            assert got == expected, f"{key}: {got} != {expected}"
        else:
            assert abs(float(got) - expected) < 1e-10, f"{key}: {got} != {expected}"
    return actual
