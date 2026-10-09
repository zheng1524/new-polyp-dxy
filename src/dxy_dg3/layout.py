"""Repository-relative locations for the archived D-G3 experiment.

The original exploratory scripts lived below a larger private worktree and
therefore used absolute experiment paths.  This small module is deliberately
the only path policy used by the vendored D-G3 implementation.  A user who
has the private raw assets may point ``DXY_DG3_DATA_ROOT`` at a directory
matching the layout documented in ``dg3/README.md``.  Outputs always go to a
new directory (``DXY_DG3_OUTPUT_ROOT`` or ``dg3/reproduced``), never to the
committed frozen result tables.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DG3_ROOT = REPO_ROOT / "dg3"
DATA_ROOT = Path(os.environ.get("DXY_DG3_DATA_ROOT", DG3_ROOT / "input_data"))
OUTPUT_ROOT = Path(os.environ.get("DXY_DG3_OUTPUT_ROOT", DG3_ROOT / "reproduced"))
ASSET_ROOT = Path(os.environ.get("DXY_DG3_ASSET_ROOT", ""))

RAW = DATA_ROOT / "raw_selector"
S6 = DATA_ROOT / "s6"
AUDIT = DATA_ROOT / "scale_audit"
IP88 = DATA_ROOT / "ip88"
FORMAL_MAINLINE = REPO_ROOT / "mainline"

EDGE_OUT = OUTPUT_ROOT / "edge_classification"
MULTIARC_OUT = OUTPUT_ROOT / "multiarc"
G3_OUT = OUTPUT_ROOT / "g3"
QUALITY_OUT = OUTPUT_ROOT / "quality"


def bind_asset_paths(frame):
    """Resolve archived relative asset pointers without storing host paths.

    The frozen CSVs use ``dxy_raw_candidates/...`` and ``raw/...`` paths.
    They remain relative when no asset root is supplied, which makes missing
    external assets fail clearly.  With ``DXY_DG3_ASSET_ROOT`` set, the same
    table can be executed unchanged on another machine.
    """
    if not ASSET_ROOT:
        return frame
    columns = ("image_path", "mask_path", "ring_fit_json", "ring_json_path", "video", "source_video")
    for column in columns:
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: str(ASSET_ROOT / value) if isinstance(value, str) and value and not Path(value).is_absolute() else value
            )
    return frame
