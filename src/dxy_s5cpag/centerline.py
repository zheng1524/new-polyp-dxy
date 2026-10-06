"""Ip8.8 fallback for a visible ring band without a closed inner contour."""
from __future__ import annotations


def reconstruct_inner_from_outer_mid(outer: dict, mid: dict) -> dict:
    """Use annular centerline symmetry; no fitted scale correction or GT.

    `inner = 2*mid - outer` is applied independently to centre and axes.  The
    physical validity requirements are exactly those frozen in the Ip8 trial.
    """
    oa, ob = float(outer["major_radius"]), float(outer["minor_radius"])
    ma, mb = float(mid["major_radius"]), float(mid["minor_radius"])
    ia, ib = 2 * ma - oa, 2 * mb - ob
    if not (ia > 0 and ib > 0 and ia < oa and ib < ob):
        raise ValueError("invalid_centerline_inner_extrapolation")
    return {
        "cx": 2 * float(mid["cx"]) - float(outer["cx"]),
        "cy": 2 * float(mid["cy"]) - float(outer["cy"]),
        "major_radius": ia,
        "minor_radius": ib,
        "angle_deg": float(mid["angle_deg"]),
    }


def add_centerline_inner_if_missing(payload: dict) -> tuple[dict, str]:
    """Return a copied payload with native or reconstructed inner geometry."""
    import json
    from polypseg.ring_fitting import skeleton_midline_ellipse, load_ring_mask, preprocess_ring_mask, select_components
    from pathlib import Path

    copied = json.loads(json.dumps(payload))
    ring = copied.get("rings", [{}])[0]
    if ring.get("inner_ellipse") and ring.get("outer_ellipse"):
        return copied, "native_inner"
    raise ValueError("centerline reconstruction requires the original ring component; use reconstruct_payload_from_mask")


def reconstruct_payload_from_mask(mask_path, base_payload: dict) -> tuple[dict, str]:
    """Build Ip8's adapted payload from the same current ring component.

    The ring preprocessing/component order is deliberately delegated to the
    existing ring-fitting module; only missing inner geometry is filled.
    """
    import json
    from pathlib import Path
    from polypseg.ring_fitting import (
        fit_single_component, load_ring_mask, parse_filename_metadata,
        preprocess_ring_mask, select_components, skeleton_midline_ellipse,
    )

    path = Path(mask_path)
    metadata = parse_filename_metadata(path.stem)
    component_mask = preprocess_ring_mask(load_ring_mask(path))
    components = select_components(component_mask, metadata["expected_rings"])
    if not components:
        raise ValueError("no_ring_component")
    current = fit_single_component(components[0])
    outer = current.get("outer_ellipse")
    mid_obj = skeleton_midline_ellipse(components[0])
    mid = mid_obj.as_dict() if mid_obj is not None else None
    if not outer or not mid:
        raise ValueError("missing_outer_or_mid_ellipse")
    inner = reconstruct_inner_from_outer_mid(outer, mid)
    payload = json.loads(json.dumps(base_payload))
    payload["metadata"] = metadata
    payload["rings"] = [{
        **current,
        "outer_ellipse": outer,
        "mid_ellipse": mid,
        "inner_ellipse": inner,
        "adapter": {"method": "centerline_symmetry_from_existing_outer_and_mid", "native_inner_present": False},
    }]
    return payload, "centerline_adapted_inner"
