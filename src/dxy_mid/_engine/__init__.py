"""Reachable numerical closure; no experimental runtime imports."""
from .components import FrameInput as FrameInput
from .contracts import RingResult as RingResult
from .v81_components import existing_ring_check as ring
from .observable_boundary import boundary_native as boundary
from .cross_ring_boundary import run as cross_ring
from .cross_ring_boundary import make_data as make_data
from .v84_pilot import configs84 as configure
from .q_distance_selection import choose_distance_body as choose
from .q_distance_selection import audit_distance_variants as audit
from .run_competition import current_baseline as baseline
from .contracts import ellipse_axes as axes
from .ring_holes import run as holes
from .q_arc_filter import apply as short_Q
from .q_arc_filter import analyze as short_Q_analyze
from .scale import run as scale
from .aggregation_interface import aggregate as fusion
