"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations



def choose_body(results):
 options=[r['chosen'] for r in results if r['chosen'] is not None]
 if options:
  chosen=old_gate.choose(options,[GateResult(True,[],{},c.provenance) for c in options],'legacy')
  return next(r for r in results if r['chosen'] is chosen)
 return max(results,key=lambda r:(r['quality_pass_count'],len(r['near'])))


# Bind relocated dependencies after defining functions (including override hooks).
from . import gate as old_gate
from .contracts import GateResult as GateResult
