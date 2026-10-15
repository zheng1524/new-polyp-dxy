"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import sys as sys


def traced_call(fn,*args):
 # Read locals at return to expose OpenCV axis order / pre-gate G3 proposals.
 # No legacy source mutation, no historical data, no replacement optimiser.
 captured={};old=sys.gettrace()
 def trace(f,event,arg):
  if f.f_code is fn.__code__:
   if event=='return':captured.update(f.f_locals)
   return trace
  return None
 sys.settrace(trace)
 try:result=fn(*args)
 finally:sys.settrace(old)
 return result,captured


# Bind relocated dependencies after defining functions (including override hooks).
