"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import math as math
import numpy as np


def bounded(p,data):
 a,b=np.exp(p[2:4]);rr=data['rr'];extent=data['extent'];return bool(np.all(np.isfinite(p)) and min(a,b)>=max(.15*rr,.08*extent) and max(a,b)<=max(3*rr,1.6*extent) and max(a,b)/min(a,b)<=3.5 and np.linalg.norm(p[:2]-data['center'])<=1.25*extent)


def weights(Q):
 ids=np.array([q['arc_id'] for q in Q]);unique=np.unique(ids)
 return np.array([1/(len(unique)*np.sum(ids==i)) for i in ids]) if len(ids) else np.empty(0)


def interval_residual(p,Q):
 if not Q:return np.empty(0)
 q=np.array([z['q'] for z in Q]);n=np.array([z['n'] for z in Q]);lo=np.array([z['lo'] for z in Q]);hi=np.array([z['hi'] for z in Q]);co,si=np.cos(p[4]),np.sin(p[4]);R=np.array([[co,-si],[si,co]]);M=R@np.diag(np.exp(-2*p[2:4]))@R.T;d=q-p[:2];alpha=np.einsum('ij,jk,ik->i',n,M,n);beta=2*np.einsum('ij,jk,ik->i',d,M,n);gamma=np.einsum('ij,jk,ik->i',d,M,d)-1;disc=beta**2-4*alpha*gamma;root=np.sqrt(np.maximum(disc,0));t1=(-beta-root)/(2*alpha);t2=(-beta+root)/(2*alpha)
 def outside(t):return np.maximum(lo-t,0)+np.maximum(t-hi,0)
 res=np.minimum(outside(t1),outside(t2));res[disc<0]+=np.sqrt(-disc[disc<0])/(2*alpha[disc<0]);return res


def residual_true(p,P,shape):
 if not len(P):return np.empty(0)
 return np.linalg.norm(P-p[:2],axis=1)-np.exp(p[2]) if shape=='circle' else RAWG3.ellipse_residual(P,p)


def vector(c):
 center,U,s,_=ellipse_axes(c);return np.array([*center,math.log(s[0]),math.log(s[1]),math.atan2(U[1,0],U[0,0])])


def circle3(pts):
 A=2*(pts[1:]-pts[0]);b=(pts[1:]**2).sum(1)-(pts[0]**2).sum()
 if abs(np.linalg.det(A))<1e-5:return None
 center=np.linalg.solve(A,b);radius=np.linalg.norm(pts[0]-center)
 return np.array([*center,math.log(max(radius,1e-9)),math.log(max(radius,1e-9)),0.])


# Bind relocated dependencies after defining functions (including override hooks).
from . import raw_g3_helpers as RAWG3
from .contracts import ellipse_axes as ellipse_axes
