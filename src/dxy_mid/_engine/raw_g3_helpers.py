"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import cv2 as cv2
import math as math
import numpy as np
MAX_CROSS = 0.75
SAMPLE_FORWARD = 0.035
SAMPLE_RADIUS = 0.025
POLY_FRAC = 0.25

def frac(mask,p,r):
 x,y=p;x0=max(0,int(round(x-r)));x1=min(mask.shape[1],int(round(x+r))+1);y0=max(0,int(round(y-r)));y1=min(mask.shape[0],int(round(y+r))+1)
 yy,xx=np.ogrid[y0:y1,x0:x1];q=(xx-x)**2+(yy-y)**2<=r*r;return float(mask[y0:y1,x0:x1][q].mean()) if q.any() else 0.


def ring_samples(ring,polyp,rr):
 cs,_=cv2.findContours(ring.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE);P=[];N=[];K=[];S=[]
 for c in cs:
  z=c.reshape(-1,2).astype(float)
  for i,p in enumerate(z):
   t=z[(i+1)%len(z)]-z[(i-1)%len(z)];n=np.linalg.norm(t)
   if n<1e-6:continue
   normal=np.array([-t[1],t[0]])/n;lab=[]
   for sign in (1,-1):
    q=beyond(ring,p,normal*sign,max(4,round(MAX_CROSS*rr)))
    lab.append('U' if q is None else ('P' if frac(polyp,q+normal*sign*max(1,round(SAMPLE_FORWARD*rr)),max(1,round(SAMPLE_RADIUS*rr)))>=POLY_FRAC else 'B'))
   if lab==['P','P']:kind='cut';sgn=0
   elif lab.count('P')==1 and lab.count('B')==1:kind='outer';sgn=1 if lab[0]=='P' else -1
   else:kind='uncertain';sgn=0
   P.append(p);N.append(normal);K.append(kind);S.append(sgn)
 return np.asarray(P),np.asarray(N),np.asarray(K,object),np.asarray(S,int)


def beyond(ring,p,v,maxd):
 h,w=ring.shape;seen=False
 for d in range(maxd+1):
  x,y=np.rint(p+v*d).astype(int)
  if x<0 or x>=w or y<0 or y>=h:return None
  if ring[y,x]:seen=True;continue
  if seen:return np.array([float(x),float(y)])
 return None


def ellipse_residual(points,p):
 cx,cy,la,lb,ang=p;a,b=np.exp(la),np.exp(lb);co,si=np.cos(ang),np.sin(ang);q=points-np.array([cx,cy]);x=q[:,0]*co+q[:,1]*si;y=-q[:,0]*si+q[:,1]*co
 return (np.sqrt((x/a)**2+(y/b)**2)-1)*math.sqrt(a*b)


def ellipse_from_cv(points):
 if len(points)<5:return None
 try:(cx,cy),(d1,d2),ang=cv2.fitEllipseAMS(points.astype(np.float32).reshape(-1,1,2))
 except cv2.error:return None
 if min(d1,d2)<=1:return None
 return np.array([cx,cy,math.log(d1/2),math.log(d2/2),math.radians(ang)],float)


# Bind relocated dependencies after defining functions (including override hooks).
