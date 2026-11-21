"""Local affine geometry with affine brightness. Matching-only; no image/camera export warps."""
import numpy as np
from scipy.optimize import least_squares

class AffinePhotometry:
    def __init__(self,source,target,source_center,target_center):
        g=np.arange(-15,16,dtype=np.float64);x,y=np.meshgrid(g,g);self.x=x.ravel();self.y=y.ravel()
        self.center=np.asarray(target_center);self.target=target;self.s=source.ev(self.y+source_center[1],self.x+source_center[0])
    def coords(self,p):return self.center[0]+p[0]+(1+p[2])*self.x+p[3]*self.y,self.center[1]+p[1]+p[4]*self.x+(1+p[5])*self.y
    def fun(self,p):
        x,y=self.coords(p);return self.target.ev(y,x)-p[6]*self.s-p[7]
    def jac(self,p):
        x,y=self.coords(p);gx=self.target.ev(y,x,dx=0,dy=1);gy=self.target.ev(y,x,dx=1,dy=0)
        return np.column_stack([gx,gy,gx*self.x,gx*self.y,gy*self.x,gy*self.y,-self.s,-np.ones_like(self.s)])
    def fit(self):return least_squares(self.fun,[0,0,0,0,0,0,1,0],jac=self.jac,bounds=([-3,-3,-.25,-.25,-.25,-.25,.25,-.5],[3,3,.25,.25,.25,.25,4,.5]),max_nfev=30,ftol=1e-9,xtol=1e-9,gtol=1e-9)
