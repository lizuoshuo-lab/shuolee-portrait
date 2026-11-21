"""Fixed-K shared-XYZ bundle model with explicit absolute-Rodrigues derivatives."""
import numpy as np
import cv2
from scipy import sparse
from scipy.spatial.transform import Rotation


class FixedKBundle:
    def __init__(self,K,R,C,X,ids,uv,fixed_center_component,depth_scale):
        self.K=np.array(K,dtype=np.float64,copy=True);self.Rfirst=np.array(R[0],copy=True);self.Cfirst=np.array(C[0],copy=True)
        self.n=len(R);self.m=len(X);self.ids=np.asarray(ids,int);self.uv=np.asarray(uv,float);self.nr=3*(self.n-1)
        self.fixed=int(fixed_center_component);self.fixed_value=float(C[1:].ravel()[self.fixed])
        self.cfree=np.delete(np.arange(3*(self.n-1)),self.fixed);self.xstart=self.nr+len(self.cfree)
        self.center_index={int(v):self.nr+i for i,v in enumerate(self.cfree)};self.depth_scale=float(depth_scale)
        assert self.K.shape==(3,3) and np.allclose(self.K[2],[0,0,1]) and self.depth_scale>0
        assert np.isfinite(self.uv).all() and len(self.ids)==len(self.uv)
        self.initial=self.pack(R,C,X)

    def pack(self,R,C,X):
        assert np.allclose(R[0],self.Rfirst) and np.allclose(C[0],self.Cfirst)
        assert np.isclose(C[1:].ravel()[self.fixed],self.fixed_value)
        return np.r_[Rotation.from_matrix(R[1:]).as_rotvec().ravel(),C[1:].ravel()[self.cfree],X.ravel()]

    def unpack(self,v):
        R=np.concatenate([self.Rfirst[None],Rotation.from_rotvec(v[:self.nr].reshape(-1,3)).as_matrix()])
        cf=np.empty(3*(self.n-1));cf[self.cfree]=v[self.nr:self.xstart];cf[self.fixed]=self.fixed_value
        C=np.vstack([self.Cfirst,cf.reshape(-1,3)]);X=v[self.xstart:].reshape(self.m,3)
        return R,C,X

    def geometry(self,v):
        R,C,X=self.unpack(v);world=X[self.ids[:,1]]-C[self.ids[:,0]];cam=np.einsum('nij,nj->ni',R[self.ids[:,0]],world)
        h=cam@self.K.T
        if not np.isfinite(h).all() or np.any(np.abs(h[:,2])<1e-9):raise ValueError('Nonfinite or near-zero projection depth')
        predicted=h[:,:2]/h[:,2,None]
        return R,C,X,world,cam,h,predicted

    def residual(self,v):
        *_,cam,h,predicted=self.geometry(v)
        penalty=1000*np.minimum(cam[:,2]/self.depth_scale-1e-5,0.)
        return np.c_[predicted-self.uv,penalty].ravel()

    def jacobian(self,v):
        R,C,X,world,cam,h,predicted=self.geometry(v)
        dR=np.zeros((self.n,3,3,3))
        for ci in range(1,self.n):
            Rc,derivative=cv2.Rodrigues(np.ascontiguousarray(v[3*(ci-1):3*ci],dtype=np.float64))
            assert derivative.shape==(3,9) and np.allclose(Rc,R[ci],rtol=1e-10,atol=1e-10)
            dR[ci]=derivative.reshape(3,3,3)
        rows=[];cols=[];values=[]
        def block(k,columns,deriv):
            for j,col in enumerate(columns):
                if col is None:continue
                for axis in range(3):
                    value=deriv[axis,j]
                    if value:rows.append(3*k+axis);cols.append(col);values.append(value)
        for k,(ci,pi) in enumerate(self.ids):
            P=np.zeros((3,3));P[:2]=(self.K[:2]*h[k,2]-h[k,:2,None]*self.K[2])/h[k,2]**2
            if cam[k,2]/self.depth_scale<1e-5:P[2,2]=1000/self.depth_scale
            if ci:
                Dr=np.stack([dR[ci,j]@world[k] for j in range(3)],axis=1)
                block(k,list(range(3*(ci-1),3*ci)),P@Dr)
                block(k,[self.center_index.get(3*(ci-1)+j) for j in range(3)],-P@R[ci])
            block(k,list(range(self.xstart+3*pi,self.xstart+3*pi+3)),P@R[ci])
        return sparse.csr_matrix((values,(rows,cols)),shape=(3*len(self.ids),len(v)))
