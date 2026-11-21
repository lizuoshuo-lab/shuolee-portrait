"""Read-only FIT numerical audit of the stalled N5c solve; no camera candidate."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import json, time, hashlib
import numpy as np
import cv2
from scipy.sparse import coo_matrix
from scipy.optimize._numdiff import approx_derivative
import shared_background122 as old

O=old.O/'numerical-audit'
def huber_cost(r):
    a=np.abs(r);return float(np.sum(np.where(a<=1,.5*r*r,a-.5)))

class Problem:
    def __init__(self):
        self.plan=old.read(old.P);self.frames=self.plan['frames']
        self.K=np.array(old.read(old.O/'result.json')['K_integer_centers'])
        d=np.load(old.O/'fit-diagnostic.npz');self.x0=d['parameters'].copy()
        self.tids=d['track_ids'];self.slices={};n=0
        for i,f in enumerate(self.frames[1:],1):
            m=5 if i==1 else 6;self.slices[f]=slice(n,n+m);n+=m
        self.ncam=n
        raw=old.read(old.B/'observations.json')['tracks'];lookup={int(t):i for i,t in enumerate(self.tids)}
        observations=[(o['frame'],lookup[t['track']],o['uv']) for t in raw if t['split']=='fit' and t['track'] in lookup for o in t['observations']]
        self.target=np.array([v for _,_,v in observations],dtype=np.float64)
        self.indices=np.array([i for _,i,_ in observations]);self.groups={f:np.where([ff==f for ff,_,_ in observations])[0] for f in self.frames}
        self.rows=[];self.cols=[]
        for r,(f,i,_) in enumerate(observations):
            cols=list(range(n+3*i,n+3*i+3))
            if f in self.slices:cols+=list(range(self.slices[f].start,self.slices[f].stop))
            for rr in [2*r,2*r+1]:
                self.rows.extend([rr]*len(cols));self.cols.extend(cols)
        self.sparsity=coo_matrix((np.ones(len(self.rows)),(self.rows,self.cols)),shape=(2*len(observations),len(self.x0))).tocsr()
    def decode(self,x):
        cams={self.frames[0]:np.eye(4)}
        for i,f in enumerate(self.frames[1:],1):
            p=x[self.slices[f]];M=np.eye(4);M[:3,:3]=cv2.Rodrigues(p[:3])[0]
            M[:3,3]=[np.cos(p[4])*np.cos(p[3]),np.cos(p[4])*np.sin(p[3]),np.sin(p[4])] if i==1 else p[3:]
            cams[f]=M
        return cams,x[self.ncam:].reshape(-1,3)
    def residual(self,x):
        cams,xyz=self.decode(x);out=np.empty_like(self.target)
        for f,ix in self.groups.items():out[ix]=old.project(self.K,cams[f],xyz[self.indices[ix]])[0]-self.target[ix]
        return out.ravel()
    def jacobian(self,x):
        cams,xyz=self.decode(x);rows=[];cols=[];values=[]
        for ci,f in enumerate(self.frames):
            ix=self.groups[f];points=xyz[self.indices[ix]];M=cams[f]
            q=points@M[:3,:3].T+M[:3,3];z=q[:,2];j=np.zeros((len(ix),2,3))
            j[:,0,0]=self.K[0,0]/z;j[:,0,2]=-self.K[0,0]*q[:,0]/z**2
            j[:,1,1]=self.K[1,1]/z;j[:,1,2]=-self.K[1,1]*q[:,1]/z**2
            jp=j@M[:3,:3]
            if ci:
                p=x[self.slices[f]]
                _,jj=cv2.projectPoints(points,p[:3],M[:3,3],self.K,None)
                jc=jj[:,:6].reshape(-1,2,6)
                if ci==1:
                    th,ph=p[3:];dt=np.array([[-np.cos(ph)*np.sin(th),-np.sin(ph)*np.cos(th)],[np.cos(ph)*np.cos(th),-np.sin(ph)*np.sin(th)],[0,np.cos(ph)]])
                    jc=np.concatenate([jc[:,:,:3],jc[:,:,3:]@dt],axis=2)
            for a,ri in enumerate(ix):
                cc=list(range(self.ncam+3*self.indices[ri],self.ncam+3*self.indices[ri]+3));vals=jp[a]
                if ci:cc+=list(range(self.slices[f].start,self.slices[f].stop));vals=np.concatenate([vals,jc[a]],axis=1)
                for k in [0,1]:rows.extend([2*ri+k]*len(cc));cols.extend(cc);values.extend(vals[k])
        return coo_matrix((values,(rows,cols)),shape=self.sparsity.shape).tocsr()

def run():
    assert not O.exists();O.mkdir();cv2.setNumThreads(2);start=time.monotonic()
    old.save(O/'plan.json',dict(registered=old.now(),scope='Read-only FIT derivative, conditioning and descent audit. No heldout data or new candidate.',
        source=old.sha(Path(__file__)),input_npz_sha256=old.sha(old.O/'fit-diagnostic.npz'),
        controls=['analytic versus SciPy default sparse finite differences','central directional derivatives at1e-5 and1e-7','one fixed dimensionless small descent control'],
        CPU_seconds=90,GPU=0,old_result_unchanged=True))
    p=Problem();x=p.x0;r=p.residual(x);ja=p.jacobian(x)
    jn=approx_derivative(p.residual,x,method='2-point',sparsity=p.sparsity)
    diff=(ja-jn).tocsr();diff.eliminate_zeros();scale=max(1,float(np.max(np.abs(ja.data))))
    cdiff=np.asarray(abs(diff).max(axis=0).toarray()).ravel();cnorm=np.sqrt(np.asarray(ja.power(2).sum(axis=0)).ravel())
    rng=np.random.default_rng(122);v=rng.normal(size=len(x))/np.maximum(cnorm,1);v/=np.max(np.abs(v));av=ja@v
    directional=[]
    for h in [1e-5,1e-7]:
        nv=(p.residual(x+h*v)-p.residual(x-h*v))/(2*h)
        directional.append(dict(h=h,max_abs=float(np.max(abs(nv-av))),relative_l2=float(np.linalg.norm(nv-av)/np.linalg.norm(av))))
    grad=ja.T@np.clip(r,-1,1);dx=-grad/np.maximum(cnorm**2,1)
    dx*=1e-4/max(np.max(abs(dx)),1e-20)
    result=dict(updated=old.now(),elapsed_seconds=time.monotonic()-start,
        dtype=dict(parameters=str(x.dtype),target=str(p.target.dtype),residual=str(r.dtype)),shape=list(ja.shape),
        cost=huber_cost(r),gradient_inf=float(np.max(abs(grad))),small_descent_cost=huber_cost(p.residual(x+dx)),
        analytic_vs_default_numeric=dict(max_abs=float(np.max(abs(diff.data))) if diff.nnz else 0,relative_to_max_jac=(float(np.max(abs(diff.data)))/scale if diff.nnz else 0),
            camera_column_max_abs=cdiff[:p.ncam].tolist(),camera_column_norm=cnorm[:p.ncam].tolist(),
            point_column_norm_percentiles=np.percentile(cnorm[p.ncam:],[0,25,50,75,100]).tolist()),
        analytic_central_directional=directional,
        state='audit_only_no_optimization',K_changed=False,old_result_unchanged=True)
    old.save(O/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':run()
