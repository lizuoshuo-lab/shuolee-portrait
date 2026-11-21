"""Noise-aware global shared-track initializer, explicit fixed rotation input."""
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

def initialize(K,R,observation_ids,uv,point_count,max_iter=100):
    n=len(R);m=point_count;count=len(uv);center_cols=3*(n-1);point_start=center_cols;depth_start=center_cols+3*m
    ids=np.asarray(observation_ids,int);u=np.c_[uv,np.ones(count)]@np.linalg.inv(K).T
    direction=np.einsum('nji,nj->ni',R[ids[:,0]],u);direction/=np.linalg.norm(direction,axis=1)[:,None]
    rows=[];cols=[];values=[]
    for k,(ci,pi) in enumerate(ids):
        for axis in range(3):
            row=3*k+axis
            if ci:rows.append(row);cols.append(3*(ci-1)+axis);values.append(-1.)
            rows.extend([row,row]);cols.extend([point_start+3*pi+axis,depth_start+k]);values.extend([1.,-direction[k,axis]])
    A=sparse.csr_matrix((values,(rows,cols)),shape=(3*count,depth_start+count))
    lower=np.r_[np.full(depth_start,-np.inf),np.ones(count)];upper=np.full(depth_start+count,np.inf)
    opt=lsq_linear(A,np.zeros(3*count),bounds=(lower,upper),tol=1e-8,max_iter=max_iter,lsq_solver='lsmr',lsmr_tol=1e-9,lsmr_maxiter=2000)
    C=np.vstack([np.zeros(3),opt.x[:center_cols].reshape(n-1,3)]);X=opt.x[point_start:depth_start].reshape(m,3)
    distance=np.linalg.norm(C[1:],axis=1);scale=float(np.median(distance))
    if not np.isfinite(scale) or scale<1e-8:raise ValueError('Joint initializer collapsed baseline')
    C/=scale;X/=scale
    return C,X,dict(method='Joint positive-depth bearing least squares, C0 fixed; all observation depths>=1 before median-baseline normalization',
        success=bool(opt.success),status=int(opt.status),message=opt.message,iterations=opt.nit,cost=float(opt.cost),optimality=float(opt.optimality),
        matrix_shape=list(A.shape),depth_min=float(opt.x[depth_start:].min()),normalizing_baseline=scale,max_iterations=max_iter,
        caveat='Conditional on input rotations. Positive ray-depth variables do not independently certify actual camera/point projections or physical correspondence.')
