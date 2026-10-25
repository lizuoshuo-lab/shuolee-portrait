"""Observation-supported translation graph, not a cloth physics model.

Weights are fixed within each Gaussian between explicit refreshes. Thus the
per-Gaussian affine Jacobian remains the body affine; graph motion adds only
translation. The same defined local affine is used for mean and covariance.
"""
import numpy as np
from scipy.spatial import cKDTree

def support_weights(points,nodes,support,world_scale):
    points=np.asarray(points)
    distance,index=cKDTree(nodes).query(points,k=min(4,len(nodes)))
    if distance.ndim==1:distance,index=distance[:,None],index[:,None]
    weight=np.exp(-.5*(distance/(.12*world_scale))**2)
    weight/=np.maximum(weight.sum(1,keepdims=True),1e-20)
    supported=cKDTree(support).query(points,k=1)[0]
    gate=np.exp(-(supported/(.12*world_scale))**2)
    return index,weight*gate[:,None]
