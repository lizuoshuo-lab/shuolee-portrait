"""Frozen-visibility Gaussian center motion for a bounded hand-flow control."""
import numpy as np,torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import quat_scale_to_covar_preci
from enerf_residual_graph import transform_gaussians

def means_at(p,model,c,t):
    i=model.base.lookup[c,t];A=model.base.A[i,p['anchor']];b=model.base.b[i,p['anchor']];body=(A@p['means'][...,None])[...,0]+b
    nodes=(model.base.A[i,model.graph.node_anchor]@model.graph.node_points[...,None])[...,0]+model.base.b[i,model.graph.node_anchor]
    R,d=model.graph((model.times[i]-726)/6);idx=p['node_indices'];weights=p['node_weights']/p['node_weights'].sum(-1,keepdim=True);g=nodes[idx]
    moved=(R[idx]@(body[:,None]-g)[...,None])[...,0]+g+d[idx]
    return (moved*weights[...,None]).sum(1)

def geometry_at(p,model,c,t):
    i=model.base.lookup[c,t];A=model.base.A[i,p['anchor']];b=model.base.b[i,p['anchor']];body=(A@p['means'][...,None])[...,0]+b
    cov,_=quat_scale_to_covar_preci(p['quats'],p['scales'].exp(),compute_covar=True,compute_preci=False,triu=False);C=A@cov@A.transpose(-1,-2)
    nodes=(model.base.A[i,model.graph.node_anchor]@model.graph.node_points[...,None])[...,0]+model.base.b[i,model.graph.node_anchor]
    R,d=model.graph((model.times[i]-726)/6)
    means,C,_=transform_gaussians(body,C,nodes,R,d,p['node_indices'],p['node_weights']/p['node_weights'].sum(-1,keepdim=True))
    assert (means-means_at(p,model,c,t)).abs().max()<1e-7
    return means,C

def project(means,W,K):
    camera=means@W[:3,:3].T+W[:3,3];q=camera@K.T;return q[:,:2]/q[:,2:],camera[:,2]

def sample(image,xy):
    h,w=image.shape[:2];grid=torch.stack([xy[:,0]*2/(w-1)-1,xy[:,1]*2/(h-1)-1],-1)[None,None]
    return F.grid_sample(image.permute(2,0,1)[None],grid,align_corners=True,mode='bilinear')[0,:,0].T

def splat(features,source,p):
    values,alpha,_=rasterization(source['means'],None,None,p['opacities'].sigmoid(),features,source['W'][None],source['K'][None],160,160,covars=source['cov'],packed=False,near_plane=.1,far_plane=20,eps2d=.3)
    return values[0],alpha[0]

def predicted_normal(p,model,pair,source):
    a=means_at(p,model,pair['camera'],pair['source_frame']);b=means_at(p,model,pair['camera'],pair['target_frame']);uv0,z0=project(a,source['W'],source['K']);uv1,z1=project(b,source['W'],source['K']);flow=uv1-uv0
    values,alpha=splat(flow,source,p);samples=sample(values,pair['xy'])/pair['alpha']
    return (samples*pair['normal']).sum(-1),samples
