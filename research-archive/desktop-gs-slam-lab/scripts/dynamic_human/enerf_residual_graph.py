"""Local residual rigid transforms for Gaussians already moved by a body prior.

This is a local research extension, not a verbatim implementation of an author
baseline. All matrices act on column vectors in world coordinates.
"""
import math
import torch

def rotation_matrix(axis_angle):
    x,y,z=axis_angle.unbind(-1);o=torch.zeros_like(x)
    K=torch.stack([o,-z,y,z,o,-x,-y,x,o],-1).reshape(*axis_angle.shape[:-1],3,3)
    angle=torch.linalg.vector_norm(axis_angle,dim=-1)
    a=torch.sinc(angle/math.pi)[...,None,None]
    b=(.5*torch.sinc(angle/(2*math.pi)).square())[...,None,None]
    return torch.eye(3,dtype=axis_angle.dtype,device=axis_angle.device)+a*K+b*(K@K)

def transform_gaussians(means,covariance,node_centers,rotations,translations,indices,weights):
    """Frozen Gaussian-to-node weights; exact affine covariance for that blend."""
    g=node_centers[indices];R=rotations[indices];d=translations[indices]
    moved=(R@(means[:,None,:]-g)[...,None])[...,0]+g+d
    result=(moved*weights[...,None]).sum(1)
    linear=(R*weights[...,None,None]).sum(1)
    cov=linear@covariance@linear.transpose(-1,-2)
    return result,cov,linear

def graph_rigidity(node_centers,rotations,translations,edges,weights):
    """Residual deformation preserves local distances when transforms agree."""
    i,j=edges.unbind(-1);offset=node_centers[j]-node_centers[i]
    error=(rotations[i]@offset[...,None])[...,0]+node_centers[i]+translations[i]-node_centers[j]-translations[j]
    return (weights*error.square().sum(-1)).sum()/weights.sum().clamp_min(1e-12)

def polynomial_transform(rotation_coefficients,translation_coefficients,tau):
    """Residual is exactly identity at reference time tau=0; no constant term."""
    basis=torch.stack([tau,tau*tau,tau*tau*tau])
    w=torch.einsum('k,nkd->nd',basis,rotation_coefficients)
    d=torch.einsum('k,nkd->nd',basis,translation_coefficients)
    return rotation_matrix(w),d
