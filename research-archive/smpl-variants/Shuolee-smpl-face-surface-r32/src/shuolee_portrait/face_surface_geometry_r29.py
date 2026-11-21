"""Measure complete Gaussian support against finite, connected facial triangles."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch.nn import functional as F

from .face_surface_association_r28 import SurfaceAssociation
from .face_surface_geometry_r28 import _frame, _rotations


FINITE_SURFACE = dict(start_step=500, end_step=15000, center_goal_m=.00075,
                      principal_goal_m=.00150, sigma_radius=1.5, loss_unit_m=.003,
                      principal_directions=6, mixed_directions=12,
                      support_weight=.65, minimum_shared_bins=2, normal_agreement_cos=.82)


@dataclass(frozen=True)
class FiniteSurfaceReference:
    association: SurfaceAssociation
    candidate_faces: torch.Tensor
    candidate_valid: torch.Tensor
    triangles: torch.Tensor
    owner_basis: torch.Tensor
    owner_scale: torch.Tensor
    owner_center: torch.Tensor
    initial_center_distance: torch.Tensor
    initial_principal_distances: torch.Tensor
    initial_mixed_distances: torch.Tensor
    initial_closest_faces: torch.Tensor

    def mask(self, stable_ids):
        return self.association.mask(stable_ids)

    def arrays(self):
        """Expose exact finite topology, posed reference and initial distances."""
        return {field.name:getattr(self,field.name).detach().cpu().numpy()
                for field in fields(self) if field.name!='association'}


def _edge_neighbors(faces):
    incidence = {}
    for face,(a,b,c) in enumerate(faces.tolist()):
        for x,y in ((a,b),(b,c),(c,a)):
            incidence.setdefault((min(x,y),max(x,y)),[]).append(face)
    neighbors = [[] for _ in faces]
    for edge_faces in incidence.values():
        if len(edge_faces)==2:
            a,b=edge_faces
            neighbors[a].append(b)
            neighbors[b].append(a)
    return neighbors


def _candidate_faces(association, faces):
    """Select immediate surface neighbors with matching signed source-bin evidence."""
    data = association.arrays()
    target,bins,support = (data[key] for key in ('target_faces','source_bins','support_views'))
    groups = np.unique(bins)
    row_bins = np.stack([support[:,bins==value].any(1) for value in groups],axis=1)
    face_bins = np.zeros((len(faces),len(groups)),dtype=bool)
    np.logical_or.at(face_bins,target,row_bins)
    adjacency = _edge_neighbors(faces)
    candidates = np.full((len(target),4),-1,dtype=np.int64)
    candidates[:,0] = target
    for row, face in enumerate(target):
        accepted = [near for near in sorted(set(adjacency[face]))
                    if (row_bins[row]&face_bins[near]).sum() >= FINITE_SURFACE['minimum_shared_bins']]
        candidates[row,1:1+len(accepted)] = accepted
    return candidates


def _distance_to_finite_triangles(points,triangles,valid):
    """Return Euclidean distance and closest triangle index for each support sample."""
    # points: [row, sample, xyz]; triangles: [row, candidate, corner, xyz]
    a,b,c = triangles[:,None].unbind(-2)
    point = points[:,:,None]
    ab,ac,ap = b-a,c-a,point-a
    aa,cc,across = (ab*ab).sum(-1),(ac*ac).sum(-1),(ab*ac).sum(-1)
    pa,pc = (ap*ab).sum(-1),(ap*ac).sum(-1)
    det = aa*cc-across.square()
    denominator = det.clamp_min(1e-20)
    u,v = (cc*pa-across*pc)/denominator,(aa*pc-across*pa)/denominator
    interior = (u>=0)&(v>=0)&(u+v<=1)&(det>1e-20)
    plane = (ap-u[...,None]*ab-v[...,None]*ac).norm(dim=-1)
    distance = torch.where(interior,plane,torch.inf)
    for left,right in ((a,b),(b,c),(c,a)):
        edge = right-left
        delta = point-left
        amount = ((delta*edge).sum(-1)/(edge*edge).sum(-1).clamp_min(1e-20)).clamp(0,1)
        distance = torch.minimum(distance,(delta-amount[...,None]*edge).norm(dim=-1))
    return distance.masked_fill(~valid[:,None],torch.inf).min(dim=-1)


def _support_samples(local,log_sigma,rotation,basis,scale,owner_center):
    center = owner_center+(basis @ local[...,None]).squeeze(-1)*scale[:,None]
    radius = log_sigma.exp()*scale[:,None]*FINITE_SURFACE['sigma_radius']
    axes = (basis @ _rotations(rotation)).transpose(-2,-1)*radius[...,None]
    x,y,z = axes.unbind(dim=1)
    first,second = torch.stack((x,x,y),dim=1),torch.stack((y,z,z),dim=1)
    diagonals = torch.cat((first+second,first-second),dim=1)*(2.**-.5)
    offsets = torch.cat((axes,-axes,diagonals,-diagonals),dim=1)
    return torch.cat((center[:,None],center[:,None]+offsets),dim=1)


@torch.no_grad()
def prepare_finite_surface(model,association,*,reference_slot=0):
    """Freeze each finite one-ring from signed association evidence."""
    association = association.to(model._xyz.device)
    if not len(association.stable_ids):
        raise ValueError('Finite facial geometry requires native-supported stable rows.')
    slots = model.densification_ids.argsort()
    index = torch.searchsorted(model.densification_ids[slots],association.stable_ids).clamp_max(len(slots)-1)
    rows = slots[index]
    if not (torch.equal(model.densification_ids[rows],association.stable_ids)
            and torch.equal(model.binding[rows],association.owner_faces)):
        raise ValueError('Finite facial patch ownership does not match the immutable source rows.')
    verts,faces = model.vertices_sequence,model.faces
    if not 0<=reference_slot<len(verts) or bool((association.target_faces>=len(faces)).any()):
        raise ValueError('Finite facial patch references an absent fitted pose or triangle.')
    candidates = torch.from_numpy(_candidate_faces(association,faces.detach().cpu().numpy())).to(faces.device)
    present = candidates>=0
    # Pad unused slots with the primary triangle; no degeneracy or fake candidate enters projection.
    triangle_ids = torch.where(present,candidates,association.target_faces[:,None])
    triangles = verts[:,faces[triangle_ids]]
    basis,_ = _frame(triangles)
    normals = basis[...,1]
    primary = normals[:,:,:1,:]
    angle = (primary*normals).sum(-1)
    valid = present[None] & (angle>=FINITE_SURFACE['normal_agreement_cos']) & (
        angle[reference_slot:reference_slot+1]>=FINITE_SURFACE['normal_agreement_cos'])
    if not bool(valid[:,:,0].all()):
        raise ValueError('The primary facial association is degenerate in a registered pose.')
    owner_triangles = verts[:,faces[association.owner_faces]]
    owner_basis,owner_scale = _frame(owner_triangles)
    owner_center = owner_triangles.mean(-2)
    local,logs,rotations = (getattr(model,key)[rows] for key in ('_xyz','_scaling','_rotation'))
    values,closest = [],[]
    for slot in range(len(verts)):
        points = _support_samples(local,logs,rotations,owner_basis[slot],owner_scale[slot],owner_center[slot])
        distances,choice = _distance_to_finite_triangles(points,triangles[slot],valid[slot])
        values.append(distances)
        closest.append(candidates.gather(1,choice))
    distances = torch.stack(values)
    if not bool(torch.isfinite(distances).all()):
        raise ValueError('Nonfinite initial finite facial support geometry.')
    principal_end = 1+FINITE_SURFACE['principal_directions']
    return FiniteSurfaceReference(association,candidates,valid,triangles,owner_basis,owner_scale,
                                  owner_center,distances[...,0],distances[...,1:principal_end],
                                  distances[...,principal_end:],torch.stack(closest))


def _schedule(step):
    cfg = FINITE_SURFACE
    t = min(1.,max(0.,(step-cfg['start_step'])/(cfg['end_step']-cfg['start_step'])))
    return t*t*(3.-2.*t)


def _measure(model,row_ids,reference,step):
    slot = int(model.timestep)
    index,valid = reference.association.indices(model.densification_ids[row_ids])
    if not bool(valid.all()) or not torch.equal(model.binding[row_ids],reference.association.owner_faces[index]):
        raise ValueError('Finite facial geometry requires original associated stable IDs and owners.')
    local,logs,rotation = (getattr(model,key)[row_ids] for key in ('_xyz','_scaling','_rotation'))
    points = _support_samples(local,logs,rotation,reference.owner_basis[slot,index],
                              reference.owner_scale[slot,index],reference.owner_center[slot,index])
    distances,choice = _distance_to_finite_triangles(
        points,reference.triangles[slot,index],reference.candidate_valid[slot,index])
    closest = reference.candidate_faces[index].gather(1,choice)
    old_center = reference.initial_center_distance[slot,index]
    old_support = reference.initial_principal_distances[slot,index]
    old_mixed = reference.initial_mixed_distances[slot,index]
    progress = _schedule(step)
    principal_end = 1+FINITE_SURFACE['principal_directions']

    def cap(initial,goal):
        return initial+progress*(initial.clamp_max(goal)-initial)

    return dict(center_distance=distances[:,0],principal_distances=distances[:,1:principal_end],
                mixed_distances=distances[:,principal_end:],center_face=closest[:,0],
                principal_faces=closest[:,1:principal_end],mixed_faces=closest[:,principal_end:],
                primary_face=reference.association.target_faces[index],
                initial_center_distance=old_center,initial_principal_distances=old_support,initial_mixed_distances=old_mixed,
                center_cap=cap(old_center,FINITE_SURFACE['center_goal_m']),
                principal_caps=cap(old_support,FINITE_SURFACE['principal_goal_m']),
                mixed_caps=cap(old_mixed,FINITE_SURFACE['principal_goal_m']))


@torch.no_grad()
def finite_surface_metrics(model,row_ids,reference,step=15000):
    """Report finite closest-surface distance and per-sample triangle migration."""
    return _measure(model,row_ids,reference,step)


def finite_surface_loss(model,row_ids,reference,step):
    """Tighten the center and sampled covariance support in metres."""
    if not len(row_ids):
        zero = model._xyz.new_zeros(())
        return zero,dict(center=zero,support=zero)
    values = _measure(model,row_ids,reference,step)

    def penalty(distance,cap):
        excess = (distance-cap).relu()/FINITE_SURFACE['loss_unit_m']
        return F.smooth_l1_loss(excess,torch.zeros_like(excess))

    center = penalty(values['center_distance'],values['center_cap'])
    principal = penalty(values['principal_distances'],values['principal_caps'])
    mixed = penalty(values['mixed_distances'],values['mixed_caps'])
    support = (principal+mixed)*.5
    return center+FINITE_SURFACE['support_weight']*support,dict(center=center,support=support)
