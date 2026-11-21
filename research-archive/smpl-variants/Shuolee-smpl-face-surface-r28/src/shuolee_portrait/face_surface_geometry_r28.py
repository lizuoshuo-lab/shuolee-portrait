"""Pose-aware geometry loss against a fixed native-supported mesh proxy."""

from dataclasses import dataclass, fields
from math import radians, sin

import torch
from torch.nn import functional as F

from .face_surface_association_r28 import SurfaceAssociation
from .face_surface_geometry import FACE_SURFACE, _tangent_major_sigma


SURFACE_R28 = dict(start_step=500, end_step=15000, center_goal_m=.00075, normal_sigma_goal_m=.00085,
                   hard_roundoff_m=2e-8, min_axis_sigma_local=1e-8, center_projection_passes=3)


@dataclass(frozen=True)
class SurfaceGeometryReference:
    association: SurfaceAssociation
    owner_to_target: torch.Tensor
    target_origin: torch.Tensor
    owner_scale: torch.Tensor
    target_scale: torch.Tensor
    initial_center: torch.Tensor
    initial_normal_sigma: torch.Tensor
    initial_tangent_sigma: torch.Tensor
    initial_local_xyz: torch.Tensor

    def arrays(self):
        return {item.name:getattr(self,item.name).detach().cpu().numpy()
                for item in fields(self) if item.name!='association'}

    def mask(self, ids):
        return self.association.mask(ids)


def _frame(triangles):
    a,b,c = triangles.unbind(-2)
    edge, second = b-a,c-a
    tangent = F.normalize(edge,dim=-1)
    normal = F.normalize(torch.cross(tangent,second,dim=-1),dim=-1)
    across = F.normalize(torch.cross(tangent,normal,dim=-1),dim=-1)
    basis = torch.stack((tangent,normal,across),dim=-1)
    scale = (edge.norm(dim=-1)+(across*second).sum(dim=-1).abs())*.5
    return basis,scale.clamp_min(1e-8)


def _rotations(quaternion):
    from roma import quat_wxyz_to_xyzw,unitquat_to_rotmat

    return unitquat_to_rotmat(quat_wxyz_to_xyzw(F.normalize(quaternion,dim=-1)))


def _values(local, logs, quaternion, rotation, origin, scale):
    center = (rotation @ local[...,None]).squeeze(-1)*scale[...,None]+origin
    axes = (rotation @ _rotations(quaternion))*(logs.exp()*scale[...,None])[...,None,:]
    normal = axes[...,1,:].norm(dim=-1)
    tangent = _tangent_major_sigma(axes.reshape(-1,3,3)).reshape(normal.shape)
    return center,normal,tangent,axes


@torch.no_grad()
def prepare_surface_geometry(model, association):
    """Snapshot the reference along the exact frozen associated poses."""
    association = association.to(model._xyz.device)
    order = model.densification_ids.argsort()
    known = model.densification_ids[order]
    rows = order[torch.searchsorted(known,association.stable_ids).clamp_max(len(known)-1)]
    if not (torch.equal(model.densification_ids[rows],association.stable_ids)
            and torch.equal(model.binding[rows],association.owner_faces)):
        raise ValueError('Frozen facial proxy rows do not match live Gaussian IDs and owners.')
    faces,verts = model.faces,model.vertices_sequence
    if bool((association.target_faces>=len(faces)).any()):
        raise ValueError('Frozen facial proxy references a missing mesh face.')
    owner = verts[:,faces[association.owner_faces]]
    target = verts[:,faces[association.target_faces]]
    source_basis,source_scale = _frame(owner)
    target_basis,target_scale = _frame(target)
    # A signed anchor point stays tied to the same target triangle material coordinates.
    anchor = (target*association.barycentric.to(verts.dtype)[None,...,None]).sum(dim=-2)
    inv = target_basis.transpose(-2,-1)
    origin = (inv @ (owner.mean(dim=-2)-anchor)[...,None]).squeeze(-1)
    rotation = inv @ source_basis
    local = model._xyz[rows].detach().clone()
    center,normal,tangent,_ = _values(local,model._scaling[rows],model._rotation[rows],
                                     rotation,origin,source_scale)
    if not all(bool(torch.isfinite(v).all()) for v in (rotation,origin,center,normal,tangent)):
        raise ValueError('Frozen facial proxy produced non-finite pose geometry.')
    if len(rows) and (bool((normal<=0).any()) or bool((torch.linalg.det(rotation)<.99).any())):
        raise ValueError('Frozen facial proxy contains a degenerate owner or target face.')
    return SurfaceGeometryReference(association,rotation,origin,source_scale,target_scale,
                                    center[...,1],normal,tangent,local)


def _rows(model,row_ids,reference):
    index,valid = reference.association.indices(model.densification_ids[row_ids])
    if not bool(valid.all()) or not torch.equal(model.binding[row_ids],reference.association.owner_faces[index]):
        raise ValueError('Associated face geometry requires unchanged target stable IDs and source owners.')
    return index


def _schedule(step):
    cfg = SURFACE_R28
    t = min(1.,max(0.,(step-cfg['start_step'])/(cfg['end_step']-cfg['start_step'])))
    return t*t*(3.-2.*t)


def _measure(model,row_ids,reference,index,step):
    slot = int(model.timestep)
    center,normal,tangent,axes = _values(
        model._xyz[row_ids],model._scaling[row_ids],model._rotation[row_ids],
        reference.owner_to_target[slot,index],reference.target_origin[slot,index],reference.owner_scale[slot,index])
    logs = model._scaling[row_ids]
    ordered,axis = logs.detach().sort(dim=-1)
    sigma = logs.exp()*reference.owner_scale[slot,index,None]
    normal_direction = axes[:,1,:]/sigma
    alignment = normal_direction.gather(1,axis[:,:1])[:,0].square().clamp(max=1.)
    reliability = (-torch.expm1(ordered[:,0]-ordered[:,1])/
                   FACE_SURFACE['relative_axis_gap_full_weight']).clamp(0,1)
    axis_excess = (1.-alignment-sin(radians(FACE_SURFACE['axis_clearance_degrees']))**2).relu()
    old_center = reference.initial_center[slot,index]
    old_normal = reference.initial_normal_sigma[slot,index]
    old_tangent = reference.initial_tangent_sigma[slot,index]
    tangent_goal = (FACE_SURFACE['tangent_support_multiple']*reference.target_scale[slot,index]).clamp(
        FACE_SURFACE['tangent_sigma_min_cap_m'],FACE_SURFACE['tangent_sigma_max_cap_m'])
    progress = _schedule(step)

    def cap(initial,goal):
        return initial+progress*(initial.minimum(torch.as_tensor(goal,device=initial.device,dtype=initial.dtype))-initial)

    return dict(center=center[:,1],normal_sigma=normal,tangent_major_sigma=tangent,
                anchor_distance=center.norm(dim=-1),axis_alignment_sq=alignment,axis_reliability=reliability,
                axis_excess=axis_excess,initial_center=old_center,initial_normal_sigma=old_normal,
                initial_tangent_major_sigma=old_tangent,center_cap=cap(old_center.abs(),SURFACE_R28['center_goal_m']),
                normal_sigma_cap=cap(old_normal,SURFACE_R28['normal_sigma_goal_m']),
                tangent_cap=cap(old_tangent,tangent_goal))


def _closest_triangle_distance(points,triangles):
    a,b,c = triangles.unbind(dim=1)
    ab,ac,delta = b-a,c-a,points-a
    ab2,ac2,abac = (ab*ab).sum(1),(ac*ac).sum(1),(ab*ac).sum(1)
    dot_ab,dot_ac = (ab*delta).sum(1),(ac*delta).sum(1)
    determinant = ab2*ac2-abac.square()
    safe = determinant.clamp_min(1e-20)
    u,v = (ac2*dot_ab-abac*dot_ac)/safe,(ab2*dot_ac-abac*dot_ab)/safe
    plane = (delta-u[:,None]*ab-v[:,None]*ac).norm(dim=1)
    valid = (determinant>1e-20)&(u>=0)&(v>=0)&(u+v<=1)
    distances = [torch.where(valid,plane,torch.inf)]
    for left,right in ((a,b),(b,c),(c,a)):
        edge = right-left
        fraction = ((points-left)*edge).sum(1)/(edge*edge).sum(1).clamp_min(1e-20)
        closest = left+edge*fraction.clamp(0,1)[:,None]
        distances.append((points-closest).norm(dim=1))
    return torch.stack(distances).amin(dim=0)


@torch.no_grad()
def surface_metrics_r28(model,row_ids,reference,step=15000):
    """Read native-pose covariance and distinguish triangle, plane and anchor distance."""
    index = _rows(model,row_ids,reference)
    values = _measure(model,row_ids,reference,index,step)
    verts = model.vertices_sequence[int(model.timestep)]
    owners = verts[model.faces[reference.association.owner_faces[index]]]
    targets = verts[model.faces[reference.association.target_faces[index]]]
    basis,scale = _frame(owners)
    local = model._xyz[row_ids]
    center = owners.mean(dim=1)+(basis @ local[...,None]).squeeze(-1)*scale[:,None]
    values.update(owner_plane=local[:,1]*scale,owner_closest_distance=_closest_triangle_distance(center,owners),
                  target_closest_distance=_closest_triangle_distance(center,targets))
    return values


def surface_loss_r28(model,row_ids,reference,step):
    """Gradually tighten the local proxy constraints while retaining photometric supervision."""
    names = ('center','thickness','tangent','orientation')
    if not len(row_ids):
        zero = model._xyz.new_zeros(())
        return zero,dict.fromkeys(names,zero)
    metrics = _measure(model,row_ids,reference,_rows(model,row_ids,reference),step)

    def penalty(value,cap):
        excess = (value-cap).relu()/FACE_SURFACE['loss_unit_m']
        return F.smooth_l1_loss(excess,torch.zeros_like(excess))

    terms = dict(center=penalty(metrics['center'].abs(),metrics['center_cap']),
                 thickness=penalty(metrics['normal_sigma'],metrics['normal_sigma_cap']),
                 tangent=penalty(metrics['tangent_major_sigma'],metrics['tangent_cap']),
                 orientation=_schedule(step)*(metrics['axis_excess']*metrics['axis_reliability']).mean())
    return sum(FACE_SURFACE[f'{name}_weight']*terms[name] for name in names),terms


def _project_centers(raw,initial,initial_center,normal,scale,good):
    """Project onto initially-good signed center constraints of all saved poses."""
    cfg = SURFACE_R28
    # Raw translation in one target-normal direction leaves tangent motion intact.
    cap = initial_center.abs()+cfg['hard_roundoff_m']
    pose = torch.arange(len(raw),device=raw.device)
    for _ in range(cfg['center_projection_passes']):
        d = initial_center+((raw-initial)[None]*normal).sum(-1)*scale
        excess = (d.abs()-cap).clamp_min(0)*good/scale
        amount,worst = excess.max(dim=0)
        raw = raw-(amount*d[worst,pose].sign())[:,None]*normal[worst,pose]
    change = ((raw-initial)[None]*normal).sum(-1)*scale
    d = initial_center+change
    outside = good & (d.abs()>cap)
    # The original raw center is feasible for every constraint. A single
    # segment interpolation resolves any remaining roundoff or coupled planes.
    ratio = (cap*d.sign()-initial_center)/torch.where(outside,change,1.)
    fraction = torch.where(outside,ratio.clamp(0,1),1.).amin(dim=0)
    return initial+fraction[:,None]*(raw-initial)


def _project_widths(logs,quaternion,normal,scale,initial_sigma,good):
    """Reduce only necessary normal covariance using the most normal axes first."""
    cfg = SURFACE_R28
    widths = logs.exp()
    variance = widths.square()
    weights = ((normal[...,None,:] @ _rotations(quaternion)).squeeze(-2)).square()
    cap = ((initial_sigma+cfg['hard_roundoff_m'])/scale).square()
    floor = torch.minimum(widths,widths.new_full((),cfg['min_axis_sigma_local'])).square()
    excess = ((weights*variance[None]).sum(dim=-1)-cap).clamp_min(0)*good
    active = excess.amax(dim=0)>0
    if not bool(active.any()):
        return logs
    v = variance[active].clone()
    w = weights[:,active]
    cap,good,floor = cap[:,active],good[:,active],floor[active]
    priority = (w*good[...,None]).sum(dim=0).argsort(dim=1,descending=True)
    for axis in priority.split(1,dim=1):
        weight = w.gather(2,axis[None].expand(len(w),-1,-1)).squeeze(2)
        excess = ((w*v[None]).sum(dim=-1)-cap).clamp_min(0)*good
        needed = torch.where(weight>1e-20,excess/weight.clamp_min(1e-20),0.).amax(dim=0)
        previous = v.gather(1,axis)[:,0]
        bound = floor.gather(1,axis)[:,0]
        v.scatter_(1,axis,(previous-needed).maximum(bound)[:,None])
    replacement = logs.clone()
    replacement[active] = torch.where(v<variance[active],.5*v.log(),logs[active])
    return replacement


@torch.no_grad()
def project_surface_envelope(model,row_ids,reference):
    """Hard-protect good proxy normals in the registered supervision poses only."""
    if not len(row_ids):
        return
    cfg = SURFACE_R28
    index = _rows(model,row_ids,reference)
    slots = reference.association.source_slots[:,None]
    normal = reference.owner_to_target[slots,index,1,:]
    scale = reference.owner_scale[slots,index]
    center = reference.initial_center[slots,index]
    normal_sigma = reference.initial_normal_sigma[slots,index]
    raw = model._xyz[row_ids]
    projected = _project_centers(raw,reference.initial_local_xyz[index],center,normal,scale,
                                 center.abs()<=cfg['center_goal_m'])
    # Do not round-trip rows that already satisfied the exact envelope.
    delta = ((raw-reference.initial_local_xyz[index])[None]*normal).sum(-1)*scale
    center_changed = ((center.abs()<=cfg['center_goal_m']) &
                      ((center+delta).abs()>center.abs()+cfg['hard_roundoff_m'])).any(dim=0)
    model._xyz[row_ids] = torch.where(center_changed[:,None],projected,raw)
    model._scaling[row_ids] = _project_widths(model._scaling[row_ids],model._rotation[row_ids],normal,scale,
                                             normal_sigma,normal_sigma<=cfg['normal_sigma_goal_m'])
