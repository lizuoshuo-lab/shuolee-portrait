"""Connected observed facial surfaces and bound, surface-projected split seeds."""

from dataclasses import dataclass, fields, replace
from heapq import heappop, heappush

import numpy as np
import torch
from torch.nn import functional as F

from .face_surface_association_r28 import SurfaceAssociation
from .face_surface_geometry_r28 import _frame, _rotations
from .face_surface_geometry_r29 import FINITE_SURFACE, _distance_to_finite_triangles, _schedule, _support_samples


CONNECTED_SURFACE = dict(**FINITE_SURFACE, minimum_radius_m=.003, maximum_radius_m=.028,
                         geodesic_padding_m=.0015, edge_curve_cos=.82, whole_patch_cos=.55,
                         projection_chunk=16, max_split_replay_distance_m=.002,
                         min_sibling_separation_m=.001, relative_covariance_floor=1e-7)


@dataclass(frozen=True)
class SurfaceSplitSeeds:
    centers: torch.Tensor
    log_sigma: torch.Tensor
    rotations: torch.Tensor
    eligible: torch.Tensor
    target_faces: torch.Tensor
    max_supported_pose_distance_m: torch.Tensor
    parent_tangent_sigma_m: torch.Tensor
    sibling_separation_m: torch.Tensor


@dataclass(frozen=True)
class ConnectedSurfaceReference:
    association: SurfaceAssociation
    reference_slot: int
    candidate_faces: torch.Tensor
    candidate_valid: torch.Tensor
    predecessor: torch.Tensor
    geodesic_distance_m: torch.Tensor
    radius_m: torch.Tensor
    triangles: torch.Tensor
    owner_basis: torch.Tensor
    owner_scale: torch.Tensor
    owner_center: torch.Tensor
    initial_distances: torch.Tensor
    initial_closest_faces: torch.Tensor

    def mask(self, stable_ids):
        return self.association.mask(stable_ids)

    def arrays(self):
        values = {field.name: getattr(self,field.name).detach().cpu().numpy() for field in fields(self)
                  if field.name not in ('association','reference_slot')}
        return dict(values,reference_slot=np.asarray(self.reference_slot,dtype=np.int64))

    @torch.no_grad()
    def with_children(self,model,parent_ids,child_ids):
        """Keep the proven parent patch for each child and measure its own initial support."""
        ids = self.association.stable_ids
        parents = torch.as_tensor(parent_ids,device=ids.device,dtype=torch.long)
        children = torch.as_tensor(child_ids,device=ids.device,dtype=torch.long)
        if parents.ndim!=1 or children.ndim!=2 or children.shape[0]!=len(parents):
            raise ValueError('Connected surface children must be aligned to their stable parents.')
        positions,selected = self.association.indices(parents)
        if not bool(selected.any()):
            return self
        parent_rows = positions[selected]
        chosen = children[selected].flatten()
        inherited = parent_rows.repeat_interleave(children.shape[1])
        retained = ~torch.isin(ids,parents[selected])
        combined = torch.cat((ids[retained],chosen))
        new_ids,order = combined.sort()
        if bool((new_ids[1:]<=new_ids[:-1]).any()):
            raise ValueError('Connected surface child stable ID already exists.')
        use = torch.cat((retained.nonzero(as_tuple=True)[0],inherited))[order]
        assoc = self.association
        new_assoc = SurfaceAssociation(new_ids,assoc.owner_faces[use],assoc.target_faces[use],assoc.barycentric[use],
                                        assoc.support_views[use],assoc.source_slots,assoc.source_bins)
        reference = replace(self,association=new_assoc,candidate_faces=self.candidate_faces[use],
            candidate_valid=self.candidate_valid[:,use],predecessor=self.predecessor[use],
            geodesic_distance_m=self.geodesic_distance_m[use],
            radius_m=self.radius_m[use],triangles=self.triangles[:,use],owner_basis=self.owner_basis[:,use],
            owner_scale=self.owner_scale[:,use],owner_center=self.owner_center[:,use],
            initial_distances=self.initial_distances[:,use],initial_closest_faces=self.initial_closest_faces[:,use])
        rows = _model_rows(model,new_assoc)
        just_children = torch.searchsorted(new_ids,chosen)
        distance,faces = _initial_metrics(model,rows[just_children],reference,just_children)
        reference.initial_distances[:,just_children] = distance
        reference.initial_closest_faces[:,just_children] = faces
        return reference


def _normal(triangles):
    e1,e2 = triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]
    n = np.cross(e1,e2)
    return n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-20)


def _connected_patches(faces,triangles,association,radius):
    """Use native-bin-compatible edge paths and a radius measured in metres."""
    data = association.arrays()
    targets,bins,support = (data[key] for key in ('target_faces','source_bins','support_views'))
    sources = np.unique(bins)
    row_bins = np.stack([support[:,bins==source].any(1) for source in sources],axis=1)
    face_bins = np.zeros((len(faces),len(sources)),dtype=bool)
    np.logical_or.at(face_bins,targets,row_bins)
    normals = _normal(triangles)
    centroids = triangles.mean(1)
    incidence = {}
    for i,(a,b,c) in enumerate(faces.tolist()):
        for x,y in ((a,b),(b,c),(c,a)):
            incidence.setdefault((min(x,y),max(x,y)),[]).append(i)
    adjacency = [[] for _ in faces]
    # Full mesh incidence excludes nonmanifold edges, even when just two
    # of their faces happen to carry signed skin evidence.
    for neighbors in incidence.values():
        if len(neighbors)==2:
            a,b = neighbors
            common = face_bins[a]&face_bins[b]
            if common.sum()>=CONNECTED_SURFACE['minimum_shared_bins'] and (
                    normals[a]@normals[b]>=CONNECTED_SURFACE['edge_curve_cos']):
                cost = max(float(np.linalg.norm(centroids[a]-centroids[b])),1e-8)
                adjacency[a].append((b,cost,common))
                adjacency[b].append((a,cost,common))
    patches = []
    for row,target in enumerate(targets):
        found,frontier = {int(target):0.},[(0.,int(target))]
        previous = {int(target):int(target)}
        while frontier:
            distance,face = heappop(frontier)
            if distance!=found[face]:
                continue
            for neighbor,cost,shared in adjacency[face]:
                total = distance+cost
                if (total<=radius[row] and total<found.get(neighbor,np.inf) and
                        (shared&row_bins[row]).sum()>=CONNECTED_SURFACE['minimum_shared_bins'] and
                        normals[target]@normals[neighbor]>=CONNECTED_SURFACE['whole_patch_cos']):
                    found[neighbor] = total
                    previous[neighbor] = face
                    heappush(frontier,(total,neighbor))
        patches.append(sorted(((face,dist,previous[face]) for face,dist in found.items()),
                              key=lambda item:(item[1],item[0])))
    width = max(map(len,patches))
    ids = np.full((len(patches),width),-1,dtype=np.int64)
    predecessor = np.zeros_like(ids)
    distances = np.zeros((len(patches),width),dtype=np.float32)
    for row,patch in enumerate(patches):
        slots = {face:j for j,(face,_,_) in enumerate(patch)}
        for j,(face,distance,parent) in enumerate(patch):
            ids[row,j],distances[row,j] = face,distance
            predecessor[row,j] = slots[parent]
    return ids,distances,predecessor


def _nearest(points,triangles,valid):
    """Use bounded exact projection chunks across all admitted patch triangles."""
    chunk = CONNECTED_SURFACE['projection_chunk']
    distance,chosen = None,None
    for start in range(0,triangles.shape[1],chunk):
        current,ids = _distance_to_finite_triangles(points,triangles[:,start:start+chunk],
                                                    valid[:,start:start+chunk])
        if distance is None:
            distance,chosen = current,ids
        else:
            improved = current<distance
            distance,chosen = torch.where(improved,current,distance),torch.where(improved,ids+start,chosen)
    return distance,chosen


def _model_rows(model,association):
    live,order = model.densification_ids.sort()
    positions = torch.searchsorted(live,association.stable_ids).clamp_max(len(live)-1)
    rows = order[positions]
    if not (torch.equal(model.densification_ids[rows],association.stable_ids)
            and torch.equal(model.binding[rows],association.owner_faces)):
        raise ValueError('Connected facial geometry requires the saved stable IDs and original owners.')
    return rows


def _indices(model,row_ids,reference):
    index,valid = reference.association.indices(model.densification_ids[row_ids])
    if not bool(valid.all()) or not torch.equal(model.binding[row_ids],reference.association.owner_faces[index]):
        raise ValueError('Connected facial geometry requires the saved stable IDs and original owners.')
    return index


def _pose_metrics(model,row_ids,reference,index,slot):
    points = _support_samples(model._xyz[row_ids],model._scaling[row_ids],model._rotation[row_ids],
        reference.owner_basis[slot,index],reference.owner_scale[slot,index],reference.owner_center[slot,index])
    distance,closest = _nearest(points,reference.triangles[slot,index],reference.candidate_valid[slot,index])
    return distance,reference.candidate_faces[index].gather(1,closest)


def _initial_metrics(model,row_ids,reference,index):
    metrics = [_pose_metrics(model,row_ids,reference,index,slot) for slot in range(len(model.vertices_sequence))]
    distances,faces = map(torch.stack,zip(*metrics,strict=True))
    if not bool(torch.isfinite(distances).all()):
        raise ValueError('Connected facial support is not finite in the named poses.')
    return distances,faces


@torch.no_grad()
def prepare_connected_surface(model,association,*,reference_slot=0):
    """Freeze evidence-connected geodesic support sized by each seed covariance."""
    association = association.to(model._xyz.device)
    if not len(association.stable_ids):
        raise ValueError('Connected facial geometry needs a signed surface association.')
    rows = _model_rows(model,association)
    verts,faces = model.vertices_sequence,model.faces
    if not 0<=reference_slot<len(verts) or bool((association.target_faces>=len(faces)).any()):
        raise ValueError('Connected patch target face or reference pose is absent.')
    # Render-space covariance is transformed from the original owner.
    source = verts[:,faces[association.owner_faces]]
    target = verts[reference_slot,faces[association.target_faces]]
    owner_basis,owner_scale = _frame(source)
    target_basis,_ = _frame(target)
    rotated = owner_basis[reference_slot] @ _rotations(model._rotation[rows])
    scale = model._scaling[rows].exp()*owner_scale[reference_slot,:,None]
    tangent = (target_basis.transpose(1,2) @ rotated)[:,(0,2)]*scale[:,None]
    tangent_sigma = torch.linalg.eigvalsh(tangent @ tangent.transpose(1,2))[:,-1].clamp_min(0).sqrt()
    bary = association.barycentric.to(verts.dtype)
    seed_anchor = (target*bary[...,None]).sum(1)
    offset = (seed_anchor-target.mean(1)).norm(dim=1)
    cfg = CONNECTED_SURFACE
    radius = (cfg['sigma_radius']*tangent_sigma+offset+cfg['geodesic_padding_m']).clamp(
        cfg['minimum_radius_m'],cfg['maximum_radius_m'])
    arrays = _connected_patches(faces.cpu().numpy(),verts[reference_slot,faces].double().cpu().numpy(),
                               association,radius.cpu().numpy())
    cand,geo,predecessor = (torch.from_numpy(x).to(faces.device) for x in arrays)
    present = cand>=0
    ids = torch.where(present,cand,association.target_faces[:,None])
    triangles = verts[:,faces[ids]]
    basis,_ = _frame(triangles)
    normals = basis[...,1]
    valid = present[None]&((normals*normals[:,:,:1]).sum(-1)>=cfg['whole_patch_cos'])
    parent_normals = normals.gather(2,predecessor[None,:,:,None].expand(len(verts),-1,-1,3))
    valid &= (normals*parent_normals).sum(-1)>=cfg['edge_curve_cos']
    row = torch.arange(len(cand),device=cand.device)
    # Dijkstra predecessors occur earlier in the sorted list. A folded
    # connection invalidates its descendant branch in that specific pose.
    for j in range(1,cand.shape[1]):
        valid[:,:,j] &= valid[:,row,predecessor[:,j]]
    if not bool(valid[:,:,0].all()):
        raise ValueError('A primary connected facial face is degenerate.')
    empty = verts.new_empty((0,))
    reference = ConnectedSurfaceReference(association,reference_slot,cand,valid,predecessor,geo,radius,triangles,
        owner_basis,owner_scale,source.mean(-2),empty,empty)
    distance,closest = _initial_metrics(model,rows,reference,torch.arange(len(rows),device=rows.device))
    return replace(reference,initial_distances=distance,initial_closest_faces=closest)


def _measure(model,row_ids,reference,step):
    slot = int(model.timestep)
    index = _indices(model,row_ids,reference)
    values,faces = _pose_metrics(model,row_ids,reference,index,slot)
    initial = reference.initial_distances[slot,index]
    goal = initial.new_full((initial.shape[1],),CONNECTED_SURFACE['principal_goal_m'])
    goal[0] = CONNECTED_SURFACE['center_goal_m']
    cap = initial+_schedule(step)*(initial.minimum(goal)-initial)
    end = 1+CONNECTED_SURFACE['principal_directions']
    return dict(center_distance=values[:,0],principal_distances=values[:,1:end],mixed_distances=values[:,end:],
        center_face=faces[:,0],principal_faces=faces[:,1:end],mixed_faces=faces[:,end:],
        primary_face=reference.association.target_faces[index],
        initial_center_distance=initial[:,0],initial_principal_distances=initial[:,1:end],
        initial_mixed_distances=initial[:,end:],center_cap=cap[:,0],principal_caps=cap[:,1:end],mixed_caps=cap[:,end:])


@torch.no_grad()
def connected_surface_metrics(model,row_ids,reference,step=15000):
    return _measure(model,row_ids,reference,step)


def connected_surface_loss(model,row_ids,reference,step):
    if not len(row_ids):
        zero = model._xyz.new_zeros(())
        return zero,dict(center=zero,support=zero)
    values = _measure(model,row_ids,reference,step)

    def penalty(key,limit):
        excess = (values[key]-values[limit]).relu()/CONNECTED_SURFACE['loss_unit_m']
        return F.smooth_l1_loss(excess,torch.zeros_like(excess))

    center = penalty('center_distance','center_cap')
    support = .5*(penalty('principal_distances','principal_caps')+penalty('mixed_distances','mixed_caps'))
    return center+CONNECTED_SURFACE['support_weight']*support,dict(center=center,support=support)


def _project_points(points,triangles,valid):
    """Return exact projected points, using the closest verified finite face."""
    _,choice = _nearest(points,triangles,valid)
    rows = torch.arange(len(points),device=points.device)[:,None]
    a,b,c = triangles[rows,choice].unbind(-2)
    ab,ac,delta = b-a,c-a,points-a
    aa,cc,across = (ab*ab).sum(-1),(ac*ac).sum(-1),(ab*ac).sum(-1)
    pa,pc = (delta*ab).sum(-1),(delta*ac).sum(-1)
    det = aa*cc-across.square()
    u,v = (cc*pa-across*pc)/det.clamp_min(1e-20),(aa*pc-across*pa)/det.clamp_min(1e-20)
    candidates = [a+u[...,None]*ab+v[...,None]*ac]
    for left,right in ((a,b),(b,c),(c,a)):
        edge = right-left
        fraction = ((points-left)*edge).sum(-1)/(edge*edge).sum(-1).clamp_min(1e-20)
        candidates.append(left+fraction.clamp(0,1)[...,None]*edge)
    candidates = torch.stack(candidates,dim=2)
    distances = (candidates-points[:,:,None]).norm(dim=-1)
    inside = (u>=0)&(v>=0)&(u+v<=1)&(det>1e-20)
    distances[...,0] = distances[...,0].masked_fill(~inside,torch.inf)
    best = distances.argmin(dim=2)
    return candidates.gather(2,best[...,None,None].expand(-1,-1,1,3)).squeeze(2),choice


@torch.no_grad()
def propose_surface_split(model,parent_rows,reference,*,tangent_shift=.5):
    """Propose two surface-projected children with inherited owners and moments."""
    from roma import quat_xyzw_to_wxyz, rotmat_to_unitquat

    if not len(parent_rows) or not 0<tangent_shift<1:
        raise ValueError('Connected surface split needs nonempty parents and a bounded shift.')
    index = _indices(model,parent_rows,reference)
    slot = reference.reference_slot
    triangles = reference.triangles[slot,index]
    valid = reference.candidate_valid[slot,index]
    owner = reference.owner_basis[slot,index]
    origin = reference.owner_center[slot,index]
    scale = reference.owner_scale[slot,index]
    local = model._xyz[parent_rows]
    logs = model._scaling[parent_rows]
    quaternion = model._rotation[parent_rows]
    world = origin+(owner@local[...,None]).squeeze(-1)*scale[:,None]
    near,face_id = _project_points(world[:,None],triangles,valid)
    nearest = near[:,0]
    selected = triangles[torch.arange(len(index),device=index.device),face_id[:,0]]
    plane,_ = _frame(selected)
    rotation = _rotations(quaternion)
    axes = (owner @ rotation)*(logs.exp()*scale[:,None])[:,None]
    tangents = plane[:,:,(0,2)]
    projected = tangents.transpose(1,2)@axes
    eig,vectors = torch.linalg.eigh(projected@projected.transpose(1,2))
    tangent_sigma = eig[:,-1].clamp_min(0).sqrt()
    direction = (tangents@vectors[:,:,-1,None]).squeeze(-1)
    offset = direction*(tangent_shift*tangent_sigma[:,None])
    raw_world = torch.stack((nearest-offset,nearest+offset),dim=1)
    centers_world,chosen = _project_points(raw_world,triangles,valid)
    centers = ((owner.transpose(1,2)@(centers_world-origin[:,None]).transpose(1,2)).transpose(1,2)/
               scale[:,None,None])
    separation = (centers_world[:,1]-centers_world[:,0]).norm(dim=1)

    # Two equally weighted children reproduce the second moment around the
    # new midpoint if subtracting their between-center covariance is feasible.
    widths_local = logs.double().exp()
    axes_local = _rotations(quaternion.double())*widths_local[:,None]
    covariance = axes_local @ axes_local.transpose(1,2)
    delta = (centers[:,1]-centers[:,0]).double()*.5
    child_cov = covariance-delta[...,None]*delta[:,None]
    eig,vectors = torch.linalg.eigh(child_cov)
    cfg = CONNECTED_SURFACE
    variance = widths_local.square()
    floor = (variance.amax(dim=1)*cfg['relative_covariance_floor']).minimum(variance.amin(dim=1)*.5)
    floor = floor.clamp_min(torch.finfo(variance.dtype).tiny)
    positive = eig[:,0]>floor
    widths = (eig.maximum(floor[:,None]).log()*.5).to(logs.dtype)
    sign = torch.linalg.det(vectors).sign()
    vectors = vectors*torch.stack((torch.ones_like(sign),torch.ones_like(sign),sign),dim=1)[:,None]
    new_quaternion = quat_xyzw_to_wxyz(rotmat_to_unitquat(vectors)).to(quaternion.dtype)
    max_replay = separation.new_zeros(len(index))
    support = reference.association.support_views[index]
    for column,pose in enumerate(reference.association.source_slots.tolist()):
        if not bool(support[:,column].any()):
            continue
        basis = reference.owner_basis[pose,index]
        point = reference.owner_center[pose,index,None]+(
            basis@centers.transpose(1,2)).transpose(1,2)*reference.owner_scale[pose,index,None,None]
        distance,_ = _nearest(point,reference.triangles[pose,index],reference.candidate_valid[pose,index])
        max_replay = torch.maximum(max_replay,distance.amax(dim=1).masked_fill(~support[:,column],0.))
    eligible = positive & (separation>=cfg['min_sibling_separation_m']) & (
        max_replay<=cfg['max_split_replay_distance_m']) & torch.isfinite(centers).flatten(1).all(1)
    faces = reference.candidate_faces[index].gather(1,chosen)
    return SurfaceSplitSeeds(centers,widths[:,None].expand(-1,2,-1),
        new_quaternion[:,None].expand(-1,2,-1),eligible,faces,max_replay,tangent_sigma,separation)
