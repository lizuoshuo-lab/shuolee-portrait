"""Freeze native-observed triangle associations without changing SMPL-X ownership."""

from dataclasses import dataclass, fields

import numpy as np
import torch


ASSOCIATION = dict(neighbor_rings=2, minimum_source_bins=2, normal_agreement_cos=.82,
                   min_view_cos=.25, max_anchor_depth_error_m=.008, max_center_depth_error_m=.035,
                   max_proxy_distance_m=.030, owner_tie_distance_m=.0005,
                   owner_good_distance_m=.0015, owner_good_required_improvement_m=.001,
                   ambiguity_anchor_gap_m=.003)


@dataclass(frozen=True)
class SurfaceView:
    slot: int
    source_bin: int
    K: np.ndarray
    w2c: np.ndarray
    box: tuple
    skin: np.ndarray
    mesh_depth: np.ndarray
    excluded: np.ndarray | None = None
    frame: str = ''


@dataclass(frozen=True)
class SurfaceAssociation:
    stable_ids: torch.Tensor
    owner_faces: torch.Tensor
    target_faces: torch.Tensor
    barycentric: torch.Tensor
    support_views: torch.Tensor
    source_slots: torch.Tensor
    source_bins: torch.Tensor

    def arrays(self):
        return {field.name: getattr(self, field.name).detach().cpu().numpy() for field in fields(self)}

    @classmethod
    def from_arrays(cls, mapping):
        names = tuple(field.name for field in fields(cls))
        if set(mapping) != set(names):
            raise ValueError('Unexpected surface association fields.')
        values = {name: torch.as_tensor(mapping[name]) for name in names}
        ids, owner, target, bary, support, slots, bins = values.values()
        count = len(ids)
        if not (ids.dtype == owner.dtype == target.dtype == slots.dtype == bins.dtype == torch.long
                and ids.ndim == 1 and owner.shape == target.shape == ids.shape
                and slots.ndim == 1 and bins.shape == slots.shape and len(slots.unique()) == len(slots)
                and bary.shape == (count, 3) and bary.dtype == torch.float32
                and support.shape == (count, len(slots)) and support.dtype == torch.bool
                and bool((ids[1:] > ids[:-1]).all()) and bool(torch.isfinite(bary).all())
                and bool((bary >= 0).all()) and bool(torch.allclose(bary.sum(1), bary.new_ones(count)))
                and not any(bool((value < 0).any()) for value in (ids, owner, target, slots, bins))):
            raise ValueError('Invalid stable surface association arrays.')
        groups = [support[:, bins == source].any(1) for source in bins.unique()]
        if count and (not groups or not bool((torch.stack(groups).sum(0) >= ASSOCIATION['minimum_source_bins']).all())):
            raise ValueError('A saved association lacks independent native skin support.')
        return cls(**values)

    def to(self, device):
        return SurfaceAssociation(**{field.name: getattr(self, field.name).to(device) for field in fields(self)})

    def indices(self, ids):
        if not len(self.stable_ids):
            return torch.zeros_like(ids), torch.zeros_like(ids, dtype=torch.bool)
        lookup = self.stable_ids
        positions = torch.searchsorted(lookup, ids.contiguous()).clamp_max(len(lookup)-1)
        return positions, lookup[positions] == ids

    def mask(self, ids):
        """Return a live row mask, including after unrelated topology changes."""
        return self.indices(ids)[1]


def _normals(triangles):
    first, second = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    vector = np.cross(first, second)
    size = np.linalg.norm(vector, axis=1)
    return np.divide(vector, size[:, None], out=np.zeros_like(vector), where=size[:, None] > 1e-10)


def _topological_candidates(faces, normals, owners):
    incidence = {}
    for face, (a, b, c) in enumerate(faces.tolist()):
        for u, v in ((a,b), (b,c), (c,a)):
            incidence.setdefault((min(u,v),max(u,v)), []).append(face)
    adjacency = [[] for _ in faces]
    for neighbors in incidence.values():
        if len(neighbors) == 2:
            a, b = neighbors
            if normals[a] @ normals[b] >= ASSOCIATION['normal_agreement_cos']:
                adjacency[a].append(b)
                adjacency[b].append(a)
    result = {}
    for owner in np.unique(owners):
        seen = {int(owner)}
        frontier = seen.copy()
        for _ in range(ASSOCIATION['neighbor_rings']):
            frontier = {candidate for current in frontier for candidate in adjacency[current]
                        if candidate not in seen and normals[candidate] @ normals[owner] >=
                        ASSOCIATION['normal_agreement_cos']}
            seen.update(frontier)
        result[owner] = sorted(seen)
    return result


def _closest_barycentric(point, triangle):
    """Exact Euclidean closest point including the three edge segments."""
    a, b, c = np.moveaxis(triangle, 1, 0)
    ab, ac, ap = b-a, c-a, point-a
    aa, bb = (ab*ab).sum(1), (ac*ac).sum(1)
    abac = (ab*ac).sum(1)
    pa, pc = (ap*ab).sum(1), (ap*ac).sum(1)
    determinant = aa*bb-abac**2
    safe = np.where(determinant > 1e-20, determinant, 1.)
    u, v = (bb*pa-abac*pc)/safe, (aa*pc-abac*pa)/safe
    plane = np.column_stack((1-u-v,u,v))
    choices = [plane]
    for start, end in ((0,1),(1,2),(2,0)):
        edge = triangle[:,end]-triangle[:,start]
        norm2 = (edge*edge).sum(1)
        t = ((point-triangle[:,start])*edge).sum(1)/np.maximum(norm2, 1e-20)
        weights = np.zeros_like(plane)
        weights[:,start], weights[:,end] = 1-t.clip(0,1), t.clip(0,1)
        choices.append(weights)
    weights = np.stack(choices, axis=1)
    projected = weights @ triangle
    distance = np.linalg.norm(projected-point[:,None], axis=2)
    distance[:,0] = np.where((plane>=0).all(1)&(determinant>1e-20), distance[:,0], np.inf)
    best = distance.argmin(1)
    rows = np.arange(len(point))
    return weights[rows,best], distance[rows,best]


def _native_samples(world, view, image, excluded, depth):
    camera = world @ view.w2c[:3,:3].T + view.w2c[:3,3]
    projected = camera @ view.K.T
    z = camera[:,2]
    xy = projected[:,:2]/np.where(z>.2, z, 1.)[:,None]
    x0,y0,x1,y1 = view.box
    valid = (z>.2)&(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)
    xy = np.floor(xy).astype(np.int64)
    x, y = (xy[:,0]-x0).clip(0,image.shape[1]-1), (xy[:,1]-y0).clip(0,image.shape[0]-1)
    supported = valid & image[y,x]
    gap = np.full_like(z,np.inf)
    gap[supported] = np.abs(z[supported]-depth[y[supported],x[supported]])
    return supported, valid & excluded[y,x], gap


def _view_arrays(view):
    x0,y0,x1,y1 = view.box
    skin = np.asarray(view.skin)
    depth = np.asarray(view.mesh_depth)
    excluded = np.zeros_like(skin) if view.excluded is None else np.asarray(view.excluded)
    if not (x0>=0 and y0>=0 and x1>x0 and y1>y0 and skin.dtype==excluded.dtype==np.bool_
            and skin.shape==excluded.shape==depth.shape==(y1-y0,x1-x0)
            and np.isfinite(view.K).all() and np.isfinite(view.w2c).all()
            and view.K.shape==(3,3) and view.w2c.shape==(4,4) and np.allclose(view.K[2],[0,0,1])
            and np.allclose(view.w2c[3],[0,0,0,1]) and not np.any(skin&excluded)
            and not np.isnan(depth[skin]).any() and np.all(depth[skin]>0)):
        raise ValueError(f'Invalid native evidence crop for pose {view.slot}.')
    return skin,excluded,depth


def _stats(values):
    return {name:float(value) for name,value in zip(('p50','p95','max'),
            np.quantile(values,(.5,.95,1.)),strict=True)} if len(values) else None


def build_surface_association(vertices_sequence, faces, stable_ids, owner_faces, world_centers,
                              trusted_rows, views, reference_slot):
    """Select a supported nearby frozen mesh proxy using only registered native views."""
    vertices = np.asarray(vertices_sequence, dtype=np.float64)
    faces, stable, owner, trusted = map(np.asarray,(faces,stable_ids,owner_faces,trusted_rows))
    centers = np.asarray(world_centers, dtype=np.float64)
    views = tuple(views)
    slots = np.asarray([view.slot for view in views],dtype=np.int64)
    bins = np.asarray([view.source_bin for view in views],dtype=np.int64)
    if not (vertices.ndim==3 and vertices.shape[2]==3 and centers.shape==(len(vertices),len(stable),3)
            and np.isfinite(vertices).all() and np.isfinite(centers).all()
            and faces.ndim==2 and faces.shape[1]==3 and faces.dtype.kind in 'iu'
            and len(faces)>0 and faces.min()>=0 and faces.max()<vertices.shape[1]
            and stable.ndim==1 and stable.dtype==np.int64 and len(np.unique(stable))==len(stable)
            and owner.shape==trusted.shape==stable.shape and owner.dtype.kind in 'iu' and trusted.dtype==np.bool_
            and np.all(stable>=0) and np.all(owner>=0) and np.all(owner<len(faces))
            and 0<=reference_slot<len(vertices) and len(np.unique(slots))==len(slots)
            and np.all(slots>=0) and np.all(slots<len(vertices)) and np.all(bins>=0)):
        raise ValueError('Surface association inputs have invalid topology, poses or stable row identities.')
    cfg = ASSOCIATION
    rows = np.flatnonzero(trusted)
    rows = rows[np.argsort(stable[rows])]
    ref_triangles = vertices[reference_slot,faces]
    normals = _normals(ref_triangles)
    neighbors = _topological_candidates(faces,normals,owner[rows])
    candidates = np.asarray([face for row in rows for face in neighbors[owner[row]]],dtype=np.int64)
    counts = np.asarray([len(neighbors[owner[row]]) for row in rows],dtype=np.int64)
    candidate_rows = np.repeat(rows,counts)
    offsets = np.r_[0,np.cumsum(counts)]
    ref_centers = centers[reference_slot,candidate_rows]
    bary, ref_distance = _closest_barycentric(ref_centers,ref_triangles[candidates])
    ref_anchors = np.einsum('ni,nij->nj',bary,ref_triangles[candidates])
    possible = (ref_distance<=cfg['max_proxy_distance_m']) & (np.linalg.norm(normals[candidates],axis=1)>.5)
    support = np.zeros((len(candidates),len(views)),bool)
    conflict = np.zeros(len(candidates),bool)
    distances = np.zeros((len(candidates),len(views)))
    owner_distances = np.zeros_like(distances)
    view_receipts = []
    for column,view in enumerate(views):
        image,excluded,depth = _view_arrays(view)
        tri = vertices[view.slot,faces[candidates]]
        anchor = np.einsum('ni,nij->nj',bary,tri)
        center = centers[view.slot,candidate_rows]
        target_n = _normals(tri)
        owner_n = _normals(vertices[view.slot,faces[owner[candidate_rows]]])
        cam_position = -view.w2c[:3,:3].T @ view.w2c[:3,3]
        direction = cam_position-anchor
        angle = (target_n*direction).sum(1)/np.maximum(np.linalg.norm(direction,axis=1),1e-12)
        anchor_skin,anchor_excluded,anchor_gap = _native_samples(anchor,view,image,excluded,depth)
        center_skin,center_excluded,center_gap = _native_samples(center,view,image,excluded,depth)
        proxy_facing = possible & (angle>=cfg['min_view_cos']) & ((target_n*owner_n).sum(1)>=
            cfg['normal_agreement_cos'])
        near_anchor = anchor_gap<=cfg['max_anchor_depth_error_m']
        near_center = center_gap<=cfg['max_center_depth_error_m']
        # Depth exists only on skin. A signed non-skin endpoint vetoes a candidate
        # when the other endpoint was independently projected near visible skin.
        conflict |= proxy_facing & ((near_center&anchor_excluded)|(near_anchor&center_excluded))
        distance = np.linalg.norm(anchor-center,axis=1)
        distances[:,column] = distance
        _,owner_distances[:,column] = _closest_barycentric(center,vertices[view.slot,faces[owner[candidate_rows]]])
        support[:,column] = proxy_facing & near_anchor & near_center & anchor_skin & center_skin & (
            distance<=cfg['max_proxy_distance_m'])
        view_receipts.append(dict(frame=view.frame,slot=view.slot,source_bin=view.source_bin,
                                  signed_skin_pixels=int(image.sum()),accepted_candidate_samples=int(support[:,column].sum())))
    groups = [support[:,bins==source].any(1) for source in np.unique(bins)]
    vote_count = np.stack(groups).sum(0) if groups else np.zeros(len(candidates),dtype=np.int64)
    qualified = (vote_count>=cfg['minimum_source_bins'])&~conflict
    median = np.full(len(candidates),np.inf)
    for index in np.flatnonzero(qualified):
        median[index] = np.median(distances[index,support[index]])
    selected,ambiguous = [],0
    for row,start,stop in zip(rows,offsets[:-1],offsets[1:],strict=True):
        indices = np.flatnonzero(qualified[start:stop])+start
        if not len(indices):
            continue
        best = min(indices,key=lambda i:(median[i],-vote_count[i],ref_distance[i],candidates[i]))
        near = indices[median[indices]<=median[best]+cfg['owner_tie_distance_m']]
        is_distinct = ((normals[candidates[near]] @ normals[candidates[best]])<cfg['normal_agreement_cos']) | (
            np.linalg.norm(ref_anchors[near]-ref_anchors[best],axis=1)>cfg['ambiguity_anchor_gap_m'])
        if is_distinct.any():
            ambiguous += 1
            continue
        original = indices[candidates[indices]==owner[row]]
        if len(original) and candidates[best]!=owner[row]:
            supported = support[best]
            owner_distance = owner_distances[best,supported]
            required = (cfg['owner_good_required_improvement_m'] if
                        np.median(owner_distance)<=cfg['owner_good_distance_m'] else cfg['owner_tie_distance_m'])
            if np.median(owner_distance-distances[best,supported]) < required:
                best = original[0]
        selected.append(int(best))
    chosen = np.asarray(selected,dtype=np.int64)
    chosen_rows = candidate_rows[chosen]
    association = SurfaceAssociation.from_arrays(dict(
        stable_ids=stable[chosen_rows],owner_faces=owner[chosen_rows].astype(np.int64),target_faces=candidates[chosen],
        barycentric=bary[chosen].astype(np.float32),support_views=support[chosen],source_slots=slots,source_bins=bins))
    owner_unchanged = candidates[chosen]==owner[chosen_rows]
    matched = support[chosen]
    receipt = dict(schema='shuolee-native-face-proxy-association/1',policy=cfg.copy(),
        eligible_trusted_rows=len(rows),candidate_faces=len(candidates),reference_slot=int(reference_slot),
        excluded_material_candidates=int(conflict.sum()),ambiguous_rows=ambiguous,associated_rows=len(chosen),
        original_owner_rows=int(owner_unchanged.sum()),neighbor_surface_rows=int((~owner_unchanged).sum()),
        native_observations=int(matched.sum()),source_views=view_receipts,
        matched_view_proxy_distance_mm=dict(owner_exact_closest=_stats(owner_distances[chosen][matched]*1000.),
            target_frozen_anchor=_stats(distances[chosen][matched]*1000.)),
        scope='This is a local association to the frozen fitted mesh proxy with corroborating native material and '
              'visibility. It does not rebind any Gaussian or establish a directly measured physical surface.')
    return association,receipt
