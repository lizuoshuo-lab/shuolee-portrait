"""Shared garment surfels and topology-limited articulated Gaussian transport."""

from copy import copy
from itertools import combinations

import numpy as np
import torch
from roma import quat_xyzw_to_wxyz, rotmat_to_unitquat
from scene.clothing_gaussian_model import ClothingGaussianModel
from torch import nn
from torch.nn import functional as F
from utils.graphics_utils import compute_face_orientation

from .free_clothing_torch import normalized_quaternion_matrix, quaternion_product


VOXEL_M = .012
NEIGHBOR_RADIUS_M = .03
FACE_HOPS = 2
ROW_NEIGHBORS = 4
GRAPH_NEIGHBORS = 8
NORMAL_MIN_SIGMA_M = .003
ARRAY_NAMES = ('node_rest_xyz', 'node_centers', 'node_owner_faces', 'node_normal_neighbors',
               'graph_edges', 'row_node_ids', 'row_node_weights', 'face_groups')
FLOAT_ARRAYS = {'node_rest_xyz', 'node_centers', 'row_node_weights'}
RAW_PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')


def _neighborhoods(faces, groups, owners):
    """Only traverse shared edges whose two incident faces support the garment."""
    adjacent = {int(face): set() for face in np.unique(owners)}
    edge_owner = {}
    for face in adjacent:
        for a, b in combinations(sorted(faces[face]), 2):
            key = (int(groups[face]), int(a), int(b))
            for previous in edge_owner.setdefault(key, []):
                adjacent[face].add(previous)
                adjacent[previous].add(face)
            edge_owner[key].append(face)
    result = {}
    for face in adjacent:
        reached, wave = {face}, {face}
        for _ in range(FACE_HOPS):
            wave = set().union(*(adjacent[i] for i in wave))-reached
            reached.update(wave)
        result[face] = tuple(sorted(reached))
    return result


def _rows_by_owner(owners):
    order = np.argsort(owners, kind='stable')
    unique, first = np.unique(owners[order], return_index=True)
    return dict(zip(unique.tolist(), np.split(order, first[1:]), strict=True))


def _candidate_nodes(node_owner, neighborhoods):
    rows = _rows_by_owner(node_owner)
    empty = np.empty(0, dtype=np.int64)
    return {face: np.concatenate([rows.get(other, empty) for other in neighbors])
            for face, neighbors in neighborhoods.items()}


def _nearest(xyz, target, limit):
    distance = np.sum((xyz[:, None]-target[None, :])**2, axis=-1)
    slots = np.argsort(distance, axis=1, kind='stable')[:, :limit]
    return slots, np.take_along_axis(distance, slots, axis=1)


def _surface_graph(xyz, opacity, faces, groups, owners, body_normals):
    neighborhoods = _neighborhoods(faces, groups, owners)
    source_rows = _rows_by_owner(owners)
    # Choose the most opaque representative in each anatomy-separated voxel.
    keys = np.column_stack((groups[owners], np.floor(xyz/VOXEL_M).astype(np.int64)))
    rank = np.argsort(-opacity, kind='stable')
    _, first = np.unique(keys[rank], axis=0, return_index=True)
    seeds = rank[first].tolist()
    seed_xyz = xyz[seeds]
    candidates = _candidate_nodes(owners[seeds], neighborhoods)
    for face, rows in source_rows.items():
        neighbors = candidates[face]
        missing = rows
        if len(neighbors):
            _, distances = _nearest(xyz[rows], seed_xyz[neighbors], 1)
            missing = rows[distances[:, 0] > NEIGHBOR_RADIUS_M**2]
        if len(missing):
            missing = missing[np.argsort(-opacity[missing], kind='stable')]
            _, first = np.unique(keys[missing], axis=0, return_index=True)
            seeds.extend(missing[first].tolist())
    seeds = np.array(sorted(set(seeds)), dtype=np.int64)
    rest, node_owner = xyz[seeds].copy(), owners[seeds].copy()
    candidates = _candidate_nodes(node_owner, neighborhoods)
    row_ids = np.empty((len(xyz), ROW_NEIGHBORS), dtype=np.int64)
    row_weights = np.zeros(row_ids.shape, dtype=xyz.dtype)
    for face, rows in source_rows.items():
        pool = candidates[face]
        closest, distance = _nearest(xyz[rows], rest[pool], ROW_NEIGHBORS)
        count = closest.shape[1]
        row_ids[rows] = pool[closest[:, :1]]
        row_ids[rows, :count] = pool[closest]
        keep = distance <= NEIGHBOR_RADIUS_M**2
        if not keep[:, 0].all():
            raise RuntimeError('A garment row is outside all legal spatial graph support.')
        weights = np.exp(-.5*(distance-distance[:, :1])/VOXEL_M**2)*keep
        row_weights[rows, :count] = weights/weights.sum(axis=1, keepdims=True)

    normal_neighbors = np.full((len(rest), 2), -1, dtype=np.int64)
    edges = set()
    for face, nodes in _rows_by_owner(node_owner).items():
        pool = candidates[face]
        closest, distances = _nearest(rest[nodes], rest[pool], GRAPH_NEIGHBORS+1)
        for node, near, distance in zip(nodes, pool[closest], distances, strict=True):
            near = near[(distance >= .001**2) & (distance <= NEIGHBOR_RADIUS_M**2)][:GRAPH_NEIGHBORS]
            edges.update(tuple(sorted((int(node), int(other)))) for other in near)
            if len(near) < 2:
                continue
            left, right = np.triu_indices(len(near), k=1)
            delta = rest[near]-rest[node]
            cross = np.cross(delta[left], delta[right])
            lengths = np.linalg.norm(delta, axis=1)
            sine = np.linalg.norm(cross, axis=1)/(lengths[left]*lengths[right])
            # Favor noncollinear, spatially local neighborhoods.
            scores = sine/(lengths[left]+lengths[right])
            best = scores.argmax()
            if sine[best] >= .2:
                pair = near[[left[best], right[best]]]
                normal_neighbors[node] = pair if cross[best]@body_normals[face] >= 0 else pair[::-1]
    return dict(node_rest_xyz=rest, node_centers=rest.copy(), node_owner_faces=node_owner,
                node_normal_neighbors=normal_neighbors, graph_edges=np.array(sorted(edges), dtype=np.int64).reshape(-1, 2),
                row_node_ids=row_ids, row_node_weights=row_weights, face_groups=groups)


class GarmentSurfaceModel(ClothingGaussianModel):
    """Free garment appearance around a jointly learned, shared surface graph."""

    def __init__(self, sh_degree=0):
        super().__init__(sh_degree)
        self.node_centers = None
        self._cached_garment_pose = None

    @torch.no_grad()
    def configure_garment_surface(self, *, face_groups=None, graph=None):
        if self.support_face_ids is None or self.max_sh_degree != 0 or (face_groups is None) == (graph is None):
            raise ValueError('Initialize triangle motion and provide either face anatomy or a stored SH0 garment graph.')
        dev, dtype = self._xyz.device, self._xyz.dtype
        faces = self.faces.detach().cpu().numpy()
        owners = self.support_face_ids.detach().cpu().numpy()
        body_basis, _ = compute_face_orientation(self.reference_vertices, self.faces, return_scale=True)
        if graph is None:
            groups = np.asarray(torch.as_tensor(face_groups).detach().cpu())
            if groups.dtype.kind not in 'iu' or groups.shape != (len(faces),) or np.any(groups < 0):
                raise ValueError('Each frozen body face requires a nonnegative integer anatomy group.')
            graph = _surface_graph(self._xyz.detach().cpu().numpy(), self.get_opacity.detach().cpu().numpy().ravel(),
                                   faces, groups.astype(np.int64), owners, body_basis[:, :, 1].cpu().numpy())
        arrays = self._checked_graph(graph, faces, owners, dtype)
        for name, value in arrays.items():
            tensor = torch.tensor(value, device=dev, dtype=dtype if name in FLOAT_ARRAYS else torch.long)
            setattr(self, name, nn.Parameter(tensor) if name == 'node_centers' else tensor)
        self._normal_fallback = body_basis[self.node_owner_faces, :, 1].detach()
        self._node_has_normal = (self.node_normal_neighbors >= 0).all(dim=1)
        self._xyz.requires_grad_(False)
        self._build_node_motion()
        self.select_mesh_by_timestep(self.reference_frame)

    @staticmethod
    def _checked_graph(graph, faces, owners, dtype):
        if set(graph) != set(ARRAY_NAMES):
            raise ValueError('The garment graph arrays differ from the executable contract.')
        arrays = {name: np.asarray(value) for name, value in graph.items()}
        expected_dtype = np.float32 if dtype == torch.float32 else np.float64
        if any(arrays[name].dtype != (expected_dtype if name in FLOAT_ARRAYS else np.int64) for name in ARRAY_NAMES):
            raise ValueError('Garment graph dtypes differ from the reference PLY or int64 identities.')
        rest, live, node_owner, normals, edges, row_ids, weights, groups = [arrays[name] for name in ARRAY_NAMES]
        count = len(rest)
        if (not count or rest.shape != (count, 3) or live.shape != rest.shape
                or not np.isfinite(rest).all() or not np.isfinite(live).all()
                or node_owner.shape != (count,) or np.any((node_owner < 0) | (node_owner >= len(faces)))
                or groups.shape != (len(faces),) or np.any(groups < 0)
                or normals.shape != (count, 2) or np.any((normals < -1) | (normals >= count))
                or np.any((normals[:, 0] < 0) != (normals[:, 1] < 0))
                or np.any((normals[:, 0] >= 0) & (normals[:, 0] == normals[:, 1]))
                or np.any(normals == np.arange(count)[:, None])
                or edges.ndim != 2 or edges.shape[1] != 2 or np.any((edges < 0) | (edges >= count))
                or np.any(edges[:, 0] >= edges[:, 1])
                or row_ids.shape != (len(owners), ROW_NEIGHBORS) or weights.shape != row_ids.shape
                or np.any((row_ids < 0) | (row_ids >= count)) or not np.isfinite(weights).all()
                or np.any(weights < 0) or np.any(weights[:, 0] <= 0)
                or not np.allclose(weights.sum(1), 1., rtol=0, atol=1e-6)):
            raise ValueError('Invalid shared garment graph geometry, normals, or row interpolation.')
        neighbors = _neighborhoods(faces, groups, np.r_[owners, node_owner])
        for owner, rows in _rows_by_owner(owners).items():
            related = node_owner[row_ids[rows]][weights[rows] > 0]
            if not np.isin(related, neighbors[owner]).all():
                raise ValueError('Garment interpolation crossed a protected anatomy or topology boundary.')
        links = np.r_[edges, np.column_stack((np.repeat(np.arange(count), 2), normals.ravel()))]
        links = links[links[:, 1] >= 0]
        for first, second in np.unique(node_owner[links], axis=0):
            if second not in neighbors[first]:
                raise ValueError('A shared garment surface edge crosses a protected support boundary.')
        return arrays

    @torch.no_grad()
    def _build_node_motion(self):
        owners, slots = self.node_owner_faces.unique(sorted=True, return_inverse=True)
        topology = self.faces[owners]
        verts = self.vertices_sequence
        basis, lengths = compute_face_orientation(verts, topology, return_scale=True)
        basis = basis.double()
        ref = self.reference_frame
        rotation = basis @ basis[ref].transpose(-1, -2)
        rotation[ref] = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        length = lengths.squeeze(-1).clamp_min(1e-8)
        scales = length/length[ref]
        quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(rotation)).to(dtype=verts.dtype)
        origins = verts[:, topology].mean(dim=2)
        if not (torch.isfinite(quat).all() and torch.isfinite(scales).all() and (scales > 0).all()):
            raise ValueError('An invalid stock face similarity entered the garment motion field.')
        self._node_face_slots, self._node_motion_origins = slots, origins
        self._node_motion_quat, self._node_motion_scale = quat, scales

    def graph_arrays(self):
        return {name: getattr(self, name).detach().cpu().numpy().copy() for name in ARRAY_NAMES}

    def select_mesh_by_timestep(self, timestep, original=False):
        super().select_mesh_by_timestep(timestep, original)
        self._cached_garment_pose = None

    def _interpolate(self, nodes):
        return (nodes[self.row_node_ids]*self.row_node_weights[..., None]).sum(dim=1)

    def canonical_centers(self):
        return self._xyz+self._interpolate(self.node_centers-self.node_rest_xyz)

    def _pose_field(self):
        if self._cached_garment_pose is None:
            frame, ref, slots = self.timestep, self.reference_frame, self._node_face_slots
            quat = self._node_motion_quat[frame, slots]
            scale = self._node_motion_scale[frame, slots]
            node_rotation = normalized_quaternion_matrix(quat)
            rotations = quat[self.row_node_ids]
            signs = torch.where((rotations*rotations[:, :1]).sum(-1, keepdim=True) < 0, -1., 1.)
            row_quat = F.normalize((rotations*signs*self.row_node_weights[..., None]).sum(1), dim=1)
            row_scale = self._interpolate(scale[:, None])
            row_rotation = normalized_quaternion_matrix(row_quat)
            origin = self._node_motion_origins[frame, slots]
            ref_origin = self._node_motion_origins[ref, slots]
            row_rest_origin = self._interpolate(self.node_rest_xyz)
            self._cached_garment_pose = (node_rotation, scale[:, None], origin, ref_origin,
                                         row_quat, row_scale, row_rotation, row_rest_origin)
        return self._cached_garment_pose

    @property
    def get_xyz(self):
        if self.node_centers is None:
            return super().get_xyz
        if self.timestep == self.reference_frame:
            return self.canonical_centers()
        rotation, scale, origin, ref, _, row_scale, row_rotation, row_rest_origin = self._pose_field()
        nodes = origin+torch.einsum('nij,nj->ni', rotation, self.node_centers-ref)*scale
        offset = torch.einsum('nij,nj->ni', row_rotation, self._xyz-row_rest_origin)*row_scale
        return self._interpolate(nodes)+offset

    @property
    def get_rotation(self):
        if self.node_centers is None:
            return super().get_rotation
        quat = F.normalize(self._rotation, dim=1)
        if self.timestep != self.reference_frame:
            quat = F.normalize(quaternion_product(self._pose_field()[4], quat), dim=1)
        return quat

    @property
    def get_scaling(self):
        if self.node_centers is None:
            return super().get_scaling
        scale = self._scaling.exp()
        if self.timestep != self.reference_frame:
            scale = scale*self._pose_field()[5]
        return scale

    def get_covariance(self, scaling_modifier=1):
        if self.node_centers is None:
            return super().get_covariance(scaling_modifier)
        axes = normalized_quaternion_matrix(self.get_rotation)*(self.get_scaling*scaling_modifier)[:, None, :]
        covariance = axes@axes.transpose(-1, -2)
        return covariance[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]

    def garment_normals(self):
        """Canonical normals from learned garment nodes, with an unsupported fallback."""
        pair = self.node_centers[self.node_normal_neighbors.clamp_min(0)]-self.node_centers[:, None]
        cross = torch.linalg.cross(pair[:, 0], pair[:, 1], dim=1)
        valid = self._node_has_normal & (cross.norm(dim=1)>1e-8)
        normals = torch.where(valid[:, None], F.normalize(cross, dim=1), self._normal_fallback)
        return F.normalize(self._interpolate(normals), dim=1)

    def garment_regularization(self, visible=None):
        delta = self.node_centers-self.node_rest_xyz
        zero = delta.sum()*0.
        edges = self.graph_edges
        smooth = F.smooth_l1_loss(delta[edges[:, 0]]/.01, delta[edges[:, 1]]/.01) if len(edges) else zero
        valid = self._interpolate(self._node_has_normal[:, None].to(dtype=delta.dtype)).squeeze(1) > .5
        valid = valid & (self.get_opacity.detach().squeeze(1) >= .05)
        if visible is not None:
            valid = valid & visible
        normal = self.garment_normals()[valid]
        sigma = self._scaling[valid].exp()
        axes = normalized_quaternion_matrix(F.normalize(self._rotation[valid], dim=1))
        dot = torch.einsum('ni,nij->nj', normal, axes)
        count = valid.count_nonzero().clamp_min(1)
        normal_sigma = (dot*sigma).norm(dim=1)
        thickness = F.smooth_l1_loss((normal_sigma/NORMAL_MIN_SIGMA_M-1.).relu(),
                                     torch.zeros_like(normal_sigma), reduction='sum')/count
        ordered, axis = sigma.detach().sort(dim=1)
        sheet = ordered[:, 0] < .8*ordered[:, 1]
        thin_dot = dot.gather(1, axis[:, :1]).squeeze(1)
        tangent = ((1.-thin_dot.square()).clamp_min(0)*sheet).sum()/sheet.count_nonzero().clamp_min(1)
        return dict(smooth=smooth, tangent=tangent, thickness=thickness)

    @torch.no_grad()
    def apply_graph_limits(self, max_displacement_m):
        if not np.isfinite(max_displacement_m) or max_displacement_m <= 0:
            raise ValueError('The shared garment displacement limit must be finite and positive.')
        delta = self.node_centers-self.node_rest_xyz
        fraction = max_displacement_m/delta.norm(dim=1, keepdim=True).clamp_min(max_displacement_m)
        self.node_centers.copy_(self.node_rest_xyz+delta*fraction)

    def preview_rows(self, source_rows, parameters):
        """Temporary executable row layout. Node geometry and the live Adam are isolated."""
        preview = copy(self)
        preview.optimizer = None
        preview.node_centers = self.node_centers.detach()
        for name in RAW_PARAMETERS:
            value = parameters[name] if name in parameters else getattr(self, name)[source_rows].detach()
            setattr(preview, name, value)
        preview.support_face_ids = self.support_face_ids[source_rows]
        preview.densification_ids = self.densification_ids[source_rows]
        preview.row_node_ids = self.row_node_ids[source_rows]
        preview.row_node_weights = self.row_node_weights[source_rows]
        preview.select_mesh_by_timestep(self.timestep)
        return preview

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                              new_scaling, new_rotation, *, constraint_parents=None):
        if self.node_centers is None or not len(new_xyz):
            return super().densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,
                                                 new_scaling, new_rotation, constraint_parents=constraint_parents)
        old_ids, old_weights = self.row_node_ids, self.row_node_weights
        if (not isinstance(constraint_parents, torch.Tensor) or constraint_parents.dtype != torch.long
                or constraint_parents.shape != (len(new_xyz),) or constraint_parents.device != old_ids.device
                or constraint_parents.min() < 0 or constraint_parents.max() >= len(old_ids)):
            raise ValueError('Every split row needs a live parent garment stencil.')
        self.row_node_ids = torch.cat((old_ids, old_ids[constraint_parents]))
        self.row_node_weights = torch.cat((old_weights, old_weights[constraint_parents]))
        self._cached_garment_pose = None
        try:
            result = super().densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities,
                                                  new_scaling, new_rotation, constraint_parents=constraint_parents)
        except Exception:
            if len(self._xyz) == len(old_ids):
                self.row_node_ids, self.row_node_weights = old_ids, old_weights
            raise
        self._xyz.requires_grad_(False)
        return result

    def prune_points(self, mask):
        result = super().prune_points(mask)
        if self.node_centers is not None:
            self.row_node_ids = self.row_node_ids[~mask]
            self.row_node_weights = self.row_node_weights[~mask]
            self._cached_garment_pose = None
            self._xyz.requires_grad_(False)
        return result
