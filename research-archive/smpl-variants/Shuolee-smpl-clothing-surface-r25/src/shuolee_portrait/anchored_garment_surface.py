"""Surface-anchored Gaussian rows with bounded normal and tangent residuals."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .free_clothing_torch import normalized_quaternion_matrix
from .garment_surface import GarmentSurfaceModel
from .splat_surface_distance import folded_normal_distance


ANCHOR_NORMAL_LIMIT_M = .006
ANCHOR_NORMAL_HARD_LIMIT_M = .032
ANCHOR_TANGENT_LIMIT_M = .002
ANCHOR_FIELDS = ('anchor_reference_normal', 'anchor_trusted', 'anchor_normal_raw', 'anchor_tangent_raw')
NORMAL_GROUP = 'anchor_normal'


class SurfaceAnchoredGarmentModel(GarmentSurfaceModel):
    """Retain the seed offset and optimize only local garment-surface corrections."""

    def __init__(self, sh_degree=0):
        super().__init__(sh_degree)
        self.anchor_normal_raw = None
        self._anchor_extension = None

    def _graph_normals(self):
        neighbors = self.node_centers[self.node_normal_neighbors.clamp_min(0)]
        delta = neighbors-self.node_centers[:, None]
        cross = torch.linalg.cross(delta[:, 0], delta[:, 1], dim=1)
        supported = self._node_has_normal & (cross.norm(dim=1)>1e-8)
        normals = F.normalize(cross,dim=1)*supported[:,None]
        blend = self._interpolate(normals)
        mass = self._interpolate(supported[:,None].to(dtype=blend.dtype))[:,0]
        magnitude = blend.norm(dim=1)
        return F.normalize(blend,dim=1), mass, magnitude/mass.clamp_min(1e-8)

    @torch.no_grad()
    def configure_surface_anchors(self, *, anchors=None):
        if self.node_centers is None:
            raise ValueError('The R24 garment graph must be configured before its surface anchors.')
        dtype, device = self._xyz.dtype, self._xyz.device
        rows, nodes = len(self._xyz), len(self.node_centers)
        if anchors is None:
            measured, supported, coherent = self._graph_normals()
            valid = measured.norm(dim=1) > .5
            normal = torch.where(valid[:,None], measured, self._normal_fallback[self.row_node_ids[:,0]])
            offset = self._xyz-self._interpolate(self.node_rest_xyz)
            normal_distance = (offset*normal).sum(dim=1)
            tangent = offset-normal_distance[:,None]*normal
            trust = valid & (supported >= .75) & (coherent >= .75) & (tangent.norm(dim=1) <= .030)
            normal_raw = self._xyz.new_zeros(rows,1)
            tangent_raw = self._xyz.new_zeros(nodes,3)
        else:
            if set(anchors) != set(ANCHOR_FIELDS):
                raise ValueError('Surface anchor arrays differ from the native drive contract.')
            expected = np.float32 if dtype == torch.float32 else np.float64
            arrays = [np.asarray(anchors[key]) for key in ANCHOR_FIELDS]
            if any(array.dtype != (np.bool_ if key == 'anchor_trusted' else expected)
                   for key,array in zip(ANCHOR_FIELDS,arrays,strict=True)):
                raise ValueError('Surface anchor dtypes differ from the executable garment graph.')
            normal, trust, normal_raw, tangent_raw = [torch.tensor(value,device=device) for value in arrays]
        if (normal.shape != (rows,3) or trust.shape != (rows,) or normal_raw.shape != (rows,1)
                or tangent_raw.shape != (nodes,3) or not trust.any()
                or not all(torch.isfinite(value).all() for value in (normal,normal_raw,tangent_raw))
                or not torch.allclose(normal.norm(dim=1),torch.ones_like(normal[:,0]),rtol=0,atol=2e-5)):
            raise ValueError('Invalid trusted anchor normals, mask or trainable residual arrays.')
        self.anchor_reference_normal, self.anchor_trusted = normal, trust
        self.anchor_normal_raw = nn.Parameter(normal_raw)
        self.anchor_tangent_raw = nn.Parameter(tangent_raw)

    def configure_anchor_optimizer(self, lr):
        if (self.anchor_normal_raw is None or self.optimizer is None or not np.isfinite(lr) or lr<=0
                or any(group['name']==NORMAL_GROUP for group in self.optimizer.param_groups)):
            raise ValueError('Configure the garment and Gaussian optimizer once before adding anchor normal Adam.')
        self.optimizer.add_param_group(dict(params=[self.anchor_normal_raw],lr=float(lr),name=NORMAL_GROUP))

    def anchor_arrays(self):
        return {key:getattr(self,key).detach().cpu().numpy().copy() for key in ANCHOR_FIELDS}

    def _live_anchor_normal(self):
        normal, mass, coherence = self._graph_normals()
        seed = self.anchor_reference_normal
        alignment = (normal*seed).sum(dim=1)
        quality = ((mass-.6)/.15).clamp(0,1)*((coherence-.5)/.25).clamp(0,1)*((alignment-.2)/.4).clamp(0,1)
        # Valid normals at construction take the exact measured branch, so
        # zero residuals preserve the original world render at every pose.
        blended = F.normalize(torch.lerp(seed,normal,quality[:,None]),dim=1)
        return torch.where((quality>=1.-1e-6)[:,None],normal,blended)

    def _anchor_terms(self):
        normal = self._live_anchor_normal()
        seed = self.anchor_reference_normal
        base = self._xyz-self._interpolate(self.node_rest_xyz)
        seed_distance = (base*seed).sum(dim=1,keepdim=True)
        normal_bound = (seed_distance.abs()+.002).clamp(ANCHOR_NORMAL_LIMIT_M,ANCHOR_NORMAL_HARD_LIMIT_M)
        normal_delta = normal_bound*self.anchor_normal_raw.tanh()
        tangent = self._interpolate(self.anchor_tangent_raw)
        tangent = tangent-(tangent*normal).sum(dim=1,keepdim=True)*normal
        norm = tangent.norm(dim=1,keepdim=True)
        factor = torch.where(norm<1e-4,1.,norm.tanh()/norm.clamp_min(1e-4))
        tangent_delta = ANCHOR_TANGENT_LIMIT_M*tangent*factor
        trust = self.anchor_trusted[:,None]
        correction = ((normal-seed)*seed_distance+normal*normal_delta+tangent_delta)*trust
        return normal, base, correction, normal_delta*trust, tangent_delta*trust

    def canonical_centers(self):
        base = super().canonical_centers()
        if self.anchor_normal_raw is None:
            return base
        return base+self._anchor_terms()[2]

    @property
    def get_xyz(self):
        base = super().get_xyz
        if self.anchor_normal_raw is None or self.timestep == self.reference_frame:
            return base
        _, _, _, _, _, scale, rotation, _ = self._pose_field()
        correction = self._anchor_terms()[2]
        return base+torch.einsum('nij,nj->ni',rotation,correction)*scale

    def garment_normals(self):
        fallback = super().garment_normals()
        if self.anchor_normal_raw is None:
            return fallback
        return torch.where(self.anchor_trusted[:,None],self._live_anchor_normal(),fallback)

    def anchor_distribution(self, visible=None):
        normal, base, correction, normal_delta, tangent_delta = self._anchor_terms()
        valid = self.anchor_trusted & (self.get_opacity.detach()[:,0]>=.05)
        if visible is not None:
            valid = valid & visible
        center = ((base+correction)*normal).sum(dim=1)
        axes = normalized_quaternion_matrix(F.normalize(self._rotation,dim=1))
        scales = self._scaling.exp()
        dot = torch.einsum('ni,nij->nj',normal,axes)
        sigma = (dot*scales).norm(dim=1)
        ordered, axis = scales.sort(dim=1)
        cosine = dot.gather(1,axis[:,:1]).abs()[:,0].clamp(0,1)
        return dict(valid=valid,center_m=center,sigma_m=sigma,
            expected_m=folded_normal_distance(center,sigma),normal_delta_m=normal_delta[:,0],
            tangent_delta_m=tangent_delta.norm(dim=1),thin_axis_angle_deg=torch.rad2deg(cosine.acos()),
            sheet=ordered[:,0]<.8*ordered[:,1])

    @torch.no_grad()
    def anchor_metrics(self):
        measured = self.anchor_distribution()
        fields = dict(signed_center_mm=measured['center_m']*1000.,abs_center_mm=measured['center_m'].abs()*1000.,
                      normal_sigma_mm=measured['sigma_m']*1000.,expected_distance_mm=measured['expected_m']*1000.,
                      normal_residual_mm=measured['normal_delta_m'].abs()*1000.,
                      tangent_residual_mm=measured['tangent_delta_m']*1000.)

        def stats(values):
            if not len(values):
                return None
            quantiles = values.quantile(values.new_tensor([.05,.5,.95])).tolist()
            return dict(zip(('p05','p50','p95','max'),(*quantiles,float(values.max())),strict=True))

        def subset(mask):
            sheet = mask & measured['sheet']
            return dict(rows=int(mask.count_nonzero()),sheet_rows=int(sheet.count_nonzero()),
                        thin_axis_angle_deg=stats(measured['thin_axis_angle_deg'][sheet]),
                        **{name:stats(value[mask]) for name,value in fields.items()})

        return dict(total_rows=len(self._xyz),trusted_rows=int(self.anchor_trusted.count_nonzero()),
                    trusted=subset(self.anchor_trusted),
                    trusted_peak_alpha_ge_point1=subset(self.anchor_trusted & (self.get_opacity[:,0]>=.1)),
                    normal_residual_base_bound_mm=ANCHOR_NORMAL_LIMIT_M*1000.,
                    normal_residual_hard_bound_mm=ANCHOR_NORMAL_HARD_LIMIT_M*1000.,
                    tangent_residual_bound_mm=ANCHOR_TANGENT_LIMIT_M*1000.)

    def garment_regularization(self, visible=None):
        result = super().garment_regularization(visible)
        if self.anchor_normal_raw is None:
            return result
        measured = self.anchor_distribution(visible)
        valid = measured['valid']
        count = valid.count_nonzero().clamp_min(1)
        center = ((measured['center_m'][valid].abs()-.001).relu()/.004)
        distribution = ((measured['expected_m'][valid]-.003).relu()/.004)
        result['surface_center'] = F.smooth_l1_loss(center,torch.zeros_like(center),reduction='sum')/count
        result['surface_distribution'] = F.smooth_l1_loss(distribution,torch.zeros_like(distribution),reduction='sum')/count
        return result

    def preview_rows(self, source_rows, parameters):
        preview = super().preview_rows(source_rows,parameters)
        if self.anchor_normal_raw is not None:
            preview.anchor_reference_normal = self.anchor_reference_normal[source_rows]
            preview.anchor_trusted = self.anchor_trusted[source_rows]
            preview.anchor_normal_raw = (parameters['anchor_normal_raw'] if 'anchor_normal_raw' in parameters
                                         else self.anchor_normal_raw[source_rows].detach())
            preview.anchor_tangent_raw = self.anchor_tangent_raw.detach()
        return preview

    def cat_tensors_to_optimizer(self, tensors_dict):
        if self._anchor_extension is not None:
            tensors_dict = {**tensors_dict,NORMAL_GROUP:self._anchor_extension}
        result = super().cat_tensors_to_optimizer(tensors_dict)
        if self._anchor_extension is not None:
            self.anchor_normal_raw = result[NORMAL_GROUP]
        return result

    def _prune_optimizer(self, mask):
        result = super()._prune_optimizer(mask)
        if self.anchor_normal_raw is not None:
            self.anchor_normal_raw = result[NORMAL_GROUP]
        return result

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                              new_scaling, new_rotation, *, constraint_parents=None):
        if self.anchor_normal_raw is None or not len(new_xyz):
            return super().densification_postfix(new_xyz,new_features_dc,new_features_rest,new_opacities,
                                                 new_scaling,new_rotation,constraint_parents=constraint_parents)
        if (not isinstance(constraint_parents,torch.Tensor) or constraint_parents.dtype != torch.long
                or constraint_parents.device != self._xyz.device or constraint_parents.shape != (len(new_xyz),)
                or constraint_parents.min()<0 or constraint_parents.max()>=len(self._xyz)
                or self.optimizer is None or not any(group['name']==NORMAL_GROUP for group in self.optimizer.param_groups)):
            raise ValueError('Surface splits require exact parent rows and the configured anchor-normal Adam group.')
        normal, trust = self.anchor_reference_normal, self.anchor_trusted
        self.anchor_reference_normal = torch.cat((normal,normal[constraint_parents]))
        self.anchor_trusted = torch.cat((trust,trust[constraint_parents]))
        self._anchor_extension = self.anchor_normal_raw[constraint_parents].detach()
        try:
            return super().densification_postfix(new_xyz,new_features_dc,new_features_rest,new_opacities,
                                                 new_scaling,new_rotation,constraint_parents=constraint_parents)
        except Exception:
            if len(self._xyz)==len(normal):
                self.anchor_reference_normal, self.anchor_trusted = normal, trust
            raise
        finally:
            self._anchor_extension = None

    def prune_points(self, mask):
        result = super().prune_points(mask)
        if self.anchor_normal_raw is not None:
            self.anchor_reference_normal = self.anchor_reference_normal[~mask]
            self.anchor_trusted = self.anchor_trusted[~mask]
        return result
