"""CPU RGB-D pair odometry with direct intensity plus point-to-plane residuals.

T_previous_from_current maps current points into the previous camera. Updates
are left perturbations exp([v_m,w_rad])@T. Geometric gates reuse the frozen
21b7 projective ICP library. Intensity interpolation and its derivatives are
analytic bilinear quantities from the same four pixels, not Sobel gradients.
Engineering residual scales/weights do not estimate covariance or information.
This library performs no dataset loading or GT evaluation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path

import numpy as np

import track_rgbd_icp as icp

FROZEN_ICP_SHA = '21b7e11bb6d6337e7c9bbfcacd8f324349bed22dd9f2fea9858caf44fe915f0a'
GRAY_RGB_WEIGHTS = np.array([.299, .587, .114], dtype=np.float64)


@dataclass(frozen=True)
class OdometryConfig:
    geometry: icp.ICPConfig = field(default_factory=icp.ICPConfig)
    sigma_geo_m: float = .01
    sigma_intensity: float = .1
    lambda_photo: float = 1.
    huber_delta_normalized: float = 1.
    max_final_intensity_rmse: float = .1
    bilinear_valid_support_guard_cells: int = 1

    def validate(self):
        values=(self.sigma_geo_m,self.sigma_intensity,self.huber_delta_normalized,self.max_final_intensity_rmse)
        if not all(math.isfinite(v)and v>0 for v in values) or not math.isfinite(self.lambda_photo)or self.lambda_photo<0:
            raise ValueError('Positive engineering scales/gates and nonnegative photo weight required')
        if self.bilinear_valid_support_guard_cells!=1:
            raise ValueError('This prototype fixes a one-cell bilinear-support trust margin')


@dataclass
class RGBDFrame:
    geometry: icp.Geometry
    gray: np.ndarray


def frozen_source_check():
    actual=hashlib.sha256(Path(icp.__file__).read_bytes()).hexdigest()
    if actual!=FROZEN_ICP_SHA:
        raise RuntimeError('Frozen geometry/correspondence implementation changed')
    return actual


def grayscale(rgb):
    a=np.asarray(rgb,dtype=np.float64)
    if a.ndim!=3 or a.shape[-1]!=3 or not np.isfinite(a).all()or a.min()<0 or a.max()>1:
        raise ValueError('Finite RGB array in [0,1] required, no clipping')
    # This fixed RGB weighted sum is not inverse-sRGB radiometric calibration.
    return a@GRAY_RGB_WEIGHTS


def make_frame(depth,valid,K,rgb,config):
    config.validate()
    geometry=icp.depth_geometry(depth,valid,K,config.geometry)
    gray=grayscale(rgb)
    if gray.shape!=geometry.valid.shape:
        raise ValueError('RGB/depth resolution mismatch')
    return RGBDFrame(geometry,gray)


def project(points,K):
    p=np.asarray(points,dtype=np.float64)
    return np.column_stack((K[0,0]*p[:,0]/p[:,2]+K[0,2],
                            K[1,1]*p[:,1]/p[:,2]+K[1,2]))


def bilinear(image,uv):
    """Intensity, exact dI/du,dI/dv and mask for 0<=uv<(W-1,H-1)."""
    image=np.asarray(image,dtype=np.float64);uv=np.asarray(uv,dtype=np.float64)
    if image.ndim!=2 or uv.ndim!=2 or uv.shape[1]!=2:
        raise ValueError('Gray image and Nx2 coordinates required')
    h,w=image.shape
    inside=np.isfinite(uv).all(axis=1)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
    ids=np.flatnonzero(inside);query=uv[ids]
    cell=np.floor(query).astype(np.int64);u,v=cell[:,0],cell[:,1]
    a=query[:,0]-u;b=query[:,1]-v
    i00=image[v,u];i10=image[v,u+1];i01=image[v+1,u];i11=image[v+1,u+1]
    value=(1-a)*(1-b)*i00+a*(1-b)*i10+(1-a)*b*i01+a*b*i11
    du=(1-b)*(i10-i00)+b*(i11-i01)
    dv=(1-a)*(i01-i00)+a*(i11-i10)
    output=np.full(len(uv),np.nan);gradient=np.full((len(uv),2),np.nan)
    output[ids]=value;gradient[ids]=np.column_stack((du,dv))
    return output,gradient,inside


def intensity_residual_jacobian(transformed,K,target_gray,source_intensity):
    p=np.asarray(transformed,dtype=np.float64);uv=project(p,K)
    sample,gradient,inside=bilinear(target_gray,uv)
    if not inside.all()or not np.isfinite(p).all()or not(p[:,2]>0).all():
        raise ValueError('Photo residual requires finite positive projected points inside bilinear domain')
    inverse_z=1/p[:,2]
    dI_dp=np.column_stack((gradient[:,0]*K[0,0]*inverse_z,
                          gradient[:,1]*K[1,1]*inverse_z,
                         -(gradient[:,0]*K[0,0]*p[:,0]+gradient[:,1]*K[1,1]*p[:,1])*inverse_z**2))
    # gradI*Jprojection*[I,-skew(p)] -> [dI_dp, p cross dI_dp].
    jacobian=np.column_stack((dI_dp,np.cross(p,dI_dp)))
    return sample-np.asarray(source_intensity,dtype=np.float64),jacobian,uv


def pair_correspondences(transform,source,target,distance_gate,config):
    pair=icp.correspondences(transform,source.geometry,target.geometry,distance_gate,config.geometry)
    p=pair['transformed'];uv=project(p,target.geometry.K)
    _,_,inside=bilinear(target.gray,uv)
    support=np.zeros(len(p),dtype=bool)
    # A one-cell guard makes fixed-correspondence local trials have support
    # around their base query, instead of one boundary pixel blocking every
    # descent direction. No trial drops pixels to lower its denominator.
    h,w=target.geometry.valid.shape
    inside &= (uv[:,0]>=1)&(uv[:,0]<w-2)&(uv[:,1]>=1)&(uv[:,1]<h-2)
    ids=np.flatnonzero(inside)
    cell=np.floor(uv[ids]).astype(np.int64);u,v=cell[:,0],cell[:,1]
    # Require all four samples to have valid target depth/normal support, so a
    # geometric nearest-pixel match does not license interpolation over holes.
    target_valid=target.geometry.valid
    guarded=np.ones(len(ids),dtype=bool)
    for dv in (-1,0,1,2):
        for du in (-1,0,1,2):guarded &= target_valid[v+dv,u+du]
    support[ids]=guarded
    selected=np.flatnonzero(support)
    for key in ('source','transformed','target','normals'):pair[key]=pair[key][selected]
    source_uv=project(pair['source'],source.geometry.K)
    source_pixel=np.rint(source_uv).astype(np.int64)
    if len(source_uv)and np.max(np.abs(source_uv-source_pixel))>1e-6:
        raise ValueError('Source depth pixel roundtrip inconsistent')
    pair['source_intensity']=source.gray[source_pixel[:,1],source_pixel[:,0]]
    pair['counts']['analytic_bilinear_support']=len(selected)
    pair['counts']['joint_correspondences']=len(selected)
    pair['ratio']=len(selected)/max(pair['counts']['valid_source_normals'],1)
    return pair


def terms(transform,pair,target):
    p=icp.transform_points(transform,pair['source'])
    geometric,j_geo=icp.residual_and_jacobian(p,pair['target'],pair['normals'])
    photo,j_photo,uv=intensity_residual_jacobian(p,target.geometry.K,target.gray,pair['source_intensity'])
    return geometric,j_geo,photo,j_photo,uv


def huber(values,delta):
    absolute=np.abs(values)
    return np.where(absolute<=delta,.5*values*values,delta*(absolute-.5*delta))


def objective(r_geo,r_photo,config):
    return float(np.mean(huber(r_geo/config.sigma_geo_m,config.huber_delta_normalized))+
                 config.lambda_photo*np.mean(huber(r_photo/config.sigma_intensity,config.huber_delta_normalized)))


def weighted_system(r_geo,j_geo,r_photo,j_photo,config):
    geo=r_geo/config.sigma_geo_m;photo=r_photo/config.sigma_intensity
    w_geo=np.minimum(1.,config.huber_delta_normalized/np.maximum(np.abs(geo),1e-15))
    w_photo=np.minimum(1.,config.huber_delta_normalized/np.maximum(np.abs(photo),1e-15))
    A=np.vstack((j_geo/config.sigma_geo_m*np.sqrt(w_geo)[:,None],
                 j_photo/config.sigma_intensity*np.sqrt(config.lambda_photo*w_photo)[:,None]))
    b=-np.r_[geo*np.sqrt(w_geo),photo*np.sqrt(config.lambda_photo*w_photo)]
    column_norm=np.linalg.norm(A,axis=0);safe=np.where(column_norm>1e-15,column_norm,1.)
    normalized=A/safe[None]
    singular=np.linalg.svd(normalized,compute_uv=False)
    rank=int(np.count_nonzero(singular>max(singular[0]*config.geometry.rank_rcond if len(singular)else 0,1e-15)))
    condition=None if len(singular)<6 or singular[-1]<=1e-15 else float((singular[0]/singular[-1])**2)
    stats={'joint_rank':rank,'column_normalized_hessian_condition':condition,
           'column_norms':column_norm.tolist(),'normalized_singular_values':singular.tolist(),
           'raw_hessian_eigenvalues':np.maximum(np.linalg.eigvalsh(A.T@A),0).tolist(),
           'plane_rmse_m':float(np.sqrt(np.mean(r_geo*r_geo))),
           'intensity_rmse':float(np.sqrt(np.mean(r_photo*r_photo))),
           'intensity_mean_residual':float(np.mean(r_photo)),
           'normalized_robust_objective':objective(r_geo,r_photo,config),
           'covariance_or_noise_information_estimated':False}
    if rank<6 or condition is None or condition>config.geometry.max_normalized_hessian_condition:
        return None,stats
    solution,_,solved_rank,_=np.linalg.lstsq(normalized,b,rcond=config.geometry.rank_rcond)
    if solved_rank!=6 or not np.isfinite(solution).all():return None,stats
    return solution/safe,stats


def fixed_correspondence_trial(transform,pair,target,config):
    p=icp.transform_points(transform,pair['source'])
    uv=project(p,target.geometry.K);_,_,inside=bilinear(target.gray,uv)
    if not inside.all()or not(p[:,2]>0).all():return None
    cell=np.floor(uv).astype(np.int64);u,v=cell[:,0],cell[:,1];valid=target.geometry.valid
    if not(valid[v,u]&valid[v,u+1]&valid[v+1,u]&valid[v+1,u+1]).all():return None
    r_geo,_,r_photo,_,_=terms(transform,pair,target)
    return objective(r_geo,r_photo,config)


def register_pair(source_pyramid,target_pyramid,config=OdometryConfig()):
    config.validate();frozen_source_check()
    if len(source_pyramid)!=3 or len(target_pyramid)!=3:
        raise ValueError('Three coarse-to-fine RGB-D levels required')
    transform=np.eye(4);history=[];levels=[];g=config.geometry
    for level,(source,target)in enumerate(zip(source_pyramid,target_pyramid)):
        updates=0;stop='iteration_budget'
        for iteration in range(g.iterations[level]):
            pair=pair_correspondences(transform,source,target,g.distance_gates_m[level],config)
            row={'level':level,'iteration':iteration,'counts':pair['counts'],'correspondence_ratio':pair['ratio'],
                 'accepted':False,'T_previous_from_current_before':transform.tolist(),'line_search_trials':[]}
            if len(pair['source'])<g.min_correspondences or pair['ratio']<g.min_correspondence_ratio:
                stop='insufficient_joint_correspondences';row['stop_reason']=stop;history.append(row);break
            r_geo,j_geo,r_photo,j_photo,_=terms(transform,pair,target)
            delta,stats=weighted_system(r_geo,j_geo,r_photo,j_photo,config);row.update(stats)
            if delta is None:
                stop='rank_deficient_or_ill_conditioned';row['stop_reason']=stop;history.append(row);break
            row['proposed_delta_twist_m_rad']=delta.tolist()
            limit=min(1.,g.max_iteration_translation_m/max(np.linalg.norm(delta[:3]),1e-15),
                      math.radians(g.max_iteration_rotation_degrees)/max(np.linalg.norm(delta[3:]),1e-15))
            delta=delta*limit;row['step_limit_factor']=float(limit)
            if np.linalg.norm(delta[:3])<g.convergence_translation_m and np.linalg.norm(delta[3:])<g.convergence_rotation_radians:
                stop='small_update';row['stop_reason']=stop;history.append(row);break
            for scale in g.line_search_scales:
                trial=icp.se3_exp(delta*scale)@transform
                cost=fixed_correspondence_trial(trial,pair,target,config)
                row['line_search_trials'].append({'scale':scale,'objective':cost,'same_correspondence_denominator':len(pair['source'])})
                if cost is not None and math.isfinite(cost)and cost<stats['normalized_robust_objective']:
                    transform=trial;updates+=1
                    row.update(accepted=True,line_search_scale=scale,accepted_delta_twist_m_rad=(delta*scale).tolist(),
                               fixed_correspondence_objective_after=cost,T_previous_from_current_after=transform.tolist())
                    break
            if not row['accepted']:stop='line_search_no_descent';row['stop_reason']=stop
            history.append(row)
            if not row['accepted']:break
        levels.append({'level':level,'iterations_recorded':iteration+1,'accepted_updates':updates,'stop_reason':stop})
    final=pair_correspondences(transform,source_pyramid[-1],target_pyramid[-1],g.distance_gates_m[-1],config)
    flags=[];stats={}
    if len(final['source'])<g.min_correspondences or final['ratio']<g.min_correspondence_ratio:
        flags.append('final_insufficient_joint_correspondences')
    else:
        r_geo,j_geo,r_photo,j_photo,_=terms(transform,final,target_pyramid[-1])
        _,stats=weighted_system(r_geo,j_geo,r_photo,j_photo,config)
        if stats['joint_rank']<6:flags.append('final_unobservable_6d_pose')
        if stats['column_normalized_hessian_condition']is None or stats['column_normalized_hessian_condition']>g.max_normalized_hessian_condition:
            flags.append('final_ill_conditioned_joint_system')
        if stats['plane_rmse_m']>g.max_final_plane_rmse_m:flags.append('final_plane_residual_exceeds_engineering_gate')
        if config.lambda_photo>0 and stats['intensity_rmse']>config.max_final_intensity_rmse:
            flags.append('final_intensity_residual_exceeds_engineering_gate')
    translation=float(np.linalg.norm(transform[:3,3]));rotation=icp.rotation_degrees(transform[:3,:3])
    if translation>g.max_pair_translation_m:flags.append('relative_translation_exceeds_threshold')
    if rotation>g.max_pair_rotation_degrees:flags.append('relative_rotation_exceeds_threshold')
    rigid_error=float(np.max(np.abs(transform[:3,:3].T@transform[:3,:3]-np.eye(3))))
    if not np.isfinite(transform).all()or rigid_error>1e-8 or abs(np.linalg.det(transform[:3,:3])-1)>1e-8:
        flags.append('nonfinite_or_nonrigid_transform')
    if levels[-1]['stop_reason']=='line_search_no_descent':flags.append('finest_level_line_search_no_descent')
    return {'accepted':not flags,'failure_flags':flags,'candidate_previous_from_current':transform.tolist(),
        'relative_translation_m':translation,'relative_rotation_degrees':rotation,'rotation_orthogonality_max_abs':rigid_error,
        'final_counts':final['counts'],'final_correspondence_ratio':final['ratio'],'final_statistics':stats,
        'levels':levels,'iteration_history':history,'configuration':asdict(config),
        'pose_direction':'current -> previous','update':'exp(delta_m_rad) @ T','GT_used':False}


def integrate_relative(previous_c2w,result):
    candidate=np.asarray(result['candidate_previous_from_current'])if result['accepted']else np.eye(4)
    return previous_c2w@candidate
