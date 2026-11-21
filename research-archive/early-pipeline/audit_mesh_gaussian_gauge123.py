"""Bounded CPU derivative/gauge audit of the exact saved mesh-bound model.

No optimizer, coefficient search, new images, or candidate model export.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import traceback
import numpy as np
from run_official_gaussians123 import PROJECT, read, save, sha, now
from train_demo_body_upstream123 import DEMO, UP, module

BASE=DEMO/'body-gaussian-smplify-eight-dc-reviewedfit'
SNAP=BASE/'gaussian-run/step-018000'
OUT=BASE/'mesh-gaussian-gauge-audit'
HERE=Path(__file__).resolve().parent


def prepare():
    assert not OUT.exists(), 'Never repeat this completed audit'
    files=[Path(__file__).resolve(),BASE/'fit/surface-cache.npz',BASE/'selection.json',BASE/'data/training-inputs.json',
           SNAP/'bound-local.ply',SNAP/'binding.npz',SNAP/'current-meshes.npz',
           UP/'scene/gaussian_model.py',PROJECT/'src/avatar_core/utils/graphics_utils.py',
           HERE/'smplify_direct_surface123.py',HERE/'facial_geometry_boundary123.py']
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),state='registered_before_execution',CPU_budget_seconds=120,
        hypothesis='Mesh triangle position/orientation/scale and Gaussian local xyz/rotation/scale jointly represent world Gaussians. Test derivatives and whether a fixed1mm hypothetical clothing deformation can be compensated by one shared local Gaussian transform.',
        baseline=str(SNAP),finite_difference_step_m=1e-6,hypothetical_reference_displacement_m=.001,
        derivative_frames='All8 frozen training frames,3 canonical translation directions restricted to the clothing mask;24 checks',
        gauge_frames=59,reference_frame='frame_000033',
        stop=['Any input/source hash change','Nonfinite output','Saved mesh replay error>=2e-6m','JVP vs central difference max absolute error>=5e-7','Nonzero protected-face displacement','CPU budget exceeded'],
        interpretation='Gauge experiment changes in-memory mesh and local Gaussian variables only, keeps reference world centers/covariances fixed, then measures remaining differences across all59 poses. This is not training, a valid new surface, pixel-image quality, or physical truth. Any introduced folds are reported, never promoted.',
        optimizer_created=False,optimization_updates=0,GPU_used=False,parameter_files_modified=False,
        new_model_exported=False,quality_accepted=False,files={str(p):sha(p) for p in files}))
    print('MESH_GAUSSIAN_GAUGE_REGISTERED',flush=True)


def run():
    assert not (OUT/'result.json').exists()
    plan=read(OUT/'plan.json')
    for p,h in plan['files'].items():assert sha(p)==h,p
    started=time.monotonic()
    import torch
    import roma
    from plyfile import PlyData
    from smplify_direct_surface123 import vertices
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    graphics=module(PROJECT/'src/avatar_core/utils/graphics_utils.py','gauge_original_graphics123')
    tree=ast.parse((UP/'scene/gaussian_model.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GaussianModel')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='get_xyz')
    fn.decorator_list=[]; ns={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(UP/'scene/gaussian_model.py'),'exec'),ns)
    actual_get_xyz=ns['get_xyz']
    with np.load(BASE/'fit/surface-cache.npz',allow_pickle=False) as z:c=dict(z)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:b=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:m=dict(z)
    rows=read(BASE/'data/training-inputs.json')['records']
    assert list(c['frame_ids'])==list(b['frame_ids'])==list(m['frame_ids'])==[r['frame'] for r in rows]
    assert rows[0]['frame']=='frame_000033'
    t=lambda x:torch.as_tensor(x,dtype=torch.float64)
    model=SimpleNamespace(surface_base=t(c['base_vertices']),surface_linear=t(c['linear']),
        surface_clothing=torch.as_tensor(c['clothing_vertices']),surface_protected=torch.as_tensor(c['protected_face_vertices']),
        static_offset=t(b['static_offset']),faces=torch.as_tensor(c['faces'].astype(np.int64)))
    binding=torch.as_tensor(b['triangle_ids'])
    ply=PlyData.read(str(SNAP/'bound-local.ply'))['vertex']
    local=t(np.stack([ply[k] for k in ('x','y','z')],axis=-1))
    scale_local=t(np.stack([ply['scale_'+str(k)] for k in range(3)],axis=-1)).exp()
    quat=t(np.stack([ply['rot_'+str(k)] for k in range(4)],axis=-1))
    rotation_local=roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(quat/quat.norm(dim=-1,keepdim=True)))
    active=model.surface_clothing & ~model.surface_protected
    cloth_faces=active[model.faces].all(-1);cloth_points=cloth_faces[binding]
    def frame(offset,index,coords=local):
        model.static_offset=offset
        v=vertices(model,index)
        R,s=graphics.compute_face_orientation(v,model.faces,return_scale=True)
        state=SimpleNamespace(binding=binding,_xyz=coords,face_orien_mat=R,face_scaling=s.clamp_min(1e-8),face_center=v[model.faces].mean(1))
        return actual_get_xyz(state),v,R,s.clamp_min(1e-8),state.face_center
    initial=t(b['static_offset']);replay=0.;derivatives=[]
    chosen=set(read(BASE/'selection.json')['selected_frames'])
    for i,row in enumerate(rows):
        _,v,_,_,_=frame(initial,i)
        replay=max(replay,float((v-t(m['vertices_camera'][i])).abs().max()))
        if row['frame'] not in chosen:continue
        for axis in range(3):
            d=torch.zeros_like(initial);d[active,axis]=1
            _,jvp=torch.autograd.functional.jvp(lambda offset:frame(offset,i)[0],initial,d)
            eps=plan['finite_difference_step_m']
            fd=(frame(initial+eps*d,i)[0]-frame(initial-eps*d,i)[0])/(2*eps)
            error=float((fd-jvp).abs().max())
            assert error<5e-7,(row['frame'],axis,error)
            derivatives.append(dict(frame=row['frame'],axis=axis,max_abs_error=error,jvp_norm=float(jvp.norm())))
        assert time.monotonic()-started<120
    assert replay<2e-6 and len(derivatives)==24
    p0,v0,R0,s0,centre0=frame(initial,0)
    # Fixed1mm outward movement in reference camera space, transformed back
    # through the saved original SMPL-X linear skinning map. Not a fitted target.
    tri=v0[model.faces];n=torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1)
    normal=torch.zeros_like(v0)
    for axis in range(3):normal.index_add_(0,model.faces[:,axis],n)
    normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-20)
    direction=torch.zeros_like(initial)
    direction[active]=torch.linalg.solve(model.surface_linear[0,active],normal[active,:,None]).squeeze(-1)*.001
    shifted=initial+direction
    _,v1,R1,s1,centre1=frame(shifted,0)
    new_local=torch.bmm(R1[binding].transpose(1,2),((p0-centre1[binding])/s1[binding])[:,:,None]).squeeze(-1)
    new_rot=R1[binding].transpose(1,2)@R0[binding]@rotation_local
    new_scale=scale_local*s0[binding]/s1[binding]
    def covariance(R,s,Q,S):
        L=(R[binding]@Q)*(S*s[binding])[:,None,:]
        return L@L.transpose(1,2)
    def stats(x):
        x=x[cloth_points]
        return dict(median=float(x.median()),p90=float(torch.quantile(x,.9)),maximum=float(x.max()))
    records=[];face_motion=0;folds=0;ref_cov_error=None;reg_changes=[]
    for i,row in enumerate(rows):
        po,vo,ro,so,_=frame(initial,i)
        uncorrected,vn,rn,sn,_=frame(shifted,i)
        corrected,_,_,_,_=frame(shifted,i,new_local)
        oldcov=covariance(ro,so,rotation_local,scale_local)
        newcov=covariance(rn,sn,new_rot,new_scale)
        covariance_relative=(newcov-oldcov).flatten(1).norm(dim=1)/oldcov.flatten(1).norm(dim=1).clamp_min(1e-20)
        face_motion=max(face_motion,float((vn[model.surface_protected]-vo[model.surface_protected]).abs().max()))
        oldtri=vo[model.faces];newtri=vn[model.faces]
        no=torch.cross(oldtri[:,1]-oldtri[:,0],oldtri[:,2]-oldtri[:,0],dim=-1)
        nn=torch.cross(newtri[:,1]-newtri[:,0],newtri[:,2]-newtri[:,0],dim=-1)
        folds+=int(((no*nn).sum(-1)<=0).sum())
        K=t(row['K_integer_centers']);E=t(row['w2c'])
        def project(p):
            cam=p@E[:3,:3].T+E[:3,3];assert (cam[:,2]>0).all()
            uv=cam@K.T;return uv[:,:2]/uv[:,2:]
        records.append(dict(frame=row['frame'],train8=row['frame'] in chosen,
            uncompensated_center_m=stats((uncorrected-po).norm(dim=-1)),
            compensated_center_m=stats((corrected-po).norm(dim=-1)),
            uncompensated_projection_px=stats((project(uncorrected)-project(po)).norm(dim=-1)),
            compensated_projection_px=stats((project(corrected)-project(po)).norm(dim=-1)),
            covariance_relative_Frobenius=stats(covariance_relative)))
        if i==0:
            ref_cov_error=float(covariance_relative.max())
            assert float((corrected-po).abs().max())<1e-10 and ref_cov_error<1e-10
        save(OUT/'progress.json',dict(updated=now(),state='read_only_cpu',frames=i+1,derivative_checks=24))
        assert time.monotonic()-started<120
    assert face_motion==0 and not torch.cuda.is_initialized()
    # Exact upstream nonmetric position penalty: a unit-radius dead zone.
    old_norm=local.norm(dim=-1);new_norm=new_local.norm(dim=-1)
    old_excess=(old_norm-1).clamp_min(0);new_excess=(new_norm-1).clamp_min(0)
    for p,h in plan['files'].items():assert sha(p)==h,p
    save(OUT/'result.json',dict(updated=now(),state='read_only_cpu_completed',seconds=time.monotonic()-started,
        derivative_checks=derivatives,saved_mesh_replay_max_m=replay,frames=records,
        shared_offset_active_vertices=int(active.sum()),clothing_points=int(cloth_points.sum()),total_points=len(local),
        reference_covariance_relative_max=ref_cov_error,protected_face_motion_m=face_motion,
        hypothetical_mesh_fold_observations=folds,
        original_position_prior=dict(threshold_local_units=1.,clothing_inside_deadzone_fraction=float((old_norm[cloth_points]<=1).double().mean()),
            before_mean_excess=float(old_excess[cloth_points].mean()),after_mean_excess=float(new_excess[cloth_points].mean()),
            scope='All clothing-bound Gaussians, not per-view visibility-filtered training loss'),
        optimization_updates=0,GPU_used=False,parameter_files_unchanged=True,quality_accepted=False,new_model_exported=False,
        limitation='Center/covariance equivalence is a representation diagnostic, not a rendered RGB test or evidence that this hypothetical mesh is the observed clothing surface.'))
    print('MESH_GAUSSIAN_GAUGE_COMPLETE',len(records),max(x['max_abs_error'] for x in derivatives),folds,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
    try:globals()[a.mode]()
    except BaseException as error:
        if OUT.exists():save(OUT/(a.mode+'-failure.json'),dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
