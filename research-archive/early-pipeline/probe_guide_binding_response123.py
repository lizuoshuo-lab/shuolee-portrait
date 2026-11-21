"""CPU-only, non-optimized perturbation: verify consequences of moving a mesh.

No candidate is exported. A 1mm bounded hypothetical offset is not training.
"""
from pathlib import Path
import argparse
import os
import sys
import time
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,OUT,SNAP,verify


def main():
    verify();assert io.read(OUT/'result.json')['state']=='complete_no_updates'
    dest=OUT/'binding-response';dest.mkdir(exist_ok=False)
    plan=dict(created=io.now(),source_sha256=io.sha(__file__),frames=59,
        audit_sha256=io.sha(OUT/'result.json'),gradient_sha256=io.sha(OUT/'offset-gradient-only.npz'),
        max_offset_perturbation_m=.001,optimizer_updates=0,cpu_only=True,max_seconds=180,
        hypothesis='Mesh motion also transports bound Gaussians. Reference-frame local-coordinate/covariance rebasing preserves one pose exactly but must be checked on all other articulated poses.',
        no_new_training_or_model=True,quality_accepted=False)
    io.save(dest/'plan.json',plan);start=time.monotonic()
    import torch
    import smplx
    import roma
    from torch.func import functional_call
    sys.path.insert(0,str(io.PROJECT/'src/avatar_core'))
    from utils.graphics_utils import compute_face_orientation
    torch.set_num_threads(4)
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:saved=dict(z)
    with np.load(OUT/'offset-gradient-only.npz',allow_pickle=False) as z:gradient=z['guide'].mean(0)
    state=torch.load(SNAP/'checkpoint.pt',map_location='cpu',weights_only=False)
    capture=state['capture'];assert state['step']==120000
    local=capture[1].detach();binding=capture[7].long();s_local=capture[4].detach().exp()
    q=capture[5].detach();q=q/q.norm(dim=-1,keepdim=True)
    Rlocal=roma.unitquat_to_rotmat(q[:,[1,2,3,0]])
    offset=state['static_offset'].detach()
    g=torch.tensor(gradient);delta=-g*(plan['max_offset_perturbation_m']/g.norm(dim=-1).max().clamp_min(1e-30))
    model=smplx.SMPLX(str(io.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,
        num_betas=10,num_expression_coeffs=10,use_pca=True,num_pca_comps=12,
        flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
        create_transl=False,dtype=torch.float32).eval().requires_grad_(False)
    faces=torch.tensor(fit['faces'],dtype=torch.long)
    rows=io.read(ROOT/'data/training-inputs.json')['records']
    slots={str(f):i for i,f in enumerate(fit['frame_ids'])}
    def forward(row,off):
        i=slots[row['frame']]
        params={k:torch.tensor(fit[k][i:i+1]) for k in ('global_orient','expression','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose')}
        params.update(body_pose=torch.tensor(fit['body_pose_axis_angle'][i:i+1]),betas=torch.tensor(fit['shared_betas']))
        output=functional_call(model,{'v_template':model.v_template+off},(),dict(params,return_verts=True,return_shaped=False))
        t=torch.tensor(fit['camera_translation'][i]);E=torch.tensor(fit['w2c'][i],dtype=torch.float32)
        return (output.vertices[0]+t-E[:3,3])@E[:3,:3],(output.joints[0]+t-E[:3,3])@E[:3,:3]
    def basis(v):
        R,s=compute_face_orientation(v,faces,return_scale=True)
        return v[faces].mean(1)[binding],R[binding],s[binding].clamp_min(1e-8)
    def xyz(frame,u):
        c,R,s=frame;return (R@u[:,:,None]).squeeze(-1)*s+c
    def cov(frame,rotation,scaling):
        _,R,s=frame;A=(R@rotation)*(scaling*s)[:,None,:];return A@A.transpose(1,2)
    def stats(x):return dict(zip(['median','p90','p99','max'],map(float,np.quantile(x,[.5,.9,.99,1]))))
    def project(x,row):
        E=torch.tensor(row['w2c'],dtype=x.dtype);K=torch.tensor(row['K'],dtype=x.dtype)
        p=(x@E[:3,:3].T+E[:3,3])@K.T;return p[:,:2]/p[:,2:].clamp_min(1e-9)-.5
    records=[]
    with torch.no_grad():
        ref=next(r for r in rows if r['frame']=='frame_000033')
        vr,_=forward(ref,offset);vc,_=forward(ref,offset+delta);old=basis(vr);new=basis(vc)
        target=xyz(old,local);worldR=old[1]@Rlocal;worldS=old[2]*s_local
        rebased=(new[1].transpose(1,2)@((target-new[0])/new[2])[:,:,None]).squeeze(-1)
        rebasedR=new[1].transpose(1,2)@worldR;rebasedS=worldS/new[2]
        for i,row in enumerate(rows):
            assert time.monotonic()-start<plan['max_seconds']
            v0,j0=forward(row,offset);v1,j1=forward(row,offset+delta)
            parity=float((v0-torch.tensor(saved['vertices_camera'][i])).abs().max());assert parity<2e-6
            old=basis(v0);new=basis(v1);oldx=xyz(old,local);naive=xyz(new,local);closed=xyz(new,rebased)
            oldcov=cov(old,Rlocal,s_local);newcov=cov(new,rebasedR,rebasedS)
            uv=project(oldx,row);w,h=row['wh'];support=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            rel=(newcov-oldcov).reshape(-1,9).norm(dim=1)/oldcov.reshape(-1,9).norm(dim=1).clamp_min(1e-20)
            records.append(dict(frame=row['frame'],old_mesh_parity_m=parity,
                mesh_motion_mm=stats((v1-v0).norm(dim=1).numpy()*1000),
                bound_Gaussian_without_rebase_mm=stats((naive-oldx).norm(dim=1).numpy()*1000),
                reference_rebased_other_pose_Gaussian_mm=stats((closed-oldx).norm(dim=1).numpy()*1000),
                reference_rebased_other_pose_projection_px=stats((project(closed,row)-uv).norm(dim=1)[support].numpy()),
                rebase_covariance_relative_Frobenius=stats(rel.numpy()),
                frozen_pose_regressed_joints_motion_px=stats((project(j1,row)-project(j0,row)).norm(dim=1).numpy())))
            print('BINDING_RESPONSE',i+1,row['frame'],flush=True)
    rr=next(r for r in records if r['frame']=='frame_000033')
    assert rr['reference_rebased_other_pose_Gaussian_mm']['max']<.002
    assert rr['rebase_covariance_relative_Frobenius']['max']<2e-5
    verify()
    result=dict(state='complete_cpu_counterfactual',updated=io.now(),frames=59,
        offset_perturbation_mm=stats(delta.norm(dim=-1).numpy()*1000),records=records,
        elapsed_seconds=time.monotonic()-start,optimizer_created=False,parameter_updates=0,
        saved_candidate=False,reference_geometry_rebase_passed=True,
        all_pose_geometry_unchanged=False,no_actual_surface_improvement_measured=True,quality_accepted=False,
        conclusion='A fixed Gaussian teacher can pull the mesh, but mesh movement also moves bound centers/covariance. Rebasing at one pose is exact only there. Pose protection and all-frame acceptance remain necessary; the frozen teacher is not clothing ground truth.')
    io.save(dest/'result.json',result);print('BINDING_RESPONSE_COMPLETE',result['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
