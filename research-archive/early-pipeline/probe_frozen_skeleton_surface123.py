"""All59 original-LBS parity and controlled clothing-displacement contract."""
import argparse
import importlib
import sys
import time
from pathlib import Path
import numpy as np
import run_official_gaussians123 as io
from probe_gaussian_surface_guidance123 import ROOT,SNAP,OUT as AUDIT,verify

OUT=AUDIT/'frozen-skeleton-contract'
HELPER=Path(__file__).with_name('frozen_skeleton_surface123.py')


def prepare():
    verify();OUT.mkdir(exist_ok=False)
    import smplx.body_models
    lbs=importlib.import_module('smplx.lbs')
    sources=[Path(__file__),HELPER,Path(smplx.body_models.__file__),Path(lbs.__file__)]
    io.save(OUT/'plan.json',dict(created=io.now(),frames=59,cpu_only=True,max_seconds=180,
        source_hashes={str(p):io.sha(p) for p in sources},gradient_sha256=io.sha(AUDIT/'offset-gradient-only.npz'),
        causal_variable='Additional surface displacement is transported by frozen original LBS transforms; skeletal joints are not recomputed from the clothing delta.',
        unchanged=['current120k baseline mesh','pose','shape','expression','cameras','Gaussian parameters','input pixels','initial shared offset'],
        maximum_hypothetical_delta_m=.001,optimizer_updates=0,no_candidate_export=True,
        thresholds=dict(saved_geometry_max_abs_m=2e-6,oracle_geometry_max_abs_m=2e-6,
            skeletal_joints_max_abs_m=2e-6,gradient_relative_L2=2e-5),
        next='A successful contract is not surface fitting. Register teacher support, image protections and a short separate pilot before mesh optimization.',quality_accepted=False))
    print('FROZEN_SKELETON_CONTRACT_PREPARED',flush=True)


def checked():
    verify();p=io.read(OUT/'plan.json')
    for f,h in p['source_hashes'].items():assert io.sha(f)==h,f
    assert io.sha(AUDIT/'offset-gradient-only.npz')==p['gradient_sha256']
    return p


def run():
    p=checked();assert not (OUT/'result.json').exists();started=time.monotonic()
    import torch,roma
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    sys.path.insert(0,str(io.PROJECT/'src/avatar_core'))
    from utils.graphics_utils import compute_face_orientation
    torch.set_num_threads(4)
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as z:fit=dict(z)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:offset=z['static_offset']
    with np.load(SNAP/'current-meshes.npz',allow_pickle=False) as z:saved=dict(z)
    with np.load(AUDIT/'offset-gradient-only.npz',allow_pickle=False) as z:g=torch.tensor(z['guide'].mean(0))
    bridge=FrozenSkeletonSurface(fit,offset,io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    zero=torch.zeros_like(bridge.initial_offset);delta=-g*(p['maximum_hypothetical_delta_m']/g.norm(dim=-1).max())
    rows=io.read(ROOT/'data/training-inputs.json')['records'];frames=[]
    for i,row in enumerate(rows):
        assert time.monotonic()-started<p['max_seconds']
        slot=bridge.slots[row['frame']];assert row['frame']==str(saved['frame_ids'][i])
        v0=bridge.vertices(zero,slot)
        saved_parity=float((v0-torch.tensor(saved['vertices_camera'][i])).abs().max())
        assert saved_parity<p['thresholds']['saved_geometry_max_abs_m']
        d=delta.clone().requires_grad_();v=bridge.vertices(d,slot);oracle,j=bridge.independent_original_forward(d,slot)
        oracle_parity=float((v-oracle).abs().max());joint_parity=float((bridge.joints(slot)-j).abs().max())
        assert oracle_parity<p['thresholds']['oracle_geometry_max_abs_m']
        assert joint_parity<p['thresholds']['skeletal_joints_max_abs_m']
        weight=torch.sin(torch.arange(v.numel(),dtype=v.dtype)).reshape_as(v)/len(v)
        a=torch.autograd.grad((v*weight).sum(),d)[0]
        b=torch.autograd.grad((oracle*weight).sum(),d)[0]
        rel=float((a-b).norm()/b.norm().clamp_min(1e-15));assert rel<p['thresholds']['gradient_relative_L2']
        frames.append(dict(frame=row['frame'],saved_mesh_parity_m=saved_parity,original_LBS_oracle_parity_m=oracle_parity,
            fixed_skeleton_parity_m=joint_parity,gradient_relative_L2=rel))
        print('FROZEN_SKELETON_PARITY',i+1,row['frame'],flush=True)
    # Same registered perturbation as the previous moving-skeleton diagnostic.
    state=torch.load(SNAP/'checkpoint.pt',map_location='cpu',weights_only=False)['capture']
    local=state[1].detach();binding=state[7].long();localS=state[4].detach().exp()
    q=state[5].detach();q=q/q.norm(dim=-1,keepdim=True);localR=roma.unitquat_to_rotmat(q[:,[1,2,3,0]])
    def basis(v):
        R,s=compute_face_orientation(v,bridge.faces,return_scale=True)
        return v[bridge.faces].mean(1)[binding],R[binding],s[binding].clamp_min(1e-8)
    def xyz(b,u):return b[0]+(b[1]@u[:,:,None]).squeeze(-1)*b[2]
    def cov(b,R,s):
        A=(b[1]@R)*(b[2]*s)[:,None,:];return A@A.transpose(1,2)
    def project(x,row):
        E=torch.tensor(row['w2c'],dtype=x.dtype);K=torch.tensor(row['K'],dtype=x.dtype)
        y=(x@E[:3,:3].T+E[:3,3])@K.T;return y[:,:2]/y[:,2:].clamp_min(1e-9)-.5
    stats=lambda x:dict(zip(['median','p90','p99','max'],map(float,np.quantile(x,[.5,.9,.99,1]))))
    all_response=[]
    with torch.no_grad():
        reference=bridge.slots['frame_000033'];old=basis(bridge.vertices(zero,reference));new=basis(bridge.vertices(delta,reference))
        target=xyz(old,local);u=(new[1].transpose(1,2)@((target-new[0])/new[2])[:,:,None]).squeeze(-1)
        R=new[1].transpose(1,2)@old[1]@localR;s=old[2]*localS/new[2]
        for row in rows:
            slot=bridge.slots[row['frame']];old=basis(bridge.vertices(zero,slot));new=basis(bridge.vertices(delta,slot))
            x0=xyz(old,local);x1=xyz(new,u);uv=project(x0,row);w,h=row['wh']
            support=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            c0=cov(old,localR,localS);c1=cov(new,R,s)
            all_response.append(dict(frame=row['frame'],mesh_motion_mm=stats((bridge.vertices(delta,slot)-bridge.vertices(zero,slot)).norm(dim=-1).numpy()*1000),
                reference_rebased_center_motion_mm=stats((x1-x0).norm(dim=-1).numpy()*1000),
                reference_rebased_center_projection_px=stats((project(x1,row)-uv).norm(dim=-1)[support].numpy()),
                covariance_relative_Frobenius=stats(((c1-c0).reshape(-1,9).norm(dim=1)/c0.reshape(-1,9).norm(dim=1).clamp_min(1e-20)).numpy())))
    # Exact differential of the cached linear map, independently in float64.
    A=bridge.linear.double();d=torch.zeros_like(zero,dtype=torch.float64,requires_grad=True)
    w=torch.cos(torch.arange(bridge.base_vertices.numel(),dtype=torch.float64)).reshape_as(bridge.base_vertices)/bridge.base_vertices.numel()
    fn=lambda x:((A@x[None,:,:,None]).squeeze(-1)*w).sum()
    grad=torch.autograd.grad(fn(d),d)[0];finite=[]
    for index in [(0,0),(5000,1),(10474,2)]:
        plus=d.detach().clone();minus=plus.clone();plus[index]+=1e-5;minus[index]-=1e-5
        fd=float((fn(plus)-fn(minus))/(2e-5));actual=float(grad[index]);assert abs(fd-actual)<1e-12
        finite.append(dict(index=list(index),finite_difference=fd,autograd=actual))
    assert not torch.cuda.is_initialized()
    checked()
    np.savez_compressed(OUT/'frozen-lbs-cache.npz',frame_ids=np.asarray(bridge.ids),
        base_vertices=bridge.base_vertices.numpy(),linear=bridge.linear.numpy(),skeletal_joints=bridge.base_skeleton.numpy(),
        full_model_joints=bridge.full_model_joints.numpy(),faces=fit['faces'],initial_offset=offset)
    result=dict(state='completed_original_LBS_contract',updated=io.now(),frames=59,parity=frames,
        binding_response=all_response,finite_difference=finite,source_plan_sha256=io.sha(OUT/'plan.json'),
        cache_sha256=io.sha(OUT/'frozen-lbs-cache.npz'),elapsed_cpu_seconds=time.monotonic()-started,
        skeletal_joints_frozen=True,skin_landmarks_not_claimed_frozen=True,optimizer_updates=0,
        trained_surface=False,cuda_initialized=False,production_source_changed=False,quality_accepted=False,
        note='New local clothing-only displacement semantics, not original GaussianAvatars defaults. Original SMPL-X solves pose and skinning; no new pose solver. Cache is input data for a separately registered experiment, not a fitted output.')
    io.save(OUT/'result.json',result);print('FROZEN_SKELETON_CONTRACT_COMPLETE',result['elapsed_cpu_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args();globals()[a.mode]()
