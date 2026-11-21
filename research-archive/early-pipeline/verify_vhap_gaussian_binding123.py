"""All59 posed binding checks after the fitted-offset loader repair.

No optimizer, image training, quality promotion or exported point cloud. Known
synthetic local Gaussian values only exercise the original coordinate map.
"""
from pathlib import Path
import fcntl
import os
import sys
import time
import traceback
import numpy as np
import torch
from run_official_gaussians123 import PROJECT,ASSETS,read,save,sha,now

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
OUT=ROOT/'smplify-vhap-gaussian-binding-verification/completed'
PLAN=ROOT/'smplify-vhap-gaussian-binding-plan.json'


def quaternion_matrix(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)


def main():
    plan=read(PLAN)
    assert plan['source_sha256']==sha(__file__) and plan['optimizer_updates']==0
    assert not OUT.exists(),'Preserve completed verification; do not repeat.'
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();start=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(6060);np.random.seed(6060)
    torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime
    runtime.configure(ASSETS)
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    fixture=ROOT/'smplify-vhap-direct-body-preflight/fitted-nonzero-offset-fixture.npz'
    with np.load(fixture,allow_pickle=False) as z:saved=dict(z)
    assert list(saved['frame_ids'])==plan['roster']
    model=SMPLXGaussianModel(3);model.load_fresh_meshes(fixture)
    model.enable_shared_static_offset(fixture,ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    init=model.initialize_surface(count=20908,seed=6060)
    assert torch.equal(model.binding,torch.arange(20908,device='cuda'))
    assert torch.equal(model.binding_counter,torch.ones(20908,dtype=torch.int32,device='cuda'))
    assert float(model._xyz.abs().max())==0
    assert float((model.get_xyz-model.face_center).abs().max())==0
    cached_gradient=torch.autograd.grad(model.get_xyz.sum(),model.static_offset,allow_unused=True)[0]
    assert cached_gradient is None  # Enable alone doesn't refresh the graph.
    offset_hash=sha(fixture)
    # An independent test object receives deterministic local probes. These
    # are neither learned parameters nor a candidate for quality evaluation.
    k=torch.arange(20908,device='cuda',dtype=torch.float32)
    local=torch.stack([.04*torch.sin(k),.02*torch.cos(k),.03*torch.sin(.5*k)],dim=-1)
    sigma=torch.tensor([.75,.3,1.1],device='cuda').expand(20908,-1)
    angle=.2+.1*torch.sin(k)
    axis=torch.tensor([1.,2.,3.],device='cuda');axis=axis/axis.norm()
    rotation=torch.cat([torch.cos(angle/2)[:,None],torch.sin(angle/2)[:,None]*axis],dim=1)
    with torch.no_grad():
        model._xyz.copy_(local);model._scaling.copy_(sigma.log());model._rotation.copy_(rotation)
    local_np=local.cpu().numpy().astype(float);sigma_np=sigma.cpu().numpy().astype(float)
    local_rot=quaternion_matrix(rotation.cpu().numpy().astype(float))
    faces=saved['faces'];records=[]
    for i,frame in enumerate(plan['roster']):
        assert time.monotonic()-start<plan['max_seconds']
        model.select_mesh_by_timestep(i)
        gradient=torch.autograd.grad(model.get_xyz.sum(),model.static_offset,retain_graph=False)[0]
        assert torch.isfinite(gradient).all() and float(gradient.abs().max())>0
        vertices=saved['vertices_world'][i].astype(float)
        tri=vertices[faces];center=tri.mean(1);e0=tri[:,1]-tri[:,0];e1=tri[:,2]-tri[:,0]
        a0=e0/np.linalg.norm(e0,axis=1,keepdims=True)
        normal=np.cross(a0,e1);a1=normal/np.linalg.norm(normal,axis=1,keepdims=True)
        a2=-np.cross(a1,a0)
        face_rot=np.stack([a0,a1,a2],axis=-1)
        scale=(np.linalg.norm(e0,axis=1)+np.abs(np.sum(a2*e1,axis=1)))/2
        scale=np.maximum(scale,1e-8)
        expected_xyz=np.einsum('nij,nj->ni',face_rot,local_np)*scale[:,None]+center
        world_rot=quaternion_matrix(model.get_rotation.detach().cpu().numpy().astype(float))
        expected_rot=face_rot@local_rot
        actual_scale=model.get_scaling.detach().cpu().numpy().astype(float)
        expected_scale=scale[:,None]*sigma_np
        expected_cov=(expected_rot*expected_scale[:,None,:]**2)@expected_rot.transpose(0,2,1)
        actual_cov=(world_rot*actual_scale[:,None,:]**2)@world_rot.transpose(0,2,1)
        xyz_error=float(np.max(np.abs(model.get_xyz.detach().cpu().numpy()-expected_xyz)))
        rotation_error=float(np.max(np.abs(world_rot-expected_rot)))
        scale_error=float(np.max(np.abs(actual_scale-expected_scale)))
        covariance_relative=float(np.max(np.linalg.norm(actual_cov-expected_cov,axis=(1,2))/np.linalg.norm(expected_cov,axis=(1,2))))
        assert xyz_error<3e-6 and rotation_error<1e-4 and scale_error<3e-6 and covariance_relative<1e-4
        assert np.all(np.linalg.eigvalsh(actual_cov)>0)
        # Inspect the optional upstream Python covariance API separately. It
        # is not enabled in the current training/export pipeline.
        packed=model.get_covariance().detach().cpu().numpy().astype(float)
        python_cov=np.zeros((20908,3,3))
        for j,(u,v) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
            python_cov[:,u,v]=python_cov[:,v,u]=packed[:,j]
        optional_error=float(np.max(np.linalg.norm(python_cov-expected_cov,axis=(1,2))/np.linalg.norm(expected_cov,axis=(1,2))))
        records.append(dict(frame=frame,gaussians=20908,xyz_max_abs_m=xyz_error,rotation_max_abs=rotation_error,
            scale_max_abs_m=scale_error,covariance_max_relative=covariance_relative,
            offset_gradient_max_abs=float(gradient.abs().max()),optional_python_covariance_max_relative=optional_error))
        save(OUT/'progress.json',dict(updated=now(),state='verifying_binding',pid=os.getpid(),frames=i+1,optimizer_updates=0))
    assert sha(fixture)==offset_hash
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    result=dict(updated=now(),state='active_binding_contract_passed',frames=59,initialization=init,
        gaussians_per_triangle=1,optimizer_updates=0,synthetic_local_probes_only=True,
        actual_render_pipeline_compute_cov3D_python=False,explicit_select_required_for_offset_gradient=True,
        consumer_enable_without_select_gradient_is_none=True,records=records,
        max_xyz_error_m=max(r['xyz_max_abs_m'] for r in records),
        max_rotation_error=max(r['rotation_max_abs'] for r in records),
        max_covariance_relative=max(r['covariance_max_relative'] for r in records),
        optional_python_covariance_api_matches_active_path=all(r['optional_python_covariance_max_relative']<1e-4 for r in records),
        seconds=time.monotonic()-start,input_hashes=plan['input_hashes'],source_sha256=sha(__file__),
        quality_accepted=False,gaussian_training_started=False,new_model_published=False,
        limits=['This validates coordinate binding and active scale/quaternion covariance, not subject geometry or image quality.',
            'The optional Python covariance API is separately reported, and is disabled in the actual current renderer.',
            'No saved fit, raw observation, camera, image, mask or learned model was modified.'])
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state=result['state'],pid=None,frames=59,optimizer_updates=0))
    print({k:v for k,v in result.items() if k not in ('records','input_hashes')},flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():
            save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
            save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,optimizer_updates=0))
        raise
