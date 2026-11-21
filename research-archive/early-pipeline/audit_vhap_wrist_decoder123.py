"""Measure wrist/body coupling in the frozen original VPoser, all59 frames.

No fitting, no neural weight change, no pose update. Differential rotations are
measured in radians using dR R^T, avoiding axis-angle coordinate magnitudes.
"""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import time
import traceback
import numpy as np
import torch
from smplx.lbs import batch_rodrigues
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
OUT=ROOT/'smplify-vhap-wrist-decoder-audit'
PLAN=ROOT/'smplify-vhap-wrist-decoder-plan.json'


def stats(x):
    a=np.asarray(x,dtype=float)
    return dict(median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max()),minimum=float(a.min()))


def main():
    plan=read(PLAN)
    assert plan['source_sha256']==sha(__file__) and plan['optimizer_updates']==0
    assert not OUT.exists(),'Never overwrite or repeat a completed audit.'
    for path,digest in plan['input_hashes'].items():assert sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();start=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(6060);torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('wrist_decoder_vposer',PROJECT/'src/avatar_core/shuolee_vposer.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    loader.MODEL=ASSETS/'vposer/V02_05';loader.CODE=PROJECT/'third_party/human_body_prior'
    decoder,_,meta=loader.load_vposer('cuda');decoder.eval();decoder.requires_grad_(False)
    checkpoint_path=ROOT/'smplify-vhap-normalized-laplacian/epoch-30/tracked-smplx-params.npz'
    expected=read(ROOT/'smplify-vhap-sequential-stage/limb-transfer-audit/result.json')['runs']['normalized_control']['records']
    with np.load(checkpoint_path,allow_pickle=False) as z:
        roster=list(z['frame_ids']);latents=z['pose_embedding'].copy()
    assert roster==plan['roster']==[r['frame'] for r in expected]
    records=[]
    for i,frame in enumerate(roster):
        assert time.monotonic()-start<plan['max_seconds']
        code=torch.tensor(latents[i],device='cuda',requires_grad=True)
        pose=decoder.decode(code[None])['pose_body'].reshape(21,3)
        parity=float(np.max(np.abs(pose.detach().cpu().numpy().flatten()-expected[i]['body_axis_angle'])))
        assert parity<5e-5,(frame,parity)
        def rotation(c):
            return batch_rodrigues(decoder.decode(c[None])['pose_body'].reshape(21,3))
        R=rotation(code)
        jac=torch.autograd.functional.jacobian(rotation,code,vectorize=True)
        assert torch.isfinite(jac).all()
        direction=torch.cos(torch.arange(32,device='cuda',dtype=torch.float32)+1)
        direction=direction/direction.norm();epsilon=.001
        with torch.no_grad():numerical=(rotation(code+epsilon*direction)-rotation(code-epsilon*direction))/(2*epsilon)
        analytic=torch.einsum('jabk,k->jab',jac,direction)
        derivative_rel=float((numerical-analytic).norm()/analytic.norm().clamp_min(1e-8))
        assert derivative_rel<.03,(frame,derivative_rel)
        r=R.detach().cpu().numpy().astype(float)
        j=jac.detach().cpu().numpy().astype(float)
        tangent=np.einsum('jabk,jcb->jack',j,r)
        omega=np.stack([(tangent[:,2,1]-tangent[:,1,2])/2,
                        (tangent[:,0,2]-tangent[:,2,0])/2,
                        (tangent[:,1,0]-tangent[:,0,1])/2],axis=1)
        item=dict(frame=frame,decoded_axis_angle_parity_rad=parity,jacobian_relative_error=derivative_rel,wrists={})
        for side,index,elbow in [('left',19,17),('right',20,18)]:
            wrist=omega[index]
            keep=[k for k in range(21) if k!=index]
            other=omega[keep].reshape(60,32)
            _,s,vh=np.linalg.svd(other,full_matrices=False)
            active=s>s[0]*1e-6;rank=int(active.sum())
            row=dict(other_joint_jacobian_rank=rank,latent_dimensions=32,
                     exact_other_joint_preserving_latent_nullity=32-rank,
                     wrist_singular_values=np.linalg.svd(wrist,compute_uv=False).tolist(),
                     other_condition_number=float(s[0]/s[active][-1]))
            if rank==32:
                # B X has orthonormal columns. Minimize ||B dz|| subject to
                # W dz=e, for each infinitesimal local wrist angular axis.
                whitened=vh.T/s[None,:]
                w=wrist@whitened
                singular=np.linalg.svd(w,compute_uv=False)
                assert singular[-1]>singular[0]*1e-6
                latent=whitened@w.T@np.linalg.inv(w@w.T)
                equality_error=float(np.max(np.abs(wrist@latent-np.eye(3))))
                assert equality_error<1e-7
                leakage=omega@latent
                row.update(unit_wrist_constraint_error=equality_error,
                    minimum_other_joint_angular_leakage_per_unit_wrist=np.linalg.norm(other@latent,axis=0).tolist(),
                    same_side_elbow_angular_leakage_per_unit_wrist=np.linalg.norm(leakage[elbow],axis=0).tolist(),
                    latent_norm_per_unit_wrist=np.linalg.norm(latent,axis=0).tolist())
            item['wrists'][side]=row
        records.append(item)
        save(OUT/'progress.json',dict(updated=now(),state='auditing_read_only',pid=os.getpid(),frames=len(records),optimizer_updates=0))
    for path,digest in plan['input_hashes'].items():assert sha(path)==digest,path
    summary={}
    for side in ('left','right'):
        rows=[r['wrists'][side] for r in records]
        full=[r for r in rows if r['other_joint_jacobian_rank']==32]
        summary[side]=dict(frames_without_exact_independent_wrist_direction=len(full),
            other_angular_leakage_per_unit_wrist=stats([x for r in full for x in r['minimum_other_joint_angular_leakage_per_unit_wrist']]) if full else None,
            elbow_angular_leakage_per_unit_wrist=stats([x for r in full for x in r['same_side_elbow_angular_leakage_per_unit_wrist']]) if full else None)
    result=dict(updated=now(),state='complete_read_only',frames=59,optimizer_updates=0,
        source_sha256=sha(__file__),input_hashes=plan['input_hashes'],checkpoint_sha256=sha(checkpoint_path),vposer=meta,
        seconds=time.monotonic()-start,summary=summary,records=records,quality_accepted=False,
        units='Infinitesimal joint-relative angular derivatives in radians, dR R^T in each parent frame. Ratios are unitless. No motion applied.',
        limits=['Dense latent coupling is an architectural property; it does not prove a unique cause of the visible wrong elbow.',
            'The equality-constrained calculation only measures local capacity; it is not a new fitting solver or an optimizer update.',
            'Changing to direct body pose changes the prior and temporal metric; it cannot be called a neutral interface change.',
            'No training quality or15min end-to-end completion follows from this diagnostic.'])
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state='complete_read_only',pid=None,frames=59,optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','input_hashes','vposer')}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        raise
