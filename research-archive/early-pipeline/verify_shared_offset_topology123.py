"""Check all94 saved poses against original CUDA integer parent residency."""
from pathlib import Path
import argparse
import fcntl
import os
import sys
import time
import traceback
import numpy as np
import torch
import run_official_gaussians123 as io


def run(plan_path):
    plan=io.read(plan_path);root=Path(plan['source_root']);out=Path(plan['output_root'])
    assert not (out/'progress.json').exists()
    start=time.monotonic()
    def progress(state,**values):
        io.save(out/'progress.json',dict(state=state,pid=None if state.startswith('complete') or state.startswith('failed') else os.getpid(),
            updated=io.now(),training_started=False,elapsed_seconds=time.monotonic()-start,**values))
    def verify():
        for path,digest in plan['source_hashes'].items():assert io.sha(io.PROJECT/path)==digest,path
        for path,digest in plan['input_hashes'].items():assert io.sha(root/path)==digest,path
        assert io.sha(__file__)==plan['verification_source_sha256']
    progress('checking')
    try:
        verify()
        with open('/tmp/shuolee-native122-gpu.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
            sys.path.insert(0,str(io.PROJECT/'src'))
            from shuolee_portrait import runtime as rt
            rt.configure(io.ASSETS)
            from scene.smplx_gaussian_model import SMPLXGaussianModel
            model=SMPLXGaussianModel(3);model.load_fresh_meshes(root/'fit/fitted.npz')
            initial=model.enable_shared_static_offset(root/'fit/fitted.npz',io.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
            with np.load(root/'gaussian-run/step-018000/shared-static-offset.npz',allow_pickle=False) as z:
                final=torch.as_tensor(z['static_offset'],device='cuda').clone()
            with torch.no_grad():model.static_offset.copy_(final)
            host=model.offset_body.parents
            assert host.device.type=='cpu' and host.dtype==torch.int64
            original=host.cuda();assert torch.equal(host,original.cpu())
            assert model.num_timesteps==94 and all(p.device.type=='cuda' for p in model.offset_body.parameters())
            adjoint=torch.randn((10475,3),device='cuda',generator=torch.Generator(device='cuda').manual_seed(6030))
            records=[]
            for i in range(model.num_timesteps):
                if time.monotonic()-start>plan['max_seconds']:raise TimeoutError('All-pose topology budget exceeded')
                model.offset_body.parents=original
                a=model.shared_offset_vertices(i)
                ga=torch.autograd.grad(a,model.static_offset,grad_outputs=adjoint)[0]
                model.offset_body.parents=host
                b=model.shared_offset_vertices(i)
                gb=torch.autograd.grad(b,model.static_offset,grad_outputs=adjoint)[0]
                row=dict(pose_index=i,vertices_max_abs_m=float((a-b).abs().max()),
                    gradient_max_abs=float((ga-gb).abs().max()),gradient_relative_L2=float((ga-gb).norm()/ga.norm().clamp_min(1e-12)),
                    finite=bool(torch.isfinite(ga).all() and torch.isfinite(gb).all()))
                records.append(row)
                if i%10==0:progress('verifying',poses_completed=i+1)
                io.save(out/'partial-poses.json',records)
                assert row['vertices_max_abs_m']==row['gradient_max_abs']==0 and row['finite']
            assert torch.equal(model.static_offset,final)
            verify()
            result=dict(state='completed_exact_parity',updated=io.now(),poses=len(records),records=records,
                zero_offset_preflight=initial,all_vertices_and_fixed_adjoint_gradients_identical=True,
                all_learned_parameters_unchanged=True,optimizer_created=False,optimizer_updates=0,
                float_model_parameters_on_cuda=True,parent_indices_on_host=True,plan_sha256=io.sha(plan_path),
                elapsed_seconds=time.monotonic()-start,production_source_hashes=plan['source_hashes'],
                quality_improvement_claimed=False,end_to_end_timing_accepted=False,
                limitation='Numerical equivalence of the exact changed LBS input on94 current poses. Raster-backward nondeterminism remains; no new training, fit, covariance or camera change.')
            io.save(out/'result.json',result);progress('completed_exact_parity')
            print(__import__('json').dumps(dict(state=result['state'],poses=len(records),seconds=result['elapsed_seconds'])),flush=True)
    except BaseException as error:
        io.save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),training_started=False))
        progress('failed_preserved');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);run(p.parse_args().plan)
