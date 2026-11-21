"""Bounded zero-update proof for a colour-direction-only upstream adapter."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from export_demo_body_snapshot123 import ROOT, process
from shuolee_portrait import runtime as rt

PARENT=ROOT/'root-gauge-audit';OUT=PARENT/'colour-only-probe';SNAP=ROOT/'gaussian-run/step-060000'
HELPER=Path(__file__).parent/'body_root_sh_colors123.py'


def plan():
    OUT.mkdir(exist_ok=False)
    assert rt.read(PARENT/'transport-cpu-result.json')['passed']
    inputs=[HELPER,SNAP/'bound-local.ply',SNAP/'binding.npz',SNAP/'current-meshes.npz',
            ROOT/'data/training-inputs.json',PARENT/'candidate-inputs/transforms.npz']
    inputs += [Path(p) for p in rt.read(ROOT/'plan.json')['source_hashes']]
    rt.save(OUT/'plan.json',dict(created=rt.now(),input_hashes={str(p):rt.sha(p) for p in inputs},
        causal_variable='SH input direction in fixed reference-root coordinates only',
        unchanged=['camera','K','mesh','Gaussian geometry/covariance/opacity','binding','losses','optimizer','schedule'],
        interface='upstream render(..., override_color=...)',frames=59,gaussian_updates=0,
        proof_controls=['Unrotated Python SH versus original CUDA SH, all59',
            'SH0 invariant to root transform, all59','Reference frame33 unchanged',
            'White override bitwise equal, all59','Autograd versus finite difference for SH and direction'],
        criteria=dict(rgb_mean_abs=2e-6,rgb_max_abs=.005,white_max_abs=0.,gradient_relative=1e-5),
        gpu_budget_seconds=180,training_changed=False,trained_quality_claim=False,
        old_coefficients_are_not_a_new_trained_model=True))
    print('COLOUR_DIRECTION_PROBE_REGISTERED',flush=True)


def checked():
    p=rt.read(OUT/'plan.json')
    for f,h in p['input_hashes'].items():assert rt.sha(f)==h,f
    return p


def replay(pid):
    p=checked();assert process(pid)[1]=='T' and not (OUT/'result.json').exists()
    import torch
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from shuolee_portrait.training import camera,image
    from body_root_sh_colors123 import root_frame_colors,render_root_frame
    from utils.sh_utils import eval_sh
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    rows=rt.read(ROOT/'data/training-inputs.json')['records'];mesh=dict(np.load(SNAP/'current-meshes.npz'))
    assert all(np.array_equal(np.array(r['w2c']),np.eye(4)) for r in rows)
    model=SMPLXGaussianModel(3);model.load_ply(str(SNAP/'bound-local.ply'))
    model.vertices_sequence=torch.tensor(mesh['vertices_camera'],device='cuda',dtype=torch.float32)
    model.faces=torch.tensor(mesh['faces'],device='cuda',dtype=torch.long);model.num_timesteps=59;model.static_offset=None
    G=torch.tensor(np.load(PARENT/'candidate-inputs/transforms.npz')['G'][:,:3,:3],device='cuda',dtype=torch.float32)
    identity=torch.eye(3,device='cuda');pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    background=torch.zeros(3,device='cuda');white=torch.ones_like(model._xyz);checks=[]
    started=time.monotonic()
    def error(a,b):
        d=(a-b).abs();m,x=float(d.mean()),float(d.max())
        assert m<p['criteria']['rgb_mean_abs'] and x<p['criteria']['rgb_max_abs'],(m,x)
        return dict(mean_abs=m,max_abs=x)
    with torch.no_grad():
        for i,r in enumerate(rows):
            model.select_mesh_by_timestep(i);cam,_=camera(r,i,pose_index=i);model.active_sh_degree=3
            original=render(cam,model,pipe,background)['render']
            same=render_root_frame(render,cam,model,pipe,background,identity)['render']
            v=dict(frame=r['frame'],original_cuda_vs_python=error(original,same))
            if i==0:
                assert torch.equal(G[i],identity)
                root_reference=render_root_frame(render,cam,model,pipe,background,G[i])['render']
                v['reference_pose_unchanged']=error(original,root_reference)
            a=render(cam,model,pipe,background,override_color=white)['render']
            b=render_root_frame(render,cam,model,pipe,background,G[i],override_color=white)['render']
            assert torch.equal(a,b);v['white_override_bitwise_equal']=True
            model.active_sh_degree=0
            a=render(cam,model,pipe,background)['render']
            b=render_root_frame(render,cam,model,pipe,background,G[i])['render']
            v['SH0_coordinate_invariance']=error(a,b)
            model.active_sh_degree=3
            if r['frame'] in ('frame_000001','frame_000018','frame_000033','frame_000059'):
                changed=render_root_frame(render,cam,model,pipe,background,G[i])['render']
                # Diagnostic only: old coefficients have not been trained in this basis.
                image(changed).save(OUT/(r['frame']+'_old_coefficients_new_directions.png'))
                v['old_coefficients_not_retrained_rgb_change_mean']=float((changed-original).abs().mean())
            checks.append(v)
    # Independent small smooth test, away from clamp boundary. No optimizer.
    torch.manual_seed(12360)
    xyz=torch.randn(8,3,dtype=torch.float64,device='cuda');xyz[:,2]+=4;xyz.requires_grad_()
    coeff=(torch.randn(8,3,16,dtype=torch.float64,device='cuda')*.02).requires_grad_()
    rot=G[1].double();weights=torch.arange(1,25,dtype=torch.float64,device='cuda').reshape(8,3)/24
    def objective(x,c):
        direction=x/x.norm(dim=-1,keepdim=True);return ((eval_sh(3,c,direction@rot.T)+.5)*weights).sum()
    value=objective(xyz,coeff);gx,gc=torch.autograd.grad(value,[xyz,coeff]);results=[]
    for name,tensor,grad,indices in [('xyz',xyz,gx,[(0,0),(3,2),(7,1)]),('SH',coeff,gc,[(0,0,1),(3,2,7),(7,1,15)])]:
        for index in indices:
            plus=tensor.detach().clone();minus=tensor.detach().clone();plus[index]+=1e-6;minus[index]-=1e-6
            argsplus=(plus,coeff.detach()) if name=='xyz' else (xyz.detach(),plus)
            argsminus=(minus,coeff.detach()) if name=='xyz' else (xyz.detach(),minus)
            fd=float((objective(*argsplus)-objective(*argsminus))/(2e-6));actual=float(grad[index])
            relative=abs(fd-actual)/max(abs(fd),abs(actual),1e-6)
            assert relative<p['criteria']['gradient_relative'],(name,index,relative)
            results.append(dict(variable=name,index=list(index),finite_difference=fd,autograd=actual,relative_error=relative))
    # Small bounded render timing, same reference camera/geometry. Not E2E FPS.
    model.select_mesh_by_timestep(0);cam,_=camera(rows[0],0,pose_index=0);model.active_sh_degree=3
    def timing(fn):
        with torch.no_grad():
            for _ in range(3):fn()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):fn()
            end.record();end.synchronize();return start.elapsed_time(end)/20
    t0=timing(lambda:render(cam,model,pipe,background))
    t1=timing(lambda:render_root_frame(render,cam,model,pipe,background,G[0]))
    checked();assert model.optimizer is None and process(pid)[1]=='T'
    rt.save(OUT/'result.json',dict(updated=rt.now(),passed=True,frames=59,checks=checks,gradient_checks=results,
        forward_only_timing_ms=dict(upstream=t0,root_frame_override=t1,scope='One native reference view, 20 forwards; excludes backward/I-O/fitting, not E2E target acceptance'),
        elapsed_seconds=time.monotonic()-started,optimizer_created=False,gaussian_updates=0,
        live_training_changed=False,quality_accepted=False))
    print('ROOT_SH_COLOUR_INTERFACE_PROBE_PASSED',flush=True)


def run(pid):
    checked();lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid);assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    start=time.monotonic();rt.save(OUT/'gpu-handoff.json',dict(started=rt.now(),pid=pid,command=cmd,identity=identity,state='suspending'))
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--pid',str(pid)],check=True,timeout=180)
    finally:
        if process(pid)[2]==identity:os.kill(pid,signal.SIGCONT)
        r=rt.read(OUT/'gpu-handoff.json');r.update(finished=rt.now(),seconds=time.monotonic()-start,state='resumed',trainer_after_state=process(pid)[1])
        rt.save(OUT/'gpu-handoff.json',r);print('TRAINER_RESUMED',pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','run','replay']);p.add_argument('--pid',type=int);a=p.parse_args()
    if a.mode=='plan':plan()
    else:globals()[a.mode](a.pid)
