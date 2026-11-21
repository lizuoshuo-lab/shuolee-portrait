"""Reproduce one failed AA derivative with identical inputs; no fitting."""
from pathlib import Path
import fcntl,os,signal,sys,time,traceback
import numpy as np
import run_official_gaussians123 as io
from probe_native_mesh_mask123 import OUT,VHAP,checked,active_gpu_pids,process


def replay():
    import torch
    import nvdiffrast.torch as dr
    import nvdiffrast.torch.ops as ops
    sys.path.insert(0,str(VHAP))
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    from native_mesh_mask123 import native_alpha
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    renderer=NVDiffRenderer(lighting_type='constant');device='cuda'
    v=torch.tensor([[[-.15,-.10,2.],[.16,-.12,2.],[.01,.15,2.]]],device=device)
    f=torch.tensor([[0,1,2]],device=device,dtype=torch.long)
    K=torch.tensor([[[300.,0.,47.3],[0.,310.,37.2],[0.,0.,1.]]],device=device)
    E=torch.diag(torch.tensor([1.,-1.,-1.,1.],device=device))[None]
    dx=torch.tensor([1.,0.,0.],device=device)
    uv=torch.full((3,2),.5,device=device);tex=torch.ones((1,3,16,16),device=device)
    epsilon=1e-4
    with torch.no_grad():
        target,_=native_alpha(renderer,v+.01*dx,f,E,K,(96,128))
        base,_=native_alpha(renderer,v,f,E,K,(96,128))
        plus,_=native_alpha(renderer,v+epsilon*dx,f,E,K,(96,128))
        minus,_=native_alpha(renderer,v-epsilon*dx,f,E,K,(96,128))
        fixed_sign=torch.sign(base-target)
        numeric=float(((plus-minus)*fixed_sign).sum()/target.sum()/(2*epsilon))
    rows=[];started=time.monotonic()
    for method in ['original_rgba','alpha_adapter','direct_cloned_1channel','direct_cloned_4channel']:
        gradients=[];positions=[];parities=[]
        for repeat in range(6):
            assert time.monotonic()-started<60
            assert active_gpu_pids()<={331,os.getpid()}
            shift=torch.tensor(0.,device=device,requires_grad=True)
            vv=v+shift*dx
            if method=='alpha_adapter':
                alpha,rast=native_alpha(renderer,vv,f,E,K,(96,128))
            else:
                rast=renderer.rasterize(vv,f,E,K,(96,128),use_cache=False)
                if method=='original_rgba':
                    alpha=renderer.render_rgba(rast,vv,f,uv,f,tex,None,enable_disturbance=False)['rgba'][...,3:]
                else:
                    channels=1 if method.endswith('1channel') else 4
                    foreground=(rast['rast_out'][...,-1:]>0).to(v.dtype)
                    colours=foreground.repeat(1,1,1,channels).contiguous()
                    clip=rast['verts_clip'].clone().contiguous()
                    raw=rast['rast_out'].clone().contiguous()
                    alpha=dr.antialias(colours,raw,clip,f.int())[...,-1:].flip(1)
            loss=(alpha*fixed_sign).sum()/target.sum()
            grad,pos=torch.autograd.grad(loss,(shift,rast['verts_clip']))
            torch.cuda.synchronize()
            gradients.append(float(grad));positions.append(pos.detach().cpu().numpy().tolist())
            parities.append(float((alpha-base).abs().max()))
        rows.append(dict(method=method,repeats=6,gradients=gradients,position_gradients=positions,
            range=max(gradients)-min(gradients),alpha_parity_max=max(parities),
            finite_difference=numeric,max_relative_derivative_error=max(abs(g-numeric)/abs(numeric) for g in gradients)))
    install=Path('/opt/gs-slam-lab/shuolee-portrait-clean-001/env/install/native-lt2pb6wk/nvdiffrast')
    sources=[install/'csrc/common/antialias.cu',install/'csrc/common/common.h',install/'csrc/torch/torch_antialias.cpp',Path(ops.__file__)]
    io.save(OUT/'gradient-repeatability-measurements.json',dict(updated=io.now(),rows=rows,optimizer_updates=0))
    result=dict(updated=io.now(),state='fixed_input_antialias_repeatability_measured',rows=rows,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
        cuda_kernel_plugin=getattr(getattr(ops,'_nvdiffrast_c',None),'__file__',None),
        installed_source_hashes={str(p):io.sha(p) for p in sources},
        exact_source_install=io.read(Path(ops.__file__).parents[2]/'nvdiffrast-0.4.0.dist-info/direct_url.json'),
        epsilon_m=epsilon,loss='fixed base sign linear objective; no L1 nondifferentiability in this objective',
        tf32=False,threads=4,optimizer_updates=0,quality_accepted=False,
        elapsed_seconds=time.monotonic()-started)
    io.save(OUT/'gradient-repeatability.json',result)
    print({r['method']:dict(range=r['range'],gradients=r['gradients'],finite_difference=numeric,alpha_parity=r['alpha_parity_max']) for r in rows},flush=True)


if __name__=='__main__':
    checked();assert (OUT/'gradient-repeatability-failure.json').exists()
    assert 'no attribute' in io.read(OUT/'gradient-repeatability-failure.json')['error']
    assert not (OUT/'gradient-repeatability-provenance-repair-plan.json').exists()
    io.save(OUT/'gradient-repeatability-provenance-repair-plan.json',dict(created=io.now(),source_sha256=io.sha(__file__),
        only_repair='Record installed0.4 extension metadata rather than the old JIT API. Measurement logic unchanged. Preserve the initial script failure.',
        cases=['unchanged original RGBA','alpha-only adapter','cloned direct AA1channel','cloned direct AA4channel'],
        repetitions=6,synthetic_triangle_and_K_unchanged=True,epsilon_m=1e-4,
        objective='fixed base sign linear loss',cpu_threads=4,tf32=False,
        no_model_or_coefficient_sweep=True,no_optimizer=True,max_gpu_seconds=60))
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(331)
    assert before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert 'train_demo_body_upstream_offset123.py train' in before['command']
    assert active_gpu_pids()<={331}
    try:
        os.kill(331,signal.SIGSTOP)
        for _ in range(50):
            if process(331)['state']=='T':break
            time.sleep(.05)
        assert process(331)['state']=='T'
        replay()
    except BaseException as error:
        io.save(OUT/'gradient-repeatability-failure.json',dict(updated=io.now(),error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        foreign=active_gpu_pids()-{331,os.getpid()}
        if not foreign:os.kill(331,signal.SIGCONT)
        io.save(OUT/'gradient-repeatability-handoff.json',dict(updated=io.now(),before=before,after=process(331),foreign=list(foreign)))
