"""One paired accumulation control; forward and derivative target are frozen."""
import fcntl,importlib.util,os,signal,sys,time,traceback
from pathlib import Path
import run_official_gaussians123 as io
from probe_native_mesh_mask123 import OUT,VHAP,checked,active_gpu_pids,process

OUT=OUT/'isolated-atomic-control'


def load_variants():
    value=io.read(OUT/'build-result.json');modules={}
    for row in value['builds']:
        assert io.sha(row['module'])==row['sha256']
        spec=importlib.util.spec_from_file_location('shuolee_aa_grad123_'+row['name'],row['module'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules[row['name']]=module
    return modules


def replay():
    import torch
    import nvdiffrast.torch.ops as ops
    sys.path.insert(0,str(VHAP))
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    from native_mesh_mask123 import native_alpha
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    modules=load_variants();original=ops._nvdiffrast_c.antialias_grad
    renderer=NVDiffRenderer(lighting_type='constant');d='cuda'
    v=torch.tensor([[[-.15,-.10,2.],[.16,-.12,2.],[.01,.15,2.]]],device=d)
    f=torch.tensor([[0,1,2]],device=d,dtype=torch.long)
    K=torch.tensor([[[300.,0.,47.3],[0.,310.,37.2],[0.,0.,1.]]],device=d)
    E=torch.diag(torch.tensor([1.,-1.,-1.,1.],device=d))[None];dx=torch.tensor([1.,0.,0.],device=d)
    with torch.no_grad():
        target,_=native_alpha(renderer,v+.01*dx,f,E,K,(96,128))
        base,_=native_alpha(renderer,v,f,E,K,(96,128))
        plus,_=native_alpha(renderer,v+1e-4*dx,f,E,K,(96,128))
        minus,_=native_alpha(renderer,v-1e-4*dx,f,E,K,(96,128))
        signs=torch.sign(base-target)
        expected=float(((plus-minus)*signs).sum()/target.sum()/2e-4)
    rows=[];started=time.monotonic()
    try:
        for name,fn in [('installed',original),('recompiled_original',modules['original'].antialias_grad),('direct_atomic',modules['direct'].antialias_grad)]:
            ops._nvdiffrast_c.antialias_grad=fn
            values=[];parity=[]
            for repeat in range(6):
                assert active_gpu_pids()<={331,os.getpid()}
                shift=torch.tensor(0.,device=d,requires_grad=True)
                a,_=native_alpha(renderer,v+shift*dx,f,E,K,(96,128))
                loss=(a*signs).sum()/target.sum()
                values.append(float(torch.autograd.grad(loss,shift)[0]))
                parity.append(float((a-base).abs().max()))
                assert time.monotonic()-started<60
            rows.append(dict(backend=name,gradients=values,gradient_range=max(values)-min(values),
                alpha_max_abs=max(parity),finite_difference=expected,
                maximum_relative_error=max(abs(g-expected)/abs(expected) for g in values)))
    finally:ops._nvdiffrast_c.antialias_grad=original
    fixed=rows[-1]
    passed=fixed['maximum_relative_error']<=.05 and fixed['alpha_max_abs']<=1e-6 and fixed['gradient_range']<=1e-4
    result=dict(updated=io.now(),state='direct_atomic_synthetic_pass' if passed else 'direct_atomic_synthetic_failed',
        rows=rows,thresholds=dict(relative_derivative=.05,absolute_alpha=1e-6,repeat_gradient_range=1e-4),
        optimizer_updates=0,forward_changed=False,installed_runtime_modified=False,
        original_baseline_modified=False,synthetic_pass=passed,full59_mask_contract_pass=False,quality_accepted=False,
        interpretation='The two accumulation-call control tests this specific installed source/toolchain/GPU. It does not establish camera calibration, facial fit, or clothing geometry.',
        elapsed_seconds=time.monotonic()-started)
    io.save(OUT/'synthetic-result.json',result);print(result,flush=True)


if __name__=='__main__':
    checked();assert not (OUT/'synthetic-plan.json').exists()
    io.save(OUT/'synthetic-plan.json',dict(created=io.now(),script_sha256=io.sha(__file__),
        build_result_sha256=io.sha(OUT/'build-result.json'),backward_only=True,
        frozen_triangle=True,frozen_K=True,fixed_sign_loss=True,epsilon_m=1e-4,repeats=6,
        relative_derivative_max=.05,repeat_gradient_range_max=1e-4,optimizer_updates=0,max_gpu_seconds=60))
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(331);assert before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert active_gpu_pids()<={331}
    try:
        os.kill(331,signal.SIGSTOP)
        for _ in range(50):
            if process(331)['state']=='T':break
            time.sleep(.05)
        assert process(331)['state']=='T';replay()
    except BaseException as error:
        io.save(OUT/'synthetic-failure.json',dict(updated=io.now(),error=repr(error),traceback=traceback.format_exc()));raise
    finally:
        foreign=active_gpu_pids()-{331,os.getpid()}
        if not foreign:os.kill(331,signal.SIGCONT)
        io.save(OUT/'synthetic-handoff.json',dict(updated=io.now(),before=before,after=process(331),foreign=list(foreign)))
