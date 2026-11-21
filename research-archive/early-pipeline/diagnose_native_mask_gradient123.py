"""One fixed source/branch audit of the failed synthetic mask derivative, no training."""
from pathlib import Path
import fcntl,os,signal,time
import numpy as np
import run_official_gaussians123 as io
from probe_native_mesh_mask123 import OUT,VHAP,checked,active_gpu_pids,process


def audit():
    import sys
    import torch
    sys.path.insert(0,str(VHAP))
    from vhap.util.render_nvdiffrast import NVDiffRenderer
    from native_mesh_mask123 import native_alpha,foreground_normalized_mask_loss
    renderer=NVDiffRenderer(lighting_type='constant');d='cuda'
    v=torch.tensor([[[-.15,-.10,2.],[.16,-.12,2.],[.01,.15,2.]]],device=d)
    f=torch.tensor([[0,1,2]],device=d,dtype=torch.long);x=torch.tensor([1.,0.,0.],device=d)
    K=torch.tensor([[[300.,0.,47.3],[0.,310.,37.2],[0.,0.,1.]]],device=d)
    RT=torch.diag(torch.tensor([1.,-1.,-1.,1.],device=d))[None]
    uv=torch.full((3,2),.5,device=d);tex=torch.ones((1,3,16,16),device=d)
    eps=1e-4
    with torch.no_grad():
        target,_=native_alpha(renderer,v+.01*x,f,RT,K,(96,128))
        baseline,br= native_alpha(renderer,v,f,RT,K,(96,128))
        plus,pr=native_alpha(renderer,v+eps*x,f,RT,K,(96,128))
        minus,mr=native_alpha(renderer,v-eps*x,f,RT,K,(96,128))
        sign=torch.sign(baseline-target)
    records=[]
    for kind in ['l1','fixed_sign_linear','squared_l2']:
        shift=torch.tensor(0.,device=d,requires_grad=True)
        vv=v+shift*x;alpha,rast=native_alpha(renderer,vv,f,RT,K,(96,128))
        original=renderer.render_rgba(rast,vv,f,uv,f,tex,None,enable_disturbance=False)['rgba'][...,3:]
        def loss(a):
            if kind=='l1':return (a-target).abs().sum()/target.sum()
            if kind=='fixed_sign_linear':return (a*sign).sum()/target.sum()
            return (a-target).square().sum()/target.sum()
        analytic=float(torch.autograd.grad(loss(alpha),shift,retain_graph=True)[0])
        upstream=float(torch.autograd.grad(loss(original),shift)[0])
        numeric=float((loss(plus)-loss(minus))/(2*eps))
        records.append(dict(kind=kind,analytic=analytic,original_rgba_gradient=upstream,finite_difference=numeric,
            alpha_parity=float((alpha-original).abs().max()),gradient_parity=abs(analytic-upstream),
            finite_difference_relative_error=abs(analytic-numeric)/max(abs(numeric),1e-8)))
    changed=(pr['rast_out'][...,-1]!=mr['rast_out'][...,-1])
    equality=(baseline==target)
    delta_alpha=(plus-minus)/(2*eps)
    result=dict(updated=io.now(),state='fixed_synthetic_mask_gradient_failure_audit_complete',records=records,
        raster_identity_changed_pixels=int(changed.sum()),raster_identity_changed_yx=torch.nonzero(changed,as_tuple=False).tolist(),
        alpha_changed_pixels=int(((plus-minus).abs()>1e-7).sum()),
        central_fd_on_L1_equal_pixels_sum_abs=float(delta_alpha[equality].abs().sum()),
        l1_plus=float(foreground_normalized_mask_loss(plus,target)),l1_minus=float(foreground_normalized_mask_loss(minus,target)),
        epsilon_m=eps,optimizer_updates=0,geometry_changed=False,quality_accepted=False,
        interpretation='Compare the adapter with unchanged original RGBA and record finite-difference branch changes. No altered threshold or fitted candidate.')
    io.save(OUT/'gradient-failure-audit.json',result);print(result,flush=True)


if __name__=='__main__':
    checked();assert not (OUT/'gradient-failure-audit.json').exists()
    io.save(OUT/'gradient-failure-audit-plan.json',dict(created=io.now(),source_hash=io.sha(Path(__file__)),
        cases=['original L1','same base fixed-sign linearization','squared L2 smooth loss'],same_triangle_and_K=True,
        finite_difference_m=1e-4,no_coefficient_or_epsilon_scan=True,optimizer_updates=0,max_seconds=60))
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=process(331);assert before['start_identity']=='9229' and before['state'] not in ('T','Z')
    assert active_gpu_pids()<={331}
    try:
        os.kill(331,signal.SIGSTOP)
        for _ in range(50):
            if process(331)['state']=='T':break
            time.sleep(.05)
        assert process(331)['state']=='T';audit()
    finally:
        foreign=active_gpu_pids()-{331,os.getpid()}
        if not foreign:os.kill(331,signal.SIGCONT)
        io.save(OUT/'gradient-audit-gpu-handoff.json',dict(updated=io.now(),before=before,after=process(331),foreign=list(foreign)))
