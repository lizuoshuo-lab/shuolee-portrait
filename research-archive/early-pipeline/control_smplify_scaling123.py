"""Bounded synthetic control: FP64, fixed per-stage Gauss-Newton column scales."""
import argparse,fcntl,json,os,time,traceback
import numpy as np
import torch
import run_smplify_reference123 as r
import control_smplify_convergence123 as c
import diagnose_smplify_conditioning123 as d
from control_smplify_precision123 import double_ref

OUT=r.O/'scaling-control'

def residuals(ref,out,state,gt,conf,jw):
    delta=gt-ref.camera(out.joints);rho=float(ref.loss.rho)
    data=(jw*conf)[...,None]*ref.loss.data_weight*rho*delta/torch.sqrt(delta.square()+rho*rho)
    bend=out.full_pose[:,3:66][:,[52,55,9,12]]*out.full_pose.new_tensor([1,-1,-1,-1])
    return torch.cat([data.reshape(-1), (state['latent']*ref.loss.body_pose_weight).reshape(-1),
      (out.betas*ref.loss.shape_weight).reshape(-1), (torch.exp(bend)*torch.sqrt(ref.loss.bending_prior_weight)).reshape(-1),
      (out.left_hand_pose*ref.loss.hand_prior_weight).reshape(-1), (out.right_hand_pose*ref.loss.hand_prior_weight).reshape(-1),
      (out.expression*ref.loss.expr_prior_weight).reshape(-1), (out.jaw_pose*ref.loss.jaw_prior_weight).reshape(-1)])

def layout(ref):
    slices={};offset=0
    for k,v in ref.param.items():slices[k]=(offset,offset+v.numel(),list(v.shape));offset+=v.numel()
    return slices

def flatten(ref):return torch.cat([v.detach().reshape(-1) for v in ref.param.values()])

def report(ref,theta,gt,conf,jw,slices):
    theta=theta.detach().requires_grad_(True);loss,out,state=d.functional(ref,theta,gt,conf,jw,slices)
    gradient=torch.autograd.grad(loss,theta,retain_graph=True)[0]
    old=ref.latent;ref.latent=state['latent'];uv,terms=c.independent_terms(ref,out,gt,conf,jw);ref.latent=old
    assert abs(sum(terms.values())-float(loss))<1e-6*max(1.,float(loss))
    return {'loss':float(loss),'errors_px':r.errors(uv,gt.cpu().numpy(),conf.cpu().numpy()),'independent_terms':terms,
      'gradient_max_abs':float(gradient.abs().max()),'minimum_joint_depth':float(out.joints[...,2].min())},out,state,uv

def optimize(ref,theta,gt,conf,slices,mon):
    started=time.monotonic();evaluations=0;stages=[];state='complete'
    initial,_,_,_=report(ref,theta,gt,conf,ref.weights(4,1520),slices)
    def bound():
        if evaluations>=4000 or time.monotonic()-started>=180:raise r.BudgetEnd('registered 180s / 4000 evaluations')
    for stage in range(5):
        jw=ref.weights(stage,1520).double();count0=evaluations;calls=[];best=theta.detach().clone();best_loss=float('inf');stop='max_outer_iterations'
        try:
            bound();probe=theta.detach().requires_grad_(True)
            loss,out,physical=d.functional(ref,probe,gt,conf,jw,slices);evaluations+=1
            rr=residuals(ref,out,physical,gt,conf,jw)
            eq=abs(float(rr.square().sum()-loss));assert eq<max(1e-6,float(loss)*1e-10)
            J=[]
            for start in range(0,rr.numel(),8):
                bound();n=min(8,rr.numel()-start);basis=torch.zeros((n,rr.numel()),device=r.DEVICE,dtype=torch.float64)
                basis[torch.arange(n,device=r.DEVICE),start+torch.arange(n,device=r.DEVICE)]=1
                J.append(torch.autograd.grad(rr,probe,grad_outputs=basis,is_grads_batched=True,retain_graph=True)[0].detach())
            J=torch.cat(J);assert torch.isfinite(J).all();diag=2*J.square().sum(0);scale=torch.rsqrt(diag.clamp_min(1))
            g1=torch.autograd.grad(loss,probe,retain_graph=True)[0];g2=2*(J.T@rr.detach())
            jac_error=float((g1-g2).abs().max())/max(1.,float(g1.abs().max()));assert jac_error<1e-8
            np.savez_compressed(OUT/f'stage{stage}-scales.npz',physical_origin=theta.cpu().numpy(),jacobian=J.cpu().numpy(),scales=scale.cpu().numpy())
            del probe,out,physical,loss,rr,J,g1,g2
            origin=theta.detach().clone();q=torch.nn.Parameter(torch.zeros_like(origin))
            class Logged(torch.optim.LBFGS):
                def step(self,closure):
                    value=super().step(closure);calls.append({'returned_loss':float(value),'gradient_max_abs':float(q.grad.abs().max()),'evaluations':evaluations-count0});return value
            opt=Logged([q],lr=1,max_iter=30,max_eval=45,tolerance_grad=1e-9,tolerance_change=1e-9,line_search_fn='strong_wolfe')
            def closure():
                nonlocal evaluations,best,best_loss
                bound();opt.zero_grad();loss,out,_=d.functional(ref,origin+scale*q,gt,conf,jw,slices);evaluations+=1
                if not torch.isfinite(loss) or not torch.isfinite(out.vertices).all() or float(out.joints[...,2].min())<=.05:raise RuntimeError('invalid trial; no parameter accepted')
                if float(loss)<best_loss:best_loss=float(loss);best=(origin+scale*q).detach().clone()
                loss.backward()
                if q.grad is None or not torch.isfinite(q.grad).all():raise RuntimeError('invalid gradient')
                return loss
            with mon(maxiters=30,ftol=1e-9,gtol=1e-9,visualize=False,model_type='smplx') as m:
                m.run_fitting(opt,closure,[q],ref.body,use_vposer=True,pose_embedding=ref.latent,vposer=ref.vposer)
            if len(calls)>1 and r.setup_rel_change(calls[-2]['returned_loss'],calls[-1]['returned_loss'])<=1e-9:stop='upstream_ftol'
        except r.BudgetEnd:state='budget_ended';stop='registered_budget'
        except Exception:state='failed';stop=traceback.format_exc()
        theta=best
        check,out,physical,uv=report(ref,theta,gt,conf,jw,slices)
        record={'stage':stage,'state':state,'stop':stop,'closure_evaluations':evaluations-count0,'outer_calls':len(calls),'steps':calls,**check}
        if 'jac_error' in locals():record.update(residual_loss_equivalence_abs=eq,residual_jacobian_gradient_relative_error=jac_error)
        stages.append(record);r.save(OUT/f'stage{stage}.json',record)
        r.save(OUT/'progress.json',{'state':'running','pid':os.getpid(),'stage':stage,'evaluations':evaluations,'seconds':time.monotonic()-started})
        print(json.dumps({'stage':stage,'state':state,'loss':check['loss'],'errors':check['errors_px'],'seconds':time.monotonic()-started}),flush=True)
        if state!='complete':break
    final,out,physical,uv=report(ref,theta,gt,conf,ref.weights(4,1520).double(),slices)
    np.savez_compressed(OUT/'final.npz',**{k:v.detach().cpu().numpy() for k,v in physical.items()},vertices=out.vertices.detach().cpu().numpy(),projected=uv,targets=gt.cpu().numpy(),K=ref.camera.K.cpu().numpy(),faces=ref.body.faces)
    return {'state':state,'seconds':time.monotonic()-started,'evaluations':evaluations,'stages':stages,'initial':initial,'final':final}

def prepare():
    assert not OUT.exists(),'No repeated scaling control'
    prior=r.read(r.O/'precision-control/result.json');assert prior['arm']['state']=='complete' and not prior['control_passed']
    OUT.mkdir()
    r.save(OUT/'plan.json',{'registered':r.now(),'purpose':'Test generic objective-derived variable scaling; not a weight or per-person parameter search.',
      'single_changed_variable_vs_fp64':'fixed linear coordinates per loss stage, scaled by 1/sqrt(max(2*diag(Jresidual.T*Jresidual),1)) at stage entry',
      'new_evidence':'FP64 directional derivatives agree, but curvature disparity remains. FP64 alone failed predefined body-median nonregression; retain that failure.',
      'fixed':['same known geometry and exact perturbed FP32 values','FP64 arithmetic','same physical loss, data, K, mapping, weights, confidence and strong-Wolfe LBFGS monitor'],
      'controls':['residual norm must equal official objective','2*J.T*r must equal official autograd gradient','finite positive-depth geometry throughout','all predeclared group median and objective safeguards from precision control remain unchanged against original repeated FP32 baseline'],
      'acceptance':'Final loss<=FP32*1.001, every group median<=FP32 median+0.01px, all five stages complete. No real/physical admission even if this passes.',
      'budget':{'synthetic_arms':1,'seconds':180,'evaluations':4000,'jacobian_builds':5,'real_fits':0,'gaussian_steps':0},
      'stop':'Invalid trial or budget ends control; no automatic scale/weight sweep.',
      'inputs':{str(p):r.sha(p) for p in [c.OUT/'ground-truth.npz',c.OUT/'repeated/result.json',r.O/'precision-control/result.json']},
      'code':{str(p):r.sha(p) for p in [__file__,d.__file__,c.__file__,r.__file__]}})

def run():
    plan=r.read(OUT/'plan.json');assert not (OUT/'progress.json').exists()
    for group in ['inputs','code']:
        for path,h in plan[group].items():assert r.sha(path)==h,path
    r.save(OUT/'progress.json',{'state':'running','pid':os.getpid(),'created':r.now()})
    parts=r.setup();r.setup_rel_change=parts[0].rel_change;data=dict(np.load(c.OUT/'ground-truth.npz'))
    params={'body_pose':np.zeros(63),'betas':np.zeros(10),'global_orient':[np.pi,0,0],'transl':[0,0,3.5]}
    ref=r.Ref(parts,data['K'],params);init={k:r.t(data[k]).clone() for k in ref.param}
    init['latent']+=r.t(.15*np.cos(np.arange(32)+1)).reshape(1,32);init['global_orient']+=r.t([[.015,-.02,.01]]);init['transl']+=r.t([[.03,-.04,.1]])
    for k in ['left_hand_pca','right_hand_pca','betas','jaw_pose','expression']:init[k].zero_()
    baseline=r.read(c.OUT/'repeated/result.json');assert c.state_hash(init)==baseline['initial_state_sha256']
    ref.restore(init);double_ref(ref);gt=r.t(data['projected']).double();conf=torch.ones((1,118),dtype=torch.float64,device=r.DEVICE);conf[:,[1,8]]=0
    arm=optimize(ref,flatten(ref),gt,conf,layout(ref),c.monitor_class(parts))
    comparison={g:{'fp32_median':v['median'],'scaled_fp64_median':arm['final']['errors_px'][g]['median']} for g,v in baseline['final']['errors_px'].items()}
    passed=arm['state']=='complete' and arm['final']['loss']<=baseline['final']['loss']*1.001 and all(v['scaled_fp64_median']<=v['fp32_median']+.01 for v in comparison.values())
    r.save(OUT/'result.json',{'created':r.now(),'state':'scaling_control_completed','control_passed':passed,'baseline_loss':baseline['final']['loss'],'scaled_fp64_loss':arm['final']['loss'],'group_comparison':comparison,'arm':arm,'source_sha256':r.sha(__file__),'real_fits':0,'gaussian_steps':0,'physical_accuracy_passed':False})
    r.save(OUT/'progress.json',{'state':'complete','pid':None,'finished':r.now(),'control_passed':passed});print(json.dumps({'control_passed':passed,'comparison':comparison}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception:
            r.save(OUT/'failure.json',{'created':r.now(),'traceback':traceback.format_exc()});r.save(OUT/'progress.json',{'state':'failed','pid':None});raise
