"""Registered fixed-state value/gradient/conditioning diagnostic; no optimizer."""
from pathlib import Path
import copy,fcntl,json,os,time,traceback
import numpy as np
import torch
import run_smplify_reference123 as r
import control_smplify_convergence123 as c
OUT=r.O/'solver-conditioning'

def winpath(path):return Path('/mnt/'+path[0].lower()+path[2:].replace('\\','/')) if ':/' in path.replace('\\','/') else Path(path)

def functional(ref,physical,gt,conf,jw,slices):
    state={k:physical[a:b].reshape(shape) for k,(a,b,shape) in slices.items()}
    out=ref.body(body_pose=ref.vposer.decode(state['latent'])['pose_body'].reshape(1,63),return_verts=True,return_full_pose=True,
        betas=state['betas'],global_orient=state['global_orient'],transl=state['transl'],left_hand_pose=state['left_hand_pca'],right_hand_pose=state['right_hand_pca'],jaw_pose=state['jaw_pose'],expression=state['expression'])
    loss=ref.loss(out,ref.camera,gt,conf,ref.body.faces_tensor,jw,use_vposer=True,pose_embedding=state['latent'])
    return loss,out,state

def curvature(ref,slices):
    diag=torch.ones(sum(b-a for a,b,s in slices.values()),dtype=torch.float64,device=r.DEVICE)
    for k,(a,b,shape) in slices.items():
        if k=='latent':v=2*float(ref.loss.body_pose_weight)**2
        elif k=='betas':v=2*float(ref.loss.shape_weight)**2
        elif k=='expression':v=2*float(ref.loss.expr_prior_weight)**2
        elif k=='jaw_pose':v=2*ref.loss.jaw_prior_weight.double().square()
        elif k=='left_hand_pca':v=2*float(ref.loss.hand_prior_weight)**2*ref.body.left_hand_components.double().square().sum(-1)
        elif k=='right_hand_pca':v=2*float(ref.loss.hand_prior_weight)**2*ref.body.right_hand_components.double().square().sum(-1)
        else:v=1. # no prior here; unit floor is for the equivalence check only
        diag[a:b]=torch.as_tensor(v,device=r.DEVICE,dtype=torch.float64).reshape(-1)
    return diag

def case(parts,rows,obs,pred,name,slot,npz):
    start=time.monotonic();evaluations=0;loaded=dict(np.load(npz));base={k:loaded[k] for k in ['global_orient','transl','latent','betas','left_hand_pca','right_hand_pca','jaw_pose','expression']}
    if slot is None:
        K=loaded['K'];gt=r.t(loaded['projected']);conf=torch.ones((1,118),device=r.DEVICE);conf[:,[1,8]]=0;params=r.parameters(pred,0)
    else:
        K=rows[slot]['K_integer_centers'];gt,conf=r.targets(obs,slot);params=r.parameters(pred,slot)
    results={};arrays={};slices={};offset=0
    for k,v in base.items():slices[k]=(offset,offset+v.size,list(v.shape));offset+=v.size
    flat=np.concatenate([base[k].reshape(-1) for k in slices]).astype(np.float64)
    for dtype in [torch.float32,torch.float64]:
        ref=r.Ref(parts,K,params);ref.restore({k:r.t(v) for k,v in base.items()});jw=ref.weights(1,1520)
        ref.body.to(dtype=dtype);ref.camera.to(dtype=dtype);ref.loss.to(dtype=dtype);ref.vposer=copy.deepcopy(parts[4]).to(dtype=dtype)
        target=gt.to(dtype);confidence=conf.to(dtype);jw=jw.to(dtype);theta=torch.tensor(flat,device=r.DEVICE,dtype=dtype,requires_grad=True)
        def ev(x,gradient=False):
            nonlocal evaluations
            if evaluations>=200 or time.monotonic()-start>=120:raise r.BudgetEnd('registered conditioning bound')
            evaluations+=1
            loss,out,state=functional(ref,x,target,confidence,jw,slices)
            assert torch.isfinite(loss) and torch.isfinite(out.vertices).all() and float(out.joints[...,2].min())>.05
            grad=torch.autograd.grad(loss,x)[0] if gradient else None
            if gradient:assert torch.isfinite(grad).all()
            return loss,out,state,grad
        loss,out,state,grad=ev(theta,True)
        uv=ref.camera(out.joints);q=out.joints.detach().cpu().numpy().astype(np.float64)@ref.camera.K.detach().cpu().numpy().astype(np.float64).T
        camera_difference=float(np.max(np.abs(q[...,:2]/q[...,2:]-uv.detach().cpu().numpy())))
        # Independent loss formula uses functional latent rather than the old Parameter.
        old_latent=ref.latent;ref.latent=state['latent'];_,terms=c.independent_terms(ref,out,target,confidence,jw);ref.latent=old_latent
        formula_diff=abs(sum(terms.values())-float(loss));assert formula_diff<max(.2,1e-5*abs(float(loss)))
        direction_checks=[]
        for k,(a,b,shape) in slices.items():
            direction=torch.zeros_like(theta);direction[a:b]=torch.sin(torch.arange(b-a,device=r.DEVICE,dtype=dtype)+1);direction/=torch.linalg.vector_norm(direction)
            analytic=float(torch.dot(grad,direction));samples=[]
            for step in [1e-4,1e-5,1e-6]:
                plus=ev(theta.detach()+step*direction)[0];minus=ev(theta.detach()-step*direction)[0]
                fd=float((plus-minus)/(2*step));samples.append({'step':step,'finite_difference':fd,'absolute_difference':abs(fd-analytic),'relative_difference':abs(fd-analytic)/max(1.,abs(fd),abs(analytic))})
            direction_checks.append({'parameter':k,'analytic_directional_derivative':analytic,'samples':samples})
        diag=curvature(ref,slices).to(dtype);scale=1/torch.sqrt(torch.clamp(diag,min=1))
        # q is a new parameterization of exactly the same physical theta.
        qvar=torch.zeros_like(theta,requires_grad=True)
        value_s,_,_,grad_s=ev(theta.detach()+scale*qvar,False)
        scaled_grad=torch.autograd.grad(value_s,qvar)[0]
        chain_error=float((scaled_grad-grad*scale).abs().max())
        relative_chain_error=chain_error/max(1.,float((grad*scale).abs().max()))
        assert relative_chain_error<1e-6 and abs(float(value_s-loss))<1e-6
        record={'dtype':str(dtype),'loss':float(loss),'independent_terms':terms,'formula_abs_difference':formula_diff,'gradient_max':float(grad.abs().max()),
          'minimum_joint_depth':float(out.joints[...,2].min()),'projection_numpy_max_px':camera_difference,'directional_checks':direction_checks,
          'prior_diagonal_curvature':{k:{'min':float(diag[a:b].min()),'max':float(diag[a:b].max())} for k,(a,b,_) in slices.items()},
          'scale_chain_rule_abs_error':chain_error,'scale_chain_rule_relative_error':relative_chain_error,'scaled_gradient_max':float(scaled_grad.abs().max()),'same_physical_parameters':True}
        results[str(dtype)]=record;arrays[str(dtype)]={'grad':grad.detach().cpu().numpy().astype(np.float64),'uv':uv.detach().cpu().numpy().astype(np.float64)}
        del ref,loss,out,state,grad,uv,theta,qvar,value_s,scaled_grad
    f32=results['torch.float32'];f64=results['torch.float64']
    value={'created':r.now(),'case':name,'state':'read_only_complete','seconds':time.monotonic()-start,'objective_evaluations':evaluations,'input_sha256':r.sha(npz),'results':results,
      'fp32_fp64_loss_difference':abs(f32['loss']-f64['loss']),'fp32_fp64_projection_max_px':float(np.max(np.abs(arrays['torch.float32']['uv']-arrays['torch.float64']['uv']))),
      'fp32_fp64_gradient_max_abs_difference':float(np.max(np.abs(arrays['torch.float32']['grad']-arrays['torch.float64']['grad']))),
      'slices':slices,'new_fit_or_camera':False,'optimizer_steps':0,'gaussian_steps':0}
    np.savez_compressed(OUT/f'{name}-gradients.npz',fp32=arrays['torch.float32']['grad'],fp64=arrays['torch.float64']['grad'],theta=flat)
    r.save(OUT/f'{name}.json',value);return value

def main():
    assert not OUT.exists(),'Do not repeat a recorded diagnostic';p=r.read(r.O/'solver-conditioning-plan.json')
    for path,h in p['input_hashes'].items():assert r.sha(winpath(path))==h,path
    OUT.mkdir();r.save(OUT/'progress.json',{'state':'running','pid':os.getpid(),'started':r.now(),'optimizer_steps':0})
    _,rows,obs,pred=r.inputs();parts=r.setup();cases=[]
    for name,slot,f in [('view31',31,r.O/'corrected-reference/slot031/stage0.npz'),('view62',62,r.O/'corrected-reference/slot062/stage0.npz'),('synthetic',None,r.O/'convergence-control/ground-truth.npz')]:
        v=case(parts,rows,obs,pred,name,slot,f);cases.append(v)
        print(json.dumps({'case':name,'seconds':v['seconds'],'evaluations':v['objective_evaluations'],'projection_difference':v['fp32_fp64_projection_max_px'],'loss_difference':v['fp32_fp64_loss_difference']}),flush=True)
    r.save(OUT/'result.json',{'created':r.now(),'state':'read_only_conditioning_complete','cases':cases,'source_sha256':r.sha(__file__),'plan_sha256':r.sha(r.O/'solver-conditioning-plan.json'),'optimizer_steps':0,'gaussian_steps':0,'physical_acceptance':False})
    r.save(OUT/'progress.json',{'state':'complete','pid':None,'finished':r.now(),'optimizer_steps':0})

if __name__=='__main__':
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception:
        if OUT.exists():r.save(OUT/'failure.json',{'created':r.now(),'traceback':traceback.format_exc()});r.save(OUT/'progress.json',{'state':'failed','pid':None})
        raise
