"""CPU-only saved-endpoint objective check. No fit or parameters updated."""
import numpy as np
import torch
import run_smplify_reference123 as r
r.DEVICE='cpu'
parts=r.setup();plan,rows,obs,pred=r.inputs();items=[]
for slot in plan['views']:
    ref=r.Ref(parts,rows[slot]['K_integer_centers'],r.parameters(pred,slot));fit=np.load(r.O/f'slot{slot:03d}-fit.npz')
    ref.restore({k:r.t(fit[k]) for k in ref.param});ref.params_on(list(ref.param));jw=ref.weights(4,rows[slot]['wh'][1]);gt,conf=r.targets(obs,slot)
    out=ref.forward();uv=ref.camera(out.joints);loss=ref.objective(out,gt,conf,jw);loss.backward()
    residual=gt-uv;weighted=(jw*conf).unsqueeze(-1)**2*ref.loss.robustifier(residual)*ref.loss.data_weight**2
    ids={'body25':np.arange(25),'left_hand21':np.arange(25,46),'right_hand21':np.arange(46,67),'face51':np.arange(67,118)}
    terms={name:float(weighted[:,ind].sum()) for name,ind in ids.items()}
    terms['latent_prior']=float(ref.latent.pow(2).sum()*ref.loss.body_pose_weight**2)
    terms['shape_prior']=float(out.betas.pow(2).sum()*ref.loss.shape_weight**2)
    terms['bend_prior']=float(ref.loss.angle_prior(out.full_pose[:,3:66]).sum()*ref.loss.bending_prior_weight)
    terms['hand_prior']=float((out.left_hand_pose.pow(2).sum()+out.right_hand_pose.pow(2).sum())*ref.loss.hand_prior_weight**2)
    terms['expression_prior']=float(out.expression.pow(2).sum()*ref.loss.expr_prior_weight**2)
    terms['jaw_prior']=float((out.jaw_pose*ref.loss.jaw_prior_weight).pow(2).sum())
    delta=abs(sum(terms.values())-float(loss));assert delta<.1
    grads={k:float(p.grad.abs().max()) for k,p in ref.param.items()}
    d=torch.linalg.vector_norm(residual,dim=-1).detach().numpy()[0]
    items.append({'slot':slot,'loss':float(loss),'components':terms,'components_sum_difference':delta,'gradient_max_abs_by_parameter':grads,'gradient_criterion_1e_9_met':max(grads.values())<1e-9,'points_euclidean_over_rho100':int((d>100).sum()),'body_or_foot_over_rho100':int((d[:25]>100).sum()),'perspective_projection_still_finite':bool(torch.isfinite(uv).all()),'any_parameters_updated':False})
value={'created':r.now(),'state':'read_only_endpoint_audit_complete','views':items,'fit_updates':0,'optimizer_contract_difference':{'official':'FittingMonitor loops maxiters times; each optimizer.step has max_iter=maxiters (official configuration30), until convergence checks. It may therefore perform many more than30 quasi-Newton iterations per stage.','local':'One torch LBFGS.step with max_iter30 perstage, hard300 closurecalls total, unlike the upstream nested convergence loop. This is a finite reference probe, not equivalent optimization.'},'conclusion':'Two views visually deteriorated and all final endpoints have large gradients. The limited adapted fit does not establish optimizer convergence or disprove SMPLify-X. Saturated robust residuals can weaken distant body/foot corrections; no exclusive causal attribution is made without an isolated control.','next':'Before another real-image attempt, test the official convergence loop and prior/initialization interaction on known synthetic full-body/hand/face geometry, with stage snapshots and a predeclared bounded budget. No parameter sweep or new Gaussian training.'}
r.save(r.O/'endpoint-audit.json',value);print(__import__('json').dumps(value))
