"""One registered synthetic comparison; no real fitting or camera modification."""
from pathlib import Path
import ast, fcntl, hashlib, json, os, time, traceback
import numpy as np
import torch
import run_smplify_reference123 as r

OUT = r.O / 'convergence-control'

def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

def monitor_class(parts):
    tree = ast.parse((r.O/'official/smplifyx/fitting.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FittingMonitor']
    assert len(nodes) == 1
    scope = {'torch': torch, 'np': np, 'time': time, 'utils': parts[0]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'pinned-official-FittingMonitor', 'exec'), scope)
    return scope['FittingMonitor']

def independent_terms(ref, out, gt, conf, jw):
    """Independent float64 numpy formula, including full K and angle prior."""
    arr = lambda x: x.detach().cpu().numpy().astype(np.float64)
    q = arr(out.joints) @ arr(ref.camera.K).T
    uv = q[..., :2] / q[..., 2:]
    sq = (arr(gt) - uv)**2; rho = float(ref.loss.rho)
    data = (arr(jw*conf)[..., None]**2 * rho*rho*sq/(sq+rho*rho) * float(ref.loss.data_weight)**2)
    groups = {'body25': np.arange(25), 'left_hand21': np.arange(25,46), 'right_hand21': np.arange(46,67), 'face51': np.arange(67,118)}
    terms = {k: float(data[:,v].sum()) for k,v in groups.items()}
    terms.update(latent_prior=float((arr(ref.latent)**2).sum()*float(ref.loss.body_pose_weight)**2),
        shape_prior=float((arr(out.betas)**2).sum()*float(ref.loss.shape_weight)**2),
        bend_prior=float((np.exp(arr(out.full_pose)[:,3:66][:,[52,55,9,12]]*np.array([1,-1,-1,-1]))**2).sum()*float(ref.loss.bending_prior_weight)),
        hand_prior=float(((arr(out.left_hand_pose)**2).sum()+(arr(out.right_hand_pose)**2).sum())*float(ref.loss.hand_prior_weight)**2),
        expression_prior=float((arr(out.expression)**2).sum()*float(ref.loss.expr_prior_weight)**2),
        jaw_prior=float((arr(out.jaw_pose)*arr(ref.loss.jaw_prior_weight)) .__pow__(2).sum()))
    return uv, terms

def checkpoint(ref, gt, conf, jw):
    for p in ref.param.values(): p.grad = None
    out = ref.forward(); uv = ref.camera(out.joints); loss = ref.objective(out,gt,conf,jw)
    loss.backward()
    uv64, terms = independent_terms(ref,out,gt,conf,jw)
    diff = abs(sum(terms.values())-float(loss))
    assert diff < max(.2, 1e-5*abs(float(loss))), (diff,float(loss))
    numpy_uv_error = float(np.max(np.abs(uv.detach().cpu().numpy()-uv64)))
    assert numpy_uv_error < .001
    err = r.errors(uv.detach().cpu().numpy(), gt.cpu().numpy(), conf.cpu().numpy())
    return {'loss': float(loss), 'independent_terms': terms, 'formula_abs_difference': diff,
        'numpy_projection_max_abs_px': numpy_uv_error, 'errors_px':err,
        'gradient_max_abs': {k:float(p.grad.abs().max()) for k,p in ref.param.items()},
        'minimum_joint_depth':float(out.joints[...,2].min())}, out, uv

def optimize_stages(ref, gt, conf, repeated, outdir, monitor, max_seconds=180, max_evals=4000, real_fits=0, height=1520, progress_root=None):
    """Same five weights and inner solver, only monitor repetition differs."""
    outdir.mkdir(parents=True, exist_ok=False)
    start=time.monotonic(); total=0; stages=[]; terminal='complete'
    progress_root = OUT if progress_root is None else progress_root
    initial=ref.snapshot(); initial_hash=state_hash(initial)
    ref.params_on(list(ref.param)); initial_check, _, initial_uv=checkpoint(ref,gt,conf,ref.weights(4,height))
    initial_uv=initial_uv.detach().cpu().numpy()
    for stage in range(5):
        jw=ref.weights(stage,height); params=ref.params_on(list(ref.param)); count0=total
        best_loss=float('inf'); best=ref.snapshot(); calls=[]
        class LoggedLBFGS(torch.optim.LBFGS):
            def step(self, closure):
                value=super().step(closure)
                calls.append({'returned_loss':float(value), 'max_abs_gradient':max(float(p.grad.abs().max()) for p in params if p.grad is not None),
                    'upstream_gtol_predicate':all(float(p.grad.reshape(-1).max().abs())<1e-9 for p in params if p.grad is not None), 'evaluations':total-count0})
                return value
        opt=LoggedLBFGS(params,lr=1,max_iter=30,max_eval=45,tolerance_grad=1e-9,tolerance_change=1e-9,line_search_fn='strong_wolfe')
        def closure():
            nonlocal total,best_loss,best
            torch.cuda.synchronize()
            if total>=max_evals or time.monotonic()-start>=max_seconds: raise r.BudgetEnd('registered per-arm budget')
            opt.zero_grad(); out=ref.forward()
            if not bool(torch.isfinite(out.vertices).all()) or float(out.joints[...,2].min())<=.05:raise RuntimeError('invalid geometry/depth')
            loss=ref.objective(out,gt,conf,jw)
            if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite objective')
            total+=1
            if float(loss)<best_loss:best_loss=float(loss);best=ref.snapshot()
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):raise RuntimeError('invalid gradient')
            return loss
        try:
            if repeated:
                with monitor(maxiters=30,ftol=1e-9,gtol=1e-9,visualize=False,model_type='smplx') as m:
                    m.run_fitting(opt,closure,params,ref.body,use_vposer=True,pose_embedding=ref.latent,vposer=ref.vposer)
                stop='max_outer_iterations'
                if calls and calls[-1]['upstream_gtol_predicate']: stop='upstream_gtol'
                if len(calls)>1 and r.setup_rel_change(calls[-2]['returned_loss'],calls[-1]['returned_loss'])<=1e-9:stop='upstream_ftol'
            else: opt.step(closure); stop='one_step_budget'
        except r.BudgetEnd:
            terminal='budget_ended';stop='registered_budget'
        except Exception:
            terminal='failed';stop=traceback.format_exc()
        # Identical best-evaluated-state retention in both arms; not an extra fit.
        ref.restore(best)
        check,out,uv=checkpoint(ref,gt,conf,jw)
        saved={k:p.detach().cpu().numpy() for k,p in ref.param.items()}
        np.savez_compressed(outdir/f'stage{stage}.npz',**saved,vertices=out.vertices.detach().cpu().numpy(),joints=out.joints.detach().cpu().numpy(),projected=uv.detach().cpu().numpy(),targets=gt.cpu().numpy(),K=ref.camera.K.cpu().numpy(),faces=ref.body.faces)
        item={'stage':stage,'stop':stop,'outer_step_calls':len(calls),'closure_evaluations':total-count0,'steps':calls,**check}
        stages.append(item);r.save(outdir/f'stage{stage}.json',item)
        r.save(progress_root/'progress.json',{'state':'running','pid':os.getpid(),'arm':outdir.name,'stage':stage,'total_closure_evaluations':total,'seconds':time.monotonic()-start,'real_fits':real_fits,'gaussian_steps':0})
        print(json.dumps({'arm':outdir.name,'stage':stage,'closure_calls':total-count0,'outer_calls':len(calls),'loss':check['loss'],'errors':check['errors_px'],'stop':stop}),flush=True)
        if terminal!='complete':break
    final, out,uv=checkpoint(ref,gt,conf,ref.weights(4,height))
    final_uv=uv.detach().cpu().numpy()
    np.savez_compressed(outdir/'final.npz',**{k:p.detach().cpu().numpy() for k,p in ref.param.items()},vertices=out.vertices.detach().cpu().numpy(),joints=out.joints.detach().cpu().numpy(),projected=final_uv,initial_projected=initial_uv,targets=gt.cpu().numpy(),confidence=conf.cpu().numpy(),K=ref.camera.K.cpu().numpy(),faces=ref.body.faces)
    result={'state':terminal,'seconds':time.monotonic()-start,'closure_evaluations':total,'initial_state_sha256':initial_hash,'initial':initial_check,'final':final,'stages':stages,
        'correct_absolute_gradient_gtol_met':max(final['gradient_max_abs'].values())<1e-9,'real_fits':real_fits,'gaussian_steps':0}
    r.save(outdir/'result.json',result)
    return result

def main():
    assert not OUT.exists(), 'A completed or partial control must not be silently repeated'
    plan=r.read(r.O/'convergence-control-plan.json')
    for path,h in plan['input_code'].items():
        p=Path('/mnt/'+path[0].lower()+path[2:].replace('\\','/')); assert r.sha(p)==h,p
    OUT.mkdir();parts=r.setup();torch.manual_seed(12302);np.random.seed(12302)
    r.setup_rel_change=parts[0].rel_change
    rows=r.read(r.R/'data/inputs.json')['records'];K=np.array(rows[0]['K_integer_centers'])
    p={'body_pose':np.zeros(63),'betas':np.zeros(10),'global_orient':[np.pi,0,0],'transl':[0,0,3.5]}
    ref=r.Ref(parts,K,p)
    gt_state=ref.snapshot()
    gt_state['latent']=r.t(.8*np.sin(np.arange(32)+1)).reshape(1,32)
    gt_state['betas']=r.t(.1*np.cos(np.arange(10)+1)).reshape(1,10)
    gt_state['left_hand_pca']=r.t(.2*np.sin(np.arange(12)+1)).reshape(1,12)
    gt_state['right_hand_pca']=-gt_state['left_hand_pca']
    gt_state['jaw_pose']=r.t([[.02,0,0]])
    gt_state['expression']=r.t(.05*np.cos(np.arange(10)+1)).reshape(1,10)
    ref.restore(gt_state);ref.params_on(list(ref.param))
    with torch.no_grad():
        out=ref.forward();gt=ref.camera(out.joints).detach();conf=torch.ones((1,118),device=r.DEVICE);conf[:,[1,8]]=0
    truth_check,truth_out,truth_uv=checkpoint(ref,gt,conf,ref.weights(4,1520))
    np.savez_compressed(OUT/'ground-truth.npz',**{k:v.cpu().numpy() for k,v in gt_state.items()},vertices=truth_out.vertices.detach().cpu().numpy(),joints=truth_out.joints.detach().cpu().numpy(),projected=gt.cpu().numpy(),K=K,faces=ref.body.faces)
    init={k:v.clone() for k,v in gt_state.items()}
    init['latent']+=r.t(.15*np.cos(np.arange(32)+1)).reshape(1,32)
    init['global_orient']+=r.t([[.015,-.02,.01]]);init['transl']+=r.t([[.03,-.04,.1]])
    for k in ['left_hand_pca','right_hand_pca','betas','jaw_pose','expression']:init[k].zero_()
    mon=monitor_class(parts);arms=[]
    r.save(OUT/'execution-contract.json',{'created':r.now(),'runner_sha256':r.sha(__file__),'plan_sha256':r.sha(r.O/'convergence-control-plan.json'),'official_fitting_sha256':r.sha(r.O/'official/smplifyx/fitting.py'),
        'weights_unchanged':True,'K':K.tolist(),'independent_seed':12302,'root_camera_init_omitted_for_both':'Known synthetic camera-space geometry already supplies the registered perturbed initialization. Only five loss stages are compared.',
        'synthetic_gt_not_map_optimum':'Nonzero priors bias the known geometry. Compare objective and recovered projection; do not equate convergence with exact parameter recovery.',
        'same_best_evaluated_retention':True,'real_fits':0,'gaussian_steps':0})
    for repeated in [False,True]:
        candidate=r.Ref(parts,K,p);candidate.restore(init)
        arms.append(optimize_stages(candidate,gt,conf,repeated,OUT/('repeated' if repeated else 'single'),mon))
    assert arms[0]['initial_state_sha256']==arms[1]['initial_state_sha256']
    result={'created':r.now(),'state':'two_arm_control_completed','arms':arms,'ground_truth_final_stage_objective':truth_check,'identical_initial_state':True,'fixed_K':True,
        'source_sha256':r.sha(__file__),'independent_physical_accuracy':False,'real_fits':0,'gaussian_steps':0,'goal_complete':False}
    r.save(OUT/'result.json',result);r.save(OUT/'progress.json',{'state':result['state'],'pid':None,'finished':r.now(),'real_fits':0,'gaussian_steps':0})

if __name__=='__main__':
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except Exception:
        if OUT.exists():r.save(OUT/'failure.json',{'created':r.now(),'traceback':traceback.format_exc()})
        raise
