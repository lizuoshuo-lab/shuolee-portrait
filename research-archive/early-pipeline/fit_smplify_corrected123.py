"""Once-only three-image repair: explicit relative scores and repeated monitor."""
import argparse, fcntl, json, os, time, traceback
import numpy as np
import torch
import run_smplify_reference123 as r
import control_smplify_convergence123 as c

OUT=r.O/'corrected-reference'

def prepare():
    assert not OUT.exists(), 'The registered repair is not a parameter sweep'
    control=r.read(c.OUT/'result.json');a,b=control['arms']
    assert all(v['state']=='complete' for v in [a,b]) and control['identical_initial_state']
    assert b['final']['loss']<a['final']['loss']
    assert all(b['final']['errors_px'][g]['median']<a['final']['errors_px'][g]['median'] for g in b['final']['errors_px'])
    OUT.mkdir()
    assessment={'created':r.now(),'implementation_controls_passed':True,'physical_accuracy_passed':False,
        'same_initial_state_and_K':True,'final_objective_single_repeated':[a['final']['loss'],b['final']['loss']],
        'stop':'Repeated arm stopped on upstream relative objective criterion, not gradient tolerance. Priors bias the known true pose; full-body maximum remains2.07px.',
        'source':str(c.OUT/'result.json'),'no_real_admission_from_synthetic':True}
    r.save(c.OUT/'assessment.json',assessment)
    r.save(OUT/'plan.json',{'registered':r.now(),'state':'registered_not_executed','purpose':'Repair invalid score/convergence interface on the same three real views, not a new calibration candidate.',
        'views':[0,31,62],'source_frames':[0,155,310],'initialization':'Fresh individual GVHMR same view, encoded with VPoser2. No old real fit or synthetic fit loaded.',
        'K':'Same full K_integer_centers in existing inputs.json, unchanged. Camera identity per independent view; root placement is body placement, not background-camera optimization.',
        'mapping':'Unchanged pinned official SMPL-X direct body/feet/hands/staticface118 mapping. Native detector coordinates. No observation removed by error.',
        'score':'Explicit previously checked relative confidence from all94 original score/flip records. Heuristic, not probability or visibility.',
        'optimizer':'Unchanged torchLBFGS inner strong_wolfe with pinned official FittingMonitor repetition and ftol/gtol1e-9,30outercalls. Root20inner, then five original stages30inner.',
        'causal_boundary':'Both score-interface and convergence repairs differ from the invalid real trial; do not attribute real change to one of them alone. Synthetic loop-only comparison is separate.',
        'visibility_review':str(r.O/'input-visual-review.json'),'physical_visibility_certified':False,
        'budget':{'real_views':3,'GPU_seconds_per_view':180,'closure_evaluations_per_view':4000,'new_downloads':0,'gaussian_steps':0},
        'stop':['nonfinite loss/gradients or jointdepth<=.05','time/evaluation budget; no extension or parameter sweep','quality failure does not trigger Gaussian training'],
        'code_hashes':{str(p):r.sha(p) for p in [__file__,c.__file__,r.__file__,r.Path(__file__).parent/'smplify_score_contract123.py']},
        'input_contract':str(r.O/'plan.json'),'no_official_full_reproduction_claim':True})
    print('REGISTERED_CORRECTED_REFERENCE',flush=True)

def root_stage(ref,parts,mon,gt,height,start):
    pars=ref.params_on(['global_orient','transl']);opt=torch.optim.LBFGS(pars,lr=1,max_iter=20,max_eval=30,tolerance_grad=1e-9,tolerance_change=1e-9,line_search_fn='strong_wolfe')
    lossfn=parts[2]['SMPLifyCameraInitLoss'](init_joints_idxs=torch.tensor([9,12,2,5]),trans_estimation=ref.body.transl.detach().clone(),data_weight=1000/height,depth_loss_weight=100).to(r.DEVICE)
    count=0;best=float('inf');saved=ref.snapshot()
    def closure():
        nonlocal count,best,saved
        torch.cuda.synchronize()
        if count>=4000 or time.monotonic()-start>=180:raise r.BudgetEnd('root budget')
        opt.zero_grad();out=ref.forward()
        if not torch.isfinite(out.vertices).all() or float(out.joints[...,2].min())<=.05:raise RuntimeError('invalid root depth')
        loss=lossfn(out,ref.camera,gt);count+=1
        if not torch.isfinite(loss):raise RuntimeError('nonfinite root loss')
        if float(loss)<best:best=float(loss);saved=ref.snapshot()
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in pars):raise RuntimeError('invalid root gradients')
        return loss
    try:
        with mon(maxiters=30,ftol=1e-9,gtol=1e-9,visualize=False,model_type='smplx') as m:m.run_fitting(opt,closure,pars,ref.body,use_vposer=True,pose_embedding=ref.latent,vposer=ref.vposer)
    finally:ref.restore(saved)
    return {'evaluations':count,'best_loss':best,'seconds':time.monotonic()-start}

def run():
    p=r.read(OUT/'plan.json');assert not (OUT/'progress.json').exists() and not (OUT/'result.json').exists()
    for path,h in p['code_hashes'].items():assert r.sha(path)==h,path
    _,rows,obs,pred=r.inputs();parts=r.setup();r.setup_rel_change=parts[0].rel_change;mon=c.monitor_class(parts);results=[]
    r.save(OUT/'progress.json',{'state':'running','pid':os.getpid(),'real_fits':0,'gaussian_steps':0})
    for slot in p['views']:
        row=rows[slot];height=row['wh'][1];assert height==1520
        K=np.array(row['K_integer_centers']);ref=r.Ref(parts,K,r.parameters(pred,row['official_prediction_slot']));gt,conf=r.targets(obs,slot)
        ref.params_on(list(ref.param));fresh_check,fresh_out,fresh_uv=c.checkpoint(ref,gt,conf,ref.weights(4,height))
        fresh_uv=fresh_uv.detach().cpu().numpy();fresh_vertices=fresh_out.vertices.detach().cpu().numpy();fresh_hash=c.state_hash(ref.snapshot())
        start=time.monotonic();root=root_stage(ref,parts,mon,gt,height,start)
        report=c.optimize_stages(ref,gt,conf,True,OUT/f'slot{slot:03d}',mon,max_seconds=180-(time.monotonic()-start),max_evals=4000-root['evaluations'],real_fits=len(results)+1,height=height,progress_root=OUT)
        with torch.no_grad():out=ref.forward();uv=ref.camera(out.joints).cpu().numpy()
        np.savez_compressed(OUT/f'slot{slot:03d}-fit.npz',**{k:v.detach().cpu().numpy() for k,v in ref.param.items()},vertices_camera=out.vertices.cpu().numpy(),joints_camera=out.joints.cpu().numpy(),projected=uv,initial_projected=fresh_uv,initial_vertices_camera=fresh_vertices,targets=gt.cpu().numpy(),confidence=conf.cpu().numpy(),K_integer_centers=K,faces=ref.body.faces)
        report.update(slot=slot,source_frame_index=row['source_frame_index'],fresh_initial=fresh_check,fresh_initial_state_sha256=fresh_hash,root=root,total_seconds=time.monotonic()-start,total_evaluations=report['closure_evaluations']+root['evaluations'],all_K_values_unchanged=np.array_equal(ref.camera.K.cpu().numpy(),K.astype(np.float32)),confidence_min=float(conf[conf>0].min()),confidence_max=float(conf.max()),confidence_valid_count=int((conf>.25).sum()),independent_physical_accuracy=False,visibility_uncertainty_preserved=True)
        r.save(OUT/f'slot{slot:03d}-result.json',report);results.append(report)
        print(json.dumps({'slot':slot,'state':report['state'],'seconds':report['total_seconds'],'evals':report['total_evaluations'],'final':report['final']['errors_px']}),flush=True)
    final={'created':r.now(),'state':'corrected_three_view_reference_complete','views':results,'source_sha256':r.sha(__file__),'same_config_across_views':True,'background_camera_changed':False,'all_K_fixed':all(x['all_K_values_unchanged'] for x in results),'gaussian_steps':0,'physical_quality_passed':False,'official_full_reproduction':False,'goal_complete':False}
    r.save(OUT/'result.json',final);r.save(OUT/'progress.json',{'state':final['state'],'pid':None,'finished':r.now(),'real_fits':len(results),'gaussian_steps':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception:
            r.save(OUT/'failure.json',{'created':r.now(),'traceback':traceback.format_exc(),'gaussian_steps':0})
            r.save(OUT/'progress.json',{'state':'failed','pid':None,'finished':r.now(),'gaussian_steps':0});raise
