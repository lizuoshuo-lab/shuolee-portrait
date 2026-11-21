"""Bounded numerical-repair verification; not a qualified Gaussian initializer."""
import argparse,fcntl,json,os,time,traceback
import numpy as np
import torch
import run_smplify_reference123 as r
import control_smplify_convergence123 as c
import control_smplify_scaling123 as s
import control_smplify_precision123 as p
import fit_smplify_corrected123 as old

OUT=r.O/'conditioned-reference'

def prepare():
    assert not OUT.exists(),'Once-only numerical repair verification'
    precision=r.read(r.O/'precision-control/result.json');scaled=r.read(r.O/'scaling-control/result.json')
    assert precision['arm']['state']==scaled['arm']['state']=='complete'
    assert not precision['control_passed'] and not scaled['control_passed']
    assert all(v['residual_jacobian_gradient_relative_error']<1e-8 for v in scaled['arm']['stages'])
    OUT.mkdir()
    r.save(OUT/'plan.json',{'registered':r.now(),'purpose':'Verify whether the observed real-image solver failure is removed by mathematically equivalent numerical conditioning. This is a diagnostic, not a quality-qualified replacement or Gaussian training admission.',
      'reason_for_separate_diagnostic':'Both synthetic numerical paths finish with finite geometry, and objective/Jacobian equivalence is verified; neither passed the frozen all-group median nonregression gate. Keep both failures unchanged. This run evaluates the earlier negative-depth failure on real views; it does not override that quality gate.',
      'views':[0,31,62],'source_frames':[0,155,310],
      'initialization':'Fresh original per-image GVHMR, encoded into VPoser2; verify identical initial FP32 values against corrected-reference. No previous fit, synthetic result or optimizer state reused.',
      'changes_from_corrected_reference':['FP64 arithmetic','fixed per-stage variable scales from exact objective residual Jacobian; same physical objective'],
      'causal_boundary':'Combined numerical repair; only the separate synthetic FP64/scaling controls isolate those individual changes. No claim about camera accuracy or historical whole-body ghosts.',
      'fixed':['same full K values, projection and source pixels','same direct official 118-point semantic map','same confidence and observations; no residual-based removal','same loss/prior weights and repeated monitor','same unknown visibility'],
      'budget':{'views':3,'seconds_per_view':360,'evaluations_per_view':8000,'gaussian_steps':0,'downloads':0},
      'implementation_limits':'Root stage retains existing 180s/4000 cap; five conditioned stages retain existing 180s/4000 cap. No stage extension or retry.',
      'stop':['Nonfinite objective/gradient or any mapped joint depth<=0.05; retain last valid state','Time/evaluation budget','No automatic extra camera candidate, per-view weights or Gaussian training'],
      'numeric_execution_acceptance':'All three finish every stage with finite positive-depth saved geometry and fixed inputs; independent of visual/physical quality.',
      'quality_acceptance':'Still requires raw-view visual review, trustworthy physical correspondence and multiview evidence; detector fit residual is not independent truth.',
      'synthetic_nonregression_passed':False,
      'code_hashes':{str(v):r.sha(v) for v in [__file__,r.__file__,c.__file__,s.__file__,p.__file__,old.__file__]},
      'input_contract':str(r.O/'plan.json'),'no_training_admission':True})

def run():
    plan=r.read(OUT/'plan.json');assert not (OUT/'progress.json').exists()
    for path,h in plan['code_hashes'].items():assert r.sha(path)==h,path
    _,rows,obs,pred=r.inputs();parts=r.setup();r.setup_rel_change=parts[0].rel_change;mon=c.monitor_class(parts);results=[]
    r.save(OUT/'progress.json',{'state':'running','pid':os.getpid(),'created':r.now(),'gaussian_steps':0})
    for slot in plan['views']:
        row=rows[slot];ref=r.Ref(parts,row['K_integer_centers'],r.parameters(pred,row['official_prediction_slot']))
        initial_hash=c.state_hash(ref.snapshot());expected=r.read(old.OUT/f'slot{slot:03d}-result.json')['fresh_initial_state_sha256']
        assert initial_hash==expected,'Fresh initialization changed'
        p.double_ref(ref);gt,conf=r.targets(obs,slot);gt=gt.double();conf=conf.double();slices=s.layout(ref)
        fresh,fresh_out,_,fresh_uv=s.report(ref,s.flatten(ref),gt,conf,ref.weights(4,1520),slices)
        initial_vertices=fresh_out.vertices.detach().cpu().numpy();started=time.monotonic();root=old.root_stage(ref,parts,mon,gt,1520,started)
        dest=OUT/f'slot{slot:03d}';dest.mkdir();s.OUT=dest
        arm=s.optimize(ref,s.flatten(ref),gt,conf,slices,mon)
        final=dict(np.load(dest/'final.npz'));ref.restore({k:torch.as_tensor(final[k],device=r.DEVICE) for k in ref.param})
        with torch.no_grad():out=ref.forward();uv=ref.camera(out.joints).cpu().numpy()
        np.savez_compressed(OUT/f'slot{slot:03d}-fit.npz',**{k:v.detach().cpu().numpy() for k,v in ref.param.items()},vertices_camera=out.vertices.cpu().numpy(),joints_camera=out.joints.cpu().numpy(),projected=uv,initial_projected=fresh_uv,initial_vertices_camera=initial_vertices,targets=gt.cpu().numpy(),confidence=conf.cpu().numpy(),K_integer_centers=ref.camera.K.cpu().numpy(),faces=ref.body.faces)
        arm.update(slot=slot,source_frame_index=row['source_frame_index'],fresh_initial=fresh,root=root,total_seconds=time.monotonic()-started,total_evaluations=root['evaluations']+arm['evaluations'],fresh_initial_state_sha256=initial_hash,
          all_K_values_unchanged=np.array_equal(ref.camera.K.cpu().numpy(),np.asarray(row['K_integer_centers'],np.float32).astype(np.float64)),synthetic_nonregression_passed=False,quality_accepted=False,gaussian_steps=0)
        r.save(OUT/f'slot{slot:03d}-result.json',arm);results.append(arm)
        r.save(dest/'progress.json',{'state':arm['state'],'pid':None,'finished':r.now()})
        r.save(OUT/'progress.json',{'state':'running','pid':os.getpid(),'completed_slots':[x['slot'] for x in results],'gaussian_steps':0})
        print(json.dumps({'slot':slot,'state':arm['state'],'seconds':arm['total_seconds'],'errors':arm['final']['errors_px']}),flush=True)
    r.save(OUT/'result.json',{'created':r.now(),'state':'three_view_numerical_repair_attempted','numeric_execution_passed':all(x['state']=='complete' for x in results),'views':results,'all_K_fixed':all(x['all_K_values_unchanged'] for x in results),'same_configuration_all_views':True,
      'synthetic_nonregression_passed':False,'physical_quality_passed':False,'gaussian_steps':0,'goal_complete':False,'source_sha256':r.sha(__file__)})
    r.save(OUT/'progress.json',{'state':'complete','pid':None,'finished':r.now(),'gaussian_steps':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception:
            r.save(OUT/'failure.json',{'created':r.now(),'traceback':traceback.format_exc()});r.save(OUT/'progress.json',{'state':'failed','pid':None});raise
