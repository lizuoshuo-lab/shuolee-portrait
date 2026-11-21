"""Once-only precision repair control on the previously frozen synthetic case."""
import argparse,copy,fcntl,json,os,traceback
import numpy as np
import torch
import run_smplify_reference123 as r
import control_smplify_convergence123 as c

OUT=r.O/'precision-control'

def double_ref(ref):
    # Preserve all physical FP32 input values; only arithmetic changes.
    ref.body.double();ref.camera.double();ref.loss.double()
    ref.latent.data=ref.latent.data.double()
    ref.vposer=copy.deepcopy(ref.vposer).double()
    assert all(p.dtype==torch.float64 for p in ref.param.values())
    return ref

def prepare():
    assert not OUT.exists(),'No repeated precision control'
    diag=r.read(r.O/'solver-conditioning/result.json')
    assert diag['state']=='read_only_conditioning_complete'
    baseline=r.read(c.OUT/'repeated/result.json')
    assert baseline['state']=='complete'
    OUT.mkdir()
    r.save(OUT/'plan.json',{'registered':r.now(),'purpose':'Change FP32 arithmetic to FP64 on the identical known-geometry synthetic optimization. No new real fit.',
      'hypothesis':'FP32 objective rounding prevents reliable line-search progress when official priors have disparate curvature.',
      'single_changed_variable':'floating point computation precision, including model/VPoser/loss/target; all stored physical input values preserved',
      'fixed':['same synthetic truth and perturbed initialization','full K and mapping','confidence','five loss stages and prior weights','same torch LBFGS, inner30/maxeval45, repeated official monitor30 ftol/gtol1e-9','same best-evaluated retention'],
      'bounds':{'synthetic_arms':1,'seconds':180,'evaluations':4000,'real_fits':0,'gaussian_steps':0},
      'acceptance':'All five stages terminate without nonfinite or behind-camera geometry; objective and every group median must not worsen against frozen repeated FP32 control beyond 0.01px/0.1% numerical allowance. Do not claim subpixel real accuracy from this test.',
      'stop':'Stop on invalid trial or budget; retain failure; no automatic sweep.',
      'inputs':{str(p):r.sha(p) for p in [c.OUT/'ground-truth.npz',c.OUT/'repeated/result.json',r.O/'solver-conditioning/result.json']},
      'code':{str(p):r.sha(p) for p in [__file__,c.__file__,r.__file__]}})

def run():
    plan=r.read(OUT/'plan.json');assert not (OUT/'progress.json').exists()
    for group in ['inputs','code']:
        for path,h in plan[group].items():assert r.sha(path)==h,path
    r.save(OUT/'progress.json',{'state':'running','pid':os.getpid(),'created':r.now(),'real_fits':0,'gaussian_steps':0})
    parts=r.setup();r.setup_rel_change=parts[0].rel_change
    data=dict(np.load(c.OUT/'ground-truth.npz'));K=data['K']
    params={'body_pose':np.zeros(63),'betas':np.zeros(10),'global_orient':[np.pi,0,0],'transl':[0,0,3.5]}
    ref=r.Ref(parts,K,params)
    init={k:r.t(data[k]).clone() for k in ref.param}
    init['latent']+=r.t(.15*np.cos(np.arange(32)+1)).reshape(1,32)
    init['global_orient']+=r.t([[.015,-.02,.01]]);init['transl']+=r.t([[.03,-.04,.1]])
    for k in ['left_hand_pca','right_hand_pca','betas','jaw_pose','expression']:init[k].zero_()
    baseline=r.read(c.OUT/'repeated/result.json')
    assert c.state_hash(init)==baseline['initial_state_sha256'],'Initialization must be identical before dtype conversion'
    ref.restore(init);double_ref(ref)
    gt=r.t(data['projected']).double();conf=torch.ones((1,118),dtype=torch.float64,device=r.DEVICE);conf[:,[1,8]]=0
    result=c.optimize_stages(ref,gt,conf,True,OUT/'fp64',c.monitor_class(parts),progress_root=OUT)
    comparison={g:{'fp32_median':v['median'],'fp64_median':result['final']['errors_px'][g]['median']} for g,v in baseline['final']['errors_px'].items()}
    passed=result['state']=='complete' and result['final']['loss']<=baseline['final']['loss']*1.001 and all(v['fp64_median']<=v['fp32_median']+.01 for v in comparison.values())
    r.save(OUT/'result.json',{'created':r.now(),'state':'precision_control_completed','control_passed':passed,'same_initial_values':True,'single_change':'FP32 to FP64 arithmetic',
      'baseline_loss':baseline['final']['loss'],'fp64_loss':result['final']['loss'],'group_comparison':comparison,'arm':result,'source_sha256':r.sha(__file__),
      'real_fits':0,'gaussian_steps':0,'physical_accuracy_passed':False})
    r.save(OUT/'progress.json',{'state':'complete','pid':None,'finished':r.now(),'control_passed':passed})
    print(json.dumps({'control_passed':passed,'state':result['state'],'fp64_loss':result['final']['loss'],'comparisons':comparison}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception:
            r.save(OUT/'failure.json',{'created':r.now(),'traceback':traceback.format_exc()});r.save(OUT/'progress.json',{'state':'failed','pid':None});raise
