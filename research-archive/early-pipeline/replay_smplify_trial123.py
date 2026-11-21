"""Forensic replay of recorded stage1 failures; never produces a new candidate."""
import argparse, fcntl, json, os, time, traceback
import numpy as np
import torch
import run_smplify_reference123 as r
import control_smplify_convergence123 as c
SRC=r.O/'corrected-reference';OUT=SRC/'failure-replay'

def prepare():
    assert not OUT.exists();OUT.mkdir()
    data=r.read(SRC/'result.json');slots=[v['slot'] for v in data['views'] if v['state']=='failed'];assert slots==[31,62]
    r.save(OUT/'plan.json',{'registered':r.now(),'purpose':'Disambiguate the combined invalid geometry/depth guard. Replay only the same saved stage0 endpoint and stage1 objective, with no accepted updates or continuation.',
        'slots':slots,'input_hashes':{str(SRC/f'slot{s:03d}'/'stage0.npz'):r.sha(SRC/f'slot{s:03d}'/'stage0.npz') for s in slots},
        'source_sha256':r.sha(__file__),'budget':{'GPU_seconds_per_slot':30,'closure_evaluations_per_slot':32,'new_fits':0,'gaussian_steps':0},
        'constraints':'Same torchLBFGS/monitor/K/score/weights as failed stage1. Save invalid trial parameters, finite flags and depth. Stop at first identical failure or budget. Never publish a fitted result or initialize training from this forensic replay.'})

def run():
    p=r.read(OUT/'plan.json');assert r.sha(__file__)==p['source_sha256'] and not (OUT/'result.json').exists()
    for path,h in p['input_hashes'].items():assert r.sha(path)==h
    _,rows,obs,pred=r.inputs();parts=r.setup();mon=c.monitor_class(parts);results=[]
    for slot in p['slots']:
        fit=dict(np.load(SRC/f'slot{slot:03d}'/'stage0.npz'));ref=r.Ref(parts,rows[slot]['K_integer_centers'],r.parameters(pred,slot));ref.restore({k:r.t(fit[k]) for k in ref.param})
        gt,conf=r.targets(obs,slot);jw=ref.weights(1,1520);pars=ref.params_on(list(ref.param));start=time.monotonic();trace=[];invalid=None
        opt=torch.optim.LBFGS(pars,lr=1,max_iter=30,max_eval=45,tolerance_grad=1e-9,tolerance_change=1e-9,line_search_fn='strong_wolfe')
        initial=ref.snapshot();start_hash=c.state_hash(initial)
        def closure():
            nonlocal invalid
            if len(trace)>=32 or time.monotonic()-start>=30:raise r.BudgetEnd('forensic bound')
            opt.zero_grad();out=ref.forward();finite=bool(torch.isfinite(out.vertices).all());z=out.joints[...,2].detach().cpu().numpy()
            rec={'call':len(trace)+1,'vertices_finite':finite,'joints_finite':bool(np.isfinite(z).all()),'minimum_joint_depth':float(np.min(z)) if np.isfinite(z).all() else None,
                'root_translation':ref.body.transl.detach().cpu().tolist(),'max_parameter_delta':{k:float((v-initial[k]).abs().max()) for k,v in ref.param.items()}}
            trace.append(rec)
            if not finite or not np.isfinite(z).all() or z.min()<=.05:
                invalid=rec
                np.savez_compressed(OUT/f'slot{slot:03d}-invalid-trial.npz',**{k:v.detach().cpu().numpy() for k,v in ref.param.items()},joints=out.joints.detach().cpu().numpy(),vertices=out.vertices.detach().cpu().numpy())
                raise RuntimeError('identical_guard_reproduced')
            loss=ref.objective(out,gt,conf,jw);rec['loss']=float(loss);loss.backward()
            rec['gradient_max_abs']=max(float(p.grad.abs().max()) for p in pars)
            if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(p.grad).all()) for p in pars):raise RuntimeError('nonfinite loss/gradient')
            return loss
        stop='finished_without_failure'
        try:
            with mon(maxiters=30,ftol=1e-9,gtol=1e-9,visualize=False,model_type='smplx') as m:m.run_fitting(opt,closure,pars,ref.body,use_vposer=True,pose_embedding=ref.latent,vposer=ref.vposer)
        except Exception as exc:stop=str(exc)
        # No resulting optimizer state is accepted or exported for fitting.
        ref.restore(initial)
        value={'slot':slot,'state':stop,'seconds':time.monotonic()-start,'initial_parameter_hash':start_hash,'trace':trace,'invalid_trial':invalid,'restored_initial_hash':c.state_hash(ref.snapshot()),'new_accepted_fits':0,'gaussian_steps':0}
        assert value['initial_parameter_hash']==value['restored_initial_hash'];results.append(value);r.save(OUT/f'slot{slot:03d}.json',value)
        print(json.dumps({'slot':slot,'state':stop,'invalid_trial':invalid}),flush=True)
    r.save(OUT/'result.json',{'created':r.now(),'state':'forensic_replay_complete','views':results,'new_accepted_fits':0,'gaussian_steps':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:
        lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
