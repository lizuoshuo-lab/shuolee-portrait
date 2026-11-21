"""Train-only ridge ARX, validation selection, then one frozen test evaluation."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-world-model-002'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load_split(split,spec):
    result=[]
    for seed in spec[split+'_seeds']:
        folder=OUT/'episodes'/f'{split}-{seed}';receipt=json.loads((folder/'receipt.json').read_text());p=folder/'trajectory.npz'
        assert receipt['status']=='completed' and sha(p)==receipt['trajectory_sha256']
        with np.load(p) as z:result.append(dict(seed=seed,**{k:z[k].copy() for k in z.files},sha256=sha(p)))
    return result

def predict(model,history,action):
    return np.asarray(model['A'])@history+np.asarray(model['B'])*action+np.asarray(model['b'])
def rollout(model,states,actions,start,horizon):
    h=np.r_[states[start],states[max(0,start-1)]];pred=[]
    for k in range(start,start+horizon):
        s=predict(model,h,actions[k]);pred.append(s);h=np.r_[s,h[:8]]
    return np.asarray(pred)
def fit(episodes,alpha,action_conditioned):
    history=[];actions=[];targets=[]
    for e in episodes:
        s=e['states'];history.extend(np.c_[s[:-1],np.r_[s[:1],s[:-2]]]);actions.extend(e['actions']);targets.extend(np.diff(s,axis=0))
    h=np.array(history);u=np.array(actions)[:,None];y=np.array(targets)
    x=np.c_[h,u] if action_conditioned else h
    xm=x.mean(0);xs=np.maximum(x.std(0),1e-8);ym=y.mean(0);ys=np.maximum(y.std(0),1e-8)
    xn=(x-xm)/xs;yn=(y-ym)/ys
    beta=np.linalg.solve(xn.T@xn+alpha*np.eye(x.shape[1]),xn.T@yn)
    physical=beta*ys[None,:]/xs[:,None];b=ym-xm@physical
    A=physical[:16].T.copy();A[:,:8]+=np.eye(8)
    B=physical[16] if action_conditioned else np.zeros(8)
    M=np.block([[A],[np.eye(8),np.zeros((8,8))]])
    return dict(type='residual-ridge-ARX' if action_conditioned else 'residual-ridge-AR',history=2,alpha=alpha,A=A.tolist(),B=B.tolist(),b=b.tolist(),augmented_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(M)))))

def main():
    assert (OUT/'dataset-completion.json').exists(),'Collection must finish first'
    assert not (OUT/'test-evaluation.json').exists(),'Frozen test already evaluated; do not silently retune'
    spec=json.loads((OUT/'protocol.json').read_text())
    fit_spec=dict(created_utc=stamp(),alphas=[1e-6,1e-4,1e-2,.1,1.,10.,100.],features='two 8-dimensional observed states; action is scalar; residual next-state regression with intercept',scaling='training-only feature and residual standard deviations, floor 1e-8',selection='mean squared error of 48-step free rollout over 4 validation episodes, all 8 features normalized by training-state standard deviation',test='horizon-end error over all valid origins for each predeclared horizon; no feedback within prediction horizon',aggregation='equal episode weights; per-episode errors also saved',state_units='normalized scene length and simulator time, not calibrated SI')
    save(OUT/'fit-protocol.json',fit_spec)
    train=load_split('train',spec);val=load_split('validation',spec)
    scale=np.maximum(np.concatenate([e['states'] for e in train]).std(0),1e-8)
    mean_time=np.mean([e['states'] for e in train],axis=0)
    selected={};scores={}
    for name,with_action in [('action',True),('no_action',False)]:
        scores[name]=[];candidates=[]
        for alpha in fit_spec['alphas']:
            model=fit(train,alpha,with_action)
            score=float(np.mean([np.mean(((rollout(model,e['states'],e['actions'],0,48)-e['states'][1:])/scale)**2) for e in val]))
            scores[name].append(dict(alpha=alpha,validation_normalized_mse=score,spectral_radius=model['augmented_spectral_radius']));candidates.append(model)
        idx=int(np.argmin([s['validation_normalized_mse'] for s in scores[name]]));selected[name]=candidates[idx]
    models=OUT/'models';models.mkdir(exist_ok=True)
    for name,model in selected.items():save(models/f'{name}.json',model)
    np.savez_compressed(models/'training-statistics.npz',scale=scale,mean_time=mean_time)
    selection=dict(frozen_utc=stamp(),selected=selected,candidate_scores=scores,train_receipts=[dict(seed=e['seed'],sha256=e['sha256']) for e in train],validation_receipts=[dict(seed=e['seed'],sha256=e['sha256']) for e in val],model_hashes={name:sha(models/f'{name}.json') for name in selected},test_read_after_this_file=True)
    save(OUT/'model-selection.json',selection)
    # First test read happens only after models and selection receipt are persisted.
    test=load_split('test',spec);metrics={};curves={}
    for name in ['action','no_action','persistence','time_mean']:
        metrics[name]={}
        for horizon in spec['prediction_horizons']:
            rows=[]
            for e in test:
                s=e['states'];a=e['actions'];w=e['foliage_readout'];errors=[]
                for start in range(len(a)-horizon+1):
                    if name in selected:pred=rollout(selected[name],s,a,start,horizon)[-1]
                    elif name=='persistence':pred=s[start]
                    else:pred=mean_time[start+horizon]
                    errors.append(pred-s[start+horizon])
                errors=np.array(errors)
                rows.append(dict(seed=e['seed'],origins=len(errors),foliage_rmse=float(np.sqrt(np.mean((errors@w)**2))),normalized_state_rmse=float(np.sqrt(np.mean((errors/scale)**2))),feature_rmse=np.sqrt(np.mean(errors**2,axis=0)).tolist()))
            metrics[name][str(horizon)]=dict(foliage_rmse=float(np.sqrt(np.mean([r['foliage_rmse']**2 for r in rows]))),normalized_state_rmse=float(np.sqrt(np.mean([r['normalized_state_rmse']**2 for r in rows]))),episodes=rows)
        for e in test:
            if name in selected:p=np.r_[e['states'][:1],rollout(selected[name],e['states'],e['actions'],0,48)]
            elif name=='persistence':p=np.repeat(e['states'][:1],49,axis=0)
            else:p=mean_time
            curves[f'{e["seed"]}_{name}']=p
    for e in test:curves[f'{e["seed"]}_truth']=e['states'];curves[f'{e["seed"]}_actions']=e['actions']
    np.savez_compressed(OUT/'test-predictions.npz',**curves)
    result=dict(status='completed',evaluated_utc=stamp(),model_selection_sha256=sha(OUT/'model-selection.json'),test_receipts=[dict(seed=e['seed'],sha256=e['sha256']) for e in test],horizons=spec['prediction_horizons'],metrics=metrics,no_test_retuning=True,interpretation='compact privileged-state predictor, single Ficus material/camera, held-out action trajectories; not learned video dynamics')
    save(OUT/'test-evaluation.json',result)
    print(json.dumps({name:{h:round(v['foliage_rmse'],7) for h,v in m.items()} for name,m in metrics.items()},indent=2))

if __name__=='__main__':main()
