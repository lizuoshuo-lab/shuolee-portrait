"""Explicit RTMW score adapter; its output is a relative weight, not probability."""
import numpy as np

def relative_confidence(scores,boxes,flip_disagreement):
    scores=np.asarray(scores,dtype=np.float64);boxes=np.asarray(boxes,dtype=np.float64);flip=np.asarray(flip_disagreement,dtype=np.float64)
    if scores.ndim!=2 or boxes.shape!=(len(scores),4) or flip.shape!=scores.shape:raise ValueError('RTMW score/box/flip shape mismatch')
    if not all(np.isfinite(x).all() for x in [scores,boxes,flip]):raise ValueError('Nonfinite RTMW reliability input')
    height=boxes[:,3]-boxes[:,1]
    if np.any(height<=0) or np.any(scores<0) or np.any(flip<0):raise ValueError('Invalid RTMW reliability input')
    q90=np.maximum(np.quantile(scores,.9,axis=0),1e-6)
    # Same existing relative-score + flip-equivariance heuristic as the generic
    # fitter. SMPLify-X squares confidence, so pass the square root of its weight.
    confidence=np.clip(scores/q90,0,1)*np.exp(-.25*(flip/(.012*height[:,None]))**2)
    return confidence.astype(np.float32),q90

def require_relative_confidence(obs):
    if 'fit_confidence' not in obs:raise ValueError('Raw RTMW scores are not OpenPose probabilities: explicit fit_confidence is required')
    c=np.asarray(obs['fit_confidence'])
    if c.shape!=obs['scores'].shape or not np.isfinite(c).all() or np.any(c<0) or np.any(c>1):raise ValueError('Invalid explicitly adapted confidence')
    return c

if __name__=='__main__':
    from pathlib import Path
    import json
    R=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
    O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control')
    obs=dict(np.load(R/'data/observations.npz'));raw=obs['scores'];conf,q90=relative_confidence(raw,obs['boxes'],obs['flip_disagreement'])
    height=obs['boxes'][:,3]-obs['boxes'][:,1]
    existing=(np.clip(raw/np.maximum(np.quantile(raw,.9,axis=0),1e-6),0,1)**2)*np.exp(-.5*(obs['flip_disagreement']/(.012*height[:,None]))**2)
    difference=float(np.max(np.abs(conf.astype(float)**2-existing)));assert difference<1e-6
    rejected=False
    try:require_relative_confidence(obs)
    except ValueError:rejected=True
    assert rejected
    checks={'raw_score_unlabeled_rejected':rejected,'sqrt_contract_matches_existing_weight_max_abs':difference}
    for kind in ['nonfinite','negative_flip','zero_height']:
        s=raw.copy();b=obs['boxes'].copy();f=obs['flip_disagreement'].copy()
        if kind=='nonfinite':s[0,0]=np.nan
        if kind=='negative_flip':f[0,0]=-1
        if kind=='zero_height':b[0,3]=b[0,1]
        rejected=False
        try:relative_confidence(s,b,f)
        except ValueError:rejected=True
        assert rejected;checks[kind+'_rejected']=True
    obs['fit_confidence']=conf;assert np.array_equal(conf,require_relative_confidence(obs))
    views=[]
    for slot in [0,31,62]:
        assert np.all(raw[slot]>1) and np.std(conf[slot])>.01
        views.append({'slot':slot,'raw_min':float(raw[slot].min()),'raw_max':float(raw[slot].max()),'wrong_clip_all133_equal_one':True,'relative_min':float(conf[slot].min()),'relative_max':float(conf[slot].max()),'relative_std':float(conf[slot].std()),'relative_below025':int((conf[slot]<.25).sum())})
    result={'state':'score_interface_defect_confirmed_and_guard_tested','checks':checks,'views':views,'original_raw_score_range':[float(raw.min()),float(raw.max())],'prior_attempt_valid_reference':False,'corrected_real_fit_run':False,'new_training_steps':0,'contract':'C=sqrt(relative_score_squared * flip_equivariance_weight). This is a heuristic inherited from generic fitting, not calibrated OpenPose confidence or anatomical visibility.','causal_limit':'Clipping discarded reliability in the invalid attempt; improvement after correction is not measured yet. Does not identify the historical failure root cause.'}
    (O/'score-contract-repair.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result))
