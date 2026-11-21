"""Replay original image-data terms on saved projections, with GPU trace oracles.

No model forward pass, optimization, image warping, or candidate selection.
Other objective terms are a residual (saved total minus replayed image data),
not independently measured priors.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import numpy as np
import audit_gaussianavatars_params123 as a

BODY=a.REC/'demo-vhap-initialization/fullbody-smplifyx'
RUN=BODY/'warmtail-guarded-sequence'
BASE=BODY/'full-sequence'


def load(path):
    with np.load(a.lp(path),allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def image_terms(projection,target,height):
    # Pinned SMPLifyLoss/GMoF and final fit_smplx stage: rho100,
    # body/feet weights1, hands/face2, ignored neck/hips, data_weight1000/H.
    residual=np.asarray(projection,dtype=np.float32).reshape(118,2)-np.asarray(target,dtype=np.float32).reshape(118,3)[:,:2]
    weight=np.ones(118,dtype=np.float32);weight[[1,9,12]]=0;weight[25:]=2
    weight*=target.reshape(118,3)[:,2].astype(np.float32)
    squared=residual**2
    robust=np.float32(100**2)*squared/(squared+np.float32(100**2))
    per_joint=(weight[:,None]**2*robust*np.float32(1000/height)**2).sum(-1)
    return dict(body_data=float(per_joint[:19].sum()),feet_data=float(per_joint[19:25].sum()),
        hands_data=float(per_joint[25:67].sum()),face_data=float(per_joint[67:].sum()))


def main():
    started=time.monotonic();out=RUN/'review/data-term-audit.json';planpath=RUN/'review/data-term-plan.json'
    assert not a.lp(out).exists() and not a.lp(planpath).exists()
    prior=a.read(BODY/'warmtail-guarded-sequence-plan.json')
    checked={}
    for path,digest in prior['source_hashes'].items():
        if any(path.endswith(name) for name in ('smplifyx/fitting.py','smplifyx/utils.py','smplifyx/fit_single_frame.py','cfg_files/fit_smplx.yaml')):
            local=path.replace('/mnt/c/','C:/');assert a.sha(local)==digest
            checked[local]=digest
    assert len(checked)==4
    paths=[Path(__file__),BASE/'fit/fitted.npz',BASE/'data/training-inputs.json']
    for name in ('stage19-original-trajectory-v2','stage19-warmstart-original-tail-v2'):
        paths.extend([BODY/name/'result.json',BODY/name/'traced-winner.npz',BODY/name/'trajectory.npz'])
    for frame in prior['temporal_roster']:
        paths.extend([RUN/'frames'/frame/'objective-selection.json',RUN/'frames'/frame/'candidate-00-temporal.npz',BASE/'frames'/frame/'winner-camera-space.npz'])
    hashes={str(p):a.sha(p) for p in paths}
    plan=dict(updated=datetime.now(timezone.utc).isoformat(),hypothesis='Original scalar image-data tradeoffs can explain lower total objective with worse body/feet medians.',
        scope='CPU saved-state diagnostic across both candidates on all59 frames. No causal claim about changing any weight.',
        oracle='Match the two previously recorded final GPU stage19 data-term decompositions within0.01 before analyzing all59.',
        budget_seconds=120,official_sources=checked,input_hashes=hashes,weights=dict(rho=100,body=1,feet=1,hands=2,face=2,ignored=[1,9,12],data='1000/native_height'),
        original_fits_unchanged=True,quality_accepted=False,goal_complete=False)
    a.save(planpath,plan)
    base=load(BASE/'fit/fitted.npz');ids=list(base['frame_ids'].astype(str))
    inputs={r['frame']:r for r in a.read(BASE/'data/training-inputs.json')['records']}
    checks=[]
    for name in ('stage19-original-trajectory-v2','stage19-warmstart-original-tail-v2'):
        folder=BODY/name;report=a.read(folder/'result.json');data=load(folder/'traced-winner.npz');target=load(folder/'trajectory.npz')['observed_native_keypoints']
        actual=image_terms(data['joints_projected_native'],target,inputs['frame_000019']['wh'][1])
        expected=report['evaluations'][-1]
        delta={key:actual[key]-expected[key] for key in actual}
        assert max(abs(v) for v in delta.values())<.01,(name,delta)
        checks.append(dict(run=name,delta=delta,passed=True))
    rows=[]
    for frame in prior['temporal_roster']:
        assert time.monotonic()-started<120
        decision=a.read(RUN/'frames'/frame/'objective-selection.json')
        target=base['observed_native_keypoints'][ids.index(frame)]
        candidates={}
        for label,index,path in [('new_candidate',0,RUN/'frames'/frame/'candidate-00-temporal.npz'),('baseline',1,BASE/'frames'/frame/'winner-camera-space.npz')]:
            d=load(path);terms=image_terms(d['joints_projected_native'],target,inputs[frame]['wh'][1])
            total=decision['original_saved_objectives'][index]
            terms['other_terms_inferred']=total-sum(terms.values());terms['total']=total
            assert terms['other_terms_inferred']>=-.01,(frame,label,terms)
            candidates[label]=terms
        delta={key:candidates['new_candidate'][key]-candidates['baseline'][key] for key in candidates['baseline']}
        rows.append(dict(frame=frame,original_selected_index=decision['selected_index'],candidates=candidates,delta=delta))
    for path,digest in hashes.items():assert a.sha(path)==digest,path
    result=dict(updated=datetime.now(timezone.utc).isoformat(),state='original_image_data_terms_replayed',frames=len(rows),oracle_checks=checks,
        plan_sha256=a.sha(planpath),rows=rows,elapsed_cpu_seconds=time.monotonic()-started,
        interpretation='Negative delta is lower loss. Per-region weighted robust sums differ from unweighted pixel medians. Other terms are inferred from total minus data and are not separately attributed to a particular prior.',
        no_parameter_changes=True,no_model_forward=True,no_gpu=True,no_fit_or_candidate_changes=True,
        quality_accepted=False,goal_complete=False)
    a.save(out,result)
    print('GPU_ORACLE_CHECKS',json.dumps(checks))
    for r in rows:
        if int(r['frame'].split('_')[-1]) in (13,18,19,20,21,22,23,40,51,52,58):
            print(json.dumps(dict(frame=r['frame'],delta=r['delta'])))
    print('COMPLETE',len(rows),'CPU_SECONDS',result['elapsed_cpu_seconds'])


if __name__=='__main__':main()
