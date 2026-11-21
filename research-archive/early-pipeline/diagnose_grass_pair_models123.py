"""Compare fixed-K essential geometry with unconstrained epipolar geometry.

Two-view diagnostics only. No camera candidate, K update or training output.
"""
from trial_grass_surface123 import ROOT,OUT,read,save,sha,now
import argparse,time,json
import numpy as np
import cv2
O=OUT/'provenance-audit'

def summary(a):
    a=np.asarray(a,float)
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(n=0)
def err(F,a,b):
    ah=np.c_[a,np.ones(len(a))];bh=np.c_[b,np.ones(len(b))];fa=ah@F.T;fb=bh@F
    return abs(np.sum(bh*fa,1))/np.maximum(np.sqrt((fa[:,:2]**2).sum(1)+(fb[:,:2]**2).sum(1)),1e-15)
def register():
    assert not (O/'pair-model-plan.json').exists()
    save(O/'pair-model-plan.json',dict(registered=now(),hypothesis='Does enforcing the conditional K damage two-view fit versus rank-two F, before any SMPL-X or3Dcamera optimization?',
        pairs=[[0,16],[31,47],[62,78]],selection='Fixed seed plus two evenly spaced windows, same16frame interval; no error-based pair choice.',
        split='Original complete appearance trackFIT/HELD preserved; everyFIT pair used. No material point transferred or deleted.',
        models=['Fixed conditional K essential, OpenCV findEssentialMat,1.5px RANSAC,.999prob,10000max iterations.',
                'Free rank2 fundamental, OpenCV USAC_MAGSAC,1.5px,.999prob,10000max iterations; diagnostic only.'],
        evaluation='Sampson distances for all observedFIT/HELD inperson/mixed/outside tracks, no postfit error exclusion. F andE parameter freedoms differ; improvement is not a unique causal proof.',
        inputs=[dict(path=str(ROOT/p),sha256=sha(ROOT/p)) for p in ['camera/solver-output.npz','camera/appearance-tracks.json']],
        budget=dict(CPU_seconds=60,pairs=3,models_per_pair=2,K_updates=0,new_camera_candidates=0,training=0),
        stop='Once only; if a partition is too small, report unavailable. No threshold scan or new frames.'))
    print('Registered three two-view projection-model diagnostics.')
def run():
    plan=read(O/'pair-model-plan.json');assert not (O/'pair-model-result.json').exists();start=time.monotonic();cv2.setNumThreads(2)
    for f in plan['inputs']:assert sha(f['path'])==f['sha256']
    cl={r['id']:r for r in read(O/'classifications.json')['records']};tracks=read(ROOT/'camera/appearance-tracks.json')['records']
    K=np.load(ROOT/'camera/solver-output.npz')['K'];inv=np.linalg.inv(K);results=[]
    for i,j in plan['pairs']:
        cv2.setRNGSeed(118);xy=[np.load(ROOT/f'camera/features-{z:03d}.npz')['xy'] for z in [i,j]];meta=[];a=[];b=[]
        for tr in tracks:
            nodes=dict(tr['nodes'])
            if i not in nodes or j not in nodes:continue
            a.append(xy[0][nodes[i]]);b.append(xy[1][nodes[j]]);meta.append(cl[tr['id']])
        a=np.array(a);b=np.array(b);fit=np.array([not r['held'] for r in meta]);row=dict(pair=[i,j],fit_count=int(fit.sum()),held_count=int((~fit).sum()),models={})
        if fit.sum()<8:row['unavailable']='Fewer than8FITtracks';results.append(row);continue
        norm=lambda v:cv2.undistortPoints(v.reshape(-1,1,2),K,None).reshape(-1,2)
        E,em=cv2.findEssentialMat(norm(a[fit]),norm(b[fit]),np.eye(3),method=cv2.RANSAC,prob=.999,threshold=1.5/np.mean([K[0,0],K[1,1]]),maxIters=10000)
        F,fm=cv2.findFundamentalMat(a[fit],b[fit],cv2.USAC_MAGSAC,1.5,.999,10000)
        for name,m,mask in [('fixed_K_E',None if E is None else inv.T@E@inv,em),('free_F',F,fm)]:
            if m is None or m.shape!=(3,3):row['models'][name]=dict(unavailable='No unique3x3 model');continue
            errors=err(m,a,b);metrics=[]
            for held in [False,True]:
                for group in ['inside','mixed','outside']:
                    take=np.array([r['held']==held and ('inside' if r['all_inside'] else 'mixed' if r['any_inside'] else 'outside')==group for r in meta])
                    metrics.append(dict(split='HELD' if held else 'FIT',group=group,errors=summary(errors[take])))
            item=dict(F=m.tolist(),inliers_fit_reported=int(np.count_nonzero(mask)) if mask is not None else None,metrics=metrics,
                      per_track=[dict(id=r['id'],held=r['held'],error=float(e)) for r,e in zip(meta,errors)])
            if name=='free_F':
                singular=np.linalg.svd(K.T@m@K,compute_uv=False);item['KTFK_singular_values']=singular.tolist();item['essential_equal_singular_ratio']=float(singular[0]/singular[1])
            row['models'][name]=item
        results.append(row);assert time.monotonic()-start<60
    for f in plan['inputs']:assert sha(f['path'])==f['sha256']
    result=dict(updated=now(),runtime_seconds=time.monotonic()-start,results=results,K_unchanged=True,new_camera_candidate=False,physical_truth=False,training_started=False)
    save(O/'pair-model-result.json',result)
    print(json.dumps(dict(seconds=result['runtime_seconds'],results=[dict(pair=r['pair'],fit=r['fit_count'],held=r['held_count'],models={k:{x:y for x,y in v.items() if x not in ['F','per_track']} for k,v in r['models'].items()}) for r in results]),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.register:register()
    elif a.run:run()
    else:p.error('Choose --register or --run.')
