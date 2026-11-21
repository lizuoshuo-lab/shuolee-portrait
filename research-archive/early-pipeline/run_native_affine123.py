"""One controlled local-affine24-query contrast; no camera or training change."""
from pathlib import Path
import json,os,signal,time,traceback
import numpy as np
import torch
from native_patch_control123 import R,read,save,sha,local,now,gray,crop_context,make_spline,TranslationPhotometry
from run_video10_matcher_contrast123 import family_indices,conflict
from roma116_coordinates import sample_direction
from native_affine_patch123 import AffinePhotometry

Q=R/'video10-correspondence-contrast';D=Q/'head-roi-contrast';O=D/'native-affine-contrast'
def main():
    assert not (O/'progress.json').exists(),'No repeated real contrast'
    assert read(O/'control-result.json')['passed']
    frozen=read(D/'native-photometric-contrast/plan.json')
    for path,h in frozen['source_sha256'].items():assert sha(local(path))==h,path
    plan=read(O/'plan.json');p=read(D/'plan.json');alloc=read(D/'review-allocation.json')['rows'];prior=read(D/'visual-review/index.json')['rows']
    for s,h in plan['source_sha256'].items():assert sha(local(s))==h,s
    assert len(alloc)==24 and all(a['feature']==b['feature'] and a['frame']==b['frame'] for a,b in zip(alloc,prior))
    images={int(f):gray(local(c['source'])) for f,c in p['crops'].items()};maps={}
    for a,b in p['pairs']:
        with np.load(Q/'run'/f'map_{a:06d}_{b:06d}.npz') as z:maps[a,b]=(torch.from_numpy(z['warp'].copy()),torch.from_numpy(z['certainty'].copy()))
    nodes,splits,trees=family_indices(p['frames']);torch.set_num_threads(2);started=time.monotonic();nfits=0;rows=[]
    save(O/'execution-admission.json',dict(updated=now(),plan_sha256=sha(O/'plan.json'),runner_sha256=sha(Path(__file__)),source_queries=24,wholeframe_model_inference=False))
    def state(s,**kw):save(O/'progress.json',dict(updated=now(),state=s,pid=os.getpid(),seconds=time.monotonic()-started,patch_fits=nfits,training_started=False,**kw))
    def alarm(*a):raise TimeoutError('Registered120second CPU budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    def sample(a,b,xy):
        if not np.isfinite(xy).all():return np.array([np.nan,np.nan]),0.,False
        wa,ce=maps[min(a,b),max(a,b)];uv,c,v=sample_direction(wa,ce,torch.tensor(np.asarray(xy)[None],dtype=torch.float32),(1520,2704),(1520,2704),'AtoB' if a<b else 'BtoA')
        return uv[0].numpy().astype(float),float(c[0]),bool(v[0])
    def context_available(a,xy):
        return np.isfinite(xy).all() and 48<=xy[0]<images[a].shape[1]-49 and 48<=xy[1]<images[a].shape[0]-49
    def refine(a,b,xy):
        nonlocal nfits
        seed,c,valid=sample(a,b,xy);record=dict(source=a,target=b,source_uv=np.asarray(xy).tolist(),seed_uv=seed.tolist(),map_confidence=c,map_valid=valid,attempted=False)
        if not valid or not context_available(a,xy) or not context_available(b,seed):
            record.update(reason='Map invalid or97px context unavailable',success=False,at_bound=False)
            return seed,record
        si,sc,sbox=crop_context(images[a],xy);ti,tc,tbox=crop_context(images[b],seed)
        m=AffinePhotometry(make_spline(si),make_spline(ti),sc,tc);opt=m.fit();nfits+=1;assert nfits<=144
        affine_matrix=np.eye(2)+opt.x[2:6].reshape(2,2);det=float(np.linalg.det(affine_matrix));good=bool(opt.success and np.isfinite(opt.x).all() and det>1e-4);bound=bool(np.any(np.abs(opt.x[:2])>=2.999) or np.any(np.abs(opt.x[2:6])>=.24999) or opt.x[6]<=.25001 or opt.x[6]>=3.99999 or abs(opt.x[7])>=.49999)
        out=seed+opt.x[:2]
        record.update(attempted=True,success=good,at_bound=bound,estimate=opt.x.tolist(),uv=out.tolist(),nfev=int(opt.nfev),status=int(opt.status),
            source_box=sbox,target_box=tbox,initial_RMS_255=float(np.sqrt(np.mean(m.fun([0,0,0,0,0,0,1,0])**2))*255),final_RMS_255=float(np.sqrt(np.mean(opt.fun**2))*255),
            translation_norm=float(np.linalg.norm(opt.x[:2])),affine_determinant=det)
        return out,record
    def pair(a,b,xy):
        uv,forward=refine(a,b,xy);back,reverse=refine(b,a,uv)
        fb=float(np.linalg.norm(back-xy));confidence=min(forward['map_confidence'],reverse['map_confidence'])
        successful=all(t['success'] and not t['at_bound'] and t['map_valid'] for t in [forward,reverse])
        geometry=bool(np.isfinite(fb) and confidence>=.9 and fb<1)
        return uv,dict(source=a,target=b,uv=uv.tolist(),back_uv=back.tolist(),fb=fb,confidence=confidence,patches=[forward,reverse],all_patch_success=successful,geometry_gate=geometry,eligible=bool(successful and geometry))
    def lineage(r,b,uv):
        tree,labels=trees[b];near=tree.query_ball_point(uv,r=8) if np.isfinite(uv).all() else []
        return conflict(r['split'],{splits[int(labels[j])] for j in near})
    def clean(obj):
        if isinstance(obj,float) and not np.isfinite(obj):return None
        if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
        if isinstance(obj,list):return [clean(v) for v in obj]
        return obj
    try:
        for i,r in enumerate(alloc):
            state('real_patch_contrast',row=i);a=r['frame'];b,c=[f for f in p['frames'] if f!=a];xy=np.array(r['uv'])
            buv,ab=pair(a,b,xy);cuv,bc=pair(b,c,buv);auv,ca=pair(c,a,cuv)
            closure=float(np.linalg.norm(auv-xy));bad=lineage(r,b,buv) or lineage(r,c,cuv)
            geometry=bool(all(t['geometry_gate'] for t in [ab,bc,ca]) and np.isfinite(closure) and closure<1)
            good=bool(all(t['eligible'] for t in [ab,bc,ca]) and geometry and not bad)
            rows.append(clean(dict(row=i,**r,cycle=[a,b,c,a],baseline=prior[i]['baseline'],refined=dict(b_uv=buv.tolist(),c_uv=cuv.tolist(),returned_uv=auv.tolist(),closure=closure,edges=[ab,bc,ca],geometry_gate=geometry,split_conflict=bad,eligible=good),physical_identity_certified=False)))
        old=np.array([r['baseline']['eligible'] and not r['baseline']['split_conflict'] for r in rows]);new=np.array([r['refined']['eligible'] for r in rows]);patches=[q for r in rows for e in r['refined']['edges'] for q in e['patches']]
        translation=read(D/'native-photometric-contrast/result.json');trans=np.array([x['refined']['eligible'] for x in translation['rows']])
        result=dict(translation_only_pass=int(trans.sum()),gained_vs_translation=int((~trans&new).sum()),lost_vs_translation=int((trans&~new).sum()),updated=now(),state='complete_pending_native_review',seconds=time.monotonic()-started,total_queries=24,patch_fits=nfits,
            baseline_pass=int(old.sum()),refined_pass=int(new.sum()),gained=int((~old&new).sum()),lost=int((old&~new).sum()),
            patch_optimizer_success=sum(q['success'] for q in patches),patch_hit_bound=sum(q['at_bound'] for q in patches),
            positive_low_RMS_is_not_accuracy=True,rows=rows,cameras_changed=False,source_queries_changed=False,old_splits_changed=False,physical_face_HELD=0,training_started=False,
            limitation='Native local affine geometry plus affine brightness. Same frozen query set is method-development evidence only. The optimized photometric residual and FB/cycle closure are self-consistency; no physical/camera/motion attribution or subpixel certification. No failures discarded.')
        save(O/'result.json',result);state('complete');x=read(O/'progress.json');x['pid']=None;save(O/'progress.json',x)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    except BaseException as ex:
        save(O/'failure.json',dict(updated=now(),error=repr(ex),rows=rows,traceback=traceback.format_exc()));state('failed');x=read(O/'progress.json');x['pid']=None;save(O/'progress.json',x);raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
