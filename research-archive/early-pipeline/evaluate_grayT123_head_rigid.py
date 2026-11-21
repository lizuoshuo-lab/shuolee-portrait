"""One preregistered diagnostic: shared vs per-frame rigid nose-proxy fit.

All sources remain read-only. The independent skin track never enters pose fitting.
This is not an estimate of ground-truth head motion or a production camera change.
"""
from datetime import datetime,timezone
import time,json,traceback
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw
from grayT123_head_rigid_common import *

def solve(fun,n):
    return least_squares(fun,np.zeros(6),loss='soft_l1',f_scale=6,max_nfev=n,
        ftol=1e-10,xtol=1e-10,gtol=1e-10)

def synthetic():
    X=np.array([[-.025,0,0],[-.015,.015,.008],[0,.025,.025],[.015,.015,.008],[.025,0,0],
        [-.02,-.02,.01],[0,-.025,.025],[.02,-.02,.01],[0,0,.035]])
    c=X.mean(0);r=float(np.sqrt(np.mean(np.sum((X-c)**2,axis=1))))
    K=np.array([[920.,0,478.25],[0,895.,353.75],[0,0,1]])
    E=np.repeat(np.eye(4)[None],3,axis=0);E[:,2,3]=.7;E[:,0,3]=[-.08,0,.08]
    known=np.array([[.025,-.035,.01,.2,-.1,.15],[-.02,.02,.035,-.1,.1,-.08],[.04,-.01,-.025,.1,.08,-.05]])
    H=np.array([transform(p,c,r) for p in known]);uv=project(X,E,K,H)[0];found=[];records=[]
    for i in range(3):
        f=lambda p:(project(X,E[i:i+1],K,transform(p,c,r))[0][0]-uv[i]).ravel()
        opt=solve(f,100);h=transform(opt.x,c,r);found.append(h)
        records.append(dict(success=bool(opt.success),nfev=opt.nfev,
            max_projection_error=float(np.max(np.linalg.norm(f(opt.x).reshape(-1,2),axis=1))),
            transform_max_abs=float(np.max(np.abs(h-H[i])))))
    testpoint=np.array([[.03,.04,.005]])
    a=project(testpoint,E,K,H)[0]
    expected=[]
    for e,h in zip(E,H):
        v=K@(e@h@np.r_[testpoint[0],1])[:3];expected.append(v[:2]/v[2])
    compose=float(np.max(np.abs(a[:,0]-expected)))
    gauge=E@H@np.linalg.inv(H[0]);reference=float(np.max(np.abs(gauge[0]-E[0])))
    passed=all(a['success'] and a['max_projection_error']<1e-5 and a['transform_max_abs']<1e-5 for a in records) and compose<1e-10 and reference<1e-12
    return dict(tests=records,independent_composition_error=compose,fixed_reference_error=reference,passed=bool(passed),physical_calibration=False)

def point_test(uv,E,K,anchor,max_nfev):
    P=K@E[:,:3,:];A=[]
    for mat,pt in zip(P[anchor],uv[anchor]):A.extend([pt[0]*mat[2]-mat[0],pt[1]*mat[2]-mat[1]])
    _,sv,V=np.linalg.svd(A);q=V[-1];x=q[:3]/q[3]
    opt=least_squares(lambda p:(project(p,E,K)[0][:,0][anchor]-uv[anchor]).ravel(),x,max_nfev=max_nfev,
        method='lm',ftol=1e-10,xtol=1e-10,gtol=1e-10)
    pred,depth=project(opt.x,E,K);pred=pred[:,0];depth=depth[:,0];err=np.linalg.norm(pred-uv,axis=1)
    return dict(point=opt.x.tolist(),success=bool(opt.success),nfev=opt.nfev,anchor_fit=stats(err[anchor]),
        held_check=stats(err[~anchor]),positive_depth_count=int((depth>0).sum()),DLT_singular_values=sv.tolist(),
        observations=[dict(frame_number=int(i+40),role='ANCHOR' if aa else 'HELD',xy=u.tolist(),projected=p.tolist(),
            residual=(p-u).tolist(),euclidean_error_px=float(e),depth=float(z)) for i,(aa,u,p,e,z) in enumerate(zip(anchor,uv,pred,err,depth))])

def main():
    start=time.monotonic()
    if (OUT/'fit-started.json').exists():raise RuntimeError('One-run diagnostic already started; inspect evidence, never rerun automatically.')
    d=load();prep=read(OUT/'preparation-result.json');review=read(OUT/'prefit-actual-review.json')
    assert prep['numerical_preparation_passed'] and review['all41_native_pairs_actually_reviewed'] and not review['proxies_are_physical_truth']
    assert review['preparation_sha256']==sha(OUT/'preparation-result.json')
    assert all(sha(Path(a['path']))==a['sha256'] for a in prep['sources'])
    save(OUT/'fit-started.json',dict(created=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
        common_sha256=sha(Path(__file__).with_name('grayT123_head_rigid_common.py')),plan_sha256=sha(PLAN),budget_seconds=120))
    try:
        sy=synthetic();save(OUT/'synthetic-result.json',sy)
        if not sy['passed']:raise RuntimeError('Synthetic rigid projection/recovery failed; no real fits.')
        X,E,K,uv,ww=d['X'],d['E'],d['K'],d['uv'],np.sqrt(d['weight'][...,None]);c=d['center'];r=d['radius']
        hfun=lambda p:transform(p,c,r)
        fun=lambda p:((project(X,E,K,hfun(p))[0]-uv)*ww).ravel()
        shared=solve(fun,d['plan']['fit']['static_global_max_nfev']);Hs=hfun(shared.x);base,basez=project(X,E,K,Hs)
        poses=[];fitrows=[];complete=bool(shared.success) and bool((basez>0).all())
        for i in range(41):
            if time.monotonic()-start>120:raise RuntimeError('CPU budget exhausted; no further fits.')
            f=lambda p:((project(X,E[i:i+1],K,hfun(p))[0][0]-uv[i])*ww[i]).ravel()
            opt=solve(f,d['plan']['fit']['per_frame_max_nfev']);H=hfun(opt.x);poses.append(H)
            pred,z=project(X,E[i:i+1],K,H);valid=bool(opt.success) and bool(np.isfinite(pred).all()) and bool((z>0).all())
            complete=complete and valid
            fitrows.append(dict(frame_number=i+40,success=bool(opt.success),status=int(opt.status),message=str(opt.message),
                nfev=opt.nfev,parameters=opt.x.tolist(),transform=H.tolist(),nose_proxy_errors=stats(np.linalg.norm(pred[0]-uv[i],axis=1)),
                positive_depth=bool((z>0).all()),weighted_cost=float(opt.cost),jacobian_singular_values=np.linalg.svd(opt.jac,compute_uv=False).tolist()))
        poses=np.asarray(poses);sharedfit=dict(success=bool(shared.success),nfev=shared.nfev,parameters=shared.x.tolist(),transform=Hs.tolist(),
            weighted_cost=float(shared.cost),nose_proxy_errors=stats(np.linalg.norm(base-uv,axis=-1).ravel()))
        frozen=dict(created=datetime.now(timezone.utc).isoformat(),shared=sharedfit,per_frame=fitrows,complete=bool(complete),
            source_evidence=[dict(path=str(f),sha256=h) for f,h in zip(d['files'],d['hashes'])],
            skin_observations_not_used_to_fit_poses=True,semantic_proxies_not_truth=True)
        save(OUT/'frozen-proxy-fits.json',frozen);frozenhash=sha(OUT/'frozen-proxy-fits.json')
        if not complete:raise RuntimeError('Incomplete convergence or invalid final projection; preserve fits, no full hypothesis acceptance.')
        sk=read(Path(d['plan']['sources']['skin_observations']))['observations'];assert [o['frame_number'] for o in sk]==d['frames'].tolist()
        suv=np.array([o['xy'] for o in sk]);anchor=np.array([o['role']=='ANCHOR' for o in sk])
        assert d['frames'][anchor].tolist()==d['plan']['held_validation']['skin_anchors']
        sharedskin=point_test(suv,E@Hs,K,anchor,100);localskin=point_test(suv,E@poses,K,anchor,100)
        old=read(Path(d['plan']['sources']['static_skin_baseline']))
        invariant=float(np.max(np.abs(np.array([o['projected'] for o in sharedskin['observations']])-np.array([o['projected'] for o in old['observations']]))))
        assert invariant<.001 and sha(OUT/'frozen-proxy-fits.json')==frozenhash
        held={o['id']:o for o in read(ROOT/'camera/held-evaluation.json')['records']}
        scene=read(Path(d['plan']['sources']['existing_scene_controls']));sc=[]
        counter=E@poses@np.linalg.inv(poses[0]);assert np.max(np.abs(counter[0]-E[0]))<1e-10
        for t in scene['tracks']:
            point=np.array(held[t['id']]['point'])
            for o in t['checks']:
                if not o['same_frame_target']:continue
                i=o['frame_number']-40;pred,z=project(point,counter[i:i+1],K);xy=np.array(o['xy'])
                oldpred,_=project(point,E[i:i+1],K);assert np.max(np.abs(oldpred[0,0]-o['projected']))<1e-7
                sc.append(dict(track_id=t['id'],category=t['category'],frame_number=o['frame_number'],xy=o['xy'],
                    baseline_error=o['euclidean_error_px'],counterfactual_projected=pred[0,0].tolist(),
                    counterfactual_error=float(np.linalg.norm(pred[0,0]-xy)),positive_depth=bool(z[0,0]>0)))
        assert all(sha(f)==h for f,h in zip(d['files'],d['hashes']))
        road=[o for o in sc if o['category']=='static_road_candidate']
        result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,synthetic_passed=sy['passed'],
            frozen_pose_fits_sha256=frozenhash,shared_proxy_fit=sharedfit,per_frame_proxy_fit=stats([o['nose_proxy_errors']['median'] for o in fitrows]),
            shared_skin_validation=sharedskin,per_frame_skin_validation=localskin,shared_skin_invariance_max_abs_px=invariant,
            original_static_skin_held=old['held_check'],background_counterfactual=sc,
            road_counterfactual=dict(baseline=stats([o['baseline_error'] for o in road]),corrected=stats([o['counterfactual_error'] for o in road])),
            complete_diagnostic=bool(sharedskin['success'] and localskin['success'] and localskin['positive_depth_count']==41),
            unique_cause_identified=False,ground_truth_motion_estimated=False,production_camera_changed=False,
            no_images_warped=True,no_training=True,quality_accepted=False,formal_training_admitted=False,
            limits=d['plan']['interpretation_limits'])
        assert result['seconds']<120
        save(OUT/'result.json',result)
        idx=read(OUT/'index.json');panels=[]
        for n,frame in enumerate([58,60,62,65]):
            i=frame-40;xy=suv[i];pp=np.array(localskin['observations'][i]['projected']);bp=np.array(old['observations'][i]['projected'])
            cx,cy=map(int,np.rint(xy));crop=[cx-128,cy-128,cx+128,cy+128]
            with Image.open(native(d['rows'][frame-1]['image'])) as im:raw=im.convert('RGB').crop(crop)
            tile=raw.copy();draw=ImageDraw.Draw(tile);a=xy-crop[:2];b=bp-crop[:2];p=pp-crop[:2]
            draw.ellipse((a[0]-4,a[1]-4,a[0]+4,a[1]+4),outline=(0,255,255),width=1)
            for pt,color in [(b,(255,176,0)),(p,(70,255,70))]:
                draw.line((pt[0]-4,pt[1],pt[0]+4,pt[1]),fill=color,width=1);draw.line((pt[0],pt[1]-4,pt[0],pt[1]+4),fill=color,width=1)
            name=f'frame_{frame:03d}_held_skin_projection.png';tile.save(OUT/name);raw.save(OUT/f'frame_{frame:03d}_held_skin_native.png');panels.append(tile)
            idx['images'].append(dict(frame=d['rows'][frame-1]['frame'],region='独立皮肤点：青圈观测／橙十字静态／绿十字逐帧头部FIT代理',file=name,
                columns=['native256px'],crop=crop))
        canvas=Image.new('RGB',(1024,256))
        for i,t in enumerate(panels):canvas.paste(t,(i*256,0))
        canvas.save(OUT/'same_frame_comparison.png');save(OUT/'index.json',idx)
        print(json.dumps({k:v for k,v in result.items() if k not in ['background_counterfactual','limits','shared_skin_validation','per_frame_skin_validation']},ensure_ascii=False))
        print(json.dumps(dict(shared_skin=sharedskin['held_check'],per_frame_skin=localskin['held_check']),ensure_ascii=False))
    except Exception as exc:
        save(OUT/'failure.json',dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start,
            error=str(exc),traceback=traceback.format_exc(),quality_accepted=False,no_training=True))
        raise

if __name__=='__main__':main()
