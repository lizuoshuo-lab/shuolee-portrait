"""Frozen 022 native detector consistency, neutral SMPL-X PnP only."""
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from official_smplifyx_entry123 import sha
from update_leafwoman129_state import FIT, NATIVE, ROOT, atomic, read

BASE=ROOT/'leafwoman-fit40-132'
OUT=BASE/'front-face-seed136'
ASSET=Path('D:/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
FIT_IDS=np.r_[np.arange(10,19),[19,22,25,28]] # iBUG 27..35 + 36/39/42/45
VARIANTS=dict(primary=FIT_IDS,one_eye_out=FIT_IDS[~np.isin(FIT_IDS,[19,22])],
    other_eye_out=FIT_IDS[~np.isin(FIT_IDS,[25,28])],lower_nose_out=FIT_IDS[~np.isin(FIT_IDS,np.arange(14,19))])
GROUPS=dict(brows=np.arange(0,10),nose=np.arange(10,19),eyes=np.arange(19,31),mouth=np.arange(31,51))


def project(q,K):
    p=q@K.T;return p[:,:2]/p[:,2:3]


def solve(X,xy,K,take):
    if len(take)<6 or np.linalg.matrix_rank(X[take]-X[take].mean(axis=0))!=3:
        raise ValueError('Insufficient 3D support for the fixed PnP subset.')
    xx=np.ascontiguousarray(X[take]);uv=np.ascontiguousarray(xy[take]);K=np.ascontiguousarray(K)
    ok,r,t=cv2.solvePnP(xx,uv,K,None,flags=cv2.SOLVEPNP_EPNP)
    if not ok:raise ValueError('EPnP failed.')
    r,t=cv2.solvePnPRefineLM(xx,uv,K,None,r,t,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,20,1e-8))
    R=cv2.Rodrigues(r)[0];t=t[:,0];q=X@R.T+t
    if not np.isfinite(q).all() or (q[:,2]<=0).any() or abs(np.linalg.det(R)-1)>1e-8:
        raise ValueError('Nonfinite or non-positive pose.')
    jac=cv2.projectPoints(xx,r,t,K,None)[1][:,:6]
    if np.linalg.matrix_rank(jac)!=6:raise ValueError('PnP local pose rank is deficient.')
    return R,t,q


def stats(e):
    return dict(n=len(e),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),max_px=float(e.max())) if len(e) else dict(n=0)


def run():
    started=time.monotonic();cv2.setNumThreads(1)
    archive=FIT/'selected-fit.npz'
    contour=BASE/'contour-interface-audit/receipt.json'
    contacts=NATIVE/'source-pose-review/source-contacts.json'
    pin={str(p):sha(p) for p in (archive,ASSET,contour,contacts,BASE/'fit-plan.json',Path(__file__))}
    assert pin[str(archive)]==read(FIT/'result.json')['fit_sha256']
    with np.load(ASSET,allow_pickle=False) as a,np.load(archive,allow_pickle=False) as b:
        f=a['f'];lf=a['lmk_faces_idx'];bary=a['lmk_bary_coords']
        assert np.array_equal(f,b['faces']) and lf.shape==(51,) and bary.shape==(51,3)
        inds=f[lf]
        model=np.einsum('lvc,lv->lc',a['v_template'][inds],bary).astype(np.float64)
        # Verify correspondence against actual saved posed vertices, not just array length.
        replay=np.einsum('flvc,lv->flc',b['vertices_camera'][:,inds],bary)
        delta=float(np.max(abs(replay-b['joints_camera'][:,67:118])))
        assert delta<1e-5
        ids=b['frame_ids'].astype(str).tolist()
        target=b['observed_native_keypoints'][:,67:118].astype(np.float64)
        old=b['joints_projected_native'][:,67:118].astype(np.float64)
        K=b['K_integer_centers'].astype(np.float64)
    assert len(ids)==24 and ids==[r['frame'] for r in read(BASE/'fit-plan.json')['front_arc']]
    assert K.shape==(3,3) and np.isfinite(target).all()
    # t_z is the depth of the declared model head center, never pelvis or true distance.
    X=model-model.mean(axis=0)
    OUT.mkdir(exist_ok=False)
    plan=dict(updated=datetime.now(timezone.utc).isoformat(),pins=pin,frames=ids,fixed_K=K.tolist(),
        landmark_order='Official static51 in iBUG17:68, archive67:118. Neutral template; no fitted shape or pose used by PnP.',
        fit_subsets={k:v.tolist() for k,v in VARIANTS.items()},
        method='One EPnP and one fixed20-iteration LM refine per subset, detector consistency diagnostic only.',
        admission='For diagnostic stability only: all four solves valid, depth spread<=5%, rotation spread<=5deg, all omitted rigid groups p90<=3 native px. Even a pass cannot supply metric head scale or pelvis depth.',
        reference_crop='Preselected frame27; display only. Source joints are same-detector predictions and no per-landmark visibility truth.',
        no_scan=True,SMPLify_X_updates=0,Gaussian_updates=0)
    atomic(OUT/'plan.json',plan)
    rtrue=cv2.Rodrigues(np.array([.12,-.23,.04]))[0];ttrue=np.array([.04,-.08,2.0])
    synth=project(X@rtrue.T+ttrue,K)
    _,t,q=solve(X,synth,K,FIT_IDS)
    control=float(np.max(np.linalg.norm(project(q,K)-synth,axis=1)))
    assert control<1e-5 and np.max(abs(t-ttrue))<1e-5

    rows=[];primary_by_frame={};rigid_held_values=[];spread_values=[];rot_values=[]
    for f,obs,prior in zip(ids,target,old):
        xy=obs[:,:2];present=(obs[:,2]>0)&(xy[:,0]>=0)&(xy[:,0]<2704)&(xy[:,1]>=0)&(xy[:,1]<1520)
        records={};rot=[];depths=[];held_p90=[]
        for label,subset in VARIANTS.items():
            take=subset[present[subset]]
            try:R,t,q=solve(X,xy,K,take)
            except (ValueError,cv2.error) as e:records[label]=dict(rejected=str(e),fit_count=len(take));continue
            uv=project(q,K);err=np.linalg.norm(uv-xy,axis=1)
            scaled=project(q*1.05,K);scale_delta=float(np.max(abs(scaled-uv)))
            assert scale_delta<1e-10
            held=present.copy();held[take]=False
            omitted=np.asarray([j for j in FIT_IDS if j not in set(take) and present[j]],dtype=int)
            metric=stats(err[omitted]);rot.append(R);depths.append(float(t[2]))
            if label!='primary':held_p90.append(metric.get('p90_px',float('inf')));rigid_held_values.extend(err[omitted])
            records[label]=dict(fit_count=len(take),head_center_z_model_units=float(t[2]),
                omitted_rigid=metric,held_by_region={k:stats(err[v[held[v]]]) for k,v in GROUPS.items()},
                scale_1p05_same_reprojection_max_px=scale_delta)
            if label=='primary':primary_by_frame[f]=uv
        depthspread=(max(depths)-min(depths))/np.median(depths) if len(depths)==4 else None
        rotation_spread=max(float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1)))) for a in rot for b in rot) if len(rot)==4 else None
        passed=depthspread is not None and depthspread<=.05 and rotation_spread<=5 and max(held_p90)<=3
        if depthspread is not None:spread_values.append(depthspread);rot_values.append(rotation_spread)
        rows.append(dict(frame=f,solutions=records,head_depth_relative_spread=depthspread,rotation_spread_deg=rotation_spread,
                         conditional_detector_seed_stable=bool(passed)))
    frame='frame_000027';i=ids.index(frame)
    if frame in primary_by_frame:
        record=next(r for r in read(contacts)['records'] if r['frame']==frame)['paths']['face']
        path=Path(record['path'].replace('/mnt/c/','C:/',1));assert sha(path)==record['sha256']
        with Image.open(path) as im:raw=im.convert('RGB')
        x0,y0,x1,y1=record['crop_original_xyxy'];w,h=raw.size;assert (w,h)==(x1-x0,y1-y0)
        panel=Image.new('RGB',(3*w,h+24),(22,25,31));d=ImageDraw.Draw(panel)
        for j,(name,pred,color) in enumerate((('Native detector',target[i,:,:2],'#ff80bd'),('Previous projection',old[i],'#50def7'),('Neutral PnP',primary_by_frame[frame],'#82f7ab'))):
            crop=raw.copy();draw=ImageDraw.Draw(crop)
            for x,y in pred-[x0,y0]:draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),outline=color)
            panel.paste(crop,(j*w,24));d.text((j*w+5,5),name,fill='white')
        panel.save(OUT/'frame27-head-seed.png')
    assert pin=={k:sha(k) for k in pin}
    result=dict(updated=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,frames=rows,
        synthetic_control_max_projection_px=control,archive_binding_max_delta=delta,
        conditional_seed_stable_count=sum(r['conditional_detector_seed_stable'] for r in rows),
        full_variant_count=len(spread_values),median_depth_relative_spread=float(np.median(spread_values)) if spread_values else None,
        median_rotation_spread_deg=float(np.median(rot_values)) if rot_values else None,
        omitted_rigid_detector_errors=stats(np.asarray(rigid_held_values)),
        independent_metric_depth=False,pelvis_depth_available=False,camera_trajectory_available=False,
        no_seed_written_to_fitting=True,SMPLify_X_updates=0,Gaussian_updates=0,quality_accepted=False)
    atomic(OUT/'result.json',result)
    print('VIDEO22_HEAD_SEED_DIAGNOSTIC',result['conditional_seed_stable_count'],result['full_variant_count'],result['median_depth_relative_spread'],result['omitted_rigid_detector_errors'])


if __name__=='__main__':run()
