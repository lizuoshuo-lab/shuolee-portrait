"""Check whether 021 limb detector points support a stationary 3D hypothesis."""
from datetime import datetime, timezone
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares

import fit_video21_native148 as portrait
from update_leafwoman129_state import atomic, read, sha


CAM = portrait.OUT/'fixedk-dense159-reference'
OUT = portrait.OUT/'static-limb-probe160'
JOINTS = {
    2:'right_shoulder',3:'right_elbow',4:'right_wrist',
    5:'left_shoulder',6:'left_elbow',7:'left_wrist',
    10:'right_knee',11:'right_ankle',13:'left_knee',14:'left_ankle',
    19:'left_big_toe',20:'left_small_toe',21:'left_heel',
    22:'right_big_toe',23:'right_small_toe',24:'right_heel',
}


def project(P, X):
    q=np.einsum('nki,i->nk',P,np.r_[X,1.])
    z=q[:,2]
    return q[:,:2]/np.maximum(z[:,None],1e-9),z


def triangulate(fit, uv, P, ray):
    ranked=[]
    for i,j in combinations(fit,2):
        if np.degrees(np.arccos(np.clip(ray[i]@ray[j],-1.,1.)))<.5:
            continue
        h=cv2.triangulatePoints(P[i],P[j],uv[i,:2].reshape(2,1),uv[j,:2].reshape(2,1))[:,0]
        if not np.isfinite(h).all() or abs(h[3])<1e-12:
            continue
        X=h[:3]/h[3]
        q,z=project(P[fit],X)
        errors=np.linalg.norm(q-uv[fit,:2],axis=1)
        inliers=(z>0)&(errors<=3.)
        if inliers.sum()>=4:
            ranked.append((int(inliers.sum()),-float(np.median(errors[inliers])),i,j,X,fit[inliers]))
    if not ranked:
        return None
    _,_,i,j,X,ok=max(ranked,key=lambda item:item[:4])
    fit_opt=least_squares(lambda x: ((project(P[ok],x)[0]-uv[ok,:2])*uv[ok,2,None]).ravel(),
                          X,loss='soft_l1',f_scale=1.,max_nfev=40)
    return fit_opt.x,ok,[i,j]


def main():
    assert not OUT.exists()
    data=CAM/'seated021/global-fixedk'
    camera=read(data/'result.json')
    assert read(CAM/'process-exit.json')['result_sha256']==sha(data/'result.json')
    assert camera['registered_count']==camera['native_frame_count']==113
    p=read(portrait.OUT/'plan.json')
    assert camera['K_integer_centers']==p['K_integer_centers']
    assert sha(portrait.ARCHIVE)==p['baseline_observation_sha256']
    with np.load(portrait.ARCHIVE,allow_pickle=False) as src:
        names=src['frame_ids'].tolist()
        keypoints=src['observed_native_keypoints'].astype(float)
        K=src['K_integer_centers']
    assert names==p['source_frames'] and not set(names)&set(p['retained_holdouts'])
    bg=camera['heldout_leave_one_view_out']['background_static_support_candidate']['per_frame']
    available=np.array([i for i,f in enumerate(names)
                        if bg[f]['n']>=10 and bg[f]['median']<=1. and bg[f]['p90']<=3.])
    held=available[::3]
    fit=np.array(sorted(set(available)-set(held)))
    matrices={row['frame']:row for row in read(data/'cameras-diagnostic.json')}
    R=np.array([matrices[f]['R_w2c'] for f in names])
    C=np.array([matrices[f]['C_world'] for f in names])
    P=K@np.concatenate([R,-(R@C[...,None])],axis=2)
    assert np.allclose(np.linalg.det(R),1.,atol=1e-8)
    plan_path=OUT/'plan.json'
    atomic(plan_path,dict(updated=datetime.now(timezone.utc).isoformat(),worker_sha256=sha(__file__),
        source_camera_result_sha256=sha(data/'result.json'),camera_sha256=sha(data/'cameras-diagnostic.json'),
        source_observations_sha256=sha(portrait.ARCHIVE),available=[names[i] for i in available],
        fit=[names[i] for i in fit],held=[names[i] for i in held],global_heldouts_untouched=p['retained_holdouts'],
        hypothesis='One stationary 3D point should predict visible leg and shoe detector positions across camera-supported views.',
        split='Chronological every third camera-eligible frame held out from limb triangulation. Existing four portrait holdouts unused.',
        fit_policy='Pair hypotheses from fit views only, at least 0.5 degree parallax and 4 positive 3px fit inliers. Robust point-only refinement.',
        cameras_fixed=True,diagnostic_detector_landmarks_only=True,physical_joint_truth=False,gaussian_started=False))
    results=[]
    for j,name in JOINTS.items():
        uv=keypoints[:,j]
        eligible_fit=fit[uv[fit,2]>0]
        eligible_held=held[uv[held,2]>0]
        unit=np.c_[uv[:,:2],np.ones(len(uv))]@np.linalg.inv(K).T
        ray=np.einsum('nji,nj->ni',R,unit)
        ray/=np.linalg.norm(ray,axis=1,keepdims=True)
        candidate=triangulate(eligible_fit,uv,P,ray)
        record=dict(index=j,name=name,eligible_fit_count=len(eligible_fit),eligible_held_count=len(eligible_held),
                    triangulated=candidate is not None)
        if candidate is not None:
            X,ok,seed=candidate
            predicted,z=project(P[eligible_held],X)
            errors=np.linalg.norm(predicted-uv[eligible_held,:2],axis=1)
            record.update(xyz_arbitrary_scale=X.tolist(),inlier_fit_frames=[names[i] for i in ok],
                seed_fit_frames=[names[i] for i in seed],
                held=[dict(frame=names[i],observed_uv=uv[i,:2].tolist(),predicted_uv=q.tolist(),
                           error_px=float(e),positive_depth=bool(depth>0))
                      for i,q,e,depth in zip(eligible_held,predicted,errors,z,strict=True)],
                held_median_px=float(np.median(errors)) if len(errors) else None,
                held_p90_px=float(np.quantile(errors,.9)) if len(errors) else None,
                max_held_px=float(max(errors)) if len(errors) else None)
        results.append(record)
        print('STATIC_JOINT',name,record['triangulated'],record.get('held_median_px'),record.get('held_p90_px'),flush=True)
    atomic(OUT/'result.json',dict(plan_sha256=sha(plan_path),rows=results,body_fitting_updates=0,
                                  physical_joint_truth=False,gaussian_started=False))


if __name__=='__main__':
    main()
