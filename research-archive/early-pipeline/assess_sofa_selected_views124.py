"""Unchanged quality-first screening of one coherent sofa selected-view fit."""
from datetime import datetime,timezone
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import audit_gaussianavatars_params123 as a
import screen_official_body_contract124 as body_screen
from select_quality_frames124 import select,POLICY
from quality_budget124 import select_budget
from screen_wrist_consistency124 import apply_to_candidates

ROOT=a.REC/'cross-subject-sh0-124';OUT=ROOT/'sofa-selected-view-pose124';SRC=ROOT/'sofa-input'


def measure():
    assert not a.lp(OUT/'numeric-screen.json').exists()
    p=a.read(OUT/'plan.json');result=a.read(OUT/'result.json');digest=a.sha(OUT/'selected-fit.npz')
    assert digest==result['fit_sha256'] and a.read(OUT/'process-exit.json')['exit_code']==0
    with np.load(a.lp(OUT/'selected-fit.npz')) as z:
        ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'].copy();proj=z['joints_projected_native'].copy()
        rotation=Rotation.from_rotvec(z['global_orient']).as_matrix()
        assert z['body_pose_axis_angle'].shape==(1,63) and z['betas'].shape==(1,10)
    assert ids==p['frames'] and not set(ids)&set(p['review_only_excluded'])
    rows={r['frame']:r for r in p['rows']};source=a.read(SRC/'selection.json')
    rank={r['frame']:r for r in source['candidates']}
    yaw=np.degrees(np.unwrap(np.arctan2(rotation[:,0,2],rotation[:,2,2])))
    weights=np.ones(118);weights[list(body_screen.IGNORED)]=0;records=[];comparison=[]
    for i,f in enumerate(ids):
        row=rows[f];height=row['initial_box_xyxy'][3]-row['initial_box_xyxy'][1]
        e=np.linalg.norm(proj[i]-obs[i,:,:2],axis=1);metrics={};flags=[]
        oldpath=ROOT/'sofa-temporal-warmstart-v2/official-fit/frames'/f/'winner-camera-space.npz'
        with np.load(a.lp(oldpath)) as z:
            assert np.array_equal(z['K_integer_centers'],row['K_integer_centers'])
            olde=np.linalg.norm(z['joints_projected_native'][0]-obs[i,:,:2],axis=1)
        oldmetrics={}
        for name,idx in [('body',body_screen.BODY),('face',np.arange(67,118)),('hands',np.arange(25,67)),('feet',np.arange(19,25))]:
            w=obs[i,idx,2]**2;assert w.sum()>0
            metrics[name+'_weighted_rms_px']=float(np.sqrt((w*e[idx]**2).sum()/w.sum()))
            metrics[name+'_supported_max_over_height']=float(e[idx][w>0].max()/height)
            oldmetrics[name+'_weighted_rms_px']=float(np.sqrt((w*olde[idx]**2).sum()/w.sum()))
            if name in ('body','feet') and metrics[name+'_supported_max_over_height']>POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        row=dict(frame=f,time_seconds=row['time_seconds'],source_fit_sha256=digest,
            yaw_proxy_deg=float(yaw[i]),face_sharpness=rank[f]['sharpness'],body_confidence=float(obs[i,body_screen.BODY,2].mean()),
            screen_flags=flags,person_box_height_px=height,all118_error_px=e.tolist(),**metrics)
        row=body_screen.apply(row,body_screen.measure(obs[i],proj[i],height,weights));records.append(row)
        comparison.append(dict(frame=f,baseline=oldmetrics,current={k:metrics[k] for k in oldmetrics}))
    records=apply_to_candidates(records,a.read(ROOT/'selected-observation-audit124/wrist-preselection/sofa.json')['records'])
    numeric=select(records,admitted_only=False)
    numeric.update(updated=datetime.now(timezone.utc).isoformat(),source_fit_sha256=digest,
        original_supported_joint_contract=True,comparison=comparison,source_rejected_frames=p['source_rejected_frames'],
        source_roster=56,all56_not_required=True,thresholds_unchanged=True,new_fit_calls=0,
        temporal_scope='One shared local body pose at these7times. Not validation of unselected motion or continuity of per-view roots.',
        current_coarse_numeric_survivors=[r['frame'] for r in numeric['records'] if r['eligible']],
        actual_pose_review=False,quality_accepted=False,gaussian_admitted=False)
    a.save(OUT/'numeric-screen.json',numeric)
    print({r['frame']:dict(body_max=r['original_body_measurement_contract']['supported_body_max_px'],feet_ratio=r['feet_supported_max_over_height'],flags=r['screen_flags']) for r in records})


def close():
    assert not a.lp(OUT/'conclusion.json').exists()
    numeric=a.read(OUT/'numeric-screen.json');review=a.read(OUT/'review-decisions.json')
    assert review['actual_visual_review'] and set(review['decisions'])==set(a.read(OUT/'plan.json')['frames'])
    result=select_budget(numeric['records'],review['decisions'])
    result.update(updated=datetime.now(timezone.utc).isoformat(),thresholds_unchanged=True,actual_visual_review=True,
        numeric_source=str(OUT/'numeric-screen.json'),no_framewise_fit_mixing=True)
    ys=[r['yaw_proxy_deg'] for r in result['records'] if r['selected']]
    result['selected_yaw_proxy_span_deg']=max(ys)-min(ys) if ys else 0.
    a.save(OUT/'selection-reviewed.json',result)
    conclusion=dict(updated=result['updated'],state='selected_sofa_fit_review_complete',selected_frames=result['selected_frames'],
        selected_count=result['actual_count'],selected_yaw_proxy_span_deg=result['selected_yaw_proxy_span_deg'],
        seconds=a.read(OUT/'result.json')['seconds'],source_rejected_frames=['frame_000053'],
        actual_native_body_face_hands_feet_and_sides_reviewed=True,decisions=review['decisions'],
        numeric_survivors=numeric['current_coarse_numeric_survivors'],original_supported_body_contract=True,
        quality_accepted=False,gaussian_admitted=False,scene_camera_physically_calibrated=False,
        finding=review['finding'],next=review['next'])
    a.save(OUT/'conclusion.json',conclusion)
    a.save(OUT/'progress.json',dict(updated=result['updated'],state=conclusion['state'],pid=None))
    print({k:conclusion[k] for k in ('selected_frames','selected_count','selected_yaw_proxy_span_deg','finding')})


if __name__=='__main__':{'measure':measure,'close':close}[sys.argv[1]]()
