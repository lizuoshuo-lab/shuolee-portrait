"""Close the one shared-beta control using reviewed, immutable image evidence."""
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import POLICY
from quality_budget124 import select_budget
from screen_wrist_consistency124 import apply_to_candidates
from publish_shared_shape_views124 import OUT, state


def main():
    assert not a.lp(OUT/'conclusion.json').exists()
    root=OUT.parent;stamp=datetime.now(timezone.utc).isoformat()
    result=a.read(OUT/'result.json');review=a.read(OUT/'review/result.json')
    evidence={r['frame']:r for r in review['records']}
    rows={r['frame']:r for r in a.read(root/'grass-temporal-refinement/official-fit/data/training-inputs.json')['records']}
    old={r['frame']:r for r in a.read(root/'grass/selection.json')['records']}
    digest=a.sha(OUT/'selected-fit.npz');assert digest==result['fit_sha256']
    with np.load(a.lp(OUT/'selected-fit.npz'),allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'].copy()
        proj=z['joints_projected_native'].copy();rots=Rotation.from_rotvec(z['global_orient']).as_matrix()
        assert z['betas'].shape==(1,10) and z['body_pose_axis_angle'].shape==(1,63)
    assert ids==a.read(OUT/'plan.json')['frames']
    yaw=np.degrees(np.unwrap(np.arctan2(rots[:,0,2],rots[:,2,2])))
    body=np.asarray([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    admitted=['frame_000030','frame_000046','frame_000049']
    records=[];decisions={}
    for i,f in enumerate(ids):
        for im in evidence[f]['images'].values():
            assert a.sha(im['path'].replace('/mnt/d/','D:/'))==im['sha256']
        e=np.linalg.norm(proj[i]-obs[i,:,:2],axis=1)
        height=rows[f]['initial_box_xyxy'][3]-rows[f]['initial_box_xyxy'][1]
        metrics={};flags=[]
        for name,idx in [('body',body),('face',np.arange(67,118)),('feet',np.arange(19,25))]:
            w=obs[i,idx,2]**2;maximum=float(e[idx][w>0].max())
            metrics[name+'_weighted_rms_px']=float(np.sqrt(np.sum(w*e[idx]**2)/w.sum()))
            metrics[name+'_supported_max_px']=maximum
            metrics[name+'_supported_max_over_height']=maximum/height
            if name in ('body','feet') and maximum/height>POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        records.append(dict(frame=f,time_seconds=rows[f]['time_seconds'],source_fit_sha256=digest,
            yaw_proxy_deg=float(yaw[i]),face_sharpness=old[f]['face_sharpness'],
            body_confidence=float(obs[i,body,2].mean()),person_box_height_px=height,screen_flags=flags,**metrics))
        status='accepted' if f in admitted else 'uncertain'
        reason='Native body/face/feet and all four synthetic views reviewed. The raised hand remains close to the chest, without the old long forward-reaching arm. Face, fingers and unclothed feet are approximate; no exact contact or anatomical-depth truth is claimed.'
        if f not in admitted:
            reason='Both overview pages and native face/feet reviewed. Existing body projection gate fails; do not promote from a small image improvement or lower total loss.'
        if f=='frame_000085':
            status='rejected';reason='Raised/other hand projection and head orientation remain wrong; feet visibly float above the photographed shoes. Body maximum139.64px and feet46.38px. Reject this whole frame.'
        decisions[f]=dict(status=status,reason=reason,depth_reviewed=f in admitted,
            evidence=evidence[f]['images'],evidence_sha256=evidence[f]['images']['sides']['sha256'],
            acceptance_scope='Coarse visible pose only. Numerical gate and selected-set coverage still apply; not independent 3D accuracy or high-quality surface acceptance.')
    wrist=a.read(root/'selected-observation-audit124/wrist-preselection/grass.json')
    records=apply_to_candidates(records,wrist['records'])
    selection=select_budget(records,decisions)
    assert selection['selected_frames']==admitted
    chosen_yaw=[r['yaw_proxy_deg'] for r in records if r['frame'] in admitted]
    selection.update(updated=stamp,policy=POLICY,thresholds_unchanged=True,no_framewise_fit_mixing=True,
        actual_visual_review=True,selected_yaw_proxy_span_deg=max(chosen_yaw)-min(chosen_yaw),
        duplicate_wrist_screen_applied=True)
    conclusion=dict(updated=stamp,state='completed_shared_shape_partial_pose_selection_incomplete',
        seconds=result['seconds'],frames_reviewed=ids,native_face_feet_reviewed=ids,
        native_body_and_four_sides_reviewed=admitted,overview_body_side_reviewed=ids,
        selected_frames=admitted,selected_count=3,selected_yaw_proxy_span_deg=selection['selected_yaw_proxy_span_deg'],
        source_fit_sha256=digest,decisions=decisions,one_shared_beta=True,
        beta_before=result['beta_before'],beta_after=result['beta_after'],beta_delta_l2=result['beta_delta_l2'],
        finding='Updating one shared shape from all eight views modestly changes proportions and yields three numerically eligible, coarsely reviewed frames. It does not resolve the poor side views or frame85. Chest-arm improvement already existed in the shared-pose baseline and must not be attributed to this beta-only control.',
        observation_contract='Body screen still includes hips9/12 ignored by the original fitting loss; they were not silently removed. Reprojection is to detector output, not independently measured material truth.',
        quality_accepted=False,gaussian_admitted=False,new_gaussian_steps=0,
        next='Freeze this completed candidate and publish it without more beta/weight sweeps. Use the input-consistency screen in future candidate rosters; keep whole-frame review and angular coverage. Before any new fit, identify a new source-backed cause of the missing side-view pose support; do not pool frames across C1/C3/this shape result or launch GS from only these three near-front views.')
    for name,value in [('selection-reviewed.json',selection),('visual-review.json',conclusion),('conclusion.json',conclusion)]:a.save(OUT/name,value)
    a.save(OUT/'progress.json',dict(updated=stamp,state=conclusion['state'],pid=None,done=8,total=8))
    winter=root/'winterhat-selected-view-pose124/body-relative-review'
    wr=a.read(winter/'result.json');assert not a.lp(winter/'visual-review.json').exists()
    assert a.sha(wr['file'].replace('/mnt/d/','D:/'))==wr['image_sha256']
    a.save(winter/'visual-review.json',dict(updated=stamp,actual_visual_review=True,
        image=wr['file'],image_sha256=wr['image_sha256'],source_result_sha256=a.sha(winter/'result.json'),
        finding='After removing global root rotation and translation, the forward-inclined torso and knee configuration remains in the new shared local pose. The prior observation is therefore not solely an effect of camera/root pitch. This does not measure gravity or prove exact real-body lean.',
        scope='Frame56 in four body-local synthetic directions; one shared body pose across eight frames. Per-frame face, fingers and placement are not all verified by this single view.',
        wrist_contact_or_depth_verified=False,quality_accepted=False,gaussian_admitted=False,new_fit_calls=0))
    summary='草地8视角共享体型对照完成141.80秒：胸前手臂保留，仅30／46／49三帧通过原筛选且视角集中；85手脚失败。四人物373帧输入一致性筛选完成，原观测未改。未启动高斯。'
    state('shared_shape_views124_closed_selection_incomplete',summary,conclusion['next'])
    print(dict(selected=admitted,yaw_span_deg=selection['selected_yaw_proxy_span_deg'],gaussian_admitted=False))


if __name__=='__main__':main()
