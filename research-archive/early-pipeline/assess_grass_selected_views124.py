"""Record actual eight-view review and apply the existing sparse-frame policy."""
from datetime import datetime,timezone
import numpy as np
from scipy.spatial.transform import Rotation
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import select,POLICY
from publish_grass_selected_views124 import OUT

def main():
    assert not a.lp(OUT/'visual-review.json').exists()
    stamp=datetime.now(timezone.utc).isoformat();root=OUT.parent
    p=a.read(OUT/'plan.json');result=a.read(OUT/'result.json');review=a.read(OUT/'review/result.json')
    evidence={r['frame']:r for r in review['records']}
    before={r['frame']:r for r in a.read(root/'grass/selection.json')['records']}
    rows={r['frame']:r for r in a.read(root/'grass-temporal-refinement/official-fit/data/training-inputs.json')['records']}
    fitpath=OUT/'selected-fit.npz';digest=a.sha(fitpath)
    assert digest==result['fit_sha256'] and len(review['records'])==8
    for r in review['records']:
        for im in r['images'].values():
            win=im['path'].replace('/mnt/d/','D:/');assert a.sha(win)==im['sha256']
    with np.load(a.lp(fitpath),allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'];proj=z['joints_projected_native']
        rots=Rotation.from_rotvec(z['global_orient']).as_matrix();shared_pose=z['body_pose_axis_angle'].copy()
    assert ids==p['frames'] and shared_pose.shape==(1,63)
    yaw=np.unwrap(np.arctan2(rots[:,0,2],rots[:,2,2]))
    body=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    decisions={};records=[]
    details={
        'frame_000004':'Side-facing source and overlay are consistent with the raised hand near hair/chest. Face and shoe alignment remain approximate; no long forward-reaching arm in the saved side views.',
        'frame_000012':'Raised hand remains near the chest in the visible source direction and saved novel side geometry. Fingers/shoe surfaces are coarse.',
        'frame_000030':'The previously extended raised forearm is visibly drawn back near the chest, while front projection remains close. No exact hand-clothing contact or anatomical surface truth is claimed.',
        'frame_000046':'Raised hand returns near the chest in side view; face/front placement remains approximate. Legs and shoes retain residual alignment error.',
        'frame_000049':'Same chest-depth improvement as46; feet and hand silhouettes are coarse but no gross limb twist in the reviewed mesh.',
        'frame_000058':'The old forward extension is reduced. Shoulder/elbow placement still differs from the photograph and numerical screening remains binding.',
        'frame_000070':'Raised wrist is near the chest rather than reaching away; body/ankle residuals remain, and the existing numerical screen still applies.',
        'frame_000085':'Rejected: feet visibly float above the actual shoes, face/head orientation is offset and a supported body residual reaches144.6px. Do not use this pose for GS.'}
    for i,f in enumerate(ids):
        e=np.linalg.norm(proj[i]-obs[i,:,:2],axis=1);height=rows[f]['initial_box_xyxy'][3]-rows[f]['initial_box_xyxy'][1]
        flags=[];metrics={}
        for name,idx in [('body',body),('face',np.arange(67,118)),('feet',np.arange(19,25))]:
            w=obs[i,idx,2]**2;assert w.sum()>0
            metrics[name+'_weighted_rms_px']=float(np.sqrt((w*e[idx]**2).sum()/w.sum()))
            maximum=float(e[idx][w>0].max());metrics[name+'_supported_max_over_height']=maximum/height
            if name in ('body','feet') and maximum/height>POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        records.append(dict(frame=f,time_seconds=rows[f]['time_seconds'],source_fit_sha256=digest,
            yaw_proxy_deg=float(np.degrees(yaw[i])),face_sharpness=before[f]['face_sharpness'],
            body_confidence=float(obs[i,body,2].mean()),screen_flags=flags,person_box_height_px=height,**metrics))
        decisions[f]=dict(status='rejected' if f=='frame_000085' else 'accepted',
            reason=details[f],depth_reviewed=True,evidence_sha256=evidence[f]['images']['sides']['sha256'],
            evidence=evidence[f]['images'],acceptance_scope='Coarse visible pose and chest-depth review only; numeric filter and selected-set coverage still apply. Not high-quality geometry acceptance.')
    selection=select(records,decisions,admitted_only=True)
    selection.update(updated=stamp,policy=POLICY,source_fit_sha256=digest,thresholds_unchanged=True,
        actual_visual_review=True,no_framewise_fit_mixing=True)
    conclusion=dict(updated=stamp,state='partial_chest_depth_improvement_selection_incomplete',
        frames_reviewed=ids,actual_native_body_face_feet_and_sides_reviewed=True,
        selected_frames=selection['selected_frames'],selected_count=selection['actual_count'],
        flags={r['frame']:r['screen_flags'] for r in records},
        decisions=decisions,seconds=result['seconds'],
        finding='Joint fitting of different recorded viewpoints reduces the erroneous forward-reaching arm while preserving approximate front alignment. Fixed local body pose also causes failures, notably floating feet in85.',
        causality_limit='Changes include shared local pose, joint observation aggregation and root initialization; do not attribute all improvement to temporal smoothing or prove precise camera calibration.',
        wrist_reference=dict(frame='frame_000030',old_local_shoulder_to_wrist_m=evidence['frame_000030']['metrics']['old_temporal']['right_wrist_relative_right_shoulder_body_local_m'],
            new_local_shoulder_to_wrist_m=evidence['frame_000030']['metrics']['shared_eight']['right_wrist_relative_right_shoulder_body_local_m'],not_measured_anatomical_depth=True),
        no_chest_contact_target=True,no_triangulated_3d_target=True,
        quality_accepted=False,gaussian_admitted=False,new_gaussian_steps=0,
        original_result_receipt_note='The inherited upstream_loss_monitor_optimizer_unchanged field describes the upstream implementations. This selected-view adapter broadcasts the local pose, averages the objective and adds per-view placement parameters; it is not the unmodified single-frame pipeline.',
        followup='Keep this coherent eight-view result frozen. Continue selected-frame quality/coverage work before GS; never reuse rejected85 or silently relax thresholds to fill slots.')
    a.save(OUT/'selection-reviewed.json',selection);a.save(OUT/'visual-review.json',conclusion)
    a.save(OUT/'conclusion.json',conclusion)
    a.save(OUT/'progress.json',dict(updated=stamp,state='completed_partial_improvement_not_admitted',pid=None))
    print(dict(selected=selection['selected_frames'],missing_bins=selection['missing_bins'],flags=conclusion['flags']))

if __name__=='__main__':main()
