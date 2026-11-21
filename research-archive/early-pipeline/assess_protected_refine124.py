"""Record inspected pose evidence; keep the registered numerical gates."""
from datetime import datetime,timezone
import numpy as np
from scipy.spatial.transform import Rotation
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import POLICY
from quality_budget124 import select_budget
from publish_protected_refine124 import OUT,update


def main():
    assert not a.lp(OUT/'conclusion.json').exists()
    stamp=datetime.now(timezone.utc).isoformat();root=OUT.parent
    p=a.read(OUT/'plan.json');result=a.read(OUT/'result.json');review=a.read(OUT/'review/result.json')
    evidence={r['frame']:r for r in review['records']}
    rows={r['frame']:r for r in a.read(root/'grass-temporal-refinement/official-fit/data/training-inputs.json')['records']}
    old={r['frame']:r for r in a.read(root/'grass/selection.json')['records']}
    digest=a.sha(OUT/'selected-fit.npz');assert digest==result['fit_sha256']
    with np.load(a.lp(OUT/'selected-fit.npz'),allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();obs=z['observed_native_keypoints'].copy()
        proj=z['joints_projected_native'].copy();rots=Rotation.from_rotvec(z['global_orient']).as_matrix()
        angles=z['body_pose_axis_angle'].reshape(8,21,3).copy()
    with np.load(a.lp(root/'grass-selected-view-pose124/selected-fit.npz')) as z:
        anchor=z['body_pose_axis_angle'].reshape(1,21,3)
    indices=np.asarray(p['protected_joint_ids'])-1
    assert np.array_equal(angles[:,indices],np.broadcast_to(anchor,(8,21,3))[:,indices])
    assert ids==p['frames'];yaw=np.unwrap(np.arctan2(rots[:,0,2],rots[:,2,2]))
    body=np.asarray([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
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
            metrics[name+'_supported_max_over_height']=maximum/height
            if name in ('body','feet') and maximum/height>POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        records.append(dict(frame=f,time_seconds=rows[f]['time_seconds'],source_fit_sha256=digest,
            yaw_proxy_deg=float(np.degrees(yaw[i])),face_sharpness=old[f]['face_sharpness'],
            body_confidence=float(obs[i,body,2].mean()),person_box_height_px=height,screen_flags=flags,**metrics))
        status='accepted' if f in ('frame_000012','frame_000030') else 'uncertain'
        reason='Coarse pose passes native body/face/feet and all four synthetic directions: raised hand stays near chest; foot and face surfaces remain approximate. Not clothing/head-shape or independent3D accuracy acceptance.' if status=='accepted' else 'Chest configuration remains protected, but existing body residual fails; no selected-pose acceptance. Overview and native face/feet reviewed.'
        if f=='frame_000085':
            status='rejected';reason='Severe global-placement/leg failure: body tilts and feet project outside the true leg/shoe region. Protected arm angles alone do not guarantee a valid whole pose.'
        decisions[f]=dict(status=status,reason=reason,depth_reviewed=True,
            evidence=evidence[f]['images'],evidence_sha256=evidence[f]['images']['sides']['sha256'])
    selection=select_budget(records,decisions)
    assert selection['selected_frames']==['frame_000012','frame_000030']
    selection.update(updated=stamp,policy=POLICY,thresholds_unchanged=True,no_framewise_fit_mixing=True)
    conclusion=dict(updated=stamp,state='completed_chest_preserved_selection_incomplete',
        seconds=result['seconds'],frames_reviewed=ids,
        native_face_feet_reviewed=ids,native_body_and_four_sides_reviewed=['frame_000012','frame_000030'],
        overview_body_side_reviewed=ids,protected_angles_identical_to_shared_seed=True,
        selected_frames=selection['selected_frames'],decisions=decisions,
        finding='Protected torso/shoulder/elbow output preserves the corrected chest-arm configuration during peripheral refinement, but broader numerical consistency remains insufficient and85has severe global/leg regression.',
        no_weight_sweep=True,quality_accepted=False,gaussian_admitted=False,new_gaussian_steps=0,
        next='Close this grass candidate without further tuning. Validate the same frozen selected multi-view fitting method on winterhat; preserve all original quality gates and explicit static-gesture assumption.')
    a.save(OUT/'selection-reviewed.json',selection);a.save(OUT/'visual-review.json',conclusion);a.save(OUT/'conclusion.json',conclusion)
    a.save(OUT/'progress.json',dict(updated=stamp,state=conclusion['state'],pid=None,done=8,total=8))
    update('protected_refine124_chest_preserved_selection_incomplete',
        '草地8帧保护精修290.69秒：胸前手臂保住，但85整体倾斜与腿脚失败；原门槛仅12／30两帧入选，未训练高斯。8帧SMPL-X到高斯坐标交接只读核验通过。',conclusion['next'])
    print('PROTECTED_CANDIDATE_REVIEWED',selection['selected_frames'])


if __name__=='__main__':main()
