"""Read-only reproducible screening; no fitting, deletion, or dataset mutation."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import audit_gaussianavatars_params123 as a

RUN = a.REC/'demo-vhap-initialization/body-gaussian-upstream-offset'
FIT_METRICS = a.REC/'demo-vhap-initialization/smplify-vhap-pose-prior-long/epoch-120/result.json'
POLICY = {
    'version': 1,
    'purpose': 'Conservative review flags; not independent physical truth or automatic frame exclusion.',
    'body_max_error_over_person_box_height': 0.03,
    'elbow_error_over_person_box_height': 0.03,
    'feet_max_error_over_person_box_height': 0.03,
    'minimum_mesh_mask_iou': 0.75,
    'maximum_target_uncovered_fraction': 0.15,
    'maximum_local_body_joint_speed_deg_per_second': 135.0,
    'temporal_ambiguity': 'Flags both endpoints. Fast real motion and ambiguous fitted rotations require review.',
    'unknown_measurement_policy': 'Mark unknown; never silently pass.',
    'input_action': 'review_required, no files removed or current training inputs changed',
    'render_action': 'Human arm/connectivity review retained; RGB brightness is not used as alpha or silhouette.',
    'future_training': 'Only use confirmed valid fits; preserve every excluded frame in evaluation and fail coverage if a necessary view is lost.',
}


def plan():
    out = RUN/'quality-screen-plan.json'
    value = dict(updated=datetime.now(timezone.utc).isoformat(), policy=POLICY,
        inputs={str(p): a.sha(p) for p in (RUN/'fit/fitted.npz', RUN/'data/training-inputs.json', FIT_METRICS)},
        frame_specific_thresholds=False, threshold_search=False, solver_changed=False,
        user_request='这种畸变的后面要加筛选机制')
    if a.lp(out).exists():
        old=a.read(out); assert old['policy']==POLICY and old['inputs']==value['inputs']
    else:
        a.save(out,value)
    print('QUALITY_SCREEN_POLICY_REGISTERED')


def main(step):
    saved=a.read(RUN/'quality-screen-plan.json'); assert saved['policy']==POLICY
    for p,d in saved['inputs'].items(): assert a.sha(p)==d,p
    metrics={r['frame']:r for r in a.read(FIT_METRICS)['records']}
    rows=sorted(a.read(RUN/'data/training-inputs.json')['records'],key=lambda r:r['time_seconds'])
    assert len(rows)==len(metrics)==59 and len({r['frame'] for r in rows})==59
    with np.load(a.lp(RUN/'fit/fitted.npz'),allow_pickle=False) as z:
        frame_ids=list(z['frame_ids']); pose=np.array(z['body_pose_axis_angle']).reshape(59,21,3)
    pose={str(f):p for f,p in zip(frame_ids,pose)}
    flags={r['frame']:[] for r in rows}; unknown={r['frame']:[] for r in rows}; records=[]; transitions=[]
    for row in rows:
        frame=row['frame']; m=metrics[frame]; box=row['initial_box_xyxy'];height=float(box[3]-box[1])
        assert height>0
        values=dict(person_box_height_px=height,body_max_over_height=m['groups']['body']['max']/height,
            elbow_over_height=m['elbow_px']/height,feet_max_over_height=m['groups']['feet']['max']/height,
            mesh_mask_iou=m['mask_iou'],target_uncovered_fraction=m['target_uncovered'])
        rules=[('body_max_over_height','body_max_error_over_person_box_height','>'),
               ('elbow_over_height','elbow_error_over_person_box_height','>'),
               ('feet_max_over_height','feet_max_error_over_person_box_height','>'),
               ('mesh_mask_iou','minimum_mesh_mask_iou','<'),
               ('target_uncovered_fraction','maximum_target_uncovered_fraction','>')]
        for key,policy_key,op in rules:
            value=values[key]; threshold=POLICY[policy_key]
            if not np.isfinite(value): unknown[frame].append(key)
            elif (value>threshold if op=='>' else value<threshold):
                flags[frame].append(dict(code=key,value=value,threshold=threshold,comparison=op))
        records.append(dict(frame=frame,metrics=values))
    for before,after in zip(rows,rows[1:]):
        fa,fb=before['frame'],after['frame']; dt=after['time_seconds']-before['time_seconds']; assert dt>0
        relative=Rotation.from_rotvec(pose[fa]).inv()*Rotation.from_rotvec(pose[fb])
        degrees=np.degrees(relative.magnitude()); joint=int(np.argmax(degrees)); speed=float(degrees[joint]/dt)
        item=dict(before=fa,after=fb,dt_seconds=dt,body_joint_index=joint,angle_deg=float(degrees[joint]),speed_deg_per_second=speed)
        transitions.append(item)
        if speed>POLICY['maximum_local_body_joint_speed_deg_per_second']:
            for f in (fa,fb): flags[f].append(dict(code='temporal_body_pose_jump',transition=item,
                threshold=POLICY['maximum_local_body_joint_speed_deg_per_second'],comparison='>'))
    for row in records:
        f=row['frame'];row.update(flags=flags[f],unknown_measurements=unknown[f],
            screening='review_required' if flags[f] or unknown[f] else 'no_configured_alarm',quality_accepted=False)
    result=dict(updated=datetime.now(timezone.utc).isoformat(),step=step,frames=59,policy=POLICY,
        input_hashes=saved['inputs'],policy_sha256=a.sha(RUN/'quality-screen-plan.json'),script_sha256=a.sha(__file__),
        records=records,transitions=transitions,flagged_count=sum(bool(flags[r['frame']]) for r in rows),
        flagged_frames=[r['frame'] for r in rows if flags[r['frame']]],
        measurement_scope='Pre-Gaussian VHAP fit metrics and frozen body pose; current Gaussian offset does not alter body pose. Render alpha is unavailable in this snapshot.',
        automatic_render_silhouette_checked=False,render_visual_confirmed_failures=['frame_000016','frame_000018','frame_000019','frame_000020','frame_000021','frame_000022','frame_000023'],
        render_visual_findings='Actual saved body contact pages 01-10 reviewed. Arm omission and disconnected hand in the listed frames; list is not an exhaustive pixel segmentation.',
        training_inputs_changed=False,threshold_tuned_to_outcomes=False,quality_accepted=False)
    out=RUN/'gaussian-run'/f'step-{step:06d}'/'quality-screen/result.json';a.save(out,result)
    print(json.dumps(dict(frames=59,flagged_count=result['flagged_count'],flagged_frames=result['flagged_frames'],output=str(out)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['plan','run']);parser.add_argument('--step',type=int,default=1000)
    args=parser.parse_args();plan() if args.mode=='plan' else main(args.step)
