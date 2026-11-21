"""CPU counterfactual over frozen saved candidates; never writes a fitted pose."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import audit_gaussianavatars_params123 as a
from audit_smplifyx_temporal_result123 import rotations, distances
from pose_candidate_gate import choose_candidate

BODY = a.REC / 'demo-vhap-initialization/fullbody-smplifyx'
RUN = BODY / 'warmtail-guarded-sequence'
BASE = BODY / 'full-sequence'
PLAN = RUN / 'review/region-gate-plan.json'
RESULT = RUN / 'review/region-gate-audit.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    assert not a.lp(PLAN).exists() and not a.lp(RESULT).exists()
    closeout = a.read(RUN / 'review/closeout.json')
    assert closeout['frames'] == 59 and not closeout['quality_accepted']
    original = BODY / 'warmtail-guarded-sequence-plan.json'
    assert a.sha(original) == '64b7d85c2881f3f5d4475eb03f57ed32b80c642b28b3e2aa34580c2061126e62'
    sources = [Path(__file__), a.H/'pose_candidate_gate.py', a.H/'audit_smplifyx_temporal_result123.py']
    inputs = [original, RUN/'review/closeout.json', RUN/'fit/fitted.npz']
    roster = a.read(original)['temporal_roster']
    for frame in roster:
        inputs.extend([RUN/'review/frames'/(frame+'.json'),
                       RUN/'frames'/frame/'objective-selection.json',
                       RUN/'frames'/frame/'winner-camera-space.npz',
                       BASE/'frames'/frame/'winner-camera-space.npz'])
    plan = dict(updated=now(), hypothesis='The scalar objective permits regional regressions. Reuse the existing body/feet/face limits as a uniform saved-candidate guard and inspect all cross-boundary SO3 changes.',
        scope='Post-hoc diagnostic, not an independently validated selector or an online sequential replay.',
        roster=roster, limits_px=dict(body17=1., feet6=1., face51=.5),
        rule='Use the current original-objective winner only if it strictly improves the original objective and every registered region is nonregressing within its existing limit; otherwise retain the same-frame independent baseline. Ties retain baseline.',
        temporal_rule='No added threshold or smoothing. Report all58 pairs, especially pairs changed by fallback; inspect whether a new jump exceeds both unmodified input sequences on that pair.',
        seed_caveat='Candidate poses retain their actual historical warmstart seeds; do not present offline selection as a new chronological fit.',
        budget=dict(cpu_seconds=120, gpu=False, fits=0, threshold_sweeps=0),
        source_hashes={str(p):a.sha(p) for p in sources}, input_hashes={str(p):a.sha(p) for p in inputs},
        outputs=['read-only all-frame decision/continuity audit'],
        no_fit_write=True, no_frame_removal=True, no_training=True, no_model_promotion=True,
        quality_accepted=False, goal_complete=False)
    a.save(PLAN,plan)
    print('PREPARED_REGION_GATE_CPU', a.sha(PLAN))


def load(path):
    with np.load(a.lp(path),allow_pickle=False) as z:
        return {key:z[key].copy() for key in z.files}


def motion(previous, current):
    joints = distances(rotations(current['body_pose_axis_angle']), rotations(previous['body_pose_axis_angle']))
    return dict(root_deg=float(distances(current['root_rotation'],previous['root_rotation'])[0]),
        joint_deg=joints.tolist(), local_joint_max_deg=float(joints.max()),
        largest_joint_index=int(joints.argmax()), left_elbow_deg=float(joints[17]), right_elbow_deg=float(joints[18]))


def audit():
    assert not a.lp(RESULT).exists()
    started=time.monotonic();plan=a.read(PLAN)
    for path,digest in {**plan['source_hashes'],**plan['input_hashes']}.items():
        assert a.sha(path)==digest,path
    rows=[];poses={};previous=None;pairs=[]
    for frame in plan['roster']:
        assert time.monotonic()-started < plan['budget']['cpu_seconds']
        meta=a.read(RUN/'review/frames'/(frame+'.json'))
        original=a.read(RUN/'frames'/frame/'objective-selection.json')
        old=load(BASE/'frames'/frame/'winner-camera-space.npz')
        fitted=load(RUN/'frames'/frame/'winner-camera-space.npz')
        assert a.sha(RUN/'frames'/frame/'winner-camera-space.npz')==meta['candidate_sha256']
        for key in ('faces','betas','K_integer_centers','w2c','camera_translation'):
            assert np.array_equal(old[key],fitted[key]),(frame,key)
        for before,after in zip(meta['baseline']['per_joint'],meta['candidate']['per_joint']):
            for key in ('index','active','confidence'):
                assert before[key]==after[key],(frame,key)
        scores=original['original_saved_objectives']
        # When the original selector already kept baseline, its saved projection
        # is not candidate0. Compare baseline to itself and retain the tie.
        selected_score=scores[original['selected_index']]
        decision=choose_candidate([selected_score,scores[1]],meta['baseline']['groups'],meta['candidate']['groups'],plan['limits_px'])
        keep=decision['selected_index']==0
        after=meta['candidate'] if keep else meta['baseline']
        changed=original['selected_index']==0 and not keep
        rows.append(dict(frame=frame,original_selected_index=original['selected_index'],**decision,
            changed_from_saved_winner=changed, metrics=after, original_metrics=meta['candidate']['groups'],
            before_groups=meta['baseline']['groups'],before_elbow=meta['baseline']['left_elbow_px'],
            after_elbow=after['left_elbow_px'], original_saved_objectives=scores,
            selected_path=str((RUN if keep else BASE)/'frames'/frame/'winner-camera-space.npz')))
        poses[frame]=dict(baseline=old,original=fitted,guarded=fitted if keep else old)
        if previous:
            pair=dict(previous=previous,frame=frame)
            for label in ('baseline','original','guarded'):
                pair[label]=motion(poses[previous][label],poses[frame][label])
            pair['affected_by_fallback']=changed or rows[-2]['changed_from_saved_winner']
            # Not an acceptance threshold: locate cross-candidate seams.
            pair['new_excesses']={key:pair['guarded'][key]-max(pair['baseline'][key],pair['original'][key])
                for key in ('root_deg','local_joint_max_deg','left_elbow_deg','right_elbow_deg')
                if pair['guarded'][key]>max(pair['baseline'][key],pair['original'][key])+1e-8}
            pairs.append(pair)
        previous=frame
    summary={label:{key:dict(median=float(np.median([p[label][key] for p in pairs])),maximum=max(p[label][key] for p in pairs))
        for key in ('root_deg','local_joint_max_deg','left_elbow_deg','right_elbow_deg')}
        for label in ('baseline','original','guarded')}
    targets=[r for r in rows if 18<=int(r['frame'].split('_')[-1])<=23]
    result=dict(updated=now(),state='read_only_counterfactual_complete',frames=len(rows),adjacent_pairs=len(pairs),
        source_plan_sha256=a.sha(PLAN),changed_frames=[r['frame'] for r in rows if r['changed_from_saved_winner']],
        kept_new_count=sum(r['selected_index']==0 for r in rows),retained_baseline_count=sum(r['selected_index']==1 for r in rows),
        target_elbows=[dict(frame=r['frame'],before=r['before_elbow'],after=r['after_elbow']) for r in targets],
        target_elbows_halved=all(r['after_elbow']<=.5*r['before_elbow'] for r in targets),
        regional_guard_passes_by_construction=True,not_an_independent_acceptance=True,
        changed_pair_count=sum(p['affected_by_fallback'] for p in pairs),
        new_seams=[p for p in pairs if p['new_excesses']],continuity=summary,rows=rows,pairs=pairs,
        elapsed_cpu_seconds=time.monotonic()-started,candidate_poses_modified=False,
        chronological_seed_policy_replayed=False,actual_visual_review=False,
        quality_accepted=False,goal_complete=False,model_promoted=False,Gaussian_training_started=False)
    for path,digest in plan['input_hashes'].items():assert a.sha(path)==digest,path
    a.save(RESULT,result)
    print(json.dumps({k:result[k] for k in ('state','frames','changed_frames','kept_new_count','retained_baseline_count','target_elbows_halved','changed_pair_count','continuity','elapsed_cpu_seconds')}))
    print('NEW_SEAMS',json.dumps([{k:p[k] for k in ('previous','frame','new_excesses')} for p in result['new_seams']]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','audit']);args=parser.parse_args()
    (prepare if args.mode=='prepare' else audit)()
