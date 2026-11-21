"""Verify actual saved outputs against each immutable candidate decision."""
from datetime import datetime, timezone
import json
import numpy as np
import audit_gaussianavatars_params123 as a


def main():
    body=a.REC/'demo-vhap-initialization/fullbody-smplifyx'
    run=body/'guarded-root-sequence'
    plan=a.read(body/'guarded-root-sequence-plan.json')
    rows=[]
    keys=('vertices_camera','joints_camera','joints_projected_native','pose_embedding',
          'global_orient','betas','camera_translation','expression','jaw_pose',
          'left_hand_pose','right_hand_pose','K_integer_centers','faces')
    for frame in plan['processing_order']:
        folder=run/'frames'/frame
        if not a.lp(folder/'result.json').exists(): break
        result=a.read(folder/'result.json')
        decision=a.read(folder/'objective-selection.json')
        assert result['state']=='complete' and decision['frame']==result['frame_id']==frame
        selected=decision['selected_index']; scores=decision['original_saved_objectives']
        assert selected==(0 if scores[0]<scores[1] else 1)
        expected=(folder/'candidate-00-temporal.npz') if selected==0 else (body/'full-sequence/frames'/frame/'winner-camera-space.npz')
        with np.load(a.lp(expected),allow_pickle=False) as chosen,np.load(a.lp(folder/'winner-camera-space.npz'),allow_pickle=False) as saved:
            for key in keys: assert np.array_equal(chosen[key],saved[key]),(frame,key)
        parameters=(folder/'candidate-00-official-result.pkl') if selected==0 else (body/'full-sequence/frames'/frame/'official-result.pkl')
        assert a.sha(parameters)==a.sha(folder/'official-result.pkl')
        assert result['winning_loss']==scores[selected] and scores[selected]<=scores[1]
        assert result['objective_selection']==decision
        rows.append(dict(frame=frame,selected_index=selected,scores=scores,
            selected_npz_sha256=a.sha(folder/'winner-camera-space.npz'),
            selected_pickle_sha256=a.sha(folder/'official-result.pkl'),all13_arrays_exact=True,
            fixed_original_objective_rule_verified=True))
    report=dict(updated=datetime.now(timezone.utc).isoformat(),state='all59_file_selection_verified' if len(rows)==59 else 'partial_actual_file_selection_verified',
        checked_frames=len(rows),total=59,rows=rows,candidate_fit_modified=False,
        original_independent_cache_explicit=True,quality_accepted=False,goal_complete=False,
        limits='Parameter identity and original-loss selection only; this is not anatomical, surface, or runtime acceptance.')
    a.save(run/'selection-audit.json',report)
    print(json.dumps(dict(checked_frames=len(rows),retained_independent=sum(row['selected_index']==1 for row in rows))))


if __name__=='__main__':main()
