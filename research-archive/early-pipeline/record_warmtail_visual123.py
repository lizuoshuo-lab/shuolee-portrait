"""Record human-reviewed native images; never infer a visual pass from metrics."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import audit_gaussianavatars_params123 as a


def main():
    run=a.REC/'demo-vhap-initialization/fullbody-smplifyx/warmtail-guarded-sequence'
    notes=a.read(a.H/'warmtail-visual-notes.json')
    review=run/'review';rows=[]
    for n,note in sorted(notes.items(),key=lambda x:int(x[0])):
        assert 1<=int(n)<=59 and isinstance(note,str) and note.strip()
        frame=f'frame_{int(n):06d}'
        record=a.read(review/'frames'/f'{frame}.json')
        fit=a.read(run/'frames'/frame/'result.json')
        assert record['frame']==frame and record['candidate_sha256']==a.sha(run/'frames'/frame/'winner-camera-space.npz')
        regions=['body','face','left_hand','right_hand','feet']
        assert [x['region'] for x in record['files']]==regions
        for f in record['files']:
            for key in ['path','record_path']:
                p=f[key].replace('/mnt/c/','C:/').replace('/mnt/d/','D:/')
                assert a.sha(p)==f['sha256'],(frame,key)
        rows.append(dict(frame=frame,regions_seen=regions,images=record['files'],findings_zh=note,
            candidate_sha256=record['candidate_sha256'],fit_receipt_sha256=record['fit_receipt_sha256']))
    now=datetime.now(timezone.utc).isoformat()
    result=dict(updated=now,state='partial_actual_review' if len(rows)<59 else 'all59_actual_review_pending_closeout',
        actual_visual_review=True,reviewed_frame_count=len(rows),reviewed_image_groups=len(rows)*5,total=59,
        frames=rows,full59_quality_passed=False,quality_accepted=False,goal_complete=False,
        no_text_on_images=True,no_resampling=True,continuous_video_reviewed=False,
        review_scope='Only listed images actually viewed; five native crop groups for each listed frame. No automated visual acceptance.',
        model_promoted=False,Gaussian_training_started=False)
    a.save(review/'visual-review.json',result)
    # Apply the preregistered protection rule to every generated frame, never remove a row.
    generated=[json.loads(p.read_text(encoding='utf-8-sig')) for p in sorted(a.lp(review/'frames').glob('frame_*.json'))]
    failures=[];targets=[]
    for row in generated:
        for group,bound in [('body17',1.0),('feet6',1.0),('face51',0.5)]:
            delta=row['candidate']['groups'][group]['median']-row['baseline']['groups'][group]['median']
            if delta>bound:failures.append(dict(frame=row['frame'],group=group,delta_px=delta,bound_px=bound))
        if 18<=int(row['frame'][-6:])<=23:
            before=row['baseline']['left_elbow_px'];after=row['candidate']['left_elbow_px']
            targets.append(dict(frame=row['frame'],before_px=before,after_px=after,halved=after<=before*.5))
    numerical=dict(updated=now,generated_frames=len(generated),total=59,
        all59_generated=len(generated)==59,protection_failures=failures,target_elbows=targets,
        all_six_target_elbows_halved=len(targets)==6 and all(t['halved'] for t in targets),
        quality_accepted=False,independent_physical_truth=False,selection_or_training_modified=False)
    a.save(review/'admission-progress.json',numerical)
    print(json.dumps(dict(actual_frames=len(rows),actual_images=len(rows)*5,generated=len(generated),
        protection_failures=failures,target_elbows=targets),ensure_ascii=False))


if __name__=='__main__':main()
