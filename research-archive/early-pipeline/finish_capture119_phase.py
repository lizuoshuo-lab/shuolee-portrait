"""Close the two finite runs as reviewed research, never as quality acceptance."""
from pathlib import Path
import json, datetime
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/clothed-surface-119')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
subjects={}
for name in ['portrait','sofa']:
    q=read(REC/f'{name}-quality-review.json');d=read(REC/f'{name}-staged-delivery.json');ui=read(REC/f'{name}-viewer-ui-review.json')
    assert d['steps']==18000 and d['local_viewer_open_file_verified'] and ui['file_opened_through_UI']
    assert q['continuous_novel_views_reviewed']==36 and ui['source_sha256']==d['sha256']==q['source_sha256']
    subjects[name]=dict(steps=18000,points=d['points'],initial_points=200000,from_scratch=True,
        actual_ui_verified=True,compatibility_passed=True,quality_accepted=False,sha256=d['sha256'],
        viewer_url=d['viewer_url'],download_url=d['download_url'],summary=q['summary'])
value=dict(updated=datetime.datetime.now().astimezone().isoformat(),state='finite_runs_reviewed_quality_not_accepted',
    subjects=subjects,training_running=False,quality_accepted=False,
    new_diagnosis='Out-of-FOV negative voting during sofa initialization; CPU counterfactual recovers1046 voxels, actual final119 model unchanged.',
    next_step='Correct visibility accounting in a separately versioned initializer, inspect complete foot support before any new from-scratch run. Keep crop/K and motion uncertainty separate.',
    deferred={'grass':'existing fixed-camera all-frame fit fails','indoor118':'external furniture occlusion unresolved'},
    scope_completed='Two authorized finite119 clothing-representation experiments, export, same-camera36angle and actual UI review.',
    overall_project_complete=False,pc00_verified=False)
(REC/'phase-result.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('TWO_FINITE_RUNS_REVIEWED; OVERALL_QUALITY_NOT_ACCEPTED')
