from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil
L=Path(__file__).resolve().parents[2]; O=L/'outputs/dynamic-human-enerf-contour-surface-103'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read(O/'comparison/summary.json'); t=read(O/'temporal/summary.json'); v=read(O/'visual-review.json')
assert c['PNG_checks']==1072 and t['PNG_checks']==104 and v['clarity_accepted'] is False
assert v['native_hand_contexts']==21
for r in v['files']: assert sha(O/r['file'])==r['sha256']
branches=['prior_control','contour_prior']
for b in branches:
    assert len(read(O/b/'native-evaluation/metrics.json')['records'])==134
    assert sha(O/b/'resume.pt')==c['checkpoints'][b]
    assert read(O/b/'frozen-state-audit.json')['all_optimizer_steps']==6000
vis=read(O/'visibility/summary.json'); assert len(vis['records'])==8
summary=dict(status='paired_contour_prior_RGB_control_complete_not_adopted',
    training={b:read(O/b/'completion.json') for b in branches},
    comparison={k:x for k,x in c.items() if k!='panels'},
    temporal={k:x for k,x in t.items() if k!='panels'},
    contribution=read(O/'contribution/summary.json')['aggregate'],
    visibility_counterfactuals=vis,
    prior_qualification='102 contour constraints are weak SAM-derived support, not finger geometry ground truth. Four mask diagnostic images were excluded only from contour loss; their keypoints were used.',
    controlled_difference='Fixed102 keypoint-only versus contour-qualified priors. Both retain the same 0.02 ceiling on6330 old context opacities,1538 new surface Gaussians,105 photos,6000 updates and schedule.',
    historical_comparison='100 context_reset differs in persistent opacity ceiling and pose optimization; its comparison is not a pure prior ablation.',
    visual_review_complete=True,visual_review='visual-review.json',adopted=False,clarity_accepted=False,goal_complete=False,
    conclusion='Contour coverage gains did not translate into clear fingers. The two native RGB results remain patchy; suppressing old context opacity also damages hand/toy context. New-only diagnostic renders remain inaccurate, so removing old geometry at display time is not a solution.',
    next_action='Qualify visible finger correspondences and per-view surface reprojection against native training observations before another RGB fit. Preserve the079 baseline. Do not continue opacity-only controls or treat keypoint/contour proxy scores as clarity. Independent scene and full-background/OmniRe validation remain required.',
    external_input_needed=False,licensed_models_exported=False)
save(O/'final-summary.json',summary)
p=L/'manifests/dynamic-human-work.json'; m=read(p)
m.update(status='103_paired_contour_RGB_evaluation_complete_not_adopted',current_runs=[],external_input_blocker=None,
    goal_turn_classification='progress',goal_blocked_consecutive_turns=0,
    last_recorded_utc=datetime.now(timezone.utc).isoformat(),next_action=summary['next_action'])
m['outdoor_contour_surface']=dict(stage='103',summary='outputs/'+O.name+'/final-summary.json',status=summary['status'],adopted=False,clarity_accepted=False)
line='103: two6000step native RGB controls and268 full-resolution evaluations completed;1072PNG checks,26continuous contexts,24consecutive transitions,21hand contexts reviewed; contour prior gains do not restore fingers. Both rejected.'
if line not in m['completed']:m['completed'].append(line)
save(p,m); shutil.copy2(__file__,O/Path(__file__).name)
print(json.dumps({s:c['splits'][s]['foreground_psnr_db'] for s in ['validation_time','validation_camera']}))
