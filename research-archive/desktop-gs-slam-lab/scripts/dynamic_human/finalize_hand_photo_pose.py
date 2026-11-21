from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-photo-pose-105'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read(O/'comparison/summary.json');t=read(O/'temporal/summary.json');v=read(O/'visual-review.json');run=read(O/'completion.json')
assert c['PNG_checks']==804 and t['PNG_checks']==78 and v['native_hand_contexts']==21
assert v['clarity_accepted'] is False and run['updates']==60
assert len(read(O/'native-evaluation/metrics.json')['records'])==134
assert sha(O/'resume.pt')==run['checkpoint_sha256']==c['checkpoints']['photo_pose']
for r in v['files']:assert sha(O/r['file'])==r['sha256']
summary=dict(status='bounded_pose_only_fit_evaluation_complete_not_adopted',training=run,comparison={k:x for k,x in c.items() if k!='panels'},temporal={k:x for k,x in t.items() if k!='panels'},visual_review_complete=True,visual_review='visual-review.json',adopted=False,clarity_accepted=False,goal_complete=False,
 conclusion='The average training objective decreased by about0.18percent. Native fingers remain mixed, broken and patchy. Pose-only refinement did not repair the fixed appearance/surface model. The original079baseline remains preferred to103and105.',
 interpretation='This bounds the usefulness of this pose-only local correction at this starting point; it does not establish that pose is accurate or that other joint optimization cannot work. All Gaussian attributes, old motion, camera/time calibration and source images remained fixed. Repeated development on this interval is not independent testing.',
 next_action='Proceed with actor2_3 independent-scene training-only geometry and timing qualification using the reserved106split, then freeze a static/dynamic comparison. Preserve full-person, detail and temporal evaluation. Do not continue opacity-only or repeated pose-only trials without new evidence. Full-background joint reconstruction and OmniRe remain unfinished.',
 external_input_needed=False,licensed_models_exported=False)
save(O/'final-summary.json',summary)
p=L/'manifests/dynamic-human-work.json';m=read(p)
m.update(status='105_pose_only_evaluation_not_adopted_106_independent_scene_reserved',current_runs=[],external_input_blocker=None,goal_turn_classification='progress',goal_blocked_consecutive_turns=0,last_recorded_utc=datetime.now(timezone.utc).isoformat(),next_action=summary['next_action'])
m['outdoor_hand_photo_pose']=dict(stage='105',summary='outputs/'+O.name+'/final-summary.json',status=summary['status'],adopted=False,clarity_accepted=False)
assert len(read(L/'outputs/dynamic-human-actor23-independent-106/training-input-audit.json')['records'])==195
m['outdoor_independent_actor23']=dict(stage='106',status='195_native_training_inputs_prepared_validation_reserved_geometry_and_time_unqualified',inventory='outputs/dynamic-human-actor23-independent-106/source-inventory.json',protocol='outputs/dynamic-human-actor23-independent-106/validation-reservation.json',training_inputs='outputs/dynamic-human-actor23-independent-106/training-input-audit.json',heldout_RGB_opened=False,training_started=False)
line='105:60fullbatch pose-only updates,134native images,804PNG checks,26continuous contexts and21hand contexts reviewed; no clarity acceptance.106:actor2_3 source/calibration inventory, independent split and195native training inputs prepared; no heldout RGB viewed and no training yet.'
if line not in m['completed']:m['completed'].append(line)
save(p,m);shutil.copy2(__file__,O/Path(__file__).name)
print(json.dumps(dict(status=summary['status'],objective_relative_drop=(run['initial_objective']-run['final_objective'])/run['initial_objective'],splits={s:c['splits'][s]['foreground_psnr_db'] for s in ['validation_time','validation_camera']})))
