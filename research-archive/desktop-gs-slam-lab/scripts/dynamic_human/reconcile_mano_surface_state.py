from pathlib import Path
from datetime import datetime,timezone
import json
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-mano-surface-099';Q=L/'outputs/dynamic-human-enerf-hand-prior-control-098';read=lambda p:json.loads(p.read_text(encoding='utf8'))
assert read(O/'final-summary.json')['visual_review_complete'];p=L/'manifests/dynamic-human-work.json';s=read(p)
s.update(status='099_paired_MANO_surface_and098_soft_prior_completed_clarity_not_accepted',current_runs=[],external_input_blocker=None,goal_turn_classification='meaningful_progress_not_blocked',last_recorded_utc=datetime.now(timezone.utc).isoformat(),updated_at=datetime.now().astimezone().isoformat(),next_action=read(O/'final-summary.json')['next_action'])
s['outdoor_hand_soft_prior']=dict(stage='098',summary='outputs/'+Q.name+'/final-summary.json',adopted=False,clarity_accepted=False)
s['outdoor_hand_surface']=dict(stage='099',summary='outputs/'+O.name+'/final-summary.json',visibility='outputs/'+O.name+'/visibility/summary.json',adopted=False,clarity_accepted=False,templates_needed=False)
for line in ['098: 3000update soft MANO prior and134native contexts complete; lower candidate-relative distance does not restore clear fingers.','099: matched6000update fixed/refined MANO surface models with1538newGaussians completed;268native evaluations,26continuous contexts and8visibility counterfactuals complete; new surface mostly occluded and no clarity acceptance.']:
 if line not in s['completed']:s['completed'].append(line)
if 'goal_blocked_consecutive_turns' in s:s['goal_blocked_consecutive_turns']=0
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('Manifest reconciled; goal active, baseline079 retained, no external input needed.')
