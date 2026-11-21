from pathlib import Path
from datetime import datetime,timezone
import json
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
plan=read(O/'evaluation-protocol.json');assert len(plan['records'])==264
save(O/'review-plan/source-window-review.json',dict(
    inspected=['camera04-frame000613-source-windows.png','camera13-frame000613-source-windows.png'],
    observations=['Both original native images show two people, clothes, shoes, hand-object contact and detailed source texture.','Standing-hand window includes the held plush and visible hands. Crouching-hand window contains hands reaching toward the ground and toy, and the knee-resting hand.','The rectangular windows also include clothing, background or props; their errors are context metrics, not finger-only accuracy.','In camera04 the source-panel crop cuts a tiny part of the crouching actor at the right margin; evaluation and detail windows are independently stored in full native canvas coordinates.'],
    unreviewed='These two images do not establish semantic coverage for all72contexts; all windows remain frozen, and other context failures must be reported rather than removed.',
    candidate_outputs_seen=False,goal_complete=False))
p=L/'manifests/dynamic-human-work.json';m=read(p)
m.update(status='106_matched_training_live_native_evaluation_prepared',last_recorded_utc=datetime.now(timezone.utc).isoformat(),goal_turn_classification='progress',goal_blocked_consecutive_turns=0,external_input_blocker=None)
m['current_runs']=[dict(stage='106',session_id=59767,status='sequential_static_nominal_clock_training;see_completion_files_for_exact_state',logs=[f'outputs/actor23-independent-106-{b}-training.log' for b in ['static','dynamic_nominal','dynamic_clock']])]
m['outdoor_independent_actor23'].update(status=m['status'],reserved_appearance_RGB_opened=True,heldout_RGB_scope='582..594usedonlyfordisjointclockcalibration;600..624nowopenedforfrozennativeevaluationandreviewwindows,nevermodelinitializationortraining.',evaluation_protocol='outputs/'+O.name+'/evaluation-protocol.json',evaluation_clock_decision='outputs/'+O.name+'/evaluation-clock-decision.json')
m['next_action']='Finish all three fixed15000step runs, independently compare264native contexts, inspect fixed whole-person/detail windows and motion continuity. Keep full-background/OmniRe requirements open.'
save(p,m);print('Evaluation source preparation complete; source-window review and current access state recorded.')
