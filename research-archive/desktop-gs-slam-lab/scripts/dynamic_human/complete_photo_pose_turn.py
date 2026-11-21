from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-photo-pose-105'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(O/'final-summary.json')['adopted'] is False
assert len(read(O/'website-qa.json')['resources'])==3
assert len(read(L/'outputs/dynamic-human-actor23-independent-106/training-input-audit.json')['records'])==195
proof=dict(status='visible_complete_figure_caption_and_rejection_conclusion_verified',url='http://localhost:8799/gs-lab/read/dynamic-human-reconstruction-benchmark#固定外观后的手部姿态修正',screenshot='website-visible.png',sha256=sha(O/'website-visible.png'),public_native_figure_visually_inspected=True,recorded_utc=datetime.now(timezone.utc).isoformat(),licensed_models_exported=False,goal_complete=False)
save(O/'website-visible-proof.json',proof)
p=L/'manifests/dynamic-human-work.json';m=read(p)
m['website_preview_pid']=read(O/'preview-process.json')['ProcessId']
m['latest_verified_site']='outputs/'+O.name+'/website-qa.json'
m['latest_visible_result']='outputs/'+O.name+'/website-visible.png'
m['outdoor_hand_photo_pose'].update(website_qa=m['latest_verified_site'],visible_proof='outputs/'+O.name+'/website-visible-proof.json',preview_pid=m['website_preview_pid'])
m['current_runs']=[];m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(p,m)
save(O/'turn-completion.json',dict(status='progress_105_closed_not_adopted_106_native_inputs_prepared',goal_status='active',goal_complete=False,trained_updates=60,native_evaluation_contexts=134,hand_contexts_visually_reviewed=21,temporal_sampled_contexts_visually_reviewed=6,wholeperson_contexts_visually_reviewed=3,PNG_checks=804,temporal_PNG_checks=78,independent_scene_training_inputs=195,independent_scene_training_started=False,independent_scene_heldout_RGB_opened=False,website_export_sync_build='passed_after_releasing_owned_preview_directory_lock',website_http_qa='2pages3assets_formula_and_anchor_checks_passed',website_visible_proof=proof,preview_pid=m['website_preview_pid'],next_action=m['next_action'],external_input_needed=False))
print('105review/website reconciled;106native preprocessing saved; goal remains active.')
