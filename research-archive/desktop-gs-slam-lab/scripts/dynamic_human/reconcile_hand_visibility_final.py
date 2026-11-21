from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
L=Path(__file__).resolve().parents[2];P=L/'outputs/dynamic-human-enerf-hand-visibility-100';O=L/'outputs/dynamic-human-hand-body-alignment-101'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
qa=read(P/'website-qa.json');proof=read(P/'website-visible-proof.json');grad=read(O/'gradient-scale/summary.json');assert len(qa['resources'])==6 and grad['source_checkpoint_unchanged']
interpretation=dict(status='read_only_gradient_assay_complete_no_new_training',source='gradient-scale/summary.json',contexts=9,
    conclusion='Hand-context normalization raises gradient norms about29x at the checked checkpoint. The regularizer is smaller than the full-person photometric gradient: mediantranslation ratio0.0316, poseratio0.0107. This doesnot support a regularizer-dominance explanation.',
    optimizer_caveat='Adam approximately cancels uniform gradient scaling through its moments. A larger gradient norm alone doesnot establish a better optimization direction or a clarity gain. The localcrop includesforearm/toy andchangesSSIMboundaryhandling; it isnot a truehandmask.',
    next='Qualify spatial hand supervision andpose/depth against actualtraining photos, then make a controlled old/new competition trial. Do not claim fixing blur by just changing learningrate, lossweight, opacityinitialization, or model-wristalignment.',
    goal_complete=False,external_input_needed=False)
save(O/'gradient-scale/interpretation.json',interpretation)
p=L/'manifests/dynamic-human-work.json';s=read(p);s.update(status='100paired_training_and101alignment_occlusion_gradient_audits_complete_no_clarity_acceptance',current_runs=[],goal_turn_classification='meaningful_progress_not_blocked',updated_at=datetime.now().astimezone().isoformat(),last_recorded_utc=datetime.now(timezone.utc).isoformat(),next_action=interpretation['next'],external_input_blocker=None)
s['outdoor_hand_body_alignment']['gradient_audit']='outputs/'+O.name+'/gradient-scale/summary.json';s['outdoor_hand_body_alignment']['gradient_interpretation']='outputs/'+O.name+'/gradient-scale/interpretation.json'
s['outdoor_hand_visibility'].update(website_qa='outputs/'+P.name+'/website-qa.json',visible_proof='outputs/'+P.name+'/website-visible-proof.json',preview_pid=int((P/'preview.pid').read_text(encoding='utf-8-sig').strip()))
if 'goal_blocked_consecutive_turns' in s:s['goal_blocked_consecutive_turns']=0
line='101additional:9trainingcontexts gradientassay complete;handcontext gradients~29xfullpersonnorm,regularizer3.16percenttranslation/1.07percentpose;scalealoneisnotAdamimprovementproof.'
if line not in s['completed']:s['completed'].append(line)
save(p,s)
save(P/'turn-completion.json',dict(status='training_evaluation_diagnostics_website_build_and_visible_check_complete',training_runs=2,steps_each=6000,native_evaluation_images=268,visual_hand_contexts=21,source_photos_unchanged=True,baseline_adopted='079',new_candidates_adopted=False,website_qa=qa['status'],visible_screenshot='website-visible.png',preview_pid=s['outdoor_hand_visibility']['preview_pid'],additional_diagnostics='outputs/'+O.name+'/diagnostic-summary.json',gradient_audit='outputs/'+O.name+'/gradient-scale/summary.json',goal_complete=False))
print('All owned jobs complete; website verified; baseline079 retained; goal active; no required user input.')
