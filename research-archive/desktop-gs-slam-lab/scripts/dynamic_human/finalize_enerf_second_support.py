from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-support-correction-081'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(O/'comparison/summary.json');v=read(O/'render-visual-review.json');a=read(O/'visual-review.json');tr=read(O/'local_bspline/completion.json');details=read(O/'comparison/detail-metrics.json')['records']
assert tr['step']==15000 and v['actual_panels_reviewed']==36 and s['PNG_checks']==960
assert sum('both_hands_and_objects' in p['file'] or 'both_person_feet' in p['file'] for p in v['panels'])==24
for p in v['panels']:assert sha(O/'comparison'/p['file'])==p['sha256']
v.update(status='36actual_native_panels_reviewed_no_clear_hand_improvement',findings_zh=['机位00、01、04、09、12的部分墙面残片减弱；保守保护区域内的墙面与脚边地面仍可能保留。','机位04帧731张开的手掌仍有半透明重影；绿色玩偶和奶牛玩偶边缘仍破碎，未见稳定细节恢复。','机位06、07的绿色玩偶上方背景残片仍在；两支的手持玩偶局部均偏软。','机位04、12抬起的黑鞋底仍有双层边缘，鞋带没有恢复到原照片的清晰程度。','既有24个手物/鞋部窗口全部实际查看；另查看12张全身图，包含两组各三个最差变化的全部案例。'],clarity_accepted=False,adopt_as_development_baseline=False)
save(O/'render-visual-review.json',v)
summary=dict(status='matched_support_intervention_complete_not_adopted_for_human_clarity',training=tr,training_input_review=a,evaluation=s['splits'],temporal_residuals=s['temporal_residuals'],detail_windows={n:dict(images=sum(r['name']==n for r in details),mean_change_db=float(np.mean([r['change_db'] for r in details if r['name']==n])),improved=sum(r['name']==n and r['change_db']>0 for r in details)) for n in ['both_hands_and_objects','both_person_feet']},actual_panels_reviewed=36,identical_initialization_and_sampling=s['identical_initialization_and_sampling'],checkpoints=s['checkpoints'],clarity_accepted=False,adopt_as_development_baseline=False,interpretation_zh='保守前景修正减少部分墙面残片，但没有稳定改善人体细节。原机位留出时刻平均变化接近零，新机位与手物窗口略退化；保留旧模型作为开发基线。不能据此否定完整、正确的语义掩码，因为本次只修正9个机位且保留了不确定区域。',next_action='Prioritize native hand-crop observations and articulated/local hand motion with cross-view consistency. Stop broad support tuning as the main route to clarity. Keep original reference and paired native evaluation; qualify any learned hand annotations before training.',goal_complete=False,updated_at=datetime.now(timezone.utc).isoformat())
save(O/'final-summary.json',summary)
snap=O/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['segment_enerf_second_support.py','protect_enerf_second_support.py','approve_enerf_second_support.py','run_enerf_second_support.sh','train_enerf_second_interval.py','evaluate_enerf_second_interval.py','compare_enerf_second_support.py','finalize_enerf_second_support.py']:
 shutil.copy2(L/'scripts/dynamic_human'/n,snap/n)
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='training_support_intervention_complete_human_clarity_still_unaccepted',current_runs=[],outdoor_support_correction='outputs/'+O.name+'/final-summary.json',next_action=summary['next_action'],updated_at=summary['updated_at'],goal_turn_classification='Progress:105training candidates generated and actually visually reviewed;63conservative training inputs accepted,42preserved;15000matched training updates,120newnative renders,120paired metric recomputations,960PNGchecks,36actualresultpanels. Partial wall residual reduction but hand clarity not improved; candidate not adopted. No external blocker.')
save(p,m);print(summary['status'])
