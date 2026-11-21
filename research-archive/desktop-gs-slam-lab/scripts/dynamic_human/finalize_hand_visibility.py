from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-visibility-100'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read(O/'comparison/summary.json');t=read(O/'temporal/summary.json');d=read(O/'depth-order/summary.json');con=read(O/'contribution/summary.json');op=read(O/'opacity-effect.json')
assert c['PNG_checks']==1072 and t['PNG_checks']==104 and len(d['records'])==4
plan=read(O/'review-plan.json');reviewed=[r for r in plan['files'] if 'temporal' not in r['file']]
assert len(reviewed)==10
for p in [O/'temporal/camera04-comparison.png',O/'temporal/camera13-comparison.png',O/'native-articulated-hand-comparison.png']+sorted((O/'contribution').glob('*.png')):
    reviewed.append(dict(file=str(p.relative_to(O)),sha256=sha(p)))
save(O/'visual-review.json',dict(status='21handcontexts6sampledcontinuousframes3wholepersoncases4contributionpanels_inspected',files=reviewed,
    native_hand_contexts=21,continuous_frames_sampled=[dict(camera=c,frames=[721,727,731]) for c in ['04','13']],
    continuous_frames_numerically_verified=26,whole_person_cases=3,video_playback_claimed=False,
    observations=['Both6000update branches retain blended and fragmented fingers in open-palm images across views.','Grasping/toy boundary remains blurred; expanded opacity initialization does not restore finger separation.','New surface contribution remains sparse and weak in all4inspected contribution panels.','Sampled early/middle/late frames and3wholeperson views show no clear visible gain; not a framewise exhaustive visual audit.'],clarity_accepted=False))
delta={s:c['splits'][s]['foreground_psnr_db']['context_reset']-c['splits'][s]['foreground_psnr_db']['narrow_reset'] for s in ['validation_time','validation_camera']}
finding='两组实际渲染仍未恢复清楚的手指。扩大旧高斯的不透明度初始化范围后，旧表面在训练中又变得较不透明，新增表面的可见贡献仍很小。在四个固定遮挡情境中，新表面的期望深度都位于旧表面之后；这只是模型之间的深度关系，不能证明哪一层符合真实人体。**下一步需要同时核对三维位置、人体连接和遮挡来源，继续改变不透明度本身不足以解决问题。**'
paragraph=f"1937点初始化与6330点初始化两组，在90张已知机位留出时刻的人物区域平均PSNR分别为{c['splits']['validation_time']['foreground_psnr_db']['narrow_reset']:.4f}和{c['splits']['validation_time']['foreground_psnr_db']['context_reset']:.4f} dB；18张留出机位图像分别为{c['splits']['validation_camera']['foreground_psnr_db']['narrow_reset']:.4f}和{c['splits']['validation_camera']['foreground_psnr_db']['context_reset']:.4f} dB。差异很小，21个手部窗口、6张连续序列抽查和3张全身图未显示清晰手指恢复。扩大初始化范围后，额外4393点的平均不透明度从初始不高于0.02重新增长至0.3466；新增高斯的平均不透明度则从0.2下降至0.0713。参数变化不能直接等同于像素贡献：实际合成的新增可见权重在21个手部上下文中仍很弱，这些窗口包含前臂和玩偶，不是语义手部真值。四个固定遮挡情境的模型期望深度差中位数为0.059至0.066场景单位，新表面位于旧表面之后；这里按可见权重平均高斯中心深度，并非真实第一表面深度。两组均不采纳为清晰重建结果。"
c['status']='134contexts_4models_1072PNGs_independently_verified_visual_review_completed'
save(O/'final-summary.json',dict(status='paired_opacity_initialization_control_complete_not_adopted',training={b:read(O/b/'completion.json') for b in ['narrow_reset','context_reset']},comparison={k:v for k,v in c.items() if k!='panels'},contribution=con['aggregate'],opacity_effect=op,depth_order=d,pair_delta_db=delta,temporal={k:v for k,v in t.items() if k!='panels'},visual_review_complete=True,visual_review='visual-review.json',knowledge_visual_finding=finding,evidence_result_paragraph=paragraph,adopted=False,clarity_accepted=False,goal_complete=False,next_action='Read-only body/MANO alignment and native per-ray occluder identity audit; no more opacity-only trials before checking which surface and parameters cause the mismatch. Preserve original source images, camera/time model and baseline079.',external_input_needed=False,licensed_models_exported=False))
shutil.copy2(__file__,O/Path(__file__).name)
p=L/'manifests/dynamic-human-work.json';s=read(p);s.update(status='100_paired_opacity_control_complete_clarity_not_accepted_101_geometry_audit',current_runs=[],external_input_blocker=None,goal_turn_classification='meaningful_progress_not_blocked',last_recorded_utc=datetime.now(timezone.utc).isoformat(),updated_at=datetime.now().astimezone().isoformat(),next_action=read(O/'final-summary.json')['next_action'])
s['outdoor_hand_visibility']=dict(stage='100',summary='outputs/'+O.name+'/final-summary.json',status='training_evaluation_and_visual_review_complete',adopted=False,clarity_accepted=False)
s['outdoor_hand_body_alignment']=dict(stage='101',summary='outputs/dynamic-human-hand-body-alignment-101/summary.json',status='105training_contexts_read_only_alignment_complete_no_refit',licensed_geometry_kept_local=True)
line='100: matched6000update narrow1937/context6330opacityinitializations,268native evaluations and26continuouscontexts complete;21handwindows retainblur; depth-order audit and105contextbody/MANO alignment measured without refitting.'
if line not in s['completed']:s['completed'].append(line)
if 'goal_blocked_consecutive_turns' in s:s['goal_blocked_consecutive_turns']=0
save(p,s);print(json.dumps(dict(pair_delta_db=delta,visual_review_complete=True,clarity_accepted=False)))
