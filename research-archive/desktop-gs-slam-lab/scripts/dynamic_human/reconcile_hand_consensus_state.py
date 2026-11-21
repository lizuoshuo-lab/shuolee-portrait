from pathlib import Path
import json,datetime,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-consensus-097';M=L/'manifests/dynamic-human-work.json'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
for f in ['final-summary.json','visual-review.json','geometry-audit.json','pixel-audit.json','website-qa.json','website-visible-proof.json']:assert (O/f).exists(),f
assert (O/'website-visible.png').exists()
shutil.copy2(M,O/'work-manifest-before.json');m=read(M);rel='outputs/dynamic-human-hand-consensus-097/'
m['status']='097_continuous_MANO_consensus_fitted_native_qualification_complete_098_soft_photometric_test_next'
m['completed'].append('097: two1200update continuous shared MANO fits (105all/84train+21excluded),66native original-photo panels132meshcomposites checked;22sheets viewed; CPU/projection audits passed. Joint-prediction residuals improve in median but tails/visible outlines/opening timing still fail. NoGSparameterschanged.')
m['current_runs']=[]
m['next_actions']=['保留079高斯开发对照；097完成连续共享MANO先验拟合，但不能作为精确手部几何。','098先核对手部高斯与MANO表面的对应及不确定区域，再尝试有界的软表面／运动约束。不得整块替换含玩偶、前臂的6330点上下文。','使用原始RGB，比较实际高斯渲染的指间分离、留出机位和奇数时刻；蓝色网格完整不作为清晰度验收。','随后继续独立场景及静态背景联合重建。']
m['next_action']='Audit hand Gaussian association to097shared MANO, then bounded soft photometric surface/motion control versus079. Preserve non-hand/toy contributions and frozen camera/estimated-time model; accept only actual native heldout/odd-time GS rendering improvement.'
m['external_input_blocker']=None;m['goal_turn_classification']='meaningful_progress_not_blocked'
m['goal_blocked_audit']={'consecutive_blocked_goal_turns':0,'classification':'meaningful_progress','reason':'Twoactual continuous MANO fits,66native visual contexts andwebsite update complete. No externalinput blocker; photometric coupling pending.'}
m['updated_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
m['latest_verified_site']=rel+'website-qa.json';m['latest_visible_result']=rel+'website-visible.png';m['website_preview_pid']=int((O/'website-preview.pid').read_text(encoding='utf-8-sig').strip())
m['outdoor_hand_consensus']={'stage':'097','status':'continuous_articulated_candidates_complete_native_source_qualification_partial','summary':rel+'final-summary.json','geometry_audit':rel+'geometry-audit.json','pixel_audit':rel+'pixel-audit.json','visual_review':rel+'visual-review.json','next_contract':rel+'next-stage-contract.json','model_input_needed':False,'Gaussian_parameters_changed':False,'clarity_accepted':False,'website_qa':rel+'website-qa.json','visible_proof':rel+'website-visible-proof.json'}
save(M,m);shutil.copy2(__file__,O/Path(__file__).name);print('Manifest reconciled:097complete,098notstarted,goalactive,noexternalinputneeded.')
