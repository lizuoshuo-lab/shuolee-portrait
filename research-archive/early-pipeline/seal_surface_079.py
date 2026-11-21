"""Make current handoff agree with the completed079 and inspected UI."""
import publish_surface_079 as p
import datetime,shutil
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
c=p.read(p.R/'completion.json');assert c['workflow_complete'] and c['local_standalone_viewer_passed']
assert p.sha(p.Path(c['standard_ply']))==c['ply_sha256']
assert p.sha(p.Path(c['deck']))==c['deck_sha256']
plan=p.B/'long-term-plan.md';s=plan.read_text(encoding='utf8')
s=s.replace('当前运行中，没有质量改善结论。','已完成，质量未通过，最终结论见下一段。')
s=s.replace('当前077沿用075接入的175帧整图SfM与全局BA相机','当前079沿用075接入的175帧整图SfM与全局BA相机')
plan.write_text(s,encoding='utf8')
status=p.read(p.B/'work-status.json')
status['frozen_depth_audit_078']['next_preflight'].update(completed=True,
    full_bank_result=str(p.B/'current/static-depth-mixture-audit-078/full-bank/result.json'),
    numerical_preflight=str(p.B/'current/static-depth-mixture-audit-078/surface-cuda-preflight.json'),
    superseded_by='079 completed evaluation; do not rerun078 or079')
status['next_bounded_work']=dict(type='diagnostic before any fresh training',
    scope=c['next_step'],latest_visual_evidence=str(p.R/'review/visual-decision.json'),
    data='existing full original image feature observations; no new download',
    same_static_goal=True,new_training_authorized_by_gate=False,
    prior_completed_rounds=['074 whole-image SfM','075 full pipeline','076 support audit','077 correspondence training','078 full-bank/depth/preflight','079 surface comparison'],
    do_not_repeat='No additional surface-loss weight or step sweep. Do not repeat whole-scene SfM just because the current camera accuracy remains unaccepted.',
    hypotheses=['global camera residual affects geometrically stable tracks across image','local face observation inconsistency from tracking/reflection or true micro-motion'],
    intended_evidence='Same observations and held-out temporal blocks, with scene/person grouped reporting; labels do not automatically exclude features',
    gaussian_optimizer_updates=0,final_goal_complete=False)
status['updated']=now;p.save(p.B/'work-status.json',status)
review=p.R/'review/final-ui-verification.json'
p.save(review,dict(updated=now,viewer_url='http://127.0.0.1:8870/?model=latest',round_shown='079',
    points_shown=305174,follow_latest_checked=True,quality_warning_visible=True,
    page_url=c['report_url'],four_new_page_images_loaded=True,download_HEAD_status=200,
    screenshots=[str(p.R/'review/viewer-final.png'),str(p.R/'review/page-final.png')],
    pc00_deployed=False,deck_sha256=c['deck_sha256']))
c.update(final_ui_verified=True,final_ui_receipt=str(review),goal_turn_classification='progress',
    repeated_blocking_condition_count=0,goal_complete=False);p.save(p.R/'completion.json',c)
shutil.copy2(p.W/'seal_surface_079.py',p.R/'source-final/seal_surface_079.py')
print('079 delivery, website, canonical report and current handoff verified')
