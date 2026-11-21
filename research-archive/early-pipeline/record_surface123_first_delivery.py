from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';R=T/'shuolee/current/robust-portrait-122'
C=H.parent/'shuolee-coordination';ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(exist_ok=True,parents=True);x=p.with_suffix(p.suffix+'.tmp');x.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');x.replace(p)
now=datetime.now(timezone.utc).isoformat();delivery=read(REC/'winterhat-staged-delivery.json');quality=read(REC/'winterhat-quality-review.json')
assert delivery['local_UI_verified'] and not quality['quality_accepted']
prog=read(ROOT/'whiteT/progress.json');assert prog['step']>1000
decision=read(ROOT/'whiteT/reviews/gaussian-step-001000-decision.json');assert decision['continue_same_run']
ppt=read(H/'surface123-18000-deck-receipt.json');assert ppt['canonical_replaced']
ledger=read(REC/'evidence-ledger.json')
entry=dict(kind='first_completed_export_published_whiteT_training_continues',date=now,delivery=delivery,quality_accepted=False,whiteT_progress=prog,whiteT_actual_1000_review=str(REC/'whiteT-train-001000-review/actual-review.json'),same52page_report=dict(path=ppt['canonical_path'],sha256=ppt['final_sha256'],changed_slides=[52],actual_last_page_visual_review=True),latest_viewer_screenshot=dict(path=str(REC/'winterhat-published-latest-ui.png'),sha256=sha(REC/'winterhat-published-latest-ui.png')))
if not any(e.get('kind')==entry['kind'] for e in ledger['entries']):ledger['entries'].append(entry)
ledger['updated']=now;save(REC/'evidence-ledger.json',ledger)
parent=read(R/'queue-state.json');parent.update(updated=now,phase='LOCAL_SURFACE123_first_model_published_next_person_training',running=dict(pid=361,subject='whiteT',command='/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python H/local_surface123.py --run',progress=str(ROOT/'whiteT/progress.json')),next='Keep same single local worker. Review whiteT6000/12000/18k and publish after final native/package/UI checks, then actual fit/surface/1000 reviews for remaining5subjects. Do not restart firstperson or old experiments.',last_goal_turn=dict(classification='progress',current='Winterhat18k exported and all36 novel plus6same-camera views actually reviewed; local ZIP OpenFile Gaussian/points and112camera verified, promotedlatest withqualityfalse. Same52pagePPT last updated. whiteT freshfit/surface/1000 reviewed andsame training continues.',time=now),training_started=True,quality_accepted=False,goal_status='active');save(R/'queue-state.json',parent)
text='\n\n<!-- LOCAL_SURFACE123_FIRST_DELIVERY -->\n戴帽蓝衣新版已完成18000步，294054个高斯与112相机配套发布到latest。6组同相机与36幅连续环绕实际审阅，衣物表面点分布成立，眼鼻嘴/鞋底重影与缺失背面仍未通过。完整协方差、三阶SH、标准PLY原生回读、112相机K/位姿/身份一致性和实际OpenFile Gaussian/points完成。原52页报告仅末页更新18000步，前51页字节与渲染保持。白T恤人物93真实相机，从头拟合、20万表面点和1000步已审阅，严重面部错位留证，继续这次有限18k研究运行。目标保持active，不恢复定时任务，不宣称物理标定或强鲁棒性完成。\n'
longplan=T/'shuolee/long-term-plan.md';old=longplan.read_text(encoding='utf-8-sig')
if '<!-- LOCAL_SURFACE123_FIRST_DELIVERY -->' not in old:longplan.write_text(old+text,encoding='utf-8')
next_action=dict(updated=now,goal_mode='active_not_timer',current_worker_pid_wsl=361,publication_worker_pid_windows=29516,first_person='winterhat18k_completed_published_quality_failed',first_model=delivery,latest_actual_checkpoint_review='whiteT-train-001000-review/actual-review.json',continuation_decision_already_consumed=True,current_subject='whiteT',observed_progress=prog,next=['Verify realPID361/log/progress; no duplicate launch or modifying frozenruntime.','Review whiteT6000/12000 snapshots when available; finish samefinite18k; no promises of ghosting correction.','At whiteT18k inspect exportvalidation,6same-camera frames and36orderednovelviews, create qualityreceipt, stage andpackage, check fullcamera roundtrip, actualOpenFileGaussian/points, thenpromotequalityseparately.','Remaining5subjects sequential; actualfit,surface,1000steps review needed.','Finalmodelpairpackage required. Latest isalreadywinterhat123, keep sofa119history andtab15.','PPTlastpageupdated18k andcanonicalverified; no repeat untilsubstantivenewresult.','Publisher29516runningafternewgalleryschemafix, trainingunmodified. Goalactive, no scheduledautomation.'],sources_and_helpers=str(H),run_root='/opt/gs-slam-lab/shuolee-local-surface-123',right_panel_tab='31',progress_tab='40',quality_accepted=False,company00_used=False)
save(REC/'next-action.json',next_action)
shared=dict(updated_at=now,owner_thread_id='01a0e316-7534-7693-a60c-ded52094785d',published_model=delivery,quality=quality,local_training=dict(subject='whiteT',pid=361,step_seen=prog['step']),scope='Local research model delivery only. No company/server113 code or installation changes. No00UI or cloudsync claim.')
save(C/'local-surface123-delivery.json',shared)
for dest in [T/'viewer/local-surface-123',Path('D:/00/工作/查看器/local-surface-123'),Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')]:
    for name in ['winterhat-quality-review.json','winterhat-viewer-ui-review.json','winterhat-package-camera-parity.json','winterhat-package-check.json','winterhat-staged-delivery.json','evidence-ledger.json','next-action.json']:
        shutil.copy2(REC/name,dest/name)
print(json.dumps(dict(published=delivery['modelId'],quality_accepted=False,next_person='whiteT',step=prog['step'],PPT_sha=ppt['final_sha256']),ensure_ascii=False))
