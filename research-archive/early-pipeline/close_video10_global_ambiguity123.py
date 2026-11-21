from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';G=R/'video10-global-initializer';REC=R.parent/'local-clothed-surface-123';C=H.parent/'shuolee-coordination'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();rec=read(H/'video10-global-ambiguity123-deck-receipt.json')
assert rec['canonical_replaced'] and rec['actual_last_page_visual_review'] and len(rec['renders'])==52
for name in ['canonical_path','mirror_path']:assert sha(Path(rec[name]))==rec['final_sha256']
viewer=(G/'browser-viewer.txt').read_text(encoding='utf-8');report=(G/'browser-report.txt').read_text(encoding='utf-8');gallery=(G/'browser-gallery.txt').read_text(encoding='utf-8')
assert all(t in viewer for t in ['最新诊断 · 123 · 相机多解与人物对齐','309,523','93 个相机','质量尚未通过验收','smplx123-video10-global-ambiguity'])
assert all(t in gallery for t in ['0.237 / 1.223','1.358 / 86.079','共享拟合点少于12','两个中心都在空调'])
assert '相机多解：5个点也能拟合出3组不同姿态' in report
ui=dict(updated=stamp,report_tab='32',viewer_tab='25',actual_report_DOM=True,actual_gallery_DOM=True,actual_viewer_model_metadata=True,
    screenshot_review=False,limitation='Existing CUA browser screenshot has recorded zero-width failure. This turn checked DOM only; actual original native crops and rendered PPT slide were visually inspected separately.',
    historical_model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,new_model=False,model_quality_passed=False,
    tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    sha256={n:sha(G/n) for n in ['browser-report.txt','browser-gallery.txt','browser-viewer.txt']})
save(G/'viewer-ui-review.json',ui);save(G/'ppt-receipt.json',rec)
p=read(G/'publication.json');p.update(updated=stamp,viewer_UI_checked=True,browser_pixel_review=False,PPT_updated=True,PPT_sha256=rec['final_sha256'],first51_preserved=True,tab15_untouched=True,company00_deployed=False);save(G/'publication.json',p)
# Next bounded read-only step. No hidden continuation of the already completed estimator.
inputs=[R/'video10-remaining-material-review/global-input-preflight/fit-observations.json',G/'relative-graph/repair-invalid-pose/pairs.json',G/'relative-graph/repair-invalid-pose/rotation-cycles.json',G/'minimal-root-diagnosis/cycle-result.json']
plan=dict(registered=stamp,state='registered_not_executed',
    hypothesis='The local three-view check resolves one two-view branch better. Check whether this support actually extends through all33views before implementing a joint branch/scale solver.',
    source_sha256={str(x):sha(x) for x in inputs},
    operation='Read-only CPU graph audit on saved FIT observations and58pair results. Enumerate the32existing closed triples and their unique FIT families, per-camera counts, exactly-three-view shared tracks, and links between triples sharing two cameras or sufficient physical tracks. Report raw counts and components, do not infer geometry just from graph connectivity.',
    preserve='Old12per-pair robust gate remains unchanged. Any union-of-triple support is explicitly a different joint-constraint diagnostic, not a relaxed pair admission. Do not merge same-family branches into one3Dpoint. No HELD coordinates or errors used in support decisions.',
    decision='Report whether each full-frame bridge has a real third-view/shared-track constraint; predeclare a global joint solver only if there is an identifiable new constraint. If not, record missing material support instead of repeating the same pair solver or random restarts.',
    budget=dict(CPU_seconds=30,passes=1,new_matching=0,new_camera_estimates=0,BA=0,training=0),
    stop='No threshold sweep, camera interpolation, reused old geometry or SMPL-X/GS fit. No claim that failed pair/triple support proves every global method impossible.')
assert not (G/'multiview-support-plan.json').exists();save(G/'multiview-support-plan.json',plan)
for root in [R,REC]:
    q=read(root/'queue-state.json');q.update(updated=stamp,next_plan=str(G/'multiview-support-plan.json'),latest_evidence=str(G/'decision.json'),running=None,training_running=False,local_GPU_task_active=False)
    q['last_goal_turn'].update(publication=str(G/'publication.json'));save(root/'queue-state.json',q)
x=read(REC/'next-action.json');x['next_plan']=str(G/'multiview-support-plan.json');save(REC/'next-action.json',x)
x=read(C/'local-research-status.json');x['updated_at']=stamp;x['current_video10_global_initializer'].update(publication=str(G/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=rec['final_sha256'],next_plan=str(G/'multiview-support-plan.json'));save(C/'local-research-status.json',x)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    for name in ['publication.json','viewer-ui-review.json','ppt-receipt.json','multiview-support-plan.json']:shutil.copy2(G/name,dest/name)
print(json.dumps(dict(publication=p,next_plan=str(G/'multiview-support-plan.json')),ensure_ascii=False))
