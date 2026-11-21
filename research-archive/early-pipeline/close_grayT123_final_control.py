from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'grayT-final-control-review';R=T/'shuolee/current/robust-portrait-122'
C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'grayT123-final-control-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review'] and sha(Path(receipt['canonical_path']))==receipt['final_sha256']
main=(O/'main-viewer-ax.txt').read_text(encoding='utf-8');report=(O/'report-ax.txt').read_text(encoding='utf-8')
assert all(s in main for s in ['309,523','93 个相机','逐帧改善，统一姿态仍未通过'])
assert '逐帧训练改善，统一姿态未通过' in report
ui=dict(updated=stamp,report_DOM_verified=True,main_viewer_DOM_verified=True,historical_model_preserved=True,
 points=309523,cameras=93,actual_browser_pixel_review=False,
 screenshot_limitation='Known browser zero-width screenshot surface; not retried or bypassed.',
 actual_saved_24_gallery_pages_reviewed=True,tab15_untouched=True,official_SIBR_independent_Spark_company00_GUI_verified=False,
 evidence=[dict(path=str(O/n),sha256=sha(O/n)) for n in ['report-ax.txt','main-viewer-ax.txt']])
save(O/'viewer-ui-review.json',ui)
d=read(O/'publication.json');d.update(updated=stamp,browser_DOM_verified=True,PPT_updated=True,PPT_sha256=receipt['final_sha256'],
 first51_slides_unchanged=True,actual_browser_pixel_review=False,company00_handoff_sent=False);save(O/'publication.json',d)
save(O/'progress.json',dict(updated=stamp,state='complete_reviewed_published',repeat=False,CPU_seconds=15.335292007,
 GPU_tasks=0,new_fit=False,new_camera=False,new_training=False,quality_accepted=False,goal_active=True,
 unit_correction_preserved=True,next=read(O/'conclusion.json')['next']))
q=read(R/'queue-state.json');q['completed'].append('GrayT final18k control read-only review:95 training faces,12canonical novel head/body angles,6bodyviews;19.35%head and1.57%whole RGB mean reduction;canonical export does not carry deformation;eyes unweighted;static quality failed;report/PPT52 updated.');save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_grayT_final_control_publication=d,
 latest_local_handoff=dict(updated=stamp,thread='01a17c8c-e70f-7822-ad39-440290b3d1da',evidence=str(O/'conclusion.json'),reply_required=False,delivered=False));save(C/'local-research-status.json',c)
n=read(REC/'next-action.json');n.update(updated=stamp,diagnostic_DOM_verified=True,actual_UI_verified=False,latest_review_publication=str(O/'publication.json'));save(REC/'next-action.json',n)
for destination in [L/'灰衣95帧与统一姿态',T/'viewer/diagnostics/grayT-final-control',Path('D:/00/工作/查看器/diagnostics/grayT-final-control')]:
 for name in ['frame_057_input_static_deformed.png','viewer-ui-review.json','publication.json','progress.json','result-completed.json']:
  shutil.copy2(O/name,destination/name)
shutil.copy2(H/'grayT123-final-control-final-52.png',L/'报告末页_灰衣95帧与统一姿态.png')
save(O/'publication-recovery.json',dict(updated=stamp,presentation_initial_error='First-party import child lacked RUNTIME_NODE_MODULES.',
 corrected='Retried finalization with bundled Node package path; all52 rendered,first51 unchanged.',
 original_audit_units='Initial numeric result preserved, corrected projection units in result-completed.json and unit-correction.json; audit not repeated.'))
print(json.dumps(dict(state='complete_reviewed_published',quality_accepted=False,new_training=False,goal='active',PPT=receipt['canonical_path']),ensure_ascii=False))
