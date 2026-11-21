from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';O=R/'native-residual-pipeline';REC=T/'shuolee/current/local-clothed-surface-123'
C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();receipt=read(H/'r7-pipeline122-deck-receipt.json')
assert receipt['canonical_replaced'] and receipt['actual_last_page_visual_review']
assert sha(Path(receipt['canonical_path']))==receipt['final_sha256']
main=(O/'main-viewer-ax.txt').read_text(encoding='utf-8');report=(O/'report-ax.txt').read_text(encoding='utf-8')
assert '309,523' in main and '93 个相机' in main and '静态对齐未通过，训练入口检查完成' in main
assert '原生训练入口检查完成；SMPL-X静态对齐仍未通过' in report
ui=dict(updated=stamp,report_DOM_verified=True,main_viewer_DOM_verified=True,historical_model_preserved=True,points=309523,cameras=93,
 actual_browser_pixel_review=False,screenshot_limitation='Existing browser reports zero-width surface; no repeated or lower-level bypass attempt.',
 actual_synthetic_images_reviewed=True,tab15_untouched=True,official_SIBR_Spark_company00_verified=False,
 evidence=[dict(path=str(O/n),sha256=sha(O/n)) for n in ['report-ax.txt','main-viewer-ax.txt']])
save(O/'viewer-ui-review.json',ui)
d=read(O/'publication.json');d.update(updated=stamp,browser_DOM_verified=True,PPT_updated=True,PPT_sha256=receipt['final_sha256'],first51_slides_unchanged=True,actual_browser_pixel_review=False,company00_handoff_sent=True);save(O/'publication.json',d)
save(O/'progress.json',dict(updated=stamp,state='engineering_complete_reviewed_published',CPU_checks=22,GPU_checks=14,person_fit=False,person_training=False,physical_quality_accepted=False,
 initial_failure_preserved=True,cache_fixed=True,source_drafts_only=False,test_executed=True,goal_active=True,repeat_this_suite=False,next=read(O/'conclusion.json')['next']))
q=read(R/'queue-state.json');q['completed'].append('R7 full residual training entry: cache bug reproduced and repaired;22CPU+14CUDA controls,96point12updates,exact native roundtrip,actual failed person rejected;UI DOM and same52page PPT updated;no person admission.');save(R/'queue-state.json',q)
c=read(C/'local-research-status.json');c.update(updated_at=stamp,current_R7_publication=d,latest_local_handoff=dict(updated=stamp,thread='01a17c8c-e70f-7822-ad39-440290b3d1da',evidence=str(O/'conclusion.json'),reply_required=False));save(C/'local-research-status.json',c)
n=read(REC/'next-action.json');n.update(updated=stamp,diagnostic_DOM_verified=True,actual_UI_verified=False,latest_engineering_publication=str(O/'publication.json'));save(REC/'next-action.json',n)
for name in ['viewer-ui-review.json','publication.json','progress.json']:shutil.copy2(O/name,L/'R7原生训练入口'/name)
shutil.copy2(H/'r7-pipeline122-final-52.png',L/'报告末页_原生训练入口核验.png')
save(O/'final-source-record.json',dict(updated=stamp,files={str(H/n):sha(H/n) for n in ['train_native_person122_v2.py','native_training122_v5.py','prepare_native_person122_v3.py','native_pipeline_binding122.py','native_training_policy122.py','check_residual_pipeline122.py','publish_local_surface123.py']},running_source_modified=False))
print(json.dumps(dict(state='engineering_complete_reviewed_published',goal='active',person_training=False,quality_accepted=False,PPT=receipt['canonical_path']),ensure_ascii=False))
