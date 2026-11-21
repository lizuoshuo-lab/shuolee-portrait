from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123'
V=REC/'pattern-preflight/native-material/bridge-control/physical-review';G=R/'video10-global-initializer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();deck=read(H/'pattern-retained123-deck-receipt.json')
assert deck['canonical_replaced'] and deck['actual_last_page_visual_review'] and len(deck['renders'])==52
assert sha(Path(deck['canonical_path']))==sha(Path(deck['mirror_path']))==deck['final_sha256']
text={n:(V/f'browser-{n}.txt').read_text(encoding='utf-8') for n in ['report','gallery','viewer']}
assert all(s in text['report'] for s in ['闭环仍可能连接不同物体','160个','3个FIT段'])
assert all(s in text['gallery'] for s in ['原帧335、375、415','674','43.99px','33个头部家族未审'])
assert all(s in text['viewer'] for s in ['闭环仍可能连接不同物体','309,523','93 个相机','质量尚未通过验收'])
ui=dict(updated=stamp,actual_report_DOM=True,actual_gallery_DOM=True,actual_viewer_metadata=True,
    report_tab='32',viewer_tab='25',new_model=False,model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,
    screenshot_review=False,new_render_quality_claim=False,
    limitation='Known zero-width browser screenshot failure not retried or bypassed. DOM/metadata only. Saved native crop pixels and slide52 were actually inspected separately. Loading label still present.',
    tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    source_sha256={n:sha(V/f'browser-{n}.txt') for n in text})
save(V/'viewer-ui-review.json',ui);save(V/'ppt-receipt.json',deck)
pub=read(V/'publication.json');pub.update(updated=stamp,viewer_UI_checked=True,viewer_pixels_reviewed=False,
    PPT_updated=True,PPT_sha256=deck['final_sha256'],first51_slides_unchanged=True,actual_native_crop_review=True,
    review_rows=212,native_views=674,contradictory_FIT_segments=3,false_complete_triangles=2,tab15_untouched=True)
save(V/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(V/'publication.json')))
for root in [R,REC]:
    q=read(root/'queue-state.json');q.update(updated=stamp,running=None,worker_pid=None,worker_command=None,
        training_running=False,local_GPU_task_active=False,previous_goal_turn_classification='progress: bounded actual material review completed; three FIT contradictions traced; original report/viewer/PPT52 updated')
    q['last_goal_turn']['publication']=str(V/'publication.json');save(root/'queue-state.json',q)
q=read(REC/'next-action.json');q.update(updated=stamp,latest_review_publication=str(V/'publication.json'),diagnostic_DOM_verified=True,
    actual_UI_verified=False,completed_this_turn=['212 native triples and38 ROI endpoints actually reviewed','Two complete false three-view cycles and one longer contradictory FIT segment traced read-only','Current subject material budget ended without camera/training admission','Existing viewer/report and slide52 updated; historical model retained'],
    publication_next='Engineering evidence adapter under registered consumer-evidence-plan only; no new feature/matching/camera/human training candidate.')
save(REC/'next-action.json',q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q.update(updated_at=stamp,training_running=False)
q['current_pattern_retained_material'].update(publication=str(V/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=deck['final_sha256'],company00_deployed=False);save(p,q)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'publication.json',dest/'publication.json')
    for n in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(V/n,dest/'pattern-preflight/native-material/bridge-control/physical-review'/n)
print(json.dumps(dict(publication=str(V/'publication.json'),PPT_sha256=deck['final_sha256'],goal_complete=False,training=False),ensure_ascii=False))
