from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123'
D=REC/'pattern-preflight/native-material/bridge-control/physical-review/consumer-audit';F=REC/'field-preflight';G=R/'video10-global-initializer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();deck=read(H/'pattern-consumer123-deck-receipt.json')
assert deck['canonical_replaced'] and deck['actual_last_page_visual_review'] and len(deck['renders'])==52
assert sha(Path(deck['canonical_path']))==sha(Path(deck['mirror_path']))==deck['final_sha256']
texts={n:(D/f'browser-{n}.txt').read_text(encoding='utf-8') for n in ['report','gallery','viewer']}
assert all(s in texts['report'] for s in ['已定位观测消费接口缺口','田地人物177帧','缺少相机准入记录'])
assert all(s in texts['gallery'] for s in ['审阅结果尚未接入相机入口','Frozen split/family required','0、220、440、660、880','人体骨架'])
assert all(s in texts['viewer'] for s in ['已定位观测消费接口缺口','309,523','93 个相机','质量尚未通过验收'])
ui=dict(updated=stamp,actual_report_DOM=True,actual_gallery_DOM=True,actual_viewer_metadata=True,
    report_tab='32',viewer_tab='25',new_model=False,model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,
    screenshot_review=False,new_render_quality_claim=False,
    limitation='Known zero-width browser screenshot failure not retried or bypassed. DOM and metadata only. Actual saved field crop pixels and slide52 inspected separately. Loading label still present.',
    tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    source_sha256={n:sha(D/f'browser-{n}.txt') for n in texts})
save(D/'viewer-ui-review.json',ui);save(D/'ppt-receipt.json',deck)
pub=read(D/'publication.json');pub.update(updated=stamp,viewer_UI_checked=True,viewer_pixels_reviewed=False,
    PPT_updated=True,PPT_sha256=deck['final_sha256'],first51_slides_unchanged=True,
    actual_field_frames_reviewed=5,actual_field_native_crops_reviewed=10,tab15_untouched=True,
    direct_consumer_integration_completed=False,physical_accuracy_improved=False)
save(D/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(D/'publication.json')))
for root in [R,REC]:
    q=read(root/'queue-state.json');q.update(updated=stamp,running=None,worker_pid=None,worker_command=None,
        training_running=False,local_GPU_task_active=False,previous_goal_turn_classification='progress: real consumer gap audited without semantic relabeling; unused field native input decoded and five frames reviewed; original report/viewer/PPT52 updated')
    q['last_goal_turn']['publication']=str(D/'publication.json');save(root/'queue-state.json',q)
q=read(REC/'next-action.json');q.update(updated=stamp,latest_review_publication=str(D/'publication.json'),diagnostic_DOM_verified=True,
    actual_UI_verified=False,completed_this_turn=['Real N3 and camera caller gap inspected; raw FIT/HELD/source/xy retained','Existing SMPLX guard actually rejects missing camera admission before fitting','Field native177frames decoded once and5originalframes/10nativecrops actually reviewed','Same report/viewer and slide52 updated; historical model retained'],
    publication_next='Implement the registered separate frozen-observation consumer integration. No feature extraction, geometric solve or human training in this engineering step.')
save(REC/'next-action.json',q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q.update(updated_at=stamp,training_running=False)
q['current_pattern_consumer'].update(publication=str(D/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=deck['final_sha256'],company00_deployed=False)
save(p,q)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'publication.json',dest/'publication.json')
    sub=dest/'pattern-preflight/native-material/bridge-control/physical-review/consumer-audit'
    for n in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(D/n,sub/n)
print(json.dumps(dict(publication=str(D/'publication.json'),PPT_sha256=deck['final_sha256'],goal_complete=False,training=False),ensure_ascii=False))
