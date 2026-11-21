from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123'
E=REC/'camera-observation-entry';C=REC/'field-preflight/coverage';G=R/'video10-global-initializer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();deck=read(H/'camera-entry123-deck-receipt.json')
assert deck['canonical_replaced'] and deck['actual_last_page_visual_review'] and len(deck['renders'])==52
assert sha(Path(deck['canonical_path']))==sha(Path(deck['mirror_path']))==deck['final_sha256']
texts={n:(E/f'browser-{n}.txt').read_text(encoding='utf-8') for n in ['report','gallery','viewer']}
assert all(s in texts['report'] for s in ['观测入口修复','田地177帧','没有实拍相机或SMPL-X对齐通过证据'])
assert all(s in texts['gallery'] for s in ['26项工程检查','后脑勺','8527','23.85px','43.99px'])
assert all(s in texts['viewer'] for s in ['观测入口修复','309,523','93 个相机','质量尚未通过验收'])
ui=dict(updated=stamp,actual_report_DOM=True,actual_gallery_DOM=True,actual_viewer_metadata=True,
    report_tab='32',viewer_tab='25',new_model=False,model='local-surface-123-whiteT-s18000',gaussians=309523,cameras=93,
    screenshot_review=False,new_render_quality_claim=False,
    limitation='Known zero-width browser screenshot failure not retried or bypassed. DOM/metadata only. Actual saved native subject crops and slide52 inspected separately. Loading label persists.',
    tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    source_sha256={n:sha(E/f'browser-{n}.txt') for n in texts})
save(E/'viewer-ui-review.json',ui);save(E/'ppt-receipt.json',deck)
pub=read(E/'publication.json');pub.update(updated=stamp,viewer_UI_checked=True,viewer_pixels_reviewed=False,
    PPT_updated=True,PPT_sha256=deck['final_sha256'],first51_slides_unchanged=True,
    actual_field_subject_crops_reviewed=5,actual_field_head_and_feet_crops_reviewed=10,tab15_untouched=True,
    direct_consumer_integration_completed=True,physical_accuracy_improved=False,
    current_model='local-surface-123-whiteT-s18000',company00_deployed=False,WPS_cloud_roundtrip_verified=False)
save(E/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(E/'publication.json')))
for root in [R,REC]:
    q=read(root/'queue-state.json');q.update(updated=stamp,running=None,worker_pid=None,worker_command=None,
        training_running=False,local_GPU_task_active=False,previous_goal_turn_classification='progress: actual frozen-observation caller repaired and26engineeringchecks passed; first field177ROI pass completed and five native contexts reviewed; report/viewer/PPT52 updated')
    q['last_goal_turn']['publication']=str(E/'publication.json');save(root/'queue-state.json',q)
q=read(REC/'next-action.json');q.update(updated=stamp,latest_review_publication=str(E/'publication.json'),diagnostic_DOM_verified=True,
    actual_UI_verified=False,completed_this_turn=['Frozen observation reader and actual prepare caller integrated, no hidden rematching fallback','26engineeringchecks passed, including actual rejection and exact evidence scope','First field177ROI pass complete, five native contexts reviewed, back-head face hallucination recorded','Existing report/viewer/PPT52 updated; original model preserved'],
    publication_next='Read the existing observation/visibility audits first. Reuse already established decoder and mask results. Trace only the actual new field observation-to-loss consumer and retained negative control; no repeat inference or fitting.')
save(REC/'next-action.json',q)
p=REC/'observation-visibility-consumer-plan.json';q=read(p);q.update(
    reuse_prior_evidence='Existing grayT observation-quality, simcc-decoder, facial-surface/visibility support audits are already completed. Read their results/source identity first; do not rerun inference, DARK control, old fitting or mask diagnostics.',
    distinct_scope='New field177 frozen observations and source440 non-visible facial negative control at the actual current consumer. If the same consumer behavior is already fully proved, record reuse and proceed to the existing missing visibility/material integration instead of repeating a test.')
save(p,q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q.update(updated_at=stamp,training_running=False)
q['current_camera_entry'].update(publication=str(E/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=deck['final_sha256'],company00_deployed=False)
save(p,q)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'publication.json',dest/'publication.json')
    shutil.copy2(REC/'observation-visibility-consumer-plan.json',dest/'observation-visibility-consumer-plan.json')
    sub=dest/'camera-observation-entry'
    for n in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(E/n,sub/n)
print(json.dumps(dict(publication=str(E/'publication.json'),PPT_sha256=deck['final_sha256'],goal_complete=False,training=False),ensure_ascii=False))
