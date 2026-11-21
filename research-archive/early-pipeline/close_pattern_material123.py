from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=R.parent/'local-clothed-surface-123';O=REC/'pattern-preflight/native-material';G=R/'video10-global-initializer'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat();deck=read(H/'pattern-material123-deck-receipt.json')
assert deck['canonical_replaced'] and deck['actual_last_page_visual_review'] and len(deck['renders'])==52
assert all(sha(Path(deck[k]))==deck['final_sha256'] for k in ['canonical_path','mirror_path'])
texts={name:(O/f'browser-{name}.txt').read_text(encoding='utf-8') for name in ['report','gallery','viewer']}
assert all(x in texts['viewer'] for x in ['309,523','93 个相机','质量尚未通过验收','SMPL-X对齐：衣服与眼睛的错误连接已隔离'])
assert all(x in texts['gallery'] for x in ['第5帧到第85帧','32,948','第490帧','614.47px'])
assert 'SMPL-X对齐：衣服与眼睛的错误连接已隔离' in texts['report']
ui=dict(updated=stamp,actual_report_DOM=True,actual_gallery_DOM=True,actual_model_metadata=True,report_tab='32',viewer_tab='25',
    screenshot_review=False,viewport_loading_label_present='载入中' in texts['viewer'],new_render_quality_claim=False,
    limitation='Known zero-width screenshot failure not retried or bypassed. Current diagnostic DOM and historical model metadata checked; canvas loading label remains. Actual saved native crops and final slide52 reviewed separately.',
    model='local-surface-123-whiteT-s18000',model_changed=False,gaussians=309523,cameras=93,quality_accepted=False,
    tab15_untouched=True,company00_deployed=False,WPS_cloud_roundtrip_verified=False,
    source_sha256={name:sha(O/f'browser-{name}.txt') for name in texts})
save(O/'viewer-ui-review.json',ui);save(O/'ppt-receipt.json',deck)
pub=read(O/'publication.json');pub.update(updated=stamp,viewer_UI_checked=True,viewer_pixels_reviewed=False,PPT_updated=True,
    PPT_sha256=deck['final_sha256'],first51_slides_unchanged=True,actual_native_crop_review=True,material_review30_completed=True,
    false_track_all10_reviewed=True,bridge_control_completed=True,tab15_untouched=True)
save(O/'publication.json',pub);save(G/'publication.json',dict(**pub,latest_stage=str(O/'publication.json')))
for root in [R,REC]:
    p=root/'queue-state.json';q=read(p);q['updated']=stamp;q['last_goal_turn']['publication']=str(O/'publication.json')
    q.update(running=None,worker_pid=None,training_running=False,local_GPU_task_active=False,
        next_preflight=str(O/'bridge-control/physical-review-plan.json'),previous_goal_turn_classification='progress: one native99frame matching pass,30physical reviews,one bridge identity control,existingviewer/log/PPT52 updated')
    save(p,q)
p=REC/'next-action.json';q=read(p);q.update(updated=stamp,latest_review_publication=str(O/'publication.json'),diagnostic_DOM_verified=True,
    actual_UI_verified=False,completed_this_turn=['One99frame native material pass','30 preregistered triples and10 false-track pixels reviewed','One topology-only bridge control; all original labels/observations preserved','Existing report,viewer DOM andPPT52 updated'],
    publication_next='Next review the retained physical material support using the preregistered finite plan. No current GPU or training. No new model accepted.');save(p,q)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p);q.update(updated_at=stamp,current_stage='pattern_bridge_control_reviewed',current_subject='pattern',training_running=False)
q['current_pattern_material'].update(publication=str(O/'publication.json'),actual_viewer_DOM_verified=True,PPT_sha256=deck['final_sha256'],company00_deployed=False);save(p,q)
for dest in [T/'viewer/diagnostics/video10-global-initializer',Path('D:/00/工作/查看器/diagnostics/video10-global-initializer'),Path('D:/00/工作日志/2026-10-29/人物对齐核验123/相机多解与人物对齐')]:
    shutil.copy2(G/'publication.json',dest/'publication.json')
    for name in ['publication.json','viewer-ui-review.json','ppt-receipt.json']:shutil.copy2(O/name,dest/'pattern-preflight/native-material'/name)
print(json.dumps(dict(publication=str(O/'publication.json'),PPT_sha256=deck['final_sha256'],goal_complete=False,training_started=False),ensure_ascii=False))
