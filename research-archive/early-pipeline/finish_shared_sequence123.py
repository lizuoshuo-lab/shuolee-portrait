from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
OUT=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame'
SYNC=Path('D:/00/工作/查看器')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
rp=H/'shared-sequence123-deck-receipt.json';r=read(rp)
assert not r['canonical_replaced'] and r['changed_slides']==[52]
assert all(not x['changed'] for x in r['renders'][:51]) and len(r['renders'])==52
assert read(H/'shared-sequence123-exact-package.json')['other_members_byte_identical']
assert sha(r['source'])==r['source_sha256'] and sha(r['finalPath'])==r['final_sha256']
shutil.copy2(r['finalPath'],r['source']);shutil.copy2(r['finalPath'],T/'shuolee/reports'/Path(r['source']).name)
assert sha(r['source'])==r['final_sha256']
r.update(canonical_replaced=True,actual_before_after_slide52_reviewed=True,updated=stamp,figure_aspect_ratio_preserved=True)
save(rp,r)
ui=dict(updated=stamp,tab_id='32',url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#shared-sequence',
    actual_existing_tab_reloaded=True,actual_AX_result_and_error_table_verified=True,loaded_images=9,
    all9_images_complete_and_native_dimensions_verified=True,source9_images_visually_reviewed=True,
    screenshot_verified=False,screenshot_limitation='Current IAB screenshot returned Cannot take screenshot with 0 width; no alternate capture path used.',
    marked_deliverable=True,tab15_untouched=True,new_PLY=False,company00_remote_verified=False,WPS_cloud_sync_verified=False)
save(OUT/'viewer-ui-review.json',ui)
p=OUT/'publication.json';q=read(p)
q.update(updated=stamp,viewer_UI_pending=True,viewer_text_and_all9_loaded_images_verified=True,
    viewer_screenshot_verified=False,viewer_UI_limitation=ui['screenshot_limitation'],
    deck_pending=False,deck=r['source'],deck_sha256=r['final_sha256'],only_slide52_changed=True,
    first51_byte_and_render_checks_passed=True)
save(p,q)
for d in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer']:
    target=d/'smplifyx-reference-control/framewise-canonical-route/shared-three-frame'
    for name in ['viewer-ui-review.json','publication.json']:shutil.copy2(OUT/name,target/name)
    p=d/'index.html';page=p.read_text(encoding='utf-8')
    page=page.replace('<section id="framewise-canonical"><h2>逐帧姿态与静态高斯输出</h2>',
        '<section id="framewise-canonical"><h2>历史路线登记：逐帧姿态与静态高斯输出</h2>')
    page=page.replace('D:/00/记录/005 当前69张图片：本次9张同帧对照与保留的60张分析图。',
        '该轮整理后为69张图片；上方共享体型试验新增9张后，D:/00/记录/005当前为78张。')
    p.write_text(page,encoding='utf-8')
    block=re.search(r'<section id="shared-sequence">.*?</section>',page,re.S).group()
    imgs=re.findall(r'<img[^>]+src="([^"]+)"',block)
    assert len(imgs)==9 and all((d/image).is_file() for image in imgs)
p=H.parent/'shuolee-coordination/local-research-status.json';q=read(p)
q.update(updated_at=stamp,latest_publication=str(OUT/'publication.json'),latest_viewer_ui_review=str(OUT/'viewer-ui-review.json'),
         current_report=r['source'],current_report_sha256=r['final_sha256'],viewer_text_and_image_load_verified=True,viewer_screenshot_verified=False)
save(p,q)
print(json.dumps(dict(record_images=78,deck_updated=True,changed_slides=[52],viewer_loaded_images=9,quality_passed=False,gaussian_steps=0,goal_active=True)))
