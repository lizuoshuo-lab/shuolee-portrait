from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=T/'shuolee/current/local-clothed-surface-123'
PRE=REC/'grass-preflight';O=PRE/'generic-input';G=R/'video10-global-initializer'
SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(H/'grass-native123-deck-receipt.json');prior=read(H/'alignment-initializer123-deck-receipt.json')
source=Path(receipt['source']);final=Path(receipt['finalPath'])
assert not receipt['canonical_replaced'] and sha(source)==prior['final_sha256']==receipt['source_sha256']
assert sha(final)==receipt['final_sha256'] and len(receipt['renders'])==52 and receipt['changed_slides']==[52]
assert read(H/'grass-native123-exact-package.json')['other_members_byte_identical']
assert (H/'grass-native123-final-validation.json').is_file()
dom=(O/'viewer-final-dom.txt').read_text(encoding='utf-8')
for token in ['草地94帧：通用输入与遮挡检查','1.322～1.477像素','没有生成新高斯点云','历史原始结果文件保持不变']:
    assert token in dom,token
expanded=(O/'viewer-dom.txt').read_text(encoding='utf-8')
assert '(expanded) 查看5个视角的脸部和脚部原像素对照' in expanded
for n,digest in read(O/'result.json')['source_hashes'].items():assert sha(P/'src/shuolee_portrait'/n)==digest,n
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,source);shutil.copy2(final,T/'shuolee/reports/02_有限视角重建优化路线.pptx')
assert sha(source)==receipt['final_sha256']
receipt.update(updated=stamp,canonical_replaced=True,actual_last_page_visual_review=True)
save(H/'grass-native123-deck-receipt.json',receipt)
save(O/'viewer-ui-review.json',dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#grass94-generic-input',
    existing_tab=32,reloaded=True,DOM_verified=True,gallery_expanded_and_verified=True,final_DOM_sha256=sha(O/'viewer-final-dom.txt'),
    screenshot_verified=False,screenshot_limitation='Previously established zero-width screenshot failure. No repeated capture or bypass. Actual source/crop images separately reviewed through local image tool.',
    new_point_cloud=False,new_renderer_acceptance=False,historical_tab15_untouched=True,cloud_sync_verified=False,company00_verified=False))
p=O/'publication.json';q=read(p);q.update(updated=stamp,diagnostic_DOM_verified=True,canonical_PPT_updated=True,PPT_path=str(source),
    PPT_sha256=receipt['final_sha256'],PPT_changed_slides=[52],other_51_preserved=True,Git_pushed=False);save(p,q)
p=REC/'next-action.json';q=read(p);q.update(actual_UI_verified=True,diagnostic_DOM_verified=True,publication_next=None,latest_grass_ui=str(O/'viewer-ui-review.json'),
    latest_engineering_publication=str(O/'publication.json'),training_running=False,current_worker_pid_wsl=None,publication_worker_pid_windows=None);save(p,q)
for dest in [G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer',LOG/'相机多解与人物对齐']:
    for name in ['publication.json','viewer-ui-review.json','visual-review.json']:shutil.copy2(O/name,dest/'grass-preflight'/name)
save(O/'verification.json',dict(updated=stamp,selected_frames_complete=94,all94_native_dimensions_verified=True,single_detector_pass=True,
    source_code_hashes_unchanged=True,actual_generic_CLI_read=True,fit_allowed=False,all_unknown_observations_inactive=True,
    actual_qualitative_native_image_review=True,pointwise_subpixel_error_not_measured=True,
    process_finished=True,CUDA_compute_tasks_none_after_pass=True,goal_stays_active=True,
    next=str(PRE/'next-action.json')))
print(json.dumps(dict(actual94frames=True,native_qualitative_review=True,training=False,canonical_PPT_updated=True,actual_DOM_verified=True,goal_stays_active=True),ensure_ascii=False))
