from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
H=Path(__file__).parent;T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'generic-2d3d-research/adapter-a0c';G=T/'shuolee/current/robust-portrait-122/video10-global-initializer'
SYNC=Path('D:/00/工作/查看器');LOG=Path('D:/00/工作日志/2026-10-29/人物对齐核验123')
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(H/'alignment-initializer123-deck-receipt.json');prior=read(H/'alignment-cohort123-deck-receipt.json')
source=Path(receipt['source']);final=Path(receipt['finalPath'])
assert not receipt['canonical_replaced'] and sha(source)==prior['final_sha256']==receipt['source_sha256']
assert sha(final)==receipt['final_sha256'] and len(receipt['renders'])==52 and receipt['changed_slides']==[52]
assert read(H/'alignment-initializer123-exact-package.json')['other_members_byte_identical']
assert (H/'alignment-initializer123-final-validation.json').is_file()
dom=(O/'viewer-dom.txt').read_text(encoding='utf-8')
for text in ['通用拟合链路：初始化绑定与完整合成验证','原17项与新增12项通过','独立NumPy重投影差值最大0.000056px','本轮没有实拍新拟合或高斯训练']:
    assert text in dom,text
result=read(O/'result.json');assert result['synthetic_full_fit']['state']=='synthetic_full_fit_passed'
for name,digest in result['source_hashes'].items():assert sha(P/'src/shuolee_portrait'/name)==digest,name
stamp=datetime.now(timezone.utc).isoformat()
shutil.copy2(final,source);shutil.copy2(final,T/'shuolee/reports/02_有限视角重建优化路线.pptx')
assert sha(source)==receipt['final_sha256']
receipt.update(canonical_replaced=True,actual_last_page_visual_review=True,updated=stamp)
save(H/'alignment-initializer123-deck-receipt.json',receipt)
ui=dict(updated=stamp,url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#generic-2d3d-initializer',
    existing_tab=32,reloaded=True,DOM_verified=True,DOM_sha256=sha(O/'viewer-dom.txt'),
    screenshot_verified=False,screenshot_limitation='Previously established zero-width screenshot failure; no repeated capture or bypass.',
    diagnostic_only=True,new_point_cloud=False,historical_tab15_untouched=True,cloud_sync_verified=False,company00_verified=False)
save(O/'viewer-ui-review.json',ui)
p=O/'publication.json';q=read(p);q.update(updated=stamp,diagnostic_DOM_verified=True,canonical_PPT_updated=True,
    PPT_path=str(source),PPT_sha256=receipt['final_sha256'],PPT_changed_slides=[52],other_51_preserved=True,
    local_source_repository=str(P),new_initializer_binding=True,Git_pushed=False);save(p,q)
p=REC/'next-action.json';q=read(p);q.update(actual_UI_verified=True,publication_next=None,latest_initializer_ui=str(O/'viewer-ui-review.json'),
    latest_engineering_publication=str(O/'publication.json'),training_running=False,current_worker_pid_wsl=None,publication_worker_pid_windows=None);save(p,q)
for dest in [G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer',LOG/'相机多解与人物对齐']:
    for name in ['publication.json','viewer-ui-review.json','visual-review.json']:
        shutil.copy2(O/name,dest/'generic-2d3d-research/adapter-a0c'/name)
    shutil.copy2(REC/'generic-2d3d-research/adaptation-plan.json',dest/'generic-2d3d-research/adaptation-plan.json')
save(O/'verification.json',dict(updated=stamp,actual_full_fit_completed=True,process_finished=True,CUDA_tasks_none_after_test=True,
    git_diff_check_passed=True,prior_module_compile_passed=True,source_hashes_verified=True,
    physical_camera_accepted=False,real_face_subpixel_accepted=False,goal_stays_active=True,
    next_scope='A0/A0b/A0c complete. Do not rerun their tests without source changes or new evidence. A1 requires independent real camera and correspondence evidence.'))
print(json.dumps(dict(A0c_complete=True,actual_full_synthetic_fit=True,real_fit=False,training=False,
    canonical_PPT_updated=True,actual_DOM_verified=True,goal_stays_active=True),ensure_ascii=False))
