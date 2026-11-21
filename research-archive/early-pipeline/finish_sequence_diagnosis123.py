from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';REC=T/'shuolee/current/local-clothed-surface-123'
OUT=REC/'smplifyx-reference-control/framewise-canonical-route/shared-three-frame'
D=OUT/'final-state-diagnosis';COORD=H.parent/'shuolee-coordination';SYNC=Path('D:/00/工作/查看器')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
receipt=H/'sequence-diagnosis123-deck-receipt.json';deck=read(receipt)
assert not deck['canonical_replaced'] and deck['changed_slides']==[52]
assert len(deck['renders'])==52 and all(not x['changed'] for x in deck['renders'][:51])
assert read(H/'sequence-diagnosis123-exact-package.json')['other_members_byte_identical']
assert sha(deck['source'])==deck['source_sha256'] and sha(deck['finalPath'])==deck['final_sha256']
shutil.copy2(deck['finalPath'],deck['source']);shutil.copy2(deck['finalPath'],T/'shuolee/reports'/Path(deck['source']).name)
deck.update(canonical_replaced=True,updated=stamp,actual_final_slide52_image_reviewed=True,
            historical_figure_preserved=True,first51_byte_and_render_checks_passed=True)
save(receipt,deck)
assert (D/'optimizer-fix-result.json').is_file() and (D/'optimizer-cpu-regression.json').is_file()
assert sha(D/'optimizer-before-fix.py')==read(D/'plan.json')['source_hashes']['/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/shuolee_portrait/sequence_optimizer.py']
assert sha(OUT/'fit.npz')==read(D/'plan.json')['source_hashes'][str(OUT/'fit.npz').replace('C:','/mnt/c').replace('\\','/')]
ui=dict(updated=stamp,tab_id='32',url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#shared-sequence-diagnosis',
  actual_existing_tab_reloaded=True,actual_AX_result_and_table_verified=True,actual_anchor_navigation=True,
  marked_deliverable=True,screenshot_verified=False,screenshot_limitation='IAB screenshot returned Cannot take screenshot with0width. No alternate capture path used.',
  tab15_untouched=True,new_PLY=False,company00_remote_verified=False,WPS_cloud_sync_verified=False)
save(D/'viewer-ui-review.json',ui)
nexttext='先按冻结三帧和可见性集合核验本地FAN面部观测，保存独立原图/翻转热图和连续坐标，只比较观测与原图、零拟合。根初始化仅剩两肩，新策略须单独登记并做合成可辨识性检查；不扩大观测、不调拟合权重或直接重跑。'
fixtext='通用求解器已修复并通过8项CPU回归：每次更新和最终恢复都重算目标/梯度，阶段完成与驻点分别记录。根初始化统一观测资格；仅有两肩时明确返回支持不足并保持初始化，未重跑实拍。'
p=D/'assessment.json';a=read(p);a.update(updated=stamp,optimizer_fix=str(D/'optimizer-fix-result.json'),
    cpu_regression=str(D/'optimizer-cpu-regression.json'),generic_fix=fixtext,real_fit_after_fix=False,next=nexttext);save(p,a)
p=D/'publication.json';pub=read(p);pub.update(updated=stamp,viewer_UI_pending=True,viewer_text_and_table_verified=True,
    viewer_screenshot_verified=False,viewer_UI_limitation=ui['screenshot_limitation'],deck_pending=False,
    deck=deck['source'],deck_sha256=deck['final_sha256'],only_slide52_changed=True,first51_byte_and_render_checks_passed=True,
    generic_optimizer_fix_cpu_passed=True,real_fit_after_fix=False)
save(p,pub)
for dest in [R/'video10-global-initializer',T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer']:
    p=dest/'index.html';s=p.read_text(encoding='utf-8')
    s=s.replace('正在修复显式检查，不暗中扩大观测或更换损失权重。',fixtext)
    s=s.replace('上方历史叠加图仍是上一轮结果。','下方历史叠加图仍是上一轮结果。')
    s=s.replace('保存解的核查见下方','保存解的核查见上方')
    s=s.replace('完成通用求解器的端点/最终状态与观测资格修复后，按冻结三帧和可见性集合核验本地FAN面部观测；先比对原图和坐标映射，不重新拟合、不调权重。根初始化仅剩两肩的支持缺口须单独登记策略后处理。',nexttext)
    p.write_text(s,encoding='utf-8')
    target=dest/'shared-sequence-diagnosis'
    for name in ['assessment.json','publication.json','viewer-ui-review.json','optimizer-fix-result.json','optimizer-cpu-regression.json']:
        shutil.copy2(D/name,target/name)
for base in [R,REC]:
    p=base/'queue-state.json';v=read(p);v.update(updated=stamp,current_stage='shared_sequence_solver_fixed_observations_pending',
      phase='shared_sequence_solver_fixed_observations_pending',next=nexttext,next_action=nexttext,
      latest_evidence=str(D/'assessment.json'),optimizer_fix=str(D/'optimizer-fix-result.json'),
      previous_goal_turn_classification='progress: exact final-state diagnosis and8CPUregression checks; no real fit after fix',
      running=None,worker_pid=None,training_running=False,local_GPU_task_active=False);save(p,v)
for base,name in [(R,'protocol.json'),(REC,'next-action.json')]:
    p=base/name;v=read(p);v.update(updated=stamp,current_stage='shared_sequence_solver_fixed_observations_pending',
      next=nexttext,next_action=nexttext,next_plan=str(D/'fan-observation-plan.json'));save(p,v)
p=COORD/'local-research-status.json';v=read(p);v.update(updated_at=stamp,current_stage='shared_sequence_solver_fixed_observations_pending',
    next_action=nexttext,latest_publication=str(D/'publication.json'),latest_viewer_ui_review=str(D/'viewer-ui-review.json'),
    current_report=deck['source'],current_report_sha256=deck['final_sha256'],optimizer_fix=str(D/'optimizer-fix-result.json'),
    company_installation_unchanged=True,viewer_text_and_table_verified=True,viewer_screenshot_verified=False);save(p,v)
p=T/'shuolee/long-term-plan.md';s=p.read_text(encoding='utf-8')
if fixtext not in s:p.write_text(s+'\n\n'+fixtext+'\n\n'+nexttext+'\n',encoding='utf-8')
print(json.dumps(dict(canonical_deck_updated=True,changed_slides=[52],generic_fix_cpu_passed=True,new_fits=0,new_gaussian_steps=0,goal_active=True)))
