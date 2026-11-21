"""Publish the actually reviewed SMPL-X failure; never approves a training stage."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,html
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/local-clothed-surface-123';P=T/'shuolee/current/robust-portrait-122'
C=H.parent/'shuolee-coordination';L=Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
mapping=read(R/'smplx-mapping-audit/summary.json');relaxed=read(R/'fixed-camera-face-relaxation/summary.json')
assert read(R/'alignment-review-supervisor-stop.json')['process_alive'] is False
gallery=R/'grayT-smplx-audit-review';idx=read(gallery/'index.json')
assert len(idx['images'])==6 and all(sha(gallery/x['file'])==x['sha256'] for x in idx['images'])
summary=('已实际审阅6组原生像素裁片：每组依次为原图、参与拟合的RTMW鼻嘴点、SMPL-X对应投影点，图上没有文字。'
 '前侧视角的检测点大体沿可见鼻嘴，帧76/95的SMPL-X点却落在脸颊或耳前；帧1投影落到轮廓外。'
 '灰衣人体可靠点中位23.85px，面部参与点中位43.99px，帧76面部中位100.42px。它们是检测器残差，不能当独立物理真值。'
 '固定全部相机、像素观测和权重，只将29个活跃面部点从人体模板解耦成独立静态3D点后，中位残差47.05px，仍未对齐。'
 '该诊断的原损失由4.775降至4.414，说明优化执行了；中位数不等于所优化的加权鲁棒目标，不能挑选均值掩盖大误差。'
 '已核验的帧/slot、面部17..67及脚部顺序、独立投影无整体错配；投影数值差<0.0004px。'
 '当前拟合缺少面部/脚部解剖可见性检查，灰衣47.1%的面部权重落在当前网格自遮挡/背向点；这是模型派生诊断，网格本身错位，不能直接据此删除观测。'
 '粗门槛允许身体中位误差达身高3%，灰衣约50–53px，而且不考核脸手脚；此前只允许它作表面包围盒初始化，不能解释为精准拟合。'
 '仅换SMPL-X参数无法说明或解决全部错位。相机实际K/E、检测语义和人物转头/表情贡献仍未唯一分开。'
 '已在灰衣表面审阅节点停止旧队列，灰衣尚未开始高斯训练，旧结果及后续人物保留，目标继续排查。')
review=dict(updated=stamp,actual_visual_review=True,reviewed_frames=[x['frame'] for x in idx['images']],
    index_sha256=sha(gallery/'index.json'),summary=summary,quality_accepted=False,physical_calibration_accepted=False,
    no_training_approval=True,model_visibility_is_not_truth=True,raw_detector_localization_is_not_material_truth=True)
save(gallery/'actual-review.json',review)
analysis=dict(updated=stamp,subject='grayT',summary=summary,mapping_audit=mapping,frozen_camera_relaxation=relaxed,
    current_training_worker=None,grayT_gaussian_training_started=False,goal_active=True,
    completed_whiteT=dict(steps=18000,points=309523,cameras=93,quality_accepted=False,staged=True,actual_package_UI_pending=True),
    next_causal_step='Using frozen grayT camera solver observations, measure foreground/static-scene support without changing splits or fitting. Review native physical same-point observations before any camera or head-pose counterfactual.',
    forbidden='No automatic visibility deletion from a wrong mesh, no fresh features/BA/training solely to hide these residuals, no old-run resume.',
    source_hashes={str(x):sha(x) for x in [H/'audit_surface123_smplx_mapping.py',H/'diagnose_surface123_fixed_camera_face.py']})
save(R/'smplx-alignment-conclusion.json',analysis)
ledger=read(R/'evidence-ledger.json')
assert not any(x.get('kind')=='smplx_alignment_native_mapping_and_relaxation' for x in ledger['entries'])
ledger['entries'].append(dict(kind='smplx_alignment_native_mapping_and_relaxation',date=stamp,summary=summary,
    evidence=str(R/'smplx-alignment-conclusion.json'),quality_accepted=False,no_unique_cause_claim=True))
ledger['updated']=stamp;save(R/'evidence-ledger.json',ledger)
q=read(P/'queue-state.json');q.update(updated=stamp,phase='LOCAL_SURFACE123_SMPLX_alignment_failure_isolated_before_grayT_training',
    running=None,next=analysis['next_causal_step'],quality_accepted=False,training_running=False)
q['completed'].append('123 frozenfit mapping audit and one fixed-camera independent-static-face relaxation on3people; allfail subpixel; actual6grayT native face panels reviewed; idle oldsupervisor stopped beforegrayTGaussian.')
save(P/'queue-state.json',q)
nextpath=R/'next-action.json';n=read(nextpath);n.update(updated=stamp,current_worker_pid_wsl=None,current_subject='grayT',
    observed_progress='camera/prior/fit/surface saved; NO grayT Gaussian training; old supervisor361 stopped atreview boundary',
    first_person='winterhat18k_completed_published_quality_failed',second_person='whiteT18k_completed_staged_quality_failed_UI_pending',
    latest_actual_checkpoint_review=str(gallery/'actual-review.json'),
    next=[analysis['next_causal_step'],'Do not restart local_surface123.py or resume oldsubjectroot. Any corrected fit must be separately registered and start fresh after the cause-specific check.',
        'Complete already-staged whiteT package actual local OpenFileGaussian/points UI and publish withqualityfalse; sourcePLY andcamera checked.',
        'Existing52pagePPT lastpage to record this alignment finding; first51unchanged. Goalactive, nottimer.'],
    diagnostics=str(R/'smplx-alignment-conclusion.json'),quality_accepted=False)
save(nextpath,n)
status=read(C/'local-research-status.json');status.update(updated_at=stamp,next=analysis['next_causal_step'],
    local_GPU_task_active=False,training_started=True,training_running=False,current_local_diagnosis=analysis)
save(C/'local-research-status.json',status)
plan=T/'shuolee/long-term-plan.md'
with plan.open('a',encoding='utf-8') as f:f.write('\n\n2026-10-28 — 本地123 SMPL-X错位核验：\n'+summary+'\n下一步：'+analysis['next_causal_step']+'\n')
for sub in ['smplx-mapping-audit','fixed-camera-face-relaxation']:
    target=L/sub;target.mkdir(parents=True,exist_ok=True)
    for src in (R/sub).glob('*.json'):shutil.copy2(src,target/src.name)
save(L/'smplx-alignment-conclusion.json',analysis)
print(summary)
