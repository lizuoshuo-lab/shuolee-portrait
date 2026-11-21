"""Record the completed observation audit and stopped material test, without refitting."""
from datetime import datetime, timezone
from pathlib import Path
import json, hashlib, shutil

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC = T/'shuolee/current/local-clothed-surface-123'
R = T/'shuolee/current/robust-portrait-122'
C = H.parent/'shuolee-coordination'
LOG = Path('D:/00/工作日志/2026-10-28/人物自由表面训练123')
OUT = REC/'grayT-temple-validation-review'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

stamp = datetime.now(timezone.utc).isoformat()
vis = read(REC/'grayT-visibility-material-review/result.json')
tr = read(OUT/'tracking-result.json')
loc = read(OUT/'localization-result.json')
assert tr['algorithmic_checks_passed'] and not loc['algorithmic_checks_passed']
assert [int(x['frame'].split('_')[-1]) for x in loc['failed_frames']] == list(range(40, 49))
assert not (OUT/'geometry-result.json').exists()
viewed = ['tracking_all41_locator.png', 'all41_localized_native.png',
          'all41_localized_locator.png', 'frames_040_041_044_048_049_054_inspection3x.png']
review = dict(updated=stamp, actual_visual_review=True, reviewed_frames=list(range(40,81)),
    reviewed_images=[dict(path=str(OUT/f), sha256=sha(OUT/f)) for f in viewed],
    summary='已审阅41帧原生裁片、跟踪/局部定位标记，并放大检查40、41、44、48、49、54帧。后段颞部斑点较清楚，40–48帧该小斑的对比度和与眉部相邻纹理的分离不足，不能把整条轨迹认证为精确的同一材料中心。光流往返最大0.200px仍不足以证明精确对应。9帧光度模型碰到预登记参数边界，按停止条件保留全部测量并停止几何检验。没有缩短序列、放宽边界或挑选通过帧重做。',
    early_frame_material_center_uncertain=True, all41_physical_identity_certified=False,
    measurement_accuracy_certified=False, geometry_executed=False, quality_accepted=False,
    prior_inventory_scope_correction='先前41帧未见明显遮挡只支持候选测量，不等于色素斑在每帧都可精确定位。本轮近距离检查和局部模型失败使早段身份/中心继续保持未确认。')
save(OUT/'actual-review.json', review)
save(OUT/'completion.json', dict(updated=stamp, state='stopped_before_geometry_on_preregistered_measurement_failure',
    tracking_runs=1, localization_runs=1, geometry_fits=0, failed_frames=list(range(40,49)),
    no_retry_or_threshold_change=True, result_sha256=sha(OUT/'localization-result.json'),
    actual_review_sha256=sha(OUT/'actual-review.json'), quality_accepted=False))

summary = ('SMPL-X仍未对齐，本轮确认拟合观测本身存在精度与可见性缺口。实际RTMW把全身输入缩为288×384，'
    '41帧脸部ROI在网络中位只有29.75像素高。argmax解码加翻转平均后，369个鼻部点全部位于约1.41原生像素间隔的解析格点上，'
    '不能当作连续的亚像素材料测量。另有12个鼻部观测位于原人物mask外0.786–3.660像素，仍因4像素余量获得正权重。'
    '原生图审阅也发现侧面点包含轮廓或遮挡推断，mask不等于解剖真值。量化和错误约束能引入跳动，但不能单独解释此前18.83像素的静态皮肤残差。'
    '新增第二处颞部色素斑的光流往返最大仅0.200像素，但40–48帧光度定位碰到预登记边界，早段材料中心也不够明确，'
    '因此停止在测量阶段，没有运行新的三维拟合。相机残差、模板形状、真实运动与对应误差尚未唯一分离。历史模型保留，灰衣高斯未启动。')
next_text = ('先核查现有RTMW的原始SimCC响应是否可从当前模型独立取得，并核对官方连续峰位解码及坐标变换；'
    '将解码量化和遮挡权重分成两个修复变量。先登记并验证已知峰位/像素变换的有界工程控制，再决定少量原生帧实测。'
    '不把连续坐标当物理真值，不修改原观测或相机，不复跑已失败M1、逐帧姿态或训练。')
con = dict(updated=stamp, summary=summary, visibility_quantization=vis['summary'],
    material_M1=dict(frames=41, tracking_max_forward_backward_px=tr['max_forward_backward_px'],
        photometric_failed_frames=list(range(40,49)), geometry_executed=False,
        center_changes_px=loc['center_changes_px'], conditional_uncertainty_not_physical_truth=True,
        actual_review=str(OUT/'actual-review.json')),
    all41_nose_and_head_reviewed=True, all369_semantic_points_labeled=True,
    mask_is_not_anatomical_truth=True, native_quantization_is_not_physical_error_bound=True,
    no_new_pose_or_camera_fit=True, no_training_or_model_change=True,
    physical_root_cause_uniquely_identified=False, quality_accepted=False, next=next_text,
    sources=[dict(path=str(p), sha256=sha(p)) for p in [
        REC/'grayT-visibility-material-review/result.json',
        REC/'grayT-visibility-material-review/actual-review.json',
        OUT/'plan.json', OUT/'tracking-result.json', OUT/'localization-result.json', OUT/'actual-review.json']])
save(REC/'observation-quality-conclusion.json', con)
save(LOG/'observation-quality-conclusion.json', con)

for base in [REC, R]:
    p=base/'evidence-ledger.json'; d=read(p)
    key='grayT123-observation-quantization-and-M1-stop'
    assert isinstance(d, dict)
    if base==REC:
        assert isinstance(d['entries'], list)
        matches=[x for x in d['entries'] if x.get('id')==key]
        if matches:
            assert len(matches)==1 and matches[0]['summary']==summary
        else:
            d['entries'].append(dict(id=key, updated=stamp, summary=summary,
                evidence=str(REC/'observation-quality-conclusion.json'), quality_accepted=False))
    else:
        assert isinstance(d['confirmed'],list) and isinstance(d['artifacts'],list)
        if summary not in d['confirmed']:d['confirmed'].append(summary)
        source=str(REC/'observation-quality-conclusion.json')
        if source not in d['artifacts']:d['artifacts'].append(source)
    d['updated']=stamp; save(p,d)
q=read(REC/'queue-state.json')
q.update(updated=stamp, state='active_alignment_diagnosis', current_stage='observation-quality-audit-completed',
    training_started=False, training_allowed=False, local_GPU_task_active=False,
    latest_diagnosis=str(REC/'observation-quality-conclusion.json'), next_action=next_text)
save(REC/'queue-state.json',q)
n=read(REC/'next-action.json')
n.update(updated=stamp, next=next_text, next_plan=None, latest_diagnosis=str(REC/'observation-quality-conclusion.json'),
    latest_actual_checkpoint_review=str(OUT/'actual-review.json'), actual_UI_verified=False,
    completed_this_turn=['41nativeframes/369nose observations: read-only visibility and analytical coordinate lattice audit.',
        'One registered second-material LK/localization measurement, failed9frames; stopped before any3Dfit.'],
    quality_accepted=False, training_allowed=False)
save(REC/'next-action.json',n)
q=read(R/'queue-state.json')
q.update(updated=stamp, current_stage='local123_observation_quality_audited', local_GPU_task_active=False,
    last_goal_turn=dict(updated=stamp, classification='progress', result=str(REC/'observation-quality-conclusion.json'),
        summary=summary, next=next_text), next_action=next_text)
save(R/'queue-state.json',q)
c=read(C/'local-research-status.json')
c.update(updated_at=stamp, next=next_text, local_GPU_task_active=False,
    current_local_observation_quality=con)
save(C/'local-research-status.json',c)
lp=T/'shuolee/long-term-plan.md'
with lp.open('a',encoding='utf-8') as f:
    f.write('\n\n### 2026-10-28 原生关键点测量核查\n\n'+summary+'\n\n下一步：'+next_text+'\n')
save(REC/'observation-quality-completion.json', dict(updated=stamp,
    state='diagnostics_completed_publication_pending', next=next_text,
    geometry_or_training_started=False, actual_UI_verified=False, PPT_updated=False,
    quality_accepted=False))
print(summary)
