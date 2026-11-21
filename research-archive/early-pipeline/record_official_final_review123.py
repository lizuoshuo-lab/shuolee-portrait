"""Record the actually reviewed first-cycle output; never starts training."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil

T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T/'shuolee/current/robust-portrait-122'
REC = T/'shuolee/current/local-clothed-surface-123'
O = REC/'smplifyx-reference-control/official-first-cycle'
F = O/'full94-shared-shape'
V = T/'viewer'
S = Path('D:/00/工作/查看器')
C = Path('C:/Users/Administrator/Documents/New project/shuolee-coordination')
NOW = datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(p)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

validation = read(F/'export/validation.json')
launch = read(F/'gaussian-launch/result.json')
assert launch['steps'] == 18000 and launch['state'] == 'exported_quality_pending'
digest = sha(F/'export/point_cloud.ply')
assert digest == validation['sha256']
snapshot = read(F/'reviews/step-018000-images/index.json')
reference = read(F/'export/final-review-images/reference-index.json')
assert reference['source_sha256'] == digest
summary = '首轮94帧官方SMPLify-X至18000步高斯已完成；正面可辨，头发、袖口、衣摆与腿脚仍有虚边，侧后方破碎，质量未通过。'

review = dict(updated=NOW, step=18000, actual_review=True,
    source_world_ply_sha256=snapshot['world_ply_sha256'],
    viewed_frames=['frame_000001','frame_000019','frame_000038','frame_000056','frame_000075','frame_000094'],
    viewed_regions=['whole_body','face','feet'],
    image_receipt=str(F/'reviews/step-018000-images/index.json'),
    extra_resize=False, image_text_drawn=False, point_count=354682,
    findings=[
        'Final matched-pose snapshots are more settled than the immediately post-densification12000 snapshot, but large hair, sleeve and hem halos remain.',
        'Legs, ankles and shoes retain fuzzy edges. Frame94 shows part of a profile where the photograph is hair-occluded.',
        'These six views use their own named poses; they must not be confused with the single exported frame32 pose.'
    ], quality_accepted=False, all_face_points_below1px=False,
    action='First18k cycle ended. Preserve its result; no automatic new loss, solver, camera or retraining.')
save(F/'reviews/gaussian-step-018000-visual-review.json', review)

quality = dict(updated=NOW, source_sha256=digest, actual_same_camera_review=True,
    reference_frame_id='frame_000032', source_frame=155, native_resolution=[2704,1520],
    same_camera_reference_review=str(F/'export/reference-review/result.json'),
    reference_image_receipt=str(F/'export/final-review-images/reference-index.json'),
    reference_regions_reviewed=['whole_body','face','left_hand','right_hand','feet','alpha'],
    continuous_novel_views_reviewed=36, novel_view_ids=list(range(36)),
    novel_view_review_mode='Six unresized3x2 contact sheets, ordered0..35 at10-degree increments; no novel-view photographs.',
    novel_image_receipt=str(F/'export/final-review-images/novel-index.json'),
    reference_findings=[
        'Same-pose native frame32 has recognizable face, dress and shoes. Facial texture remains speckled; hair is blurred and extended beyond the photograph.',
        'Raised hand and sleeve are diffuse and partly missing. Lower hand is noisy. Hem, shoulders, calves and ankles have substantial translucent extensions.',
        'Alpha inspection confirms the extensions are rendered occupancy, not merely a chosen background colour.'
    ], novel_view_findings=[
        'Views0..23 show severe side/back holes, fragmented surfaces, elongated splats and an unreliable raised arm.',
        'Views24..35 return to the more recognizable front, but hair/shoulder/hem halos and unsupported volume remain.',
        'The continuous orbit is visually unstable; unobserved or weakly observed back surfaces have no matching ground truth.'
    ], quality_accepted=False, gaussian_centers_surface_adherence_accepted=False,
    physical_calibration_accepted=False, independent_face_subpixel_accepted=False,
    full94_images_individually_reviewed=False, format_passed=validation['format_passed'],
    local_adapter=True, upstream_complete_reproduction=False, summary=summary,
    cause_boundary='The failed fit/occlusion observations and broad Gaussian support are evidence, but do not identify a unique camera, correspondence, motion or loss cause.',
    timing=dict(official_full94_fit_seconds=3881.2469,
        gaussian_launch_to_export_seconds=launch['elapsed_seconds'],
        gaussian_time_includes_review_wait=True, fresh_video_end_to_end_measured=False,
        end_to_end_15min_accepted=False),
    local_viewer_UI='pending', official_SIBR_GUI_verified=False,
    independent_Spark_GUI_verified=False, company00_verified=False)
save(F/'export/quality-review.json', quality)

for p in [R/'queue-state.json', REC/'queue-state.json', C/'local-research-status.json']:
    d=read(p)
    d.update(updated=NOW, updated_at=NOW, state='official_first_cycle_exported_quality_failed_UI_pending',
        current_stage='final_delivery_UI', training_running=False, fitting_running=False,
        local_GPU_task_active=False, running=[], worker_pid=None, current_worker_pid_wsl=None,
        worker_command=None, gaussian_steps=18000, goal_active=True,
        quality_accepted=False, next_action='Package completed PLY and94 cameras, actually open Gaussian/points, publish reviewed candidate; then analyze minimum necessary correction.',
        next='No restart of the completed first18k cycle. Goal remains unmet: quality defects and15min target.',
        latest_official_final_review=str(F/'export/quality-review.json'))
    d['current_official_gaussian_run']=dict(root=str(F), state='complete_quality_failed_UI_pending', steps=18000,
        pid=None, source_sha256=digest, points=354682, quality_accepted=False)
    save(p,d)

plan=read(O/'plan.json')
plan.update(updated=NOW, current_step='first_cycle_exported_quality_reviewed',
    next_step='Actual local OpenFile Gaussian/points and paired-camera delivery',
    next='No retraining before the final evidence and minimum correction decision are concrete.')
plan['final_reference_review']=dict(executed=True, result=str(F/'export/reference-review/result.json'),
    actual_visual_review=str(F/'export/quality-review.json'), model_updates=0)
save(O/'plan.json',plan)

ledger=read(R/'evidence-ledger.json')
ledger['official_first_cycle_final']=dict(updated=NOW, result=str(F/'result.json'),
    quality_review=str(F/'export/quality-review.json'), quality_accepted=False,
    steps=18000, points=354682, source_sha256=digest, UI_pending=True)
save(R/'evidence-ledger.json',ledger)

ident='local-surface-123-grass-official-smplifyx-s18000'
section='''<section id="official-gaussian-final"><h2>首轮完成：94帧 SMPLify-X → 18000步高斯 → 静态模型</h2>
<p class="status">训练与导出完成；质量未通过，15分钟目标未达到。当前是可检查的失败候选。</p>
<p>现有mask、原生2D和完整相机 → 官方SMPLify-X五阶段拟合 → GaussianAvatars_SMPLX本地入口适配。GVHMR未参与，共用同一体型；200000点从头训练，最终354682点。静态模型固定在frame_000032（原视频155帧），配套94个相机。</p>
<p>实际查看结果：正面可以辨认，面部仍有斑点，头发、袖口、衣摆、腿脚仍有虚边；侧后方有严重破碎、拉长高斯和错误体积。连续36视角已审阅。不能把训练结束、PLY可读或检测点拟合残差当质量通过。</p>
<p>94帧拟合64.7分钟；高斯启动至导出45.0分钟，含1000步人工审阅等待。使用已有mask与相机，尚不是新视频到导出的端到端计时。</p>
<p><a href="/?model='''+ident+'''">打开本轮静态模型</a> · <a href="/models/'''+ident+'''/scene.zip?download=1">下载点云＋camera完整包</a> · <a href="official-first-cycle/final/quality-review.json">实际质量结论</a></p>
<p>下列静态对照均使用frame32自己的原图、姿态和相机；原生2704×1520的同一整数裁剪，无额外变形、无图内文字。</p>'''
for region,label in [('body','全身'),('face','面部'),('left-hand','左手'),('right-hand','右手'),('feet','双脚'),('body-alpha','全身alpha')]:
    section+=f'<figure><img src="official-first-cycle/final/reference32-{region}.png" alt="第32帧原图与最终{label}对照"><figcaption>{label}：左原图，右最终静态高斯'+('占据率。' if region=='body-alpha' else '。')+'</figcaption></figure>'
section+='<h3>连续转视角</h3><p>每图左至右、上至下，共36视角，每10度一帧。固定同一静态姿态；没有这些角度的同步原图真值，背面不作可靠恢复声明。</p>'
for i in range(6):
    section+=f'<figure><img src="official-first-cycle/final/novel-contact-{i:02d}.png" alt="最终模型连续视角{i*6}至{i*6+5}"><figcaption>视角{i*6}—{i*6+5}；3×2原尺寸拼接。</figcaption></figure>'
section+='<h3>训练视角的同帧姿态对照</h3><p>这六帧使用各自拟合姿态，不是单一静态PLY。原图与渲染在同一整数裁片内对照。</p>'
for frame in [1,19,38,56,75,94]:
    section+=f'<figure><img src="official-first-cycle/final/frame_{frame:06d}_body.png" alt="第{frame}帧原图与18000步同姿态高斯"><figcaption>frame_{frame:06d}：左原图、右该帧姿态高斯。</figcaption></figure>'
section+='</section>'

pages=[R/'video10-global-initializer',V/'diagnostics/video10-global-initializer',S/'diagnostics/video10-global-initializer']
for folder in pages:
    p=folder/'index.html';text=p.read_text(encoding='utf-8')
    if '<section id="official-gaussian-final">' not in text:
        text=text.replace('<section id="full-sequence-fit">',section+'<section id="full-sequence-fit">',1)
    text=re.sub(r'<p id="official-gaussian-phase">.*?</p>', '<p id="official-gaussian-phase">本轮18000步已完成。<a href="#official-gaussian-final">查看最终静态模型与原图对照</a>；质量未通过，未启动下一轮。</p>',text,count=1)
    text=text.replace('SMPLify-X → GaussianAvatars_SMPLX：94帧拟合完成，进入高斯阶段','SMPLify-X → GaussianAvatars_SMPLX：本轮拟合与高斯已完成')
    text=text.replace('现已完成完整94帧拟合，正在同一轮高斯训练。','现已完成完整94帧拟合和同一轮18000步高斯训练。')
    text=text.replace('高斯训练已启动；当前还没有发布新点云。','本轮高斯训练与导出已完成；最终质量结论见上方。')
    p.write_text(text,encoding='utf-8')
    dest=folder/'official-first-cycle/final';dest.mkdir(parents=True,exist_ok=True)
    for name in ['reference32-'+s+'.png' for s in ['body','face','left-hand','right-hand','feet','body-alpha']]+[f'novel-contact-{i:02d}.png' for i in range(6)]:
        shutil.copy2(F/'export/final-review-images'/name,dest/name)
    for frame in [1,19,38,56,75,94]:
        name=f'frame_{frame:06d}_body.png';shutil.copy2(F/'reviews/step-018000-images'/name,dest/name)
    shutil.copy2(F/'export/quality-review.json',dest/'quality-review.json')

for folder in [V,S]:
    p=folder/'latest-result.json';d=read(p)
    d['latestDiagnostic']=dict(updated=NOW, round='123-official-s18000', title='首轮18000步已完成，质量未通过',
        url='diagnostics/video10-global-initializer/index.html#official-gaussian-final',
        report='diagnostics/video10-global-initializer/index.html#official-gaussian-final',
        reportUrl='diagnostics/video10-global-initializer/index.html#official-gaussian-final',
        summary=summary, status=summary, qualityAccepted=False)
    save(p,d)

longplan=T/'shuolee/long-term-plan.md'
with longplan.open('a',encoding='utf-8') as f:
    f.write('\n\n2026-10-30 首轮正式流程完成：94帧官方SMPLify-X（3881.25秒）→现有本地SMPLX高斯适配18000步（启动至导出2702.22秒，含审阅等待）。最终354682点，固定frame32；实际同帧原图、脸/手/脚/alpha和36连续视角检查仍有明显虚边及侧后破碎，quality_accepted=false。PLY完整协方差/SH原生回读通过。开始当前本机OpenFile与94相机实际交付检查；不重启本轮，不以缓存输入宣称15分钟全流程。\n')
print(json.dumps(dict(source_sha256=digest, quality_accepted=False, reviewed_novel_views=36, viewer_UI_pending=True),ensure_ascii=False))
