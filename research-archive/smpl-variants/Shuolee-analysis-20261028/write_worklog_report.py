"""Keep the source gallery beside current, separately labelled diagnostic evidence."""
from pathlib import Path
import datetime,json,shutil
BASE=Path('/mnt/d/Projects/Shuolee-analysis-20261028')
OUT=Path('/mnt/d/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图')
FIG=OUT/'修复对照';FIG.mkdir(exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
metrics=read(BASE/'gray_current/metrics.json');alignment=read(BASE/'alignment-control-summary/report.json')
baseline=read(BASE/'gray-reproduction.json');controls={}
for key,suffix in [('rtmw',''),('fan','-fan')]:
    run=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029')/f'gray-dynamic-face{suffix}'
    comparisons=BASE/f'dynamic-face{suffix}-render-comparison'
    available=[step for step in [1000,6000,12000,18000] if (comparisons/f'step-{step:06d}/report.json').exists()]
    latest=max(available) if available else None
    controls[key]=dict(run=str(run),training=read(run/'status.json'),latest_comparison_step=latest)
    if latest:
        controls[key]['render']=read(comparisons/f'step-{latest:06d}/report.json')
        for source,dest in [('head-comparison.png','头部渲染对照'),('full-comparison.png','完整画面对照')]:
            shutil.copyfile(comparisons/f'step-{latest:06d}'/source,FIG/f'{key}_{latest}步_{dest}.png')
    if (run/'export/validation.json').exists():controls[key]['export']=read(run/'export/validation.json')
    if (comparisons/'canonical-head-novel-views.png').exists():
        shutil.copyfile(comparisons/'canonical-head-novel-views.png',FIG/f'{key}_通用PLY新视角.png')
    all_views=BASE/f'dynamic-face{suffix}-all-views/report.json'
    if all_views.exists():controls[key]['all_view_validation']=read(all_views)
    for source,dest in [('all-frame-render-errors.png','全95帧_渲染误差.png'),('worst-foreground-regressions.png','前景退步帧对照.png')]:
        image=all_views.parent/source
        if image.exists():shutil.copyfile(image,FIG/dest)
for source,dest in [('dynamic-face-control/face-alignment-comparison.png','原检测_逐帧对齐.png'),
                    ('dynamic-face-fan-control/face-alignment-comparison.png','脸部检测_逐帧对齐.png'),
                    ('face-local-detector/detector-comparison.png','脸部观测对照.png'),
                    ('alignment-control-summary/all-frame-alignment.png','全95帧_对齐误差.png')]:
    shutil.copyfile(BASE/source,FIG/dest)
report=dict(updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_frames=6,individual_source_pngs=24,native_wh=[1520,2704],
    baseline_training=baseline['result'],baseline_face=metrics['regions']['face']['used_observations'],
    baseline_body=metrics['regions']['body']['used_observations'],alignment_controls=alignment,training_controls=controls,
    server_results_loaded=False,original_video_changed=False,camera_matrices_changed_in_motion_controls=False,quality_accepted=False,
    cpu_cuda_fix=dict(commit='9225e1da6a18fe6ae8aac63ec48bafd65a3794a1',tests=85,ci='success',published=True),
    limitation='Keypoint residual and training-view image metrics do not establish physical calibration or unseen-view quality.')
if (BASE/'dataset-inventory-current.json').exists():report['dataset_inventory']=read(BASE/'dataset-inventory-current.json')
if (BASE/'arrived-videos-validation.json').exists():report['new_video_decode_validation']=read(BASE/'arrived-videos-validation.json')
if (BASE/'source-invariants.json').exists():report['source_invariants']=read(BASE/'source-invariants.json')
if (BASE/'intermediate-ply-cleanup.json').exists():report['completed_control_cleanup']=read(BASE/'intermediate-ply-cleanup.json')
(OUT/'修复证据.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
training=controls['fan']['training']
progress=f"{training.get('step',training.get('steps',0)):,} / 18,000 步" if training['state']=='training' else '已完成 18,000 步' if training['state']=='completed' else training['state']
sections=[]
for key,title in [('fan','原生分辨率脸部检测 + 逐帧头颈、下颌与表情'),('rtmw','原全身检测器 + 逐帧头颈、下颌与表情')]:
    entry=controls[key];step=entry['latest_comparison_step']
    if not step:continue
    render=entry['render'];mean=render['mean_across_six_views']
    change=1-mean['head_motion']['head_crop_masked_rgb_l1']/mean['static']['head_crop_masked_rgb_l1']
    foreground=1-mean['head_motion']['head_foreground_rgb_l1']/mean['static']['head_foreground_rgb_l1']
    whole=1-mean['head_motion']['whole_foreground_rgb_l1']/mean['static']['whole_foreground_rgb_l1']
    rows=''.join(f"<tr><td>{r['frame']}</td><td>{r['static']['head_crop_masked_rgb_l1']:.4f}</td><td>{r['head_motion']['head_crop_masked_rgb_l1']:.4f}</td><td>{1-r['head_motion']['head_crop_masked_rgb_l1']/r['static']['head_crop_masked_rgb_l1']:+.1%}</td></tr>" for r in render['records'])
    extra=''
    if 'export' in entry:
        extra+='<p>该控制的通用 PLY 已完成导出和六视角数值回读检查。数值一致性只验证格式与坐标变换，不等于新视角画质通过。</p>'
    if (FIG/f'{key}_通用PLY新视角.png').exists():
        extra+=f'<p><a href="修复对照/{key}_通用PLY新视角.png">查看通用 PLY 的 12 个新视角</a>（没有对应真值，包含未充分拍摄的角度，不能用于定量画质验收。）</p>'
    if 'all_view_validation' in entry:
        av=entry['all_view_validation'];crop=av['improvement']['head_crop_masked_rgb_l1'];fg=av['improvement']['head_foreground_rgb_l1']
        extra+=f'<p>全部 {av["frames"]} 个原训练视角回放：头部裁剪整体误差平均降低 {crop["relative_improvement_of_mean"]:.1%}，{crop["improved_frames"]} 帧改善；头部前景平均降低 {fg["relative_improvement_of_mean"]:.1%}，{fg["improved_frames"]} 帧改善、{fg["regressed_frames"]} 帧退步。回读渲染与训练保存的六帧 PNG 完全一致。</p><img src="修复对照/全95帧_渲染误差.png" alt="全部帧的渲染误差"><p><a href="修复对照/前景退步帧对照.png">查看前景误差退步最多的六帧</a>，保留失败细节，不通过删帧改变统计。</p>'
    sections.append(f'''<section><h2>{title}：{step:,} 步</h2>
<p>相同步数、原图、相机、未训练初始点云、抽帧顺序和 18,000 步学习率计划；均从空优化器开始。</p>
<p>六个预先固定的训练视角：头部裁剪整体误差降低 <strong>{change:.1%}</strong>；头部前景误差降低 {foreground:.1%}；全身前景误差降低 {whole:.1%}。负数表示变差，不能只看平均数。</p>
<img src="修复对照/{key}_{step}步_头部渲染对照.png" alt="相同步数头部对照"><p><a href="修复对照/{key}_{step}步_完整画面对照.png">打开完整画面对照</a></p>
<details><summary>逐帧头部裁剪误差</summary><table><tr><th>帧</th><th>原版</th><th>本控制</th><th>降低</th></tr>{rows}</table></details>{extra}</section>''')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SMPL-X 对齐检查与修复验证</title>
<style>body{{max-width:1150px;margin:32px auto;padding:0 20px;font:17px/1.65 system-ui;background:#f7f9fa;color:#18222b}}h1{{font-size:28px}}h2{{font-size:21px}}section{{background:white;border:1px solid #dae0e5;border-radius:10px;padding:22px;margin:24px 0}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dce3e8;text-align:left}}img{{width:100%;height:auto}}a{{color:#17628c}}.status{{background:#fff4da;padding:14px;border-radius:8px}}summary{{cursor:pointer}}</style>
<h1>SMPL-X 对齐检查与修复验证</h1><p><a href="查看原图.html">打开六组独立原图</a> · <a href="修复证据.json">查看完整数值证据</a></p>
<p class="status">本地灰衣 video3 原版已从头完成 95 帧、18,000 步及导出。脸部专用检测控制：{progress}。对齐已明显改善，重建质量仍未认定合格。</p>
<section><h2>原图交付</h2><p>六帧各自独立文件夹，每帧包含完整原图、检测点完整图、SMPL-X 投影完整图和单帧脸部对照，共 24 张 PNG。完整画面为原生 1520 × 2704 像素，未修改源视频或对人像进行图像形变。</p></section>
<section><h2>已确认的问题</h2><p>原版全帧共用一套静态人体姿态，没有优化逐帧下颌和表情。固定原相机复算后，脸部投影明显错位；第 95 帧脸部残差中位数约 26.08 像素。帧顺序与投影公式复算一致，没有发现简单的帧错配或投影坐标计算错误。</p>
<p>原准入检查只看身体关键点的粗误差，本片允许的身体中位误差约 29–31 像素，因此任务能够完成，却仍有脸部错位和渲染拖影。实际渲染支持逐帧面部建模能够改善该片；尚不能把剩余问题唯一归因于人物运动或相机外参。</p></section>
<section><h2>固定背景相机的对齐验证</h2><table><tr><th>模型</th><th>脸部残差中位数</th><th>同一皮肤标记留出误差</th></tr>
<tr><td>原版共享静态姿态</td><td>10.50 像素（RTMW） / 10.05 像素（FAN）</td><td>18.66 原生像素</td></tr>
<tr><td>逐帧颈部与头部旋转</td><td>2.30 像素（RTMW）</td><td>6.06 原生像素</td></tr>
<tr><td>逐帧颈部、头部、下颌与表情；原检测器</td><td>1.36 像素（RTMW）</td><td>5.12 原生像素</td></tr>
<tr><td>同样逐帧模型；原生脸部检测器</td><td>1.04 像素（FAN）</td><td>2.42 原生像素</td></tr></table>
<p>脸部指标覆盖全部 95 帧、2,752 个有效观测，使用 899 × 1600 分析域。两个检测器不是同一套真值，表中标明目标来源；下面每个子图对所有模型使用同一目标。</p>
<p>沿用的有效权重只包含鼻部与嘴部，眼睛和眉毛未参与此拟合。这里的残差降低不代表眼部或眨眼重建已验收。</p>
<p>皮肤轨迹使用 1520 × 2704 原生域。6 个定位观测只用于定位该轨迹的三维点，另 35 个观测用于留出验证；均未用于拟合头部或表情。它仅覆盖一条局部轨迹，后续高斯训练会看到原始视频，不能称为独立的新视角画质测试。</p>
<img src="修复对照/脸部检测_逐帧对齐.png" alt="脸部观测和原版、逐帧投影"><p>人体形状、根位姿、背景相机与原始 RGB 保持固定。</p>
<img src="修复对照/全95帧_对齐误差.png" alt="全部95帧同目标误差"></section>
{''.join(sections)}
<section><h2>相机与观测核查</h2><p>脸部专用检测器与原全身检测器的观测相差中位 2.64 像素。其模型在隔离诊断目录运行，未替换公共部署依赖。<a href="修复对照/脸部观测对照.png">查看检测器对照</a>。</p>
<p>背景内参和相机联合微调仅使留出背景轨迹中位误差从 0.328 降到 0.313 像素，且在计算预算内未收敛，未用于修复训练。背景误差小或栏杆方向看齐，都不能单独证明头部附近相机外参准确；仍缺少充分的独立物理对应约束。</p></section>
<section><h2>工程与验证范围</h2><p>CPU 人体检测导致后续 CUDA 不可见的修复已发布：提交 <code>9225e1d</code>，85 项测试及线上工程检查通过。逐帧脸部改动仍在本地验证，尚未发布到部署流程，未放宽质量门槛。</p>
<p>服务器 10 个任务完成 6 个是用户报告的任务数。本地最初只有 8 个源视频；10 月 29 日重新检查时，video9 和 video10 已同步，当前共 10 个。本次新增完整复现的是同源灰衣 video3，并复查过另两份横、竖屏本地完整结果。新到的两段视频未完成本地训练，也没有读取服务器全部模型，未声称解决了所有失败任务。</p>
<p>新到的两段视频均可完整解码。video10 的空输出检查出现时间戳取整警告，但其全部 860 帧原始时间严格递增；这不是输入帧时间倒退或文件损坏的证据。</p>
<p>原视频、原模型、原拟合和完整成功结果均保留；失败预处理目录约 364 MB 与重复下载片段已清理。工作日志保存图像与必要验证证据。</p></section></html>'''
if 'completed_control_cleanup' in report:
    size=report['completed_control_cleanup']['bytes_removed']/1e6
    page=page.replace('工作日志保存图像与必要验证证据。',f'完成最终检查后另清理 {size:.1f} MB 中间点云与重复图，三份控制的最终点云、各步渲染、日志与校验值保留。工作日志保存图像与必要验证证据。')
(OUT/'修复进展.html').write_text(page,encoding='utf-8')
index=OUT/'查看原图.html';body=index.read_text(encoding='utf-8')
if '修复进展.html' not in body:
    body=body.replace('<h1>灰衣 SMPL-X 对齐原图</h1>','<h1>灰衣 SMPL-X 对齐原图</h1><p><a href="修复进展.html">查看对齐分析与修复验证</a></p>');index.write_text(body,encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'修复进展.html'),training=progress,quality_accepted=False),ensure_ascii=False))
