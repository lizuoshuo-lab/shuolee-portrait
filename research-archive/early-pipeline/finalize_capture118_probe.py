from pathlib import Path
import datetime,hashlib,json,shutil
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=TOP/'shuolee/current/new-capture-118';LOG=Path('D:/00/工作日志/2026-10-26/新视频118')
R=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-new-capture-118-support')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
now=datetime.datetime.now().astimezone().isoformat();validation=read(R/'export/validation.json')
quality=dict(updated=now,source_sha256=validation['sha256'],steps=1000,quality_accepted=False,
 summary='实际审阅仍见衣服轮廓光晕、脸部模糊与腿脚破碎；仅缩小高斯会减轻光晕但增加内部缺口，未升级完整训练。',
 same_camera_review_frames=['frame_000001','frame_000018','frame_000036','frame_000053','frame_000071','frame_000089'],
 novel_keyframes_reviewed=[0,9,18,27],novel_frames_generated=36,continuous_full_orbit_acceptance=False,
 observations=['Front/side/back body appearance is learned; the six source images and rendered crops retain identical camera geometry and aspect ratio.',
  'Shoulder, torso and hair boundaries still show broad translucent glow; facial detail is blurred, not accepted as ghosting fixed.',
  'Lower legs and feet are incomplete and splattered in the novel views. Source crop/furniture occlusion remains a separate support issue.',
  'No extra import-time limb deformation observed in local standalone Gaussian and points modes; this is format/coordinate compatibility only.'],
 counterfactual=dict(report='gaussian-footprint-audit.json',variable='Gaussian scale multiplier 1.0 -> 0.5; frozen positions/opacity/SH/cameras.',
  measured_outside_alpha_fraction_full_range=[.1006860584,.1597567052],measured_outside_alpha_fraction_half_range=[.0273818318,.0602762215],
  half_scale_foreground_pixels_under_alpha_half_range=[.0551567862,.1184027281],
  conclusion='Broad footprint is one direct image-level source of halo; shrinking it alone is not a geometry repair.'),
 limitations=['At 1000 steps the configured densification start (10000) has not been reached; not a final verdict on the 18k method.',
  'No independent physical camera calibration or verified full-orbit subject staticity; cannot uniquely attribute ghosting to intrinsics/extrinsics/micro-motion.',
  'The pretrained/body adapter is local SMPL-X code, not a complete official GaussianAvatars_SMPLX reproduction.',
  'New video differs from historical113/114 subjects; their errors are not controlled A/B measurements.'],
 decision='Stop this 1000-step bounded probe as planned, preserve the current reviewable PLY, do not automatically promote to full 18k or best model.',
 next_diagnostic='Measure per-view Gaussian-center/footprint support at shirt contours and occluded legs before choosing one fresh training change.')
save(O/'gaussian-quality-review.json',quality)
screens=[]
for n in ['viewer-openfile-gaussian.png','viewer-openfile-points.png','viewer-openfile-scale-half.png']:
 p=O/n;assert p.is_file();screens.append(dict(path=str(p),sha256=sha(p)));shutil.copy2(p,LOG/n)
ui=dict(updated=now,source_sha256=validation['sha256'],file_opened_through_UI=True,
  browser_url='http://127.0.0.1:8870/',source_path=str(TOP/'viewer/models/new-capture-118-probe/shuolee_118_s01000.ply'),
  modes=['Gaussian','points'],observed_count=200000,gaussian_scale_control_tested=[1.,.5],
  model_upright=True,format_verified=True,quality_accepted=False,pc00_opened_this_round=False,
  screenshots=screens,notes='Actual OpenFile file chooser used; no model library camera was required for standalone geometry. Scale restored to 1.0 after test.')
save(O/'viewer-ui-review.json',ui)
queue=read(O/'queue-state.json');queue.update(updated=now,state='gaussian_probe_reviewed_not_accepted',stage='quality-review',
 pid=None,supervisor_pid=None,steps=1000,training_started=True,training_running=False,
 formal_training_admitted=False,quality_accepted=False,viewer_ui_review='viewer-ui-review.json',
 next_action=quality['next_diagnostic']);save(O/'queue-state.json',queue)
for n in ['gaussian-quality-review.json','viewer-ui-review.json','gaussian-footprint-audit.json','queue-state.json']:
 shutil.copy2(O/n,LOG/n)
for dest in [TOP/'viewer/new-capture-118',Path('D:/00/工作/查看器/new-capture-118'),LOG]:
 (dest/'gaussian-1000').mkdir(parents=True,exist_ok=True);shutil.copy2(R/'export/review/orbit.gif',dest/'gaussian-1000/orbit.gif')
 (dest/'footprint-check').mkdir(exist_ok=True)
 for n in ['frame_000001_scale_1.0.png','frame_000001_scale_0.5.png']:
  shutil.copy2(R/'alignment/gaussian-footprint-audit'/n,dest/'footprint-check'/n)
plan=TOP/'shuolee/long-term-plan.md';text=plan.read_text(encoding='utf-8')
marker='<!-- CAPTURE118_CURRENT_BEGIN -->';end='<!-- CAPTURE118_CURRENT_END -->'
section='''<!-- CAPTURE118_CURRENT_BEGIN -->
## 2026-10-26 · 新视频118当前结果

已用89个真实整图相机完成本轮全新GVHMR先验、两次1200步共享静态SMPL-X拟合，并从零初始化20万高斯完成1000步短程训练。未复用旧人物高斯或优化器；正式18k准入仍未通过。

发现135/228个高权重足部观测落在人物mask外；仅抑制这些观测后从头拟合，589个共同关节观测的P90仍由0.01899增至0.01968个人物框高度，不能称明显拟合改进。人体模板和相机不再以粗拟合分数冒充物理精度。

实际1000步结果：衣服/头部轮廓存在光晕，腿脚破碎。冻结点位和颜色、只把高斯尺度设为0.5后，六视角的轮廓外alpha占比从10.1%–16.0%降至2.7%–6.0%，但前景内部低覆盖像素增至5.5%–11.8%；这是覆盖范围诊断，不是缩放修复。当前尚未达到原设10000步增密阶段，不能据此宣判完整18k方法失败。

已生成+Y PLY，完整协方差/SH旋转与原生回读渲染通过。实际本地打开文件入口的Gaussian、points和高斯球缩放已检查；格式通过、质量未通过，未核验公司00部署。右侧模型为new-capture-118-probe，下载名shuolee_118_s01000.ply。证据统一在current/new-capture-118，资料同步目录D:/00/工作日志/2026-10-26/新视频118；完整环境与预训练权重仍在WPS之外。

下一步先量化衣服轮廓处高斯中心/椭圆范围和遮挡腿部的冲突，再选单一训练变量并从头验证；不把1000步PLY作为完整18k成果，不因颜色损失下降自动加训。
<!-- CAPTURE118_CURRENT_END -->
'''
if marker in text:
 a=text.index(marker);b=text.index(end,a)+len(end);text=text[:a]+section+text[b:]
else:text+='\n'+section
plan.write_text(text,encoding='utf-8')
print(json.dumps(dict(state=queue['state'],ui=True,quality=False,pc00=False)))
