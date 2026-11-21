from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil

REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
LOG=Path('D:/00/工作日志/2026-10-27/新人物118')
CAL=Path('D:/00/工作日志/2026-10-27/内参文件解析')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(ROOT/'export/validation.json');r=read(ROOT/'gaussian-run/result.json')
assert v['format_passed'] and r['steps']==18000
images=[]
for p in sorted((REC/'fixedk-gaussian-review').glob('*.jpg')):
    images.append(dict(path=str(p),sha256=sha(p)))
assert len(images)==6
review=dict(updated=datetime.now(timezone.utc).isoformat(),steps=18000,
    initial_points=200000,points=v['points'],from_scratch=True,old_checkpoint_loaded=False,
    source_sha256=v['sha256'],quality_review_completed=True,quality_accepted=False,
    formal_training_admitted=False,physical_calibration_accepted=False,
    reviewed_frames=['frame_000001','frame_000018','frame_000036','frame_000053','frame_000071','frame_000089'],
    same_camera_raw_reference=True,identical_crop=True,nonuniform_image_warp=False,
    image_text_burned_in=False,reviewed_novel_views=list(range(36)),novel_view_ground_truth=False,
    summary='本轮已从头完成18000步、239602个高斯；同相机侧脸鼻嘴前方仍有亮色散点及局部重复边缘，衣袖/腰边仍有外扩光晕，下肢与脚部严重破碎。质量未通过，不能认定重影已修复。',
    observations={
      'face':'Frames18/71: detached light/skin fragments at the forehead, nose, lips and chin silhouette. Front views1/89 have recognizable features but softened duplicate eyebrow/eye detail. Head rear views36/53 remain blurred at the silhouette.',
      'body':'Torso and clothing color broadly follow the source. Sleeve/shoulder/waist edges retain halos; the elbow-to-torso gaps contain floaters.',
      'feet':'Raw frames often crop or occlude feet behind sofa/table. Some raised-looking rear-foot positions also exist in the original view, so image height alone is not a geometry verdict. Nevertheless reconstructed lower legs have large holes, scattering and inconsistent surfaces across the full orbit.',
      'novel':'All36 rendered orbit images reviewed. Face/body remain recognizable, but fragmented legs, lower-body gaps and protruding shirt-edge splats persist. No novel-view ground truth or causal quality improvement is claimed.'},
    causal_limits=['Factory K is used under an unverified crop/rectified-export hypothesis; D8 is not applied.',
      'Projection implementation consistency does not prove physical calibration.',
      'Camera error, ambiguous material correspondence, furniture occlusion and subject micro-motion are not uniquely separated.',
      'This run and the old1000-step probe have different inputs/cameras/training lengths; no single-factor improvement attribution.'],
    export=dict(format_passed=True,position_inverse_max=v['position_inverse_max'],
      covariance_inverse_max=v['covariance_inverse_max'],sh_rotation_error=v['sh_basis_rotation_max'],
      max_render_error=max(x['max_abs'] for x in v['native_render_checks']),
      basis=v['coordinate_space'],body_up_residual_degrees=v['body_up_residual_degrees'],
      origin=v['origin_world'],training_geometry_changed=False,
      reason='An exact proper axis permutation without recentering passes the original strict renderer checks. Arbitrary body-root alignment failed sparse per-pixel numeric parity and is retained as a rejected export attempt.'),
    images=images,viewer_ui_review='pending',pc00_deployed=False,
    next_evidence=['FISHEYE_8 projection/inverse-projection definition and coefficient order.',
      'Per-video calibration applicability plus exact crop, rectification, EIS and zoom settings.',
      'Visible physical correspondences and independent evidence on subject staticity/occluded lower legs.'],
    budget_exhausted_for_this_run=True,automatic_additional_training=False)
save(REC/'fixedk-gaussian-quality-review.json',review)
save(LOG/'本轮质量审阅.json',review)
for name in ['fixedk-research-training-result.json','fixedk-export-validation.json','fixedk-export-roundoff-repair.json']:
    if (REC/name).exists():shutil.copy2(REC/name,LOG/name)
note='''2026-10-27 新人物118：固定新内参研究训练
输入：df776c03-91a4_1792572380952.mp4，白衣人物；89张采样原图，非884帧全部注册。
从头人体拟合、200000初始高斯、空优化器；18000步完成，最终239602个高斯。
新点云：点云/shuolee_118_fixedK_s18000.ply（59,422,827字节）。
查看：在shuolee查看器点“打开文件”，选择此PLY；可切换“高斯渲染”和“点云·观察空间分布”。配套模型库另带89个相机，单独PLY不包含相机信息。
图片：过程图片；同相机对比均为左原图、右本轮渲染，使用相同裁剪和等比显示，没有在图上写字。未知视角共36帧，每10度一帧，无实拍真值。
质量结论：未通过。侧脸鼻嘴前方仍有散点/重复边缘，衣袖边界光晕、下肢破碎仍明显，不能称重影解决。
内参：用了厂家K换算后的固定K；居中裁切、已矫正导出仍是假设，8个FISHEYE_8畸变系数尚未应用。完整解析见同日“内参文件解析”目录。
导出：完整协方差、degree3 SH、6个原生相机回读通过；采用精确轴翻转保留原始几何和倾斜，非实测重力定向。人物先验向上方向与+Y尚有约8.58度差，不可把显示倾斜当局部变形。
没有在公司电脑00实际验证本轮；写入同步目录不等于已完成云端同步/部署。
本轮18k预算结束；不自动追加步数或无界扫参。继续改进前应核实镜头模型、视频裁切/防抖，以及遮挡和真实材料对应。
'''
(LOG/'说明.txt').write_text(note,encoding='utf-8')
htmlfile=CAL/'01_内参文件解析与待核实项.html'
text=htmlfile.read_text(encoding='utf-8')
block='<section id="latest-training"><h2>7. 本轮1.8万步实测结果</h2><p>2026-10-27：已从头完成200000初始高斯、18000步训练，最终239602个高斯。六个相机的原图/渲染与36个未知视角已实际审阅。</p><p><strong>质量仍未通过：</strong>侧脸鼻嘴前方散点、衣袖光晕及下肢破碎仍在。提供K不等于该K已适用于导出视频；不能凭训练完成判断标定正确。</p><p>PLY完整协方差、SH和原生渲染一致性通过，采用保持坐标精度的轴变换；没有修改训练几何。详见同日“新人物118”目录的说明、过程图片与点云。</p></section>'
if '<section id="latest-training">' not in text:
    htmlfile.write_text(text.replace('</main>',block+'</main>'),encoding='utf-8')
print(json.dumps(dict(quality_accepted=False,steps=18000,points=v['points'],reviewed_images=len(images)),ensure_ascii=False))
