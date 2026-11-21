"""Persist evidence-based outcome of the two completed, bounded avatar runs."""
import datetime,hashlib,json,shutil
from pathlib import Path
import numpy as np

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');PAIR=B/'current/avatar-repository-pair-114';V=B.parent/'viewer'
BUILD=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=lambda:datetime.datetime.now().astimezone().isoformat()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)

for arm in ['gaussianavatar','smplx']:
    result=read(PAIR/arm/'gaussian-run/result.json');assert result['steps']==18000 and result['from_scratch'] and result['shared_static']
    assert sum(result['per_frame_updates'])==18000 and min(result['per_frame_updates'])==102 and max(result['per_frame_updates'])==103
assert read(PAIR/'gaussianavatar/data/training-inputs.json')['records']==read(PAIR/'smplx/data/training-inputs.json')['records']
assert sha(Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee/avatar_pair_114.py'))=='0c4894e5d1d6c5e64560f558a01dfca736209de3a0dd420cda9f1cdada2db576'

root=PAIR/'smplx';delivery=read(root/'gaussian-run/viewer-export/delivery.json')
assert delivery['points']==215693 and delivery['compatibility_passed']
screens=[BUILD/'avatar114b-openfile-gaussian.png',BUILD/'avatar114b-openfile-points.png']
assert all(p.exists() for p in screens)
save(root/'viewer-ui-review.json',dict(created=stamp(),file_opened_through_UI=True,format_verified=True,source_sha256=delivery['sha256'],observed_count=215693,modes=['Gaussian','points'],screenshots=[dict(path=str(p),sha256=sha(p)) for p in screens],observations='Actual Open File loaded the standalone +Y PLY in Gaussian and points modes. Body topology and orientation match native review; existing facial and silhouette artifacts persist. No additional local deformation attributed to import.',quality_accepted=False,pc00_opened_this_round=False))

comparison=read(PAIR/'comparison/result.json');a=comparison['models']['gaussianavatar']['metrics'];b=comparison['models']['smplx']['metrics']
ref=read(PAIR/'gaussianavatar/evaluation/result.json')['summary']['113']
ratios=dict(head_vs_ga_percent=(b['head_mae']/a['head_mae']-1)*100,foreground_vs_ga_percent=(b['foreground_mae']/a['foreground_mae']-1)*100,head_vs_113_percent=(b['head_mae']/ref['head_mae']-1)*100,foreground_vs_113_percent=(b['foreground_mae']/ref['foreground_mae']-1)*100,alpha_vs_113_percent=(b['alpha_mae']/ref['alpha_mae']-1)*100)
bands={}
for arm in ['gaussianavatar','smplx']:
    rows=[r for r in read(PAIR/arm/'evaluation/result.json')['records'] if r['model']==arm]
    bands[arm]={name:{k:float(np.mean([r['bands'][name][k] for r in rows if r['bands'][name][k] is not None])) for k in ['outside_alpha','inside_missing']} for name in ['head','shoulder-torso','hem-thigh','legs-shoes']}
observed=[PAIR/'comparison'/x for x in ['face-000070.jpg','face-000100.jpg','face-000136.jpg','body-000070.jpg','body-000100.jpg','continuous-review/heads-1.jpg','continuous-review/heads-2.jpg']]
observed += [PAIR/arm/'gaussian-run'/x for arm in ['gaussianavatar','smplx'] for x in ['heads-current.jpg','review-current.jpg']]
observed += [PAIR/'gaussianavatar/evaluation'/x for x in ['novel-above25.jpg','novel-below25.jpg','continuous-review/heads-1.jpg','continuous-review/heads-2.jpg']]
summary='两组均从头完成18,000步。SMPL-X本地适配的实拍视角纹理较清楚，但鼻口、眼镜及身体边界仍有重影，新俯仰视角存在半透明和表面不稳定；GaussianAvatar更平滑但细节和几何偏差明显。两组均未通过重影修复验收。'
decision=dict(created=stamp(),round='114',state='both_trained_exported_UI_reviewed_quality_not_passed',actual_visual_review_completed=True,comparison_complete=True,quality_accepted=False,bounded_experiment_complete=True,broader_reconstruction_goal_achieved=False,summary=summary,
  quantitative_summary=dict(gaussianavatar=a,smplx=b,reference113=ref,relative_percent=ratios,boundary_bands=bands,all_175_are_training_views=True),
  visual_findings=dict(gaussianavatar='六个相机的完整穿衣人体可见，但眼睛/镜框被平滑到头发和面部中；侧面鼻嘴出现重复、前伸轮廓；肩袖和裤腿粗糙。',smplx='六个实拍相机的发丝、镜框、衣服文字与褶皱较清楚；第100帧鼻嘴前方与眼镜外缘仍有重复层/亮斑，第70帧肩袖与头部有外扩，第136/175帧仍见侧面错位。',novel_views='同一150帧路径的16个等间隔及终点抽帧已实际比较。GaussianAvatar较不透明但脸形有误；SMPLX在低俯仰段80—120帧可见脸、衣物透明与混层，侧后段有衣物不稳定。未知视角没有实拍真值。'),
  inspected_images=[dict(path=str(p),sha256=sha(p)) for p in observed],
  continuous_review=dict(decoded_frames=150,reviewed_frame_indices=read(PAIR/'comparison/continuous-review/decode.json')['sampled_frames'],framewise_human_video_review=False,video_sha256=sha(PAIR/'comparison/continuous-ab.mp4')),
  conclusions=['标准世界PLY和+Y导出保留完整位置、协方差及SH；实际打开文件未引入额外肢体变形。重影已存在于原生训练渲染。','SMPLX训练视角的颜色误差较低，但不能据此声称几何正确或新视角问题解决。','这两组使用固定相机和固定粗人体。仅换高斯参数化不能验证或修正实际外参、人体微动及初始拟合残差；本次没有联合优化相机或人体。','本地GaussianAvatar为静态第一阶段适配，统一目标省略LPIPS；SMPLX公开库缺失入口用本地适配补齐。不能外推为作者完整基准结果或单因素因果比较。'],
  action='保留两条最终可复查候选，结束本次已授权对比；不自动追加训练或扫描参数。不将任何一组标成质量最佳。',
  next_analysis_recommendation='下一次若继续，应先定量分离跨帧共同投影偏差与局部观测不一致，再判断相机、人体微动或模型拟合的贡献；不要把颜色误差下降当作空间对齐。',
  slide_title='两条人像路线：重影尚未消除',
  slide_summary='175视角、20万初始点，各自从头训练18,000步',
  slide_observation='SMPL-X实拍纹理较清楚；鼻口、镜框和身体边界仍有重影。',
  slide_conclusion='新俯仰视角仍不稳定；两组均未通过，不追加训练。')
decision['slide_notes']=[summary,
 '比较条件：175张074整图真实注册相机固定，不插值；统一1536高、20万初始点、18,000步、随机种子6060与相同抽帧顺序。每张图参与102或103次更新。两组各自从原始二维检测和官方人体预测重新拟合一个共享静态人体；旧高斯、旧拟合和旧优化器均未加载。',
 'GaussianAvatar采用aipixel/GaussianAvatar公开POP_no_unet第一阶段：UV特征输入神经网络，输出规范空间偏移、RGB及各向同性尺度；通过骨骼线性蒙皮变换到世界。规范点q与偏移delta给出q+delta，每个骨骼变换按权重w加权。这里固定一个姿态，只有网络、UV特征和高斯外观被学习；不是时变网络。固定不透明度1、RGB相当于SH0。初始20万样本来自有效UV texel，结束仍20万点。',
 'GaussianAvatars_SMPLX采用公开网格绑定高斯和明确标注的本地SMPL-X适配。高斯位置mu=c_f+s_f R_f u：c_f是绑定三角形中心，R_f是三角形局部坐标到世界的旋转，s_f是面尺度，u是待优化局部位置。这使点围绕人体三角面学习偏移，不是把点锁死在裸人体表面。颜色为SH3、各向异性协方差、可学习不透明度，保留增密。初始按面面积抽样并覆盖全部20908面，最终215693点。',
 '统一颜色目标为0.8 L1+0.2 SSIM，只用人物mask内部；SSIM窗口必须完全位于内部。另用0.1软mask alpha损失。两组保留方法各自的偏移/尺度等正则。为了统一实验目标，GaussianAvatar官方LPIPS项未开启；静态SMPL-X底模适配、采样和参数化也不同，不能将结果当单一模块消融。',
 '175训练视角：GaussianAvatar全身前景MAE=0.024265、面部MAE=0.033526；SMPLX分别0.008015、0.014135。SMPLX相对GaussianAvatar降低约67.0%和57.8%；相对113参考面部约降低12.4%，但全图alpha误差约增加21.2%。这些是训练视角颜色和覆盖统计，不是真实三维精度。',
 '实际检查六个同相机面部和全身、已解码150帧视频及16个预定连续路径抽帧。SMPLX在训练视角保留纹理较好，但鼻口、镜框仍错位；低角度段出现半透明、混层和衣物不稳定。GaussianAvatar偏模糊且肩袖、鼻嘴偏差明显。两个方法均没有达到无重影要求。未声称逐帧人工播放验收或有未知视角实拍真值。',
 '图源：本轮comparison/face-000100.jpg。左原照片、中GaussianAvatar、右GaussianAvatars_SMPLX。相同相机、像素裁剪、黑色背景和等比例放大，无单独二维对齐。',
 '导出将所有位置、完整协方差和全部球谐系数共同转为+Y，保持世界标准PLY。两组原生与世界PLY渲染对照通过，+Y的少量差异留档为浮点近深度排序/贡献阈值，不称像素完全相同。实际在本机打开文件核验Gaussian和points，两组点数与文件哈希对应。未在00重新部署。',
 '结论边界：训练渲染已有重影，不能归因于单独导出；但公式投影和实现一致也不证明真实估计外参准确。固定相机、固定粗人体及观测不一致仍共同影响结果，本轮不够区分相机误差、微动或拟合残差。',
 '来源：https://github.com/aipixel/GaussianAvatar @ d981c62238ef64e89dcc04719d2ebbb4758b080a；https://github.com/jmaraval/GaussianAvatars_SMPLX @ e44c1a6c5fbb7f76089f14982eba98d94e553214。后者公开库缺train_smplx.py与SMPL-X模型入口，使用本地补齐；不称完整官方复现。证据目录：'+str(PAIR)]
save(PAIR/'decision.json',decision)
for arm in ['gaussianavatar','smplx']:
    r=read(PAIR/arm/'evaluation/result.json');r.update(visual_review_pending=False,quality_accepted=False,quality_review=str(PAIR/'decision.json'));save(PAIR/arm/'evaluation/result.json',r)
comparison.update(visual_review_pending=False,quality_accepted=False,review_completed=True,conclusion=summary);save(PAIR/'comparison/result.json',comparison)
shutil.copy2(PAIR/'comparison/result.json',V/'body-avatar114/comparison/result.json')
save(V/'body-avatar114/comparison/decision.json',decision)
print(json.dumps(dict(decision=str(PAIR/'decision.json'),relative_percent=ratios,quality_accepted=False),ensure_ascii=False))
