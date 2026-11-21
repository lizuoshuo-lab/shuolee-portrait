from pathlib import Path
import json,datetime,hashlib,shutil
import numpy as np
from PIL import Image
BASE=Path(r'\\?\C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
ROOT=BASE/'current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/crown-attribution'
BUILD=Path(r'C:\Users\Administrator\Documents\New project\shuolee-slide-update')
WORK=BUILD.parent
VIEWER=BASE.parent/'viewer'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.datetime.now().astimezone().isoformat()
assert not (ROOT/'motion-review.json').exists()
result=read(ROOT/'measured-patch-pilot/result.json')
assert result['local_control_optimizer_updates']==472 and result['main_preserved']
keepers={
 'main':(ROOT.parent/'run18000/current.pt','676675cc2f38913ee95e00f4ceb2ba1fb7d488bf6a1fbaa711516d0e69411ec2'),
 'clothing':(ROOT.parent/'observed-clothing-pilot/matched-training/independent_dynamic_clothing.pt','0685685a9a47f148e0831c1e65d10c9cc1ef02f94e5d5ac0ce469264c4260284'),
 'viewer':(VIEWER/'models/whole-body-clothing-candidate/point_cloud.ply','338b0816e1bde4a064a49b18d55473fe36b369d89646ae92cc2dc66051e1f844')}
for k,(p,h) in keepers.items():assert sha(p)==h,k
parity=[]
for i in range(1,56):
    a=np.asarray(Image.open(BUILD/f'crown-final-renders/{i:02d}.png'))
    b=np.asarray(Image.open(BUILD/f'motion-final-renders/{i:02d}.png'))
    assert a.shape==b.shape and np.array_equal(a,b),i
    parity.append(i)
final=WORK/'shuolee-slide-output/02_有限视角重建优化路线_20261022_运动对应修订.pptx'
assert sha(final)=='bad81c8a5743df30a50b77acc07717b6e23e938607818813ecd786337e64c7ee'
receipt=read(BUILD/'motion-final-validation.json')
assert receipt['packageIntegrity']['status']=='pass' and receipt['presentationLayout']['finding_count']==0
inspection=(BUILD/'motion-final-inspection.ndjson').read_text(encoding='utf-8')
assert '[DYNAMIC_REAL_MOTION_20261022]' in inspection
canonical=BASE/'reports/02_有限视角重建优化路线.pptx'
assert sha(canonical)=='6b5ecf3bf9c704504744d6cbe4be17bc3ca391acdf7cc68d8ae95d49a9c5f9fb'
shutil.copyfile(final,canonical);assert sha(canonical)==sha(final)
ppt=dict(updated=stamp,slides=55,notes_updated_slide=55,all55_rendered=True,visible_slide_pixel_parity=parity,
         notes_evidence_checked=True,source_sha256='6b5ecf3bf9c704504744d6cbe4be17bc3ca391acdf7cc68d8ae95d49a9c5f9fb',
         final_sha256=sha(final),final_bytes=final.stat().st_size,nativePowerPointVerified=False,
         canonical_path=str(canonical),validated_output=str(final))
save(BUILD/'motion-visual-review.json',ppt)
assets={
 'real-flow-intermediate/real-track-coverage.png':'crown-flow-coverage.png',
 'measured-patch-pilot/pitch-45-comparison.png':'crown-measured-pitch.png',
 'measured-patch-pilot/000023-comparison.png':'crown-measured-source.png',
 'measured-patch-pilot/head-pitch.mp4':'crown-measured-pitch.mp4'}
for src,dst in assets.items():
    shutil.copyfile(ROOT/src,VIEWER/'body-dynamic'/dst)
    assert sha(ROOT/src)==sha(VIEWER/'body-dynamic'/dst)
summary={k:dict(**v['summary'],selected_track_depth_median_mm=v['depth']['76']['median_abs_mm'],
               selected_track_depth_max_mm=v['depth']['76']['max_abs_mm']) for k,v in result['metrics'].items()}
review=dict(updated=stamp,state='reviewed_no_visible_gain_not_adopted',main_training_steps=18000,source_frames=59,
 gaussian_parameter_updates=0,local_control_optimizer_updates=472,steps_per_arm=236,affected_hair_points=800,
 motion_unchanged=True,face_parameters_unchanged=True,background_unchanged=True,main_preserved=True,
 quality_accepted=False,deployed_to_00=False,eyewear_optimization=False,
 decision='原视频中间帧扩充了真实对应，但单片局部深度约束未改善未知俯仰碎片。保留原模型，停止该局部续训，不采用候选。',
 static_rigid_local_all_loo_tracks=[4,7,7],
 flow=dict(direct_tracks=30,direct_all_loo_passed=22,intermediate_tracks=119,intermediate_all_loo_passed=28,
           precise_tracks=3,decoded_frames=176,calibrated_ROIs_pixel_exact=59,added_calibrated_cameras=0),
 metrics=summary,displacements=result['canonical_displacement'],
 rendered_native_RGB_forwards=result['native_rgb_forwards'],depth_forwards=result['added_depth_forwards'],
 actual_visual_review=['measured-patch-pilot/000023-comparison.png','measured-patch-pilot/000045-comparison.png',
                       'measured-patch-pilot/pitch-15-comparison.png','measured-patch-pilot/pitch-45-comparison.png',
                       'real-flow/real-track-coverage.png','real-flow-intermediate/real-track-coverage.png',
                       'local-motion/track-6-reprojection.png','local-motion/track-13-reprojection.png'],
 caveats=['全部59张标定图均参与原模型训练，无独立新视角GT。',
          '三角化深度依赖现有相机和动态模型；像素扰动分析未包括全部系统误差。',
          '透射率加权高斯中心深度不同于真实表面深度。',
          '局部深度差中位数下降但最大差变大，不能宣称所有观测的几何改善。',
          '对应仍集中在少量正面与侧面区域，没有创造新的俯仰观测。'],
 next_experiment='检验头发轮廓、遮挡与协方差支持范围的关系，寻找跨视图一致的几何约束；先做覆盖诊断，不重复已失败的单独方向权重搜索、直接平滑或整发替换。',
 protected_sha256={k:h for k,(_,h) in keepers.items()},ppt=ppt,
 web_url='http://127.0.0.1:8870/body-dynamic.html#motion-evidence',
 web_assets={v:sha(VIEWER/'body-dynamic'/v) for v in assets.values()},browser_verified=False,
 execution_repairs=['Corrected a camera tensor device mismatch before any optimization.',
                    'Corrected the existing video recipe path before any video-based run.'])
save(ROOT/'motion-review.json',review)
page=VIEWER/'body-dynamic.html';html=page.read_text(encoding='utf-8-sig')
assert 'id="motion-evidence"' not in html
html=html.replace('<details><summary>真实特征对应与被排除的调色方案</summary>',
                  '<details id="motion-evidence"><summary>真实视频对应与局部约束对照</summary>',1)
old='<p>38帧获得414个内部特征，形成17条至少4帧的轨迹。相同轨迹的静态三角化通过6条，现有动态补偿通过8条。它们只通过本视频内的一致性检查，尚不足以监督整个头顶，也不是真实深度或独立测试结果。下一步核对局部运动误差并扩展实拍对应。</p>'
assert old in html
new='''<p>17条SIFT候选改为逐个观测留出后，静态相机通过4条，整头刚体与局部三角形运动各通过7条。运动仍由同一训练视频估计，这些对应不能作为独立深度真值。匹配数量和可见次数也不能代替未知俯仰的覆盖。</p>
<figure><a href="body-dynamic/crown-flow-coverage.png"><img src="body-dynamic/crown-flow-coverage.png" alt="六个真实视频观测及通过逐观测几何检查的头发特征位置"></a><figcaption>绿点为实拍光流对应。解码176个原始视频帧，59张标定图的对应区域逐像素一致。中间帧只用于连接跟踪，相机和训练图仍采用原59帧。</figcaption></figure>
<p>直接跟踪得到30条候选、22条通过检查；利用原视频中间帧连接后得到119条候选、28条通过，其中3条在当前像素扰动检查下深度较稳定。对应主要集中在侧面，仅补充了少量正面位置，仍不覆盖整个头顶。</p>
<figure><a href="body-dynamic/crown-measured-pitch.png"><img src="body-dynamic/crown-measured-pitch.png" alt="同一未知俯仰下原动态模型、仅光度约束与加入实拍深度约束的实际渲染"></a><figcaption>同一时刻23、同一未知俯仰。两组各236次局部控制点更新，均只调整附近800个头发点的位置；图中“局部颜色对照”指只使用光度损失，颜色参数始终固定。主轮仍为18,000步。</figcaption></figure>
<p>一处特征的深度差中位数由2.98降至2.01模型毫米，但最大差由4.92升至5.29毫米；头顶碎片没有可见改善。该数值是相对于带运动假设的三角化估计，不是真实深度误差。两组均未采用，原动态模型继续保留。</p>
<figure><video id="measured-pitch-video" controls preload="metadata" poster="body-dynamic/crown-measured-pitch.png" src="body-dynamic/crown-measured-pitch.mp4" aria-label="局部约束的连续俯仰对照"></video><figcaption>固定人物时刻的61帧连续俯仰路径；这段视频检验新视角外观，不表示新增动作观测。</figcaption></figure>
<figure><a href="body-dynamic/crown-measured-source.png"><img src="body-dynamic/crown-measured-source.png" alt="真实正面人物与三个版本的相同训练观测对照"></a><figcaption>头发训练L1为原模型0.01507587、光度对照0.01507424、加入深度约束0.01508431。脸部参数、原人体运动与背景均保持不变。全部59帧为训练资料。</figcaption></figure>'''
html=html.replace(old,new,1);page.write_text(html,encoding='utf-8')
plan=BASE/'long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
old_start='本次增量：已完成原高斯的实际合成贡献回溯'
a=text.index(old_start);b=text.index('<!-- END_CURRENT_HEAD_HAIR_REVIEW -->',a)
newplan='''本次增量：在同一动态模型上保留真实对应，17条内部SIFT轨迹改为逐观测留出检查后，静态通过4条、刚体与局部三角形补偿各通过7条。原视频每3帧取1帧的配方已核实；解码176帧，59张标定图的头部ROI逐像素一致。以真实中间帧连接光流，候选由30条增至119条，严格几何检查通过28条，仍主要在侧面。没有插值新相机或制造新俯仰数据。

在一处误差超出当前容限的特征周围，设置5个平滑控制点影响800个可靠头发高斯。仅光度与附加实拍深度约束各更新236次，共472次；位置最大改变约0.124与0.683模型毫米，原主轮仍18,000步。颜色、协方差、脸部、人体运动与背景固定。该特征深度差中位数2.98降到2.01毫米，但最大差4.92升至5.29毫米；头发L1由0.01507587略升到0.01508431，未知俯仰仍有头顶碎片。该约束来自带运动假设的三角化，不是真值。本候选不采用、不续训、不部署00，保留原模型与可复用实拍对应。

下一步检验头发轮廓、遮挡与协方差支持范围的关系，寻找跨视图一致的几何约束；先做覆盖诊断，不重复已失败的单独方向权重搜索、直接平滑或整发替换。原实拍、相机与1.8万步断点保护不变，眼镜专项停止。

[实拍运动与局部约束对照](http://127.0.0.1:8870/body-dynamic.html#motion-evidence) · 同一55页PPT仅补充技术备注，正文版式不变。
'''
text=text[:a]+newplan+text[b:];plan.write_text(text,encoding='utf-8')
ws=read(BASE/'work-status.json')
ws.update(updated=stamp,local_training_running=False,current_phase='dynamic_real_motion_reviewed_local_patch_not_adopted',goal_accepted=False)
ws['dynamic_measured_motion']=review
ws['latest_report']=ppt
ws['latest_local_execution']=dict(kind='real_video_motion_and_bounded_hair_patch',running=False,
 main_training_steps=18000,local_control_updates=472,rendered_rgb_forwards=result['native_rgb_forwards'],
 depth_forwards=result['added_depth_forwards']+19,geometric_fits=740+1204+2509,
 source_frames=59,original_video_decoded_frames=176,main_preserved=True,eyewear_optimization=False,deployed_to_00=False)
ws['goal_turn_audit']=dict(last_turn_classification='progress',previous_turn_classification='progress',
 consecutive_blocked_turns=0,concrete_progress='Completed740local-motion fits,1204direct-flow fits,2509intermediate-flow fits,19depth checks and472bounded control updates; model not promoted.',
 quality_complete=False)
ws['next_experiment']=review['next_experiment']
save(BASE/'work-status.json',ws)
print(json.dumps(dict(review=review['state'],ppt_sha256=sha(canonical),all55_visible_parity=True,web_assets=len(assets),local_training_running=False),ensure_ascii=False))

