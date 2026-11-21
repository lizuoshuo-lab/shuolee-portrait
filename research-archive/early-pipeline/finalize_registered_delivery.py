"""Record verified072 delivery and073 diagnostic limits; preserve model bytes."""
from pathlib import Path
import datetime,hashlib,json,shutil
import numpy as np
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
V=B.parent/'viewer';U=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
R=B/'current/person-static-registered-camera-fresh-072';O=B/'current/pixel-mapping-audit-073'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)

# Validate immutable outputs and finish the genuine independent-import receipt.
rows=read(R/'data/training-inputs.json')['records'];meta_path=V/'model-whole-body-static-registered-camera.json'
meta=read(meta_path);receipt=read(R/'gaussian-run/viewer-export/delivery.json')
T=np.asarray(receipt['world_to_viewer']['rotation']);o=np.asarray(receipt['world_to_viewer']['origin_world'])
assert len(rows)==len(meta['cameras'])==175
assert sha(V/meta['model'])==receipt['sha256']==meta['sha256']
rng=np.random.default_rng(72);maximum=0.
for row,cam in zip(rows,meta['cameras']):
    E,K=np.asarray(row['w2c']),np.asarray(row['K'])
    assert (row['source_frame_index'],row['frame'],row['camera_kind'])==(cam['raw_source_frame'],cam['img_name'],cam['camera_kind'])
    Kv=np.array([[cam['fx'],0,cam['width']/2],[0,cam['fy'],cam['height']/2],[0,0,1]])
    np.testing.assert_allclose(K,Kv,atol=1e-10,rtol=0)
    xc=rng.uniform(-1,1,(64,3));xc[:,2]=rng.uniform(1,10,64)
    xw=(xc-E[:3,3])@E[:3,:3];xv=(xw-o)@T.T
    nv=(xv-np.asarray(cam['position']))@np.asarray(cam['rotation'])
    a=xc@K.T;b=nv@Kv.T
    maximum=max(maximum,float(abs(a[:,:2]/a[:,2:]-b[:,:2]/b[:,2:]).max()))
assert maximum<1e-8
save(R/'gaussian-run/viewer-camera-roundtrip.json',dict(checked_at=now,cameras=175,synthetic_rays_per_camera=64,
    max_projection_difference_px=maximum,passed=True,training_camera_counts=dict(calibrated=59,registered=116,interpolated=0),
    claim_boundary='Coordinate and metadata consistency only; not accuracy of estimated camera parameters'))
screens=[U/'registered-072-viewer-gaussians.png',U/'registered-072-viewer-points.png']
assert all(p.is_file() and p.stat().st_size>10000 for p in screens)
visual_path=R/'review/visual-decision.json';visual=read(visual_path)
visual.update(local_independent_ply_import='passed',local_viewer_checked_at=now,local_viewer_points=323174,
    local_viewer_modes=['gaussian','point_cloud'],local_viewer_screenshots=[str(p) for p in screens],
    quality_accepted=False,pc00_opened_this_round=False,
    next_step='Independently verify face image correspondences and camera estimation; do not infer calibration accuracy from metadata parity.',
    viewer_status='072：鼻口与眼镜仍有多重几何，质量未通过；相机外参与面部对应关系复核中。',
    report='body-smplx.html#camera-estimation-current')
save(visual_path,visual)
registry_path=B/'gaussian-deliveries.json';registry=read(registry_path)
registry['rounds']['072'].update(local_standalone_open_verified=True,local_gaussian_and_centers_verified=True,
    local_viewer_proof=[dict(path=str(p),sha256=sha(p)) for p in screens],quality_accepted=False)
registry['updated']=now;save(registry_path,registry)

# Replace the one canonical deck, only if it is still the version we edited.
deck=read(U/'registered-deck-receipt.json');validation=read(U/'registered-validation.json')
src,dst=Path(deck['finalPath']),Path(deck['source'])
assert sha(src)==deck['final_sha256']==validation['finalSha256']
assert validation['packageIntegrity']['status']=='pass' and validation['firstPartyImport']['passed']
assert sha(dst)==deck['source_sha256'],'Canonical changed after editing; do not overwrite.'
assert sum(not r['changed'] for r in deck['renders'])==49
review=dict(reviewed_at=now,final_sha256=sha(src),actual_render_inspection=True,slides=52,
    inspected_changed_slides=[20,51,52],render_files=[str(U/f'registered-final-{s}.png') for s in [20,51,52]],
    unchanged_slide_render_hashes=49,passed=True,native_powerpoint_verified=False,pc00_deployed=False)
assert all(Path(p).is_file() for p in review['render_files'])
save(U/'registered-visual-review.json',review)
shutil.copy2(src,dst);assert sha(dst)==deck['final_sha256']
deck.update(canonical_replaced=True,canonical_replaced_at=now);save(U/'registered-deck-receipt.json',deck)
manifest_path=B/'reports/report-manifest.json';manifest=read(manifest_path)
report=next(x for x in manifest['reports'] if x['key']=='reconstruction-optimization')
report.update(sha256=sha(dst),bytes=dst.stat().st_size,slides=52,
    local_validation=str(U/'registered-validation.json'),local_visual_review=str(U/'registered-visual-review.json'),
    latest_scope='逐帧相机072从头训练结果、同相机对照与未通过的边界和新视角质量；相机准确性仍在排查',
    current_local_report_deployed_to_00=False)
manifest['updated']=now;manifest['last_local_revision']=dict(path=str(dst),sha256=sha(dst),bytes=dst.stat().st_size,
    slides=52,remote_delivered=False,transfer_performed=False,local_review=str(U/'registered-visual-review.json'),updated=now)
save(manifest_path,manifest)

# Preserve confirmed diagnostic findings without prematurely naming a root cause.
camera=read(O/'camera-parameters/result.json');face=read(O/'face-rays/result.json')
ba=read(O/'joint-calibration/result.json');vf=read(O/'variable-focal/result.json')
support=read(O/'multiview-face-support/result.json')
decision=read(O/'causal-decision.json')
decision.update(updated=now,diagnosis_completed=False,root_cause_established=False,
    camera_accuracy_accepted=False,new_training_ready=False,
    confirmed=[
        '175 source-to-training pixel and matrix associations pass; this tests implementation, not estimated calibration accuracy.',
        'Frozen model and independent PLY show the duplicate face; delivery parity passes, so import alone does not explain the native ghost.',
        'Earlier fixed-map feature audit contained a3px anchor-reprojection prefilter; its low residual cannot certify original camera accuracy.',
        'Fresh1031point joint BA without that extra prefilter lowers held lower-face feature median1.69774to1.31541px with pose+point refinement; shared f/principal/k1 adds no material benefit.',
        'Reviewed noisy FAN nose/lip rays give blocked-held median5.21482px andp9014.57193px; fixed-map camera proposal does not improve them.',
        'Additional per-frame focal candidate yields1.31124px vs1.31756px shared-intrinsic BA, hits60evaluation budget, and is not accepted.',
        'At final snapshot, face geometry-mask support exists in at least one neighbor for>99.8percent of observed interior pixels in three views; complete lack of this support is not supported.',
        '2770 exterior contributors reduce alpha from0.46093to0.26550 when temporarily hidden; no delivered point is deleted.'
    ],
    next_required=[
        'Build face correspondences directly from original image texture with forward-backward and occlusion checks; exclude misleading glasses reflections and detector-only landmarks.',
        'Use those observations to distinguish external-pose errors from correspondence bias and model inconsistency, including block-held reprojection and body/background protection.',
        'Only after a camera hypothesis produces a clear independent improvement, preregister one fresh Gaussian control and verify whether duplicate nose/mouth geometry disappears.'
    ])
for relative in ['camera-parameters/result.json','face-rays/result.json','joint-calibration/result.json','variable-focal/result.json','multiview-face-support/result.json']:
    p=str(O/relative)
    if p not in decision['evidence']:decision['evidence'].append(p)
save(O/'causal-decision.json',decision)
asset=V/'body-camera-current'
for name in ['landmark-rays-000070.jpg','landmark-rays-000100.jpg','landmark-rays-000136.jpg']:
    shutil.copy2(O/'face-rays'/name,asset/name)
shutil.copy2(O/'multiview-face-support/face-support-000100.jpg',asset/'face-support-100.jpg')
user_image=Path('C:/Users/ADMINI~1/AppData/Local/Temp/codex-clipboard-23bce962-9e06-40d1-8b0d-bedf69ff053b.png')
if user_image.is_file():shutil.copy2(user_image,asset/'reported-face-duplication.png')
page=V/'body-smplx.html';html=page.read_text(encoding='utf8')
anchor='<h3 id="pixel-mapping">'
assert html.count(anchor)==1
addition='''<h3 id="camera-estimation-current">鼻口重影：相机估计尚未通过面部核验</h3>
<p><strong>当前点云仍有分离的鼻口和眼镜结构，不能作为合格结果。</strong>175帧的图像、mask、矩阵读取与缩放对应正确，不等于相机内参、外参估计准确。本轮继续冻结高斯排查，没有启动新的训练。</p>
<figure><a href="body-camera-current/reported-face-duplication.png"><img src="body-camera-current/reported-face-duplication.png" alt="用户指出的多层面部和分离眼镜结构"></a><figcaption>实际查看器中的点云，作为当前失败现象保留。点云显示与原生渲染中的重影都需要解决。</figcaption></figure>
<p>外参仍是优先假设。去掉原先“原相机重投影误差小于3像素”的额外筛选后，使用1,031条原图特征轨迹，重新三角化三维点并进行联合束调整。只放开位姿与点位置，留出的鼻口区域纹理误差中位数由1.70降至1.32个原图像素；再放开共享焦距、主点与径向畸变，仍约1.32像素。逐帧焦距候选也只达到1.31像素，且在60次评估预算内未收敛。它们都是诊断候选，未替换训练相机，也未解决多层脸。</p>
<p>另一项检查直接使用鼻尖、鼻底和上下唇中点：将连续10个观测帧留出，用其余帧拟合一个固定三维点，再投影到留出帧。当前相机的误差中位数5.21像素、90分位14.57像素；此前固定地图的位姿微调没有改善。这些是经过可见性审阅的FAN检测候选，并非人工精确标注，不能把全部误差直接归给相机。</p>
<figure><a href="body-camera-current/landmark-rays-000070.jpg"><img src="body-camera-current/landmark-rays-000070.jpg" alt="真实鼻口观测与留出连续帧的三维重投影"></a><figcaption>蓝圈：原图上的检测观测。红十字：不使用该连续帧组拟合的三维点投影。左为当前相机，右为固定地图的相机微调候选；同一裁片，没有为展示重新对齐。</figcaption></figure>
<p>下一步先从原图纹理建立可核验的面部跨帧对应，检查双向跟踪、遮挡和眼镜反光，再分别验证外参、内参和观测偏差。最终须通过同一鼻口结构的跨视角一致性与从头训练对照，才能认定原因和修复成立。暂不恢复动态网络、添加眼镜模型或增加训练步数。</p>
<details><summary>本轮诊断的适用范围</summary><p>联合束调整仍基于原SfM保留下来的图像轨迹；虽然去掉了额外的3像素筛选，但不能视作完全独立的真实标定。最终快照中三个面部区域超过99.8%的内部像素至少有一个邻居通过几何一致性遮挡门槛，因此“脸部完全没有多视图约束”不受该检查支持；它仍不证明对应到了同一个真实鼻子，也不代表NCC全部通过。</p></details>
'''
html=html.replace(anchor,addition+anchor,1)
html=html.replace('第100帧可靠头部纹理匹配的位移中位数为0.42个训练像素，不能用统一平移解释外围重影。','第100帧可靠头部纹理匹配的位移中位数为0.42个训练像素；该统计不包含未匹配的重影像素，不能据此认定整张脸已经对齐。')
page.write_text(html,encoding='utf8')
status=read(B/'work-status.json');status['updated']=now
status['current_phase']='073 camera-estimation diagnosis unfinished;072 geometry rejected'
status['latest_local_execution']=dict(running=False,state='bounded_diagnostics_finished_root_cause_unresolved',directory=str(O),training_steps=0)
status['latest_completed_static_delivery'].update(local_viewer_passed=True,quality_accepted=False,
    camera_export_check=str(R/'gaussian-run/viewer-camera-roundtrip.json'),review=str(visual_path))
status['iteration_policy'].update(new_training_ready=False,read_only_diagnosis=str(O/'causal-decision.json'),
    required_next_protocol='Independently verified face correspondences and camera-estimation validation before a single fresh Gaussian test')
status['latest_report']=dict(path=str(dst),sha256=sha(dst),changed_slides=deck['changed_slides'],slides=52,updated=now,remote_deployed=False)
status['pending_report_delivery']=dict(status='local_updated_remote_deferred',source=str(dst),file=dst.name,sha256=sha(dst),bytes=dst.stat().st_size,remote_delivered=False)
status['camera_estimation_diagnosis']=dict(root_cause_established=False,exterior_pose_error_still_priority=True,
    shared_intrinsics_not_proven_correct=True,nose_mouth_correspondence_needs_independent_validation=True,
    source=str(O/'causal-decision.json'),new_training_started=False)
save(B/'work-status.json',status)
plan=B/'long-term-plan.md';text=plan.read_text(encoding='utf8');mark='<!-- CAMERA_REGISTRATION_CURRENT_END -->'
text=text.replace(mark,'\n073后续相机诊断：输入/矩阵映射通过不代表估计精度通过。去掉旧3px锚点筛选、重新三角化1031条轨迹并联合调整位姿/点，鼻口区域留出特征中位误差1.69774降至1.31541原图像素；增加共享内参/畸变仍1.31756，逐帧焦距候选1.31124且未在60评估内收敛。独立鼻口检测点的连续帧留出误差中位5.21482、p9014.57193px，检测点本身亦有偏差。尚未找到能消除重影的根因修复，下一步优先从原图建立可靠面部跨帧对应，再检验外参；不把背景小误差或输入配对正确当作人脸已对齐。暂不新增训练。\n'+mark,1)
plan.write_text(text,encoding='utf8')
# Pointer last, after the model, cameras, status and diagnostics are all consistent.
meta.update(status=visual['viewer_status'],report=visual['report'],qualityAccepted=False,localStandaloneVerified=True)
save(meta_path,meta)
pointer=read(V/'latest-result.json');assert pointer['round']=='072' and pointer['sha256']==receipt['sha256']
pointer.update(revision=sha(meta_path),updated=now,report=visual['report'],qualityAccepted=False);save(V/'latest-result.json',pointer)
print(json.dumps(dict(registered_round='072',camera_roundtrip_max_px=maximum,local_import_verified=True,
    root_cause_established=False,quality_accepted=False,canonical_ppt_sha256=sha(dst)),ensure_ascii=False))
