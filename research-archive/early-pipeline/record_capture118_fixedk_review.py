"""Keep one concise camera-candidate ledger and publish evidence without a fake PLY."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/new-capture-118'
W=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.now(timezone.utc).isoformat()
trials=[]
for slug,label in [('fixedk-opencv','初始固定间隔选帧'),('fixedk-opencv-seedrepair','修正初始化选帧'),('fixedk-opencv-observationrepair','逐视角观测验证'),('fixedk-opencv-loopclosure','加入首尾视觉对应')]:
 p=W/('shuolee-new-capture-118-'+slug)
 rec=dict(id=slug,label=label,run_root=str(p),plan=read(p/'plan.json'),new_gaussian_training_started=False)
 if (p/'failure.json').exists():rec['failure']=read(p/'failure.json')
 if (p/'solver-result.json').exists():
  rec['solver']=read(p/'solver-result.json');rec['evaluation']=read(p/'evaluation/result.json')
  rec['observation_provenance']=read(p/'observation-provenance.json')
 if (p/'anatomical-ray-review/result.json').exists():
  rec['anatomical_rays']=[dict(name=x['name'],state=x['state'],fit_views=x['fit_views'],held_views=x['held_views'],methods={name:{k:v for k,v in value.items() if k!='predicted_raw_uv'} for name,value in x['methods'].items()}) for x in read(p/'anatomical-ray-review/result.json')['landmarks']]
 trials.append(rec)
last=W/'shuolee-new-capture-118-fixedk-opencv-loopclosure'
review=dict(updated=now,state='fixed_intrinsic_camera_candidate_rejected',trials=trials,
 summary='固定厂家K的居中裁切针孔假设已用OpenCV从头求解89个外参；补齐逐视角观测验证和首尾对应后优化收敛，但身体、背景留出保护及整体可观测性未通过。未启动新高斯训练。',
 fixed_intrinsics_used=True,raw_D8_applied=False,physical_calibration_verified=False,
 old_pose_or_point_initialization_used=False,gaussian_retrained=False,formal_training_admitted=False,quality_accepted=False,
 confirmed_findings=[
  'The hardcoded 0/8 seed had only 23 direct matches; selecting the largest declared temporal hop with >=160 appearance matches produced a supported 0/4 initialization. This failure was not evidence against factory K.',
  'The new OpenCV implementation originally checked only triangulation anchors before admitting every track observation to BA. Applying the same positive-depth/3px check to every fitting observation fixed convergence without deleting any validation observation.',
  'Adding first8/last8 appearance edges reduced whole-orbit anatomical-ray inconsistency in this new solver. Endpoint poses were not forced equal.',
  'Local face texture residuals below 1px did not establish consistent whole-orbit nose/mouth geometry.',
  'Inspected left-heel detections include calf and image/occlusion boundary locations. Foreground-mask membership does not establish anatomical visibility.'],
 limits=[
  'FISHEYE_8 formula, centered crop and exported video rectification/stabilization remain unverified.',
  'Baseline vs new solve changes both lens model and solver; it does not isolate the physical cause of the previous Gaussian ghosting.',
  'Anatomical label identities and old-template visibility selection are hypotheses, not manually surveyed material ground truth.',
  'No candidate passed all protection gates. No new body fit, Gaussian, native PLY readback or 00 deployment is claimed.'],
 next_action='Keep 118 probe as the labeled historical result. Before another fresh training, validate anatomical visibility/identity at the heel and nose and separate whole-orbit consistency from local track residuals. Do not keep retrying the same fixed-K camera hypothesis or relax the held-out gates.',
 viewer_model_unchanged=True,company00_verified=False)
save(O/'fixedk-opencv-review.json',review)
visual=dict(updated=now,reviewer='assistant',reviewed=[
 dict(path=str(W/'shuolee-new-capture-118-fixedk-opencv-seedrepair/evaluation/review/04_face_5f12c903.png'),finding='Native face/chin appearance patches checked; small local residual does not cover all nose/mouth points.'),
 dict(path=str(W/'shuolee-new-capture-118-fixedk-opencv-observationrepair/anatomical-ray-review/nose_tip_raw_projection.png'),finding='First and final frontal frames show conspicuous drift in the temporal-only fixed-K candidate.'),
 dict(path=str(last/'anatomical-ray-review/nose_tip_raw_projection.png'),finding='Endpoint drift reduced after adding image-based loop edges. Remaining multi-view nose residual is similar to baseline; no ghosting acceptance.'),
 dict(path=str(last/'anatomical-ray-review/left_heel_raw_projection.png'),finding='Several heel labels sit on calf or near image/occlusion edge; original-pixel panels preserve scale. Not a valid visible heel in those reviewed frames.')],
 new_gaussian_reviewed=False,quality_accepted=False)
save(O/'fixedk-visual-review.json',visual)
q=read(O/'queue-state.json');q.update(updated=now,state='fixed_intrinsic_camera_candidate_rejected',stage='fixed-K-global-camera-and-anatomical-validation',pid=None,supervisor_pid=None,training_running=False,new_training_started_this_review=False,formal_training_admitted=False,quality_accepted=False,camera_candidate_running=False,camera_candidate_root=str(last),camera_candidate_report='fixedk-opencv-review.json',next_action=review['next_action'],process_check='All bounded OpenCV processes exited; no body fit or Gaussian task was launched in this review.')
save(O/'queue-state.json',q)
protocol=read(O/'protocol.json');protocol['current_stage']=q['stage'];protocol['fixed_intrinsic_camera_review']=dict(report='fixedk-opencv-review.json',accepted=False,iterations_in_existing_probe=1000,new_training_started=False);save(O/'protocol.json',protocol)
evidence=O/'fixedk-evidence';evidence.mkdir(exist_ok=True)
for name in ['nose_tip_raw_projection.png','mouth_left_raw_projection.png','left_heel_raw_projection.png','right_heel_raw_projection.png']:
 shutil.copy2(last/'anatomical-ray-review'/name,evidence/name)
shutil.copy2(last/'evaluation/result.json',evidence/'held-result.json')
shutil.copy2(last/'anatomical-ray-review/result.json',evidence/'anatomical-result.json')
shutil.copy2(last/'candidate-cameras.json',evidence/'candidate-cameras-not-admitted.json')
for dest in [T/'viewer/new-capture-118/fixedk-evidence',Path('D:/00/工作/查看器/new-capture-118/fixedk-evidence'),Path('D:/00/工作日志/2026-10-26/新视频118/固定内参核验')]:
 dest.mkdir(parents=True,exist_ok=True)
 for f in evidence.iterdir():shutil.copy2(f,dest/f.name)
plan=T/'shuolee/long-term-plan.md';text=plan.read_text(encoding='utf-8')
marker='<!-- CAPTURE118_CURRENT_END -->';assert text.count(marker)==1
start='<!-- CAPTURE118_FIXEDK_REVIEW_BEGIN -->';end='<!-- CAPTURE118_FIXEDK_REVIEW_END -->'
text=re.sub(re.escape(start)+'.*?'+re.escape(end),'',text,flags=re.S)
paragraph='''<!-- CAPTURE118_FIXEDK_REVIEW_BEGIN -->

2026-10-26 固定内参验证已实际执行：采用厂家K的居中裁切针孔假设，固定K，以OpenCV五点法/PnP和固定K束调整从头估计89个外参，不使用旧位姿、三维点或人物模型。FISHEYE_8的8个系数仍未应用，不能称物理标定完成。

先发现并修正不够支持的初始化帧对，再补齐整个拟合轨迹的逐视角正深度/3像素验证；最后只新增首尾图像的外观匹配边，检验环绕漂移。最后候选26次迭代收敛，拟合均值0.385px。相同面部留出46个观测的中位误差0.768→0.499px，但身体76观测0.432→0.542px、背景对应子集0.309→0.385px，且可观测轨迹248→245，故未通过保护门槛。不能把拟合分数或局部面部误差下降当作高斯重影解决。

鼻尖全圈固定分组检验的P90由无首尾约束候选16.865px降至6.363px；旧相机为6.686px，仍未明显解决原问题。鼻尖中位数旧2.816px、新2.876px。实际原图核验发现若干脚跟检测落在小腿或遮挡/画幅边缘，mask内支持不足以保证解剖点正确。以上说明新求解器局部低误差不能代替整体一致性，不证明此前重影只由相机造成。

本次没有新人体拟合或高斯训练；既有1000步PLY及历史模型身份保持不变。完成本轮有界相机核验，不重复扫描同一假设或放宽门槛。下一步先解决鼻/脚跟的真实对应与可见性证据，再决定从头训练。精简证据：current/new-capture-118/fixedk-opencv-review.json及fixedk-visual-review.json。
<!-- CAPTURE118_FIXEDK_REVIEW_END -->
'''
text=text.replace(marker,paragraph+marker);plan.write_text(text,encoding='utf-8')
print(json.dumps(dict(state=q['state'],new_training_started=False,training_running=False,evidence=str(evidence)),ensure_ascii=False))
