from pathlib import Path
import json, datetime, hashlib, re, shutil

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/portrait-alignment-open-source-116';V=B.parent/'viewer';D=V/'body-camera116'
BUILD=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
capture=read(O/'capture-provenance.json');assert capture['source_unchanged'] and capture['wire_parse_complete']
assert capture['timing_check']['field_3_1_1_minus_first_vs_video_pts_max_abs_ticks']==0
assert read(B/'current/avatar-repository-pair-114/queue-state.json')['state']=='both_complete_actual_ui_and_quality_review_finished'

sources=read(O/'research-sources.json')
assert len(sources['sources'])==4 and all(s['pinned_commit'] for s in sources['sources'])
candidates=[
 {'id':'P1','name':'PixSfM','repository':'https://github.com/cvg/pixel-perfect-sfm',
  'purpose':'先细化已匹配二维特征的位置，再用特征一致性优化场景点与相机；与109单纯几何BA区分。',
  'entry':'pixsfm.refine_colmap: keypoint_adjuster / bundle_adjuster',
  'license':'Apache-2.0; pretrained components checked separately',
  'pilot':'先只细化特征、固定074相机评测，再独立测试相机/点联合细化。两步分别保存结果，不把组合收益归给单一变量。',
  'risk':'官方依赖COLMAP3.8及较老pycolmap/Ceres接口；在隔离环境构建，不降级当前训练环境。错误对应与微动不会自动消失。',
  'local_runtime_verified':False},
 {'id':'P2','name':'RoMa + 跨视角轨迹验证','repository':'https://github.com/Parskatt/RoMa',
  'purpose':'用稠密图像匹配核对鼻嘴、衣物纹理的实际材料对应，验证同一三维点能否解释多视角。',
  'entry':'romatch.roma_outdoor; match / sample / to_pixel_coordinates',
  'license':'MIT repository code; DINOv2 and weights retain their own terms',
  'pilot':'先在少量左右侧及正面重叠图像做双向/三视角循环和原像素检查；通过后覆盖175帧，独立三角化与相机优化。',
  'risk':'它是匹配器，不是人体或相机重建系统；需显式接入BA。高置信度、同名关键点及循环一致都不是物理对应的充分证明。',
  'local_runtime_verified':False},
 {'id':'P3','name':'HSfM','repository':'https://github.com/hongsukchoi/HSfM_RELEASE',
  'purpose':'联合人体、场景与相机全局坐标，作为P1/P2未通过后的结构性对照。',
  'entry':'align_world_env_and_smpl_hsfm_optim.py; run_hsfm.sh',
  'license':'MIT repository code; SMPL/SMPL-X/HMR2/DUSt3R and other dependencies checked separately',
  'pilot':'先复核现有数据能否满足其多视角同一人体状态假设；从公开模板与原始观测重新拟合，不继承旧个体形状。',
  'risk':'顺序拍摄视频并非同步多相机；身体先验不等于衣物表面，也不保证眼镜与脸部对齐。不能为适配它恢复动态主线。',
  'local_runtime_verified':False},
 {'id':'reference','name':'TrackNeRF','repository':'https://github.com/Wayne-Mai/traf_public',
  'purpose':'参考全局特征轨迹重投影约束：每条可见材料轨迹关联一个共享三维点。',
  'entry':'source/training/core/base_track_loss.py',
  'license':'Apache-2.0 repository code',
  'pilot':'本轮只做源码参考，不另开NeRF训练。移植到3DGS时明确标注本地适配。',
  'risk':'官方实现为NeRF，不可直接输出当前shuolee查看器所需的高斯PLY。',
  'local_runtime_verified':False}]
save(O/'candidates.json',candidates)
protocol={
 'id':'portrait-alignment-open-source-116','created':now,'status':'research_and_original_metadata_audit',
 'objective':'定位并改善整身重影，重点区分采集相机信息、估计标定误差和跨视角材料映射；先对齐验收，再从头训练。',
 'source_video':capture['video'],'source_sha256':capture['video_sha256'],
 'active_data':'175 real raw frames 1..175; full-frame 074 reconstruction; no interpolated camera substitution',
 'parent_diagnosis':str(B/'current/camera-feature-root-cause-115'),
 'not_established':['Physical correctness of074 K/R/t','All named facial detections correspond to the same material point','Single cause of ghosting','hovr fields contain usable original K/R/t'],
 'capture_priority':{
   'fact':'hovr track with179 messages;177 repeated records, numeric frame indexes0..176 and clocks agree exactly with video PTS after origin subtraction.',
   'not_fact':'The remaining unnamed fields have not been decoded as intrinsics, poses, IMU, exposure or stabilization.',
   'next':'Find official schema/SDK or originaltdsp export; compare decoded camera model against actual exported pixels, timestamps, distortion/stabilization and coordinate conventions.',
   'bounded_search':'One targeted schema/sidecar investigation; if unavailable retain unresolved status and continue independent image alignment without invented calibration.',
   'preserve_original':True},
 'fixed_conditions':[
   'Shared static clothed person; no per-frame deformation network or generated training views.',
   'Do not warp, independently center or stretch portraits to conceal misalignment.',
   'Freeze175 real acceptance image identities, masks, source-pixel coordinate chain and canonical evaluation views.',
   'Copy source COLMAP database and reconstruction before any refinement; always pass a separate output path.',
   'No change to running source code; inspect process command/PID/progress before launching. At most one GPU workload.',
   'No data upload to external services. Local first; reuse compatible official detectors/templates, not old subject fits.',
   'Raw/world camera changes require a single recorded gauge transform, including both cameras and points.'],
 'stages':[
   {'id':'S0','task':'Original capture metadata + open-source entry/license/provenance audit','state':'partial_done_schema_pending'},
   {'id':'S1','task':'Pre-register a small diagnostic protocol per candidate; verify original-image correspondence, round trips, visibility, failure counterexamples and budget','state':'next'},
   {'id':'S2','task':'Run bounded P1, then P2, then P3 only if needed; assess all175 views and immutable reference observations','state':'not_started'},
   {'id':'S3','task':'Only for an accepted alignment candidate, freshly fit/init/train the two user-approved avatar routes with identical inputs','state':'gated'},
   {'id':'S4','task':'Same-camera whole body/face and continuous novel view comparison; world/+Y PLY, native readback, actual OpenFile Gaussian+points UI checks; publish each completed round','state':'gated'}],
 'diagnostic_gate':{
   'projection_round_trip_max_training_px':0.02,
   'required_final_frame_count':175,
   'lower_face_independent_texture_median_reduction_fraction':0.20,
   'lower_face_independent_texture_p90_must_not_worsen':True,
   'background_held_p90_max_increase_source_px':0.2,
   'torso_and_legs_held_p90_max_increase_source_px':0.5,
   'no_selective_dropping':'Report coverage, visibility and errors on the same acceptance observations. Do not discard high error points using candidate cameras.',
   'selection_vs_test':'Old115/101 held observations have already informed selection: label them validation, not an unseen test. Reserve new independent image tracks/visibility blocks before optimization for the final check.',
   'visual':'Original-image overlay and unchanged face/body crops required; nose/mouth/glasses and shoulder/sleeve/hem/legs must be reviewed.',
   'physical_claim_limit':'Passing these screens is evidence of improvement, not proof of metric camera ground truth.'},
 'fresh_training':{
   'arms':['aipixel/GaussianAvatar','jmaraval/GaussianAvatars_SMPLX'],
   'initial_points_per_arm':200000,'steps_per_arm':18000,
   'person_fit':'fresh from raw observations and public/licensed prior','gaussians':'fresh','optimizer':'empty',
   'smplx_disclosure':'Public repository entry is incomplete; existing local adapter must remain explicitly identified.',
   'color':'RGB/SSIM restricted to foreground support; silhouette supervision retained outside, no blank-color learning.',
   'camera_policy':'Same accepted physical-camera candidate for both arms; preserve074 as immutable control. No camera interpolation.',
   'maximum_full_training_pairs':1},
 'budget_and_stop':{
   'maximum_alignment_candidates':3,'smoke_gpu_minutes_per_candidate':30,
   'full175_diagnostic_hours_per_candidate':4,'setup_hours_per_candidate':2,
   'no_automatic_hyperparameter_sweeps':True,
   'stop_on':'Stop any run with invalid frame mapping, NaN, changed source hashes, or violated input assumptions; record cause before fixing.',
   'end_followup':'Disable this bounded follow-up after a reviewed final pair or after all3 candidates are rejected/unavailable. Report any missing source data and remaining uncertainty. Do not keep retrying unchanged failures.'},
 'viewer':{'page':'http://127.0.0.1:8870/body-smplx.html#alignment116','preserve_tab15':True,
           'latest_model':'Do not replace model for diagnostic-only rounds. Label quality separately from format.',
           'delivery':'World/+Y standardPLY, full covariance andSH, native readback, actual OpenFile Gaussian/points. No00 deployment claim without current remote verification.'},
 'reports':{'canonical_plan':str(B/'long-term-plan.md'),'single_ppt':str(B/'reports/02_有限视角重建优化路线.pptx'),
            'ppt_policy':'Update existing52-page deck last page when substantive experimental evidence is ready; preserve other51pages. No new decks.',
            'cleanup':'Preserve source data, code, current/best and necessary controls. Remove only identified failed bulky duplicates under prior authorized policy with a keeper audit.'},
 'new_training_started':False}
save(O/'protocol.json',protocol)
state_path=O/'queue-state.json'
if not state_path.exists():save(state_path,{'updated':now,'state':'source_metadata_found_prepare_alignment_pilot','stage':'S0_to_S1','active_pid':None,'training_started':False,'candidate_index':0,'completed_candidates':[],'quality_accepted':False,'automation_id':None,'next_action':'Verify hovr schema/originaltdsp if available; independently prepare PixSfM dependency and correspondence diagnostic in isolated environment.'})

# Bounded local file inventory: source input and copied viewer bundle, not the entire machine.
roots=[Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/inputs/computer00-local-pipeline'),Path('C:/Users/Administrator/Downloads/05_mini_splatting2_viewer')]
matches=[]
for root in roots:
    if not root.exists():continue
    for p in root.rglob('*'):
        if p.is_file() and (p.suffix.lower() in ('.tdsp','.proto','.xmp','.srt') or re.search('calib|intrin|extrin|camera',p.name,re.I)):
            matches.append({'path':str(p),'bytes':p.stat().st_size})
save(O/'local-sidecar-inventory.json',{'roots':[str(x) for x in roots],'matches':matches,'remote00_checked':False,
 'interpretation':'Copied viewer camera JSONs are reconstruction/export artifacts, not independently verified device calibration. No tdsp or hovr schema was found in these roots.'})

current='''<!-- PORTRAIT_ALIGNMENT_116_CURRENT_BEGIN -->
当前主线：原始相机元数据核验与开源跨视角人像对齐（116）。175帧图像和相机传递在115通过一致性核验，但真实标定精度、材料对应及重影根因仍未唯一确定。114两条人像路线已完成，质量均未通过，不继续等待旧监督进程。

已发现原视频内嵌hovr轨道：179条消息包含177条逐帧结构记录，数值帧序列与视频时钟一致。原始K、畸变、R/t的字段语义尚未确认；现用074相机来自COLMAP估计。先查官方元数据定义或原始tdsp，同时在隔离环境准备PixSfM；需要时比较RoMa真实材料对应与HSfM联合人体/场景/相机对齐。TrackNeRF只作全局轨迹约束参考。本轮最多三条有界标定候选，不扫权重。

候选须通过原像素映射、175帧覆盖、独立面部及全身对应与背景保护检查，才允许两条已授权人像路线用同一候选相机重新拟合、各20万初始点和1.8万步训练。保持静态穿衣人物；导出完整协方差/SH的world与+Y标准PLY，真实打开文件验证后再更新latest。每轮更新现有右侧评测页，不能把格式通过当重影修复。

协议与进度：current/portrait-alignment-open-source-116/protocol.json、queue-state.json。原始元数据证据保存在同目录；公开研究登记research-sources.json。下方历史计划不触发旧任务。
<!-- PORTRAIT_ALIGNMENT_116_CURRENT_END -->

'''
plan=B/'long-term-plan.md';s=plan.read_text(encoding='utf8')
a='<!-- PORTRAIT_ALIGNMENT_116_CURRENT_BEGIN -->';z='<!-- PORTRAIT_ALIGNMENT_116_CURRENT_END -->'
if a in s:s=s[:s.index(a)]+current+s[s.index(z)+len(z):].lstrip()
else:s=current+s
plan.write_text(s,encoding='utf8')

D.mkdir(exist_ok=True)
for name in ['capture-provenance.json','hovr-frame-timing-check.json','candidates.json','protocol.json']:
    shutil.copy2(O/name,D/name)
html='''<section id="alignment116"><h2>116 · 原始相机信息与跨视角人像对齐</h2>
<p class="notice"><strong>新发现：原视频内嵌了 HOVER VERSA 的逐帧元数据。尚未确认其中有可用的原始内外参。</strong>当前175帧的相机来自COLMAP重建。先验证原始数据与真实材料对应，再决定从头训练；现有高斯尚未通过重影质量验收。</p>
<h3>数据里到底有什么相机信息？</h3>
<p>HOVER_VERSA_0102.mp4 包含177帧视频（1520×2704）和名为 Hover meta 的hovr数据轨道，标注schema 1.1.0。已完整提取179条消息，其中177条有相同逐帧结构；一个数值字段依次为0—176，另一个字段减去首帧后与视频时间戳完全一致。首尾两条是不同的配置结构，不能直接用消息下标当视频帧号。</p>
<p><strong>已找到逐帧元数据，不等于已经取得相机标定。</strong>目前缺少这些数字字段的官方定义，尚不能把它们命名为焦距、畸变、旋转、平移或防抖参数。视频标签还提到HOVER_VERSA_0102.tdsp，本机已核查的输入与传回资料目录中没有这份原始文件；本轮没有检查00上的完整采集目录。MCTF开启只是记录到的标签，不能据此断定它造成重影。</p>
<p>现有074相机的来源可以确认：整图SIFT匹配、SfM及束调整，优化了共享焦距、畸变、主点和各帧外参；训练器再做世界坐标、去畸变和分辨率转换。115已验证这条传递链路一致，但没有证明估计参数是物理真值。</p>
<h3>按什么顺序尝试开源方法？</h3>
<ol><li><strong><a href="https://github.com/cvg/pixel-perfect-sfm">PixSfM</a>：先改善二维对应，再检查相机。</strong>先细化同一匹配点在各图像中的落点，再用深层特征一致性优化相机与场景点。与此前只优化几何重投影的BA区别明确；先做固定相机的特征诊断，再做相机候选，分别记录收益。</li>
<li><strong><a href="https://github.com/Parskatt/RoMa">RoMa</a>：核对实际材料，而非仅依赖关键点编号。</strong>检查鼻嘴、眼镜和衣物纹理的双向匹配及多视角循环，构造共享三维轨迹；它是匹配器，需要接入三角化和标定。不会把人像二维拉伸或平移成“看起来对齐”。</li>
<li><strong><a href="https://github.com/hongsukchoi/HSfM_RELEASE">HSfM</a>：人体、场景与相机联合优化。</strong>作为前两项未通过时的结构性对照。先核实顺序视频与其多视角假设是否相容；人体先验不代表衣物表面，也不保证精细五官正确。</li></ol>
<p><a href="https://github.com/Wayne-Mai/traf_public">TrackNeRF</a>的全局轨迹约束用于方法参考：同一条可见材料轨迹关联一个三维点。其官方训练器是NeRF，本轮不另开NeRF训练，也不把它说成已经接入了高斯。</p>
<p>上述项目的公开入口、版本和代码许可已核查；<strong>尚未完成本地运行验证，也没有本项目上的改善结果。</strong>依赖和模型许可单独处理。PixSfM使用较老的COLMAP/Ceres接口，安装会与当前训练环境隔离。</p>
<h3>什么条件下才开始下一轮训练？</h3>
<p>先固定验收图像和观测划分，报告面部、肩袖、衣摆、腿脚、背景的同像素误差与覆盖。候选需让下半脸独立纹理的中位误差下降至少20%，同时P90不恶化，并保护躯干、腿脚和背景。旧的留出点已用于方案选择，因此继续称验证集；另保留新图像轨迹作最终核验。阈值只是试验准入条件，不是物理真值证明。</p>
<p>通过后，两条用户指定的人像路线重新拟合人物、各自初始化20万高斯、训练1.8万步，以同一组相机比较。始终采用共享静态穿衣人物；每轮输出标准world与+Y高斯PLY，实际检查打开文件的高斯和点云模式。几何、重影、新视角质量与格式兼容分别记录。</p>
<p>长期跟进最多验证三条标定候选；通过后完成一次双路线训练对照，或候选全部失败后给出结论并结束这轮跟进。不无限追加试验。当前未启动新的高斯训练。</p>
<p><a href="body-camera116/capture-provenance.json">原始数据核查</a> · <a href="body-camera116/hovr-frame-timing-check.json">逐帧时间戳核验</a> · <a href="body-camera116/candidates.json">候选与边界</a> · <a href="body-camera116/protocol.json">长期协议</a> · <a href="#camera115-source">已有映射与特征证据</a> · <a href="./?model=latest">查看现有高斯</a></p></section>'''
path=V/'body-smplx.html';s=path.read_text(encoding='utf8');a='<!-- ALIGNMENT116_BEGIN -->';z='<!-- ALIGNMENT116_END -->';block=a+'\n'+html+'\n'+z+'\n'
if a in s:s=s[:s.index(a)]+block+s[s.index(z)+len(z):]
else:
    marker='<!-- CAMERA115_BEGIN -->';assert marker in s;s=s.replace(marker,block+marker,1)
path.write_text(s,encoding='utf8')
latest=read(V/'latest-result.json');old_id=latest.get('modelId',latest.get('model'))
latest['latestDiagnostic']={'round':'116','report':'body-smplx.html#alignment116','status':'发现原视频hovr逐帧元数据，内外参字段语义待确认；开源对齐路线已核查，未启动新高斯训练。','updated':now}
save(V/'latest-result.json',latest)
state=read(B/'work-status.json');state.update(updated=now,local_training_running=False,current_phase='portrait_alignment_open_source_116',goal_accepted=False)
state['next_bounded_work']={'run':str(O),'phase':'capture_metadata_and_open_source_alignment','new_training_started':False,'remaining':'Decode capture metadata only with verified semantics; run bounded correspondence/calibration diagnostics before fresh avatar training.'}
save(B/'work-status.json',state)
save(O/'publication.json',{'updated':now,'url':'http://127.0.0.1:8870/body-smplx.html#alignment116','latest_model_unchanged':True,'new_gaussian':False,'visual_review_pending':True})
print(json.dumps({'published':True,'url':'http://127.0.0.1:8870/body-smplx.html#alignment116','new_training':False,'capture_metadata':capture['metadata_packets']},ensure_ascii=False))
