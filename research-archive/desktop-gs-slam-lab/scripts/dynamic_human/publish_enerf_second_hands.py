"""Publish a concise theory update and linked, bounded hand evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-motion-082';SITE=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=read(O/'final-summary.json');assert s['actual_panels_reviewed']==38 and not s['adopt_as_development_baseline']
# Editorial correction only: the diagnostic actually optimized21parameters.
for p in [L/'scripts/dynamic_human/fit_enerf_rigid_palm.py',O/'fit_enerf_rigid_palm.py',O/'palm-protocol.json']:
    txt=p.read_text(encoding='utf8');txt=txt.replace('23? exactly9rotation+12translation=21parameters','9rotation+12translation=21parameters');p.write_text(txt,encoding='utf8')
shutil.copy2(L/'scripts/dynamic_human/audit_finalize_enerf_second_hands.py',O/'audit_finalize_enerf_second_hands.py')
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md')
a=txt.index('![原生手部照片与全身姿态模型预测，遮挡处仍可能出现完整手指]');b=txt.index('对人物 $a$ 的关节 $j$，二维观测约束可概括为',a)
txt=txt[:a]+'''![原生手掌与全身及局部手部模型的预测](../outputs/dynamic-human-enerf-hand-motion-082/candidate-panels/camera04-candidates.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与 RTMW、RTMPose-Hand 实测 · 机位04，帧720、726、732 · 原生像素；彩色点为预测。

每行从左至右是原照片、全身模型、局部手部模型，以及缩小和扩大裁剪范围后的局部预测。上两行的手正在抓握玩偶，部分预测落到玩偶表面；仅改变裁剪范围，某些关节就明显移动。最下行张开的手掌较清楚，局部预测也较稳定，但稳定并不等于每根手指都对应正确。

网络的响应分数不能直接解释成“这个指尖真实可见的概率”。不可见区域也可能得到完整的21点结构；多人交互时，还必须确认预测属于哪一个人。相同模型在不同裁剪下保持一致，只检验了对裁剪的敏感性，不能代替跨机位的独立核验。

'''+txt[b:]
a=txt.index('细节还受到检测输入分辨率限制：',txt.index('#### 手部观测必须区分可见性与预测分数'));b=txt.index('局部图像跟踪也不能直接替代手指观测。',a)
txt=txt[:a]+r'''细节还受到检测输入分辨率限制：全身检测器缩放整个人后，手指只占其中很小一块。局部检测能让网络使用更多输入像素描述这只手，却不能创造被遮挡的指尖。跨机位核验时，应把目标相机从三角化与拟合中完全排除，再检查投影是否与该相机的照片一致；没有形成有效三维假设的点也必须计入覆盖率，不能只报告成功点的平均误差。

在这一运动片段中，局部检测改善了手腕的跨机位一致性，指尖却反而变差。裁剪稳定性筛选还会失去大量指尖观测。由这些预测得到的平滑三维曲线仍可能偏离真实皮肤，因而没有作为高斯训练的手指约束。

掌部结构先验可以进一步限制不合理形变。例如，短时间内将手腕和四个掌指关节的参考位置记为 $\overline{\mathbf X}_j$，刚性掌部近似为

$$
\mathbf X_j(t)=\mathbf R(t)\overline{\mathbf X}_j+\mathbf t(t),\qquad
\|\mathbf X_i(t)-\mathbf X_j(t)\|_2
=\|\overline{\mathbf X}_i-\overline{\mathbf X}_j\|_2.
$$

$\mathbf R(t)\in\mathrm{SO}(3)$ 为掌部旋转，$\mathbf t(t)\in\mathbb R^3$ 为平移；各点间距离固定。它约束的是掌部的局部形状，手指屈伸还需要独立的关节运动。形状不再拉伸，也不保证位置正确：本地五点掌部拟合在未参与拟合的相机中没有改善投影，因此同样没有用于驱动高斯。完整条件与原生照片见[连续手部观测与掌部运动核验](../experiments/37_dynamic_human_reconstruction.md#连续手部观测与掌部运动核验)。

'''+txt[b:]
p.write_text(txt,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';txt=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 连续手部观测与掌部运动核验' not in txt
txt+='''

### 连续手部观测与掌部运动核验

对象限定为 actor5_6 的浅衣人物右手，帧720至732。先查看15个原机位、三个训练时刻的90个左右手原生裁图，排除抓握玩偶且遮挡严重的左手。右手195张原生照片分别采用三种裁剪范围，完成585次官方 RTMPose-Hand 推理；105张偶数帧用于几何拟合候选，90张奇数帧仅用于检查。原测试机位00、06、12均未进入这项诊断。输入RGB与模型发布包的标准归一化流程逐元素一致。

![同一只手的原照片与裁剪敏感性](../outputs/dynamic-human-enerf-hand-motion-082/candidate-panels/camera04-candidates.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与 RTMPose-Hand 实测 · 从左至右：原照片、全身预测、局部预测、较小裁剪、较大裁剪。

保留已有相机时间偏移，只用偶数帧在五个共同物理时刻进行插值。每个关节每次排除一个目标相机，以其余相机三角化，要求至少四个支持机位、重投影误差小于3像素、正深度及足够的视线夹角。三种观测方式共4725项检查，所有失败项也计入总数。这是检测预测的一致性，不能当作真实关节位置精度。

全身模型与局部模型的手腕误差中位数分别为3.06、2.47像素；指尖则由4.81变为5.94像素。以全部375个指尖目标为分母，低于3像素的比例由24.8%降到18.1%。只保留裁剪变化不超过3像素的支持点后，有效指尖覆盖率仅40%，不能把筛选后的较小误差解释为完整手部恢复。

进一步将04机位整段排除，使用其他机位的偶数帧拟合连续三维轨迹。裁剪稳定方案只有9个关节得到足够支持，不能构成完整的手指运动。原照片对照仍能看见部分曲线离开皮肤、落到玩偶或背景。

![没有参加拟合的04机位中的手部轨迹](../outputs/dynamic-human-enerf-hand-motion-082/motion-panels/camera04-motion.png)
> 图源：同上 · 原照片、局部检测、全身预测的三维拟合、局部预测的三维拟合、裁剪稳定点的三维拟合 · 各行均为未拟合的奇数帧。

再对手腕及四个掌指关节采用固定形状的刚性掌部模型，实际优化21个运动参数，使用324个训练观测。五点间距离保持不变，但04机位65个对应投影的误差中位数由独立曲线的4.47变为5.37像素，未改善独立机位一致性；该模型也没有表达指尖屈伸。

![固定掌部形状前后的原生投影检查](../outputs/dynamic-human-enerf-hand-motion-082/motion-panels/camera04-palm.png)
> 图源：同上 · 原照片、局部检测、独立三维掌部点、固定形状掌部。

**局部手指标签和掌部运动均未加入高斯训练，人物清晰度仍未通过。**本轮没有新的人体RGB重建，以上彩色线条是预测投影。已实际查看38张原生照片对照，并独立复算195个输入文件、4725项跨机位记录、9750项轨迹投影和975项掌部投影。下一项重建干预应优先处理清晰张手区间的局部图像对应与遮挡，不能把这些不可靠指尖直接加成强约束。

记录：[结论与决策](../outputs/dynamic-human-enerf-hand-motion-082/final-summary.json)、[跨机位一致性](../outputs/dynamic-human-enerf-hand-motion-082/geometry-summary.json)、[输入与投影复算](../outputs/dynamic-human-enerf-hand-motion-082/numerical-audit.json)。
''';p.write_text(txt,encoding='utf8')
files=['final-summary.json','geometry-summary.json','numerical-audit.json','candidate-panels/camera04-candidates.png','motion-panels/camera04-motion.png','motion-panels/camera04-palm.png']
p=SITE/'scripts/sync-gs-lab.mjs';txt=p.read_text(encoding='utf8');assert O.name+'/${name}' not in txt;lines=txt.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-support-correction-081/${name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+f".map(name => [`outputs/{O.name}/${{name}}`, '连续手部观测与掌部运动核验']),\n");p.write_text(''.join(lines),encoding='utf8')
save(O/'published-resources.json',dict(paths={O.name:files},licensed_models_exported=False))
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='hand_observation_and_motion_candidates_rejected_clarity_unresolved',outdoor_hand_motion='outputs/'+O.name+'/final-summary.json',current_runs=[],next_action=s['next_action'],goal_turn_classification='Progress:195native hand observations585crops;4725independent camera checks;three3Dtrajectory fits andfixedshape5point palm control;38realpanels reviewed; allresiduals independentlyrecomputed. No accepted Gaussian or clarity improvement.',updated_at=datetime.now(timezone.utc).isoformat());m['completed'].append('082: native hand195images585crops,4725crosscamera checks,3trajectory variants andrigidpalm control;38nativepanels reviewed. Wristconsistency improves but tipsregress; labels andpalm motion notadopted; nonewGaussiantraining.');save(p,m)
save(O/'recovery-notes.json',dict(items=['A print-only undefined variable interrupted aftergeometry; resumed cached4725geometry records andcompletedtrajectoryfits withsameprotocol.','Audit completed itsdata andprojectionchecks, thenmanifestreadhitWindowsdefaultGBK; fixedUTF8read andpublishedmanifestseparately without changing experimentresults.','Removed stray textual23? frompalmprotocol; actualoptimizedparameters always21.']))
shutil.copy2(__file__,O/Path(__file__).name);print('Knowledge section replaced; evidence appended;6assets allowlisted; manifest updated')
