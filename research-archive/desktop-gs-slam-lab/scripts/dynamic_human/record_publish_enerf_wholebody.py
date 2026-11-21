from pathlib import Path
import json,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];SITE=Path('D:/Projects/EngineLab/engine-home');O=LAB/'outputs/dynamic-human-enerf-wholebody-059';P=LAB/'outputs/dynamic-human-enerf-wholebody-prior-060';R=LAB/'outputs/dynamic-human-enerf-forearm-control-061'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
assert read(O/'independent-audit.json')['accepted_landmarks']==64
save(O/'visual-review.json',dict(status='bounded_actual_training_photo_review',detector_cameras=['01','04','07','13','17'],triangulated_cameras=['04','07','17'],findings='Separateperson RTMW recovers darkleftelbow and consistentfootpoints. Occluded darkhead/limbs can attachtofrontperson. Lightperson back-facing nose/eyes hallucinated; excludedfromfit. Multiview agreement isnotvisibilitycertification.',goal_complete=False))
save(O/'final-summary.json',dict(status='training_only_wholebody_landmarks_acquired_and_geometrically_checked',images=15,persons_per_image=2,keypoints_per_person=133,accepted_selected_landmarks=64,total_selected_landmarks=66,accepted_body_head_feet=46,median_inlier_error_px=read(O/'independent-audit.json')['median_accepted_inlier_error_px'],scope='Predicted coordinates and coarse3Dhypotheses, notreconstructed appearance orGT.',goal_complete=False))
save(P/'visual-review.json',dict(status='4of15training_projection_panels_actually_reviewed',cameras=['01','04','07','17'],findings='Headtilt,darkleftelbow andfootorientation moreconsistent withphotographs than058. Palmrotation/openhandsremainwrong; clothedsilhouette,hair,shoeandtoyabsent. Forearmmiddleposition plausible forasoftboundedtest, notahardbodyconstraint.',goal_complete=False))
fit=read(P/'independent-audit.json')
save(P/'final-summary.json',dict(status='2500update_structural_prior_refined_not_a_clothed_reconstruction',updates=2500,used_predicted_landmarks=43,initial_predicted_landmark_error_world=fit['initial_mean_valid_error_world'],final_predicted_landmark_error_world=fit['final_mean_valid_error_world'],projection_review='4of15trainingcameras; noheldoutfit',goal_complete=False))
a=read(R/'training-state-audit.json');r=read(R/'paired-comparison/summary.json');assert a['step']==6000 and read(R/'native-evaluation/independent-image-audit.json')['full_pngs']==24
save(R/'visual-review.json',dict(status='6wholeperson_and17fixed_native_detail_panels_actually_reviewed',cameras=['00','06','12','01','08','17'],findings='Localforearmshapechangesminor;00darkforearmstillstretched,06crossedarmsstillmisplaced,12shirtprintpartlychanges. Hands,toys,shoesandoccludedshoulderretainstreaking/blur. No clearhumanreconstruction acceptance.',decision='Do notpromote localforearmpriorasclaritysolution. Check cross-camera motioncorrespondence usingreservedtime trajectoryvalidation, then considerarticulateddynamicrepresentation.',goal_complete=False))
save(R/'final-summary.json',dict(status='6000update_soft_forearm_control_completed_small_mixed_changes_clarity_not_accepted',updates=6000,points=a['points'],checkpoint_sha256=a['checkpoint_sha256'],heldout_foreground_delta_db={x['camera']:x['foreground_delta_db'] for x in r['records'] if x['split']=='heldout'},training_foreground_delta_db={x['camera']:x['foreground_delta_db'] for x in r['records'] if x['split']=='training_diagnostic'},scope='Same original15nativeimages and initialtensors as055, only forearmpriorlossadded. Developmentheldoutcamerasrepeatedlyinspected, notfreshfinaltest.',goal_complete=False))
p=LAB/'experiments/37_dynamic_human_reconstruction.md';text=p.read_text(encoding='utf8')
addition='''

### 全身关键点与人体结构先验的修正

使用 RTMW 的 133 个全身关键点模型，对同一帧 15 台训练相机中的两个人分别推理。人物框沿用此前固定的提示，不使用留出照片。对身体、头部、脚部及指尖的 66 个候选三维点进行鲁棒三角化，64 个获得至少四个视角支持；其中身体、头部和脚部的 46 个候选均有几何支持，补上了此前缺失的黑衣人物左肘。990 个重投影误差已重算，接受点支持观测的中位残差为 2.762 像素。几何一致不等于语义正确：背对相机的白衣人物仍有被模型猜测出来的鼻子和眼睛。

因此，人体拟合排除了白衣人物的鼻子、双眼以及全部指尖，只使用 43 个预测点完成 2,500 次优化。基本 SMPL 没有独立的手指关节，不能用张开的模板手掌强行拟合抓握动作。相对同一组新目标点，原先验平均残差为 0.07770，修正后为 0.01564，单位为数据世界坐标单位，未独立认证为米。保存参数在 CPU 上重新执行完整 SMPL 前向后，顶点最大差异小于 0.000001。

![真实训练照片与修正后的人体结构先验投影](../outputs/dynamic-human-enerf-wholebody-prior-060/camera04-body-prior.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站拟合 · SMPL 模板已获用户下载授权 · 结构投影，不是外观重建。

检查了 15 张投影中的 01、04、07、17 四台相机：头部俯仰、左肘与脚部方向改善，手掌、衣物、鞋和玩具仍不匹配。该先验只用于有限范围的柔性约束试验，不能作为真实衣物表面。见[关键点结论](../outputs/dynamic-human-enerf-wholebody-059/final-summary.json)、[重投影核验](../outputs/dynamic-human-enerf-wholebody-059/independent-audit.json)、[观测检查范围](../outputs/dynamic-human-enerf-wholebody-059/visual-review.json)、[先验结论](../outputs/dynamic-human-enerf-wholebody-prior-060/final-summary.json)、[前向与误差复算](../outputs/dynamic-human-enerf-wholebody-prior-060/independent-audit.json)和[投影检查范围](../outputs/dynamic-human-enerf-wholebody-prior-060/visual-review.json)。模板、网格和拟合参数保留在本地。

### 裸露前臂的柔性表面约束

保持 15 张原生训练图、相机、掩码、初始高斯、采样顺序、优化器、增密计划和表面正则项一致，新增 6,000 次更新的对照。唯一训练改动是从第 1,401 次更新起，对肘至腕中间一段的高斯加入相对人体表面切平面的距离惩罚。容许带宽为前臂长度的 4%，权重为 0.1；手掌、头部和衣物不作为该项的监督目标。空间筛选近似仍可能误含邻近物体，不能视为精确皮肤分割。

![原照片、表面基线与增加前臂约束后的原生人物对照](../outputs/dynamic-human-enerf-forearm-control-061/paired-comparison/camera06-frame000750-full-person.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站渲染 · 开发留出机位 06 · 无锐化或生成式增强。

最终得到 208,468 个面元，峰值训练分配显存约 1.53 GB。三个开发留出机位 00、06、12 的前景 PSNR 分别变化 −0.014、+0.039、+0.145 dB。24 张原生输出、12 张比较图和同一组 17 个细节区域通过像素核验；六张完整人物对照和全部 17 个细节均已查看。黑衣人物前臂仍拉长，交叉手臂的位置错误和玩具拖影基本保留，不作为达标结果。下一步用连续帧中的人体对应检验跨相机残余时序差异，不能仅根据这组负结果断言模糊由时间误差造成。

见[固定训练条件](../outputs/dynamic-human-enerf-forearm-control-061/protocol.json)、[结果汇总](../outputs/dynamic-human-enerf-forearm-control-061/final-summary.json)、[模型和优化器核验](../outputs/dynamic-human-enerf-forearm-control-061/training-state-audit.json)、[像素与指标重算](../outputs/dynamic-human-enerf-forearm-control-061/native-evaluation/independent-image-audit.json)、[同区域逐项比较](../outputs/dynamic-human-enerf-forearm-control-061/paired-comparison/summary.json)和[看图结论](../outputs/dynamic-human-enerf-forearm-control-061/visual-review.json)。
'''
if '### 全身关键点与人体结构先验的修正' not in text:p.write_text(text+addition,encoding='utf8')
p=LAB/'knowledge/32_dynamic_human_scene_reconstruction.md';text=p.read_text(encoding='utf8')
addition=r'''

### 人体关键点约束的是结构，不是真实衣物表面

人体关节或脚部标志点的多视角预测，可以先通过三角化得到三维假设，再用于拟合身体姿态。设 $\widehat{\mathbf X}_j$ 为第 $j$ 个预测三维点，$\mathbf J_j(\boldsymbol\beta,\boldsymbol\theta)$ 为人体模型对应点，基本拟合目标可写为

$$
\underset{\boldsymbol\beta,\boldsymbol\theta,s,\mathbf t}{\operatorname{minimize}}
\quad\sum_j w_j\,\rho\!\left(
\frac{\left\|s\mathbf J_j(\boldsymbol\beta,\boldsymbol\theta)+\mathbf t-\widehat{\mathbf X}_j\right\|_2}{\ell}
\right)+\lambda\mathcal P(\boldsymbol\beta,\boldsymbol\theta,s).
$$

$s$ 为尺度，$\mathbf t$ 为平移，$\ell$ 是用于归一化残差的身体长度；$w_j$ 表示观测可信度，$\rho$ 为鲁棒损失，$\mathcal P$ 为体形与姿态先验。这是结构拟合的概括表达。多视角预测相互接近，并不能证明被遮挡的关节真实可见；背面人物的面部猜测尤其不应自动加入约束。基础 SMPL 的手指自由度也不足以解释所有抓握动作。

![真实照片与人体结构先验的投影，展示身体与衣物之间的差别](../outputs/dynamic-human-enerf-wholebody-prior-060/camera04-body-prior.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap 与本站拟合 · SMPL 结构检查，不是外观重建。

上图中，头部、肘部和脚部可以对齐，而宽松衣服、头发、鞋和玩具仍位于模型表面之外。因此，更小的关节点误差不能替代清晰外观的验证；表面约束也需要允许真实衣物偏离人体模板。本地限定在裸露前臂的柔性约束仅产生小幅、混合变化，未解决交叉手臂和物体拖影。仍应分别检查身体对应、可见表面与跨时刻运动，而不是把整个人强制压到模板上。[实际拟合与同预算对照](../experiments/37_dynamic_human_reconstruction.md#全身关键点与人体结构先验的修正)。

'''
if '### 人体关键点约束的是结构，不是真实衣物表面' not in text:
 anchor='### 相机与人物必须对应同一个拍摄时刻';assert anchor in text;text=text.replace(anchor,addition+anchor);p.write_text(text,encoding='utf8')
p=SITE/'scripts/sync-gs-lab.mjs';text=p.read_text(encoding='utf8')
if 'dynamic-human-enerf-wholebody-059/${name}' not in text:
 lines=text.splitlines(keepends=True);idx=next(i for i,l in enumerate(lines) if 'dynamic-human-enerf-body-observations-057/${name}' in l)+1
 lines.insert(idx,"  ...['final-summary.json','independent-audit.json','visual-review.json'].map(name => [`outputs/dynamic-human-enerf-wholebody-059/${name}`, '全身关键点几何检查']),\n  ...['final-summary.json','independent-audit.json','visual-review.json','camera04-body-prior.png'].map(name => [`outputs/dynamic-human-enerf-wholebody-prior-060/${name}`, '人体结构先验修正']),\n  ...['protocol.json','final-summary.json','training-state-audit.json','native-evaluation/independent-image-audit.json','paired-comparison/summary.json','visual-review.json','paired-comparison/camera06-frame000750-full-person.png'].map(name => [`outputs/dynamic-human-enerf-forearm-control-061/${name}`, '裸露前臂柔性约束对照']),\n");p.write_text(''.join(lines),encoding='utf8')
p=LAB/'manifests/dynamic-human-work.json';m=read(p);m['outdoor_wholebody_observations']='outputs/dynamic-human-enerf-wholebody-059/final-summary.json';m['outdoor_refined_body_prior']='outputs/dynamic-human-enerf-wholebody-prior-060/final-summary.json';m['outdoor_forearm_control']='outputs/dynamic-human-enerf-forearm-control-061/final-summary.json'
m['status']='wholebody_prior_refined_forearm_training_completed_clarity_not_accepted';m['external_input_blocker']=None
m['goal_turn_classification']='Progress:059actualRTMWinferenceandtriangulation,0602500updateSMPLfit,0616000updateforearmcontrol completed andaudited. No clarityclaim.'
m['next_actions']=['Complete062training-only13frameposeobservations and independentlyreservedtime trajectory/timingcheck; do notassume a timingcause from061negative result.','Use qualifiedmotioncorrespondence tobuildarticulatedGaussianappearance across time; retaincloth/objectfreedom andrequireactualheldoutcamera/time clarity.','After credibleforegroundreconstruction, integratefullscene; noadditionalhardwarecurrentlyrequired.']
for item in ['05915nativeviews30people133keypoints;64of66selected3Dlandmarks verified;back-facingfacepredictions excludedfromfit.','0602500updateSMPLrefinement43predictedtargets;4trainingprojectionsreviewed;hands/clothes/toysremainunmodeled.','0616000updateforearmpriorcontrol208468points;6whole17detailsreviewed;smallmixedheldoutchanges,claritynotaccepted.']:
 if item not in m['completed']:m['completed'].append(item)
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(p,m);shutil.copy2(__file__,R/'source-snapshot'/Path(__file__).name)
print('059060061 findings and knowledge sources updated')
