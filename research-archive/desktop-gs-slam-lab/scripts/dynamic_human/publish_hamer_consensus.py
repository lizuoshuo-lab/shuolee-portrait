from pathlib import Path
import json,shutil
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-consensus-097';SITE=Path('D:/Projects/EngineLab/engine-home')
assert json.loads((O/'final-summary.json').read_text(encoding='utf8'))['native_visual_contexts']==66
p=L/'knowledge/32_dynamic_human_scene_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'knowledge-before.md')
start=s.index('手部参数模型可以把任意点的运动');end=s.index('### 相机与人物必须对应同一个拍摄时刻',start)
replacement=r'''手部参数模型用共同的形状和关节连接，约束不同视角中的候选手形。HaMeR逐图预测MANO参数；要形成同一个动态三维对象，还需在世界坐标系中共享形状，并让姿态只随拍摄时刻变化。以本地连续拟合为例，第$i$个手部顶点为

$$
\mathbf v_i(t)=s\mathbf R(t)
\left[\mathcal M_i\!\left(\boldsymbol\beta,\boldsymbol\theta(t)\right)
-\mathbf J_0\!\left(\boldsymbol\beta,\boldsymbol\theta(t)\right)\right]
+\mathbf p(t).
$$

$\mathcal M_i$为MANO生成的局部顶点，$\mathbf J_0$为手腕关节；两者相减使局部坐标以手腕为原点。$\boldsymbol\beta$是整个区间共享的10维形状参数，$\boldsymbol\theta(t)$包含15个局部关节的轴角旋转，$s$将MANO长度换算为场景长度，$\mathbf R(t)$将手部方向变换到世界坐标，$\mathbf p(t)$为世界坐标中的手腕位置。关节角以弧度计，位置与场景坐标使用相同长度单位。

本地模型用三次B样条表示连续轨迹：$\mathbf p(t)=\sum_k B_k(t)\mathbf p_k$，各关节的轴角也由同一时间基函数展开。同一拍摄时刻，所有相机看到的是**同一套三维顶点**，而不是各自独立预测的手。这里的$t$以帧为单位并包含现有相机时间偏移；时间偏移仍是估计值。网络给出的单目深度没有直接当作场景深度。

![单视图预测与共享手部模型在原照片中的投影](../outputs/dynamic-human-hand-consensus-097/shared-hand-versus-source.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；HaMeR、MANO与本地拟合 · 上为03机位726帧，下为04机位732帧 · 原照片、单视图预测、15机位共享模型、排除04／09／13机位的共享模型 · 蓝色为结构先验。

上排共享模型把单视图较伸展的预测收拢为抓握候选，下排显示张手时的连续手指结构。但部分指尖与原照片轮廓仍不重合；蓝色材质也没有恢复皮肤纹理。手被玩偶遮挡时，当前叠加没有使用玩偶深度，不能凭蓝色覆盖关系确认隐藏手指的位置。

共享模型可通过鲁棒重投影误差拟合：

$$
\mathcal L_{\mathrm{proj}}=
\frac{\sum_{c,f,j} w_{cfj}\,
\rho\!\left(\left\|\pi_c\!\left(\mathbf J_j(t_{cf})\right)
-\widehat{\mathbf u}_{cfj}\right\|_2\right)}
{\sum_{c,f,j}w_{cfj}},\qquad
\rho(r)=a^2\ln\!\left(1+\frac{r^2}{a^2}\right).
$$

$c,f,j$分别索引相机、照片帧和关节；$\widehat{\mathbf u}_{cfj}$是HaMeR预测的二维关节，$\pi_c$使用原标定相机，$t_{cf}$为估计拍摄时刻。权重$w_{cfj}=\sigma_0^2/(\sigma_0^2+\delta_{cfj}^2)$随裁剪扰动$\delta_{cfj}$增大而减小；本地诊断取$\sigma_0=2$像素、$a=4$像素，另加形状与运动平滑正则。这是本地检验目标，并非HaMeR原论文的训练损失；裁剪稳定也不能证明关节可见或预测正确。

**重投影一致性与照片重建质量需要分别验证。** 本地留出3个机位后，预测关节残差的中位数下降，但抓握到张手的细节仍有偏差。共享结构因此只保留为软约束候选；它是否改善高斯表面，要由原始RGB训练、遮挡处理及留出视角的实际渲染判断。[先验接入](../experiments/37_dynamic_human_reconstruction.md#手部关节先验的接入与可用性)与[共享模型检查](../experiments/37_dynamic_human_reconstruction.md#连续手部模型的跨视角检验)保留实验条件。

'''
s=s[:start]+replacement+s[end:];s=s.replace('下一步引入手部关节结构前，需要先验证','引入手部关节结构时，仍需验证');p.write_text(s,encoding='utf8')
p=L/'experiments/37_dynamic_human_reconstruction.md';s=p.read_text(encoding='utf8');shutil.copy2(p,O/'evidence-before.md');assert '### 连续手部模型的跨视角检验' not in s
s+='''

### 连续手部模型的跨视角检验

在固定相机内外参与估计曝光时间下，完成两组连续MANO拟合，各1200次更新。主候选使用15个训练机位的105张偶数帧照片；诊断候选完全排除04、09、13机位，只用12机位84张照片初始化和拟合，其余21张仅作检查。00、06、12机位与所有奇数帧均不参与拟合。模型共享10维手形与尺度，使用9个三次B样条控制点表示手腕运动和16组轴角旋转，共470个可训练参数；没有逐相机形变，也没有采用HaMeR单目深度。

目标是裁剪扰动加权的Cauchy重投影损失，带形状、尺度及轨迹平滑正则。主候选对预测关节的残差中位数从4.56降至2.64像素；诊断候选在排除的3机位上从5.63降至3.91像素，但第90百分位从14.93升至16.05像素。**这些是对神经网络关节预测的一致性指标，并非真实指尖精度或高斯画质指标。** 部分大误差仍未解决，不能只引用中位数宣称准确。

![抓握与张手时的共享手部候选](../outputs/dynamic-human-hand-consensus-097/shared-hand-versus-source.png)
> 图源：ENeRF-Outdoor / 4K4D、EasyVolcap；HaMeR、MANO与本地拟合 · 从左到右为原照片、单视图预测、全部15机位拟合、排除04／09／13机位拟合 · 原生像素，非高斯重建。

实际检查66组原生窗口：15机位的720、726、732帧共45组，04／09／13机位的6个奇数时刻共18组，以及00／06／12机位732帧共3组。抓握位置的共享候选较单视图预测更收拢，但张手时的指尖和轮廓仍存在偏移，04／09机位729帧仍明显偏屈曲。蓝色手模型没有玩偶／身体深度遮挡，亦未优化皮肤外观，不能作为照片一致的手部表面。

两组参数均通过CPU恢复、独立相机投影、旋转矩阵与连续轨迹检查；66张原图窗口和132个共享网格合成图逐像素核对。相同物理时刻的顶点完全相同，未引入按相机变化的手形。原高斯参数没有更新，当前候选不作为硬几何替换。[结果与适用范围](../outputs/dynamic-human-hand-consensus-097/final-summary.json)、[实际看图记录](../outputs/dynamic-human-hand-consensus-097/visual-review.json)。

后续仅尝试有界的软表面／运动约束，先核对手部高斯与模型的对应关系，再与原RGB损失联合优化；手部上下文中还包含前臂和玩偶，不能整块删除或强制绑定。是否采用以留出机位、奇数时刻及原生手部实际渲染为准。
''';p.write_text(s,encoding='utf8')
root='dynamic-human-hand-consensus-097';files=['final-summary.json','visual-review.json','shared-hand-versus-source.png'];p=SITE/'scripts/sync-gs-lab.mjs';s=p.read_text(encoding='utf8');assert root+'/${name}' not in s;lines=s.splitlines(keepends=True);i=next(i for i,l in enumerate(lines) if 'dynamic-human-hand-prior-096/${name}' in l)+1;lines.insert(i,'  ...'+json.dumps(files)+'.map(name => [`outputs/'+root+'/${name}`, \'共享手部结构检验\']),\n');p.write_text(''.join(lines),encoding='utf8');(O/'published-resources.json').write_text(json.dumps(dict(paths={root:files},licensed_models_exported=False),indent=2)+'\n')
shutil.copy2(__file__,O/Path(__file__).name);print('Knowledge section replaced; no new reading page; only3 non-model assets allowed.')
