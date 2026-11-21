from pathlib import Path
import json
H=Path(__file__).parent
assert not (H/'update_deck_nativeTraining122.mjs').exists()
s=(H/'update_deck_materialGeometry122.mjs').read_text(encoding='utf-8-sig').replace('materialGeometry122','nativeTraining122')
s=s.replace("build+'/headSupport122-deck-receipt.json'","build+'/materialGeometry122-deck-receipt.json'")
s=s.replace("const figure=evidence+'/../clip03-head-level-support/facade_post_context.png';","const figure=evidence+'/../../native-training-integration/existing-backend-path/synthetic_comparison.png';")
a=s.index("replace('头部高度");b=s.index('const images=',a)
s=s[:a]+"""replace('对应证据接入三维评测','原生训练与小高斯剔除错误');
replace('沿用围栏反例','合成目标 / 从头初始化 / 120步后；上排训练视角，下排留出。');
replace('逐视角检查','20万点被误剔除；仅改裁剪后，30681点获得位置更新。');
replace('已修复0.024','生产修复待做。该后端尚未用于历史人物，不能归因旧重影。');
"""+s[b:]
s=s.replace("alt:'远楼与重复围栏立柱的三帧原生上下文，图片无新增文字'","alt:'已知合成目标、从头初始化和120步结果，上为训练视角，下为留出视角；图片无新增文字'")
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index('const candidatePath=',a)
notes=[
 '2026-10-28 T0a/T0b。保持用户确认的第一YAML4000×3000基准，经下采样和居中裁剪。实际K_rect及整数裁剪实现仍未独立实证，本轮没有修改人物相机、重新BA或人物训练。',
 '只读源码核查：历史119自由衣物中心训练已经直接优化世界XYZ，但使用传统屏幕空间gaussian_renderer。新native_training122组件连接已有gsplat1.5.3 UT和ray-3D内核，使用自由XYZ、完整协方差、SH3、空Adam、固定完整K与外参、准确前景RGB和轮廓有效区域、1000步审阅钩子。没有修改119、113或运行中的原生查看器。',
 'T0a预登记一次已知合成96点、4训练视角和2不参与优化视角、120步。两个启动实现错误分别是Windows到WSL路径斜杠和未加入已存在gsplat源码路径，都在合成优化前停止并保留日志；仅做必要修复，无依赖下载。随后实际优化120步完成，每个训练视角30次，优化器由0状态到5参数组。全部96中心移动，位移中位数0.037419世界单位。',
 '训练目标比初始0.122605338544，留出目标比初始0.421893873318；图中左右分别为已知目标/初始/训练后，上下为训练相机0及留出相机4，每格160×160原始像素。实际审阅可见形状和颜色改善，但留出结果仍有模糊与差异。12个轨道视角已审阅，是粗糙合成椭球，不代表人体新视角能力。教师与学生同用已知后端，不是独立物理标定真值。',
 'T0a有11项先行控制通过，包括前景外RGB无损失/梯度、遮挡区RGB和alpha不监督、全部训练视角/空优化器/中心更新、6记录相机完整协方差SH的PLY原生回读、审阅拒绝在第2步停止、缺1000步审阅拒绝、旧输出不可覆盖、合成KB4有限非零几何梯度。RGB回读最大9.54e-7、alpha1.07e-6。整套T0a仍失败：200000点64×64一步检查所有位置梯度为0，不能以小规模通过作为完整接入通过。',
 'T0b注册单因子诊断，冻结同一20万点、相机K/外参、尺度0.004/0.005/0.003、透明度0.1、eps2d0.3、SH及所有CUDA代码，仅切换fully_fused_projection_with_ut的可选透明度参数。原后端与底层算子重接原规则的RGB、alpha及位置梯度完全一致，均全部剔除。',
 '源码ProjectionUT3DGSFused.cu约169至180行在包围范围计算中先以opacity乘二维模糊补偿，小于1/255则剔除。但classic ray-3D后续使用原opacity。20万个点的补偿后透明度最小0.001563876、中位0.002363345、最大0.003685487，全部低于1/255。仅关闭该提前裁剪后20万点全部获得有效范围，30681点有非零位置梯度且在相同Adam学习率下一步更新，所有参数梯度有限。未通过改相机/放大尺度/提高透明度或改阈值获取结果。',
 '另一个独立NumPy控制将一颗同尺度各向异性高斯放在精确像素射线上，按完整偏心K和刚体相机直接计算Mahalanobis射线响应。预期中心alpha约0.1，原后端给0，关闭错误裁剪的CUDA给0.100000024，整图RGB/alpha最大差2.235e-8。5项T0b控制通过，解释的是新合成零梯度错误。这是响应数值差，不是重投影像素误差。',
 '生产修复仍待做；本轮低层连接只是诊断，没有替换native_portrait_renderer122/scene包约定或当前查看器。该后端尚未训练历史人物模型，不能称这是历史多鼻子/全身白边的根因。人物独立材料、静止性、实际输出K及全帧注册仍不通过，当前7帧面部最大13.05px。',
 '下一步登记响应一致的保守包围范围适配，明确新后端版本并同步原生包回读，验证正常尺度/SH/协方差/针孔/KB4/梯度/回读和这同一小高斯反例。不要修改运行源码、静默替换后端或整套重复T0a，不新增人体拟合/训练或参数扫描。原生生产接口、实拍全分辨率容量、增密、初始化与训练准入尚未完成。119历史点云保持。',
 '来源：robust-portrait-122/native-training-integration/plan.json、existing-backend-path/result.json和synthetic_comparison.png、subpixel-culling-diagnosis/plan.json/result.json与review.json。以下保留历史诊断：'
]
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');\n"+s[b:]
s=s.replace('02_有限视角重建优化路线_122对应证据接入.pptx','02_有限视角重建优化路线_122原生训练接入.pptx')
(H/'update_deck_nativeTraining122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_materialGeometry122.py').read_text(encoding='utf-8-sig').replace('materialGeometry122','nativeTraining122')
(H/'preserve_deck_nativeTraining122.py').write_text(p,encoding='utf-8')
print('T0_DECK_PREPARED')
