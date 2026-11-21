from pathlib import Path
import json
H=Path(__file__).parent;tag='video10-convergence123'
s=(H/'update_deck_video10_joint_tracks123.mjs').read_text(encoding='utf-8').replace('video10-joint-tracks123',tag)
s=s.replace('video10-global-ambiguity123-deck-receipt.json','video10-joint-tracks123-deck-receipt.json')
s=s.replace("read(root+'/joint-track-candidate/visual-review/review.json')","read(root+'/joint-track-candidate/convergence-repair/visual-review/review.json')")
a=s.index("replace('相同5个点");b=s.index('const figure=',a)
s=s[:a]+"""replace('共享三维点后，相机仍未对齐','相机误差降低，人体对齐仍未通过');
replace('留出观测128/161未低于1px','留出中位3.36降到1.29px，93/161个观测仍不小于1px。');
replace('拟合中位0.78px','33个观测变差。头部原生裁片未改善匹配，尚未训练。');
replace('第617／644／671帧','第484／510／537帧：保留误差增大的材料对应。');
"""+s[b:]
s=s.replace("root+'/joint-track-candidate/visual-review/heldout_1126_frames_617_644_671.png'","root+'/joint-track-candidate/convergence-repair/visual-review/heldout_834_part_1.png'")
s=s.replace('三个384px原生裁片，中心是桌面袋状物附近的冻结材料查询位置；无缩放、扭曲或图内文字。袋状材料物理固定性与精确对应未认证。','三个384px原生裁片，中心是盆栽及灯箱／柜体附近的冻结查询。第537帧误差由2.83增至42.07px，精确身份尚未认证，保留反例和图边界填充。')
notes=[
'2026-10-29，相同K、初值、563FIT及161HELD观测的唯一数值修复。相机改善但SMPL-X对齐、训练准入和人像质量均未通过。',
'首先以既有33相机162点可见图执行完整合成控制：原生观测噪声0.2px，旋转扰动1度，位置和点扰动0.02。解析Jacobian12列独立有限差分通过，175评估收敛，最终重投影均值0.1485px，最大0.4649px。合成通过仅验证实现。',
'真实修复从同一原始初始化重启，不使用之前80次的优化相机。初始残差差异小于4.55e-13，12列实拍解析Jacobian核验通过。K、观察、权重、第一相机和尺度固定。解析导数和数值预算一起变化，不能冒充单因素物理实验。',
'一次登记的2000评估／120秒预算，实际41.03秒。终止status0、successfalse、nfev2000、njev1951，cost73.51、optimality0.56445，仍未收敛。FIT中位0.3350px、最大4.7676px，全部正深度。不继续增加该轮预算。',
'候选冻结后再评测同45条HELD的161次观察。用其余全部视角三角化预测留一视角。中位3.3604降到1.2913px，P90为3.5772px，最大414.5369降到42.0674px。68个严格小于1px，93个不通过。物理面部HELD仍0，33视角不能替代860帧注册。',
'128个观测改善，33个变差。8个原通过点变为不通过，43个原失败点通过。全部反例保留，不按残差删点，不用HELD选参数、根或观察。',
'相对旋转698至752从错误分支附近移到此前FIT三视图更支持的根附近，角差0.1267度。事后比较，未使用HELD选根，不等于物理真值或唯一解。',
'本页为误差反增族834在484、510、537帧的无缩放384px裁片。第537帧2.83增至42.07px。盆栽和灯箱／柜体边缘可辨认，黑条是图外填充，另查192px精确中心仍不足以证明错接。另实际审阅77镜框、978冰箱和1126袋状物，共17个观察、7张图。',
'额外完成一次冻结查询的原生头部裁片对照。原整图2704x1520压到798x448粗特征及1344x756细特征，头部粗特征约97至112乘130至159。新768整数裁片以448粗、768细匹配，局部上下文和有效分辨率共同改变。未检测新点或修改训练图。',
'三对图、每帧600相同查询、六方向。六个有向三视图门槛计数都下降，示例644至671至698从62降23。实际查看预登记24组、120张192px原生裁片，有可辨认耳部／镜腿／胡须，也有背景和遮挡误配。少数新通过只代表自洽，仍未认证亚像素物理精度。该ROI方案不采用，无重跑或门槛扫描。',
'旧灰衣SMPL-X参与面部点残差中位43.99px，粗身体门槛容许约50至53px，缺解剖可见性。帧、编号、独立投影数值差小于0.0004px。当前新视频10诊断不能直接证明旧灰衣唯一根因。',
'后续仅登记原生纹理二维可测性及已知亚像素变换合成控制。实拍局部光度细化须单独登记，仍不拟合新相机或训练人体。证据保存在video10-global-initializer/joint-track-candidate/convergence-repair和video10-correspondence-contrast/head-roi-contrast。前51页不变。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123视频10共享点联合核验.pptx','02_有限视角重建优化路线_123视频10数值修复与局部匹配.pptx')
(H/'update_deck_video10_convergence123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_video10_joint_tracks123_deck.py').read_text(encoding='utf-8').replace('video10-joint-tracks123',tag)
(H/'commit_video10_convergence123_deck.py').write_text(c,encoding='utf-8')
print('Prepared one slide52 edit; canonical source unchanged.')
