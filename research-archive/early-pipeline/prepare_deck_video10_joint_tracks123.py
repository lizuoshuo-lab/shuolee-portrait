from pathlib import Path
import json
H=Path(__file__).parent;tag='video10-joint-tracks123'
s=(H/'update_deck_video10_global_ambiguity123.mjs').read_text(encoding='utf-8').replace('video10-global-ambiguity123',tag)
s=s.replace('video10-global-preflight123-deck-receipt.json','video10-global-ambiguity123-deck-receipt.json')
s=s.replace("read(root+'/visual-review/review.json')","read(root+'/joint-track-candidate/visual-review/review.json')")
s=s.replace("if(!review.actual_saved_image_review||review.families.length!==7)","if(!review.actual_saved_native_crop_review||review.rows.length!==4)")
start=s.index("replace('相机初始化与重复纹理错配'");end=s.index('const figure=',start)
s=s[:start]+"""replace('相同5个点，算出3组不同相机','共享三维点后，相机仍未对齐');
replace('三组解都使拟合误差接近零','留出观测128/161未低于1px，当前相机不能用于人体拟合。');
replace('三视角约束使该点极线误差','拟合中位0.78px，留出中位3.36px。优化仍未收敛。');
replace('第698／725／752帧','第617／644／671帧：桌面材料仍需确认精确对应。');
"""+s[end:]
s=s.replace("root+'/visual-review/heldout_floor_1297_frames_698_725_752.png'","root+'/joint-track-candidate/visual-review/heldout_1126_frames_617_644_671.png'")
s=s.replace('三个原生384像素裁片，每格中心为冻结的地面HELD查询坐标。只作材料检查，模糊及重复纹理未达到亚像素认证。图内无新增文字。','三个384px原生裁片，中心是桌面袋状物附近的冻结材料查询位置；无缩放、扭曲或图内文字。袋状材料物理固定性与精确对应未认证。')
notes=[
'2026-10-29，本轮在固定条件K上，检查视频10的多视角共享XYZ联合求解。相机、SMPL-X和人像质量均未通过，不替换历史点云。',
'材料图有33视角、162FIT轨迹、563FIT观测。保存的57条相对姿态边形成32个三角闭环，但107、510、537、564、590、617六帧没有这样的闭环。另有159组三视角共享材料组合未进入两视图姿态图，各1至4条共同轨迹。原每对12点保护未放宽。',
'单次初始化复用本轮新估的57条相对旋转，不改变五点根。连接Laplacian求全局旋转，再用全部FIT世界射线共同求中心和共享XYZ。没有旧人物几何或旧优化相机初值，没有K优化、重新匹配或留出反馈。',
'首先在已有合成几何和已知旋转上加0.2px噪声，seed12311，正深度位置求解最大残差0.6145px、均值0.1764px，控制通过。它仅验证已知旋转时的位置实现，不等于完整联合优化或实拍通过。',
'实拍单次稀疏BA固定第一相机和一个尺度分量，soft_l1尺度1px，80次函数评估预算。全部563FIT保留，拟合中位5.6136降至0.7813px，最大仍107.63px，全部正深度。优化返回status0、successfalse、nfev80、optimality2613.52，未收敛。',
'候选冻结后对45HELD轨迹的161次观测逐次排除一个视角，用其余全部观察三角化并预测排除视角。161个预测均正深度，33个误差严格小于1px、128个不小于1px，中位3.3604px、P90为32.7319px、最大414.5369px。这是原生欧氏重投影，非此前Sampson距离。没有按误差删点或挑视角。',
'事后只读实际Jacobian为1126×677，固定7维相似变换自由度。12列中心有限差分通过，列归一化SVD秩677，条件数1498.21，局部切空间鲁棒梯度最大2872.03。实际模型未达驻点，满秩不证明真实材料、K、唯一解、精度或无静态解。',
'本页图是误差最大材料族1126在617、644、671帧的原生384px裁片。中心在棕黄色袋状物附近，桌面固定不代表袋状材料一定静止或精确对应。另实际查看244通风口、696地砖地毯边缘、397木架金属器具共12个原生裁片。保留反光、模糊、软物体和重复纹理的不确定性，没有据残差重分类。',
'灰衣SMPL-X的已存实拍诊断显示面部检测残差中位43.99px，部分鼻嘴投影在脸颊或耳前。帧/编号和独立投影数值差小于0.0004px；粗身体门槛容许约50至53px，缺解剖可见性。新视频10的相机失败不是旧灰衣唯一原因的证明。',
'物理可信面部HELD仍0，860帧未完整注册。所有候选仅诊断，不准入身体拟合或高斯训练。下一步只验证解析Jacobian与完整合成联合恢复，不自动延长本轮80评估预算，也不把失败认定为数据无解。',
'证据：robust-portrait-122/video10-global-initializer/joint-track-candidate/result.json、stationarity-audit/result.json、visual-review/review.json、decision.json。上一轮三根证据和全部原图仍保留在同一工作日志。前51页保持原样。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123视频10相机多解核验.pptx','02_有限视角重建优化路线_123视频10共享点联合核验.pptx')
s=s.replace("b+'/preserve_deck_video10_global_preflight123.py'","b+'/preserve_deck_video10_joint_tracks123.py'")
(H/'update_deck_video10_joint_tracks123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_video10_global_ambiguity123_deck.py').read_text(encoding='utf-8').replace('video10-global-ambiguity123',tag)
(H/'commit_video10_joint_tracks123_deck.py').write_text(c,encoding='utf-8')
print('Prepared slide52 edit, canonical file retained until actual rendered review.')
