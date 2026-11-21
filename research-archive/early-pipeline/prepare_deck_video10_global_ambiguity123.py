from pathlib import Path
import json
H=Path(__file__).parent
s=(H/'update_deck_video10_global_preflight123.mjs').read_text(encoding='utf-8')
s=s.replace('video10-global-preflight123','video10-global-ambiguity123').replace('video10-remaining123-deck-receipt.json','video10-global-preflight123-deck-receipt.json')
s=s.replace("robust-portrait-122/video10-remaining-material-review'","robust-portrait-122/video10-global-initializer'")
s=s.replace("read(root+'/visual-review.json')","read(root+'/visual-review/review.json')")
s=s.replace("if(review.reviewed_rows!==222||review.rows.length!==222)throw Error('Review missing');","if(!review.actual_saved_image_review||review.families.length!==7)throw Error('Review missing');")
s=s.replace("replace('相机初始化与重复纹理错配','相机初始化与重复纹理错配');","replace('相机初始化与重复纹理错配','相同5个点，算出3组不同相机');")
s=s.replace("replace('新审阅222组材料','新审阅222组材料；同一物体仍可能匹配到错误部位。');","replace('新审阅222组材料','三组解都使拟合误差接近零，旋转却相差约42°。');")
s=s.replace("replace('相同轨迹的合成连接结构可解','旧脚本仍先逐帧注册；全局初始化待补齐，实拍精度未过关。');","replace('旧脚本仍先逐帧注册','三视角约束使该点极线误差由86.08降至1.22px，仍未过关。');")
s=s.replace("replace('第725／752／779帧','第725／752／779帧：飞镖盘裁片中心落在不同扇区。');","replace('第725／752／779帧','第698／725／752帧：同组地面留出点的原生裁片。');")
s=s.replace("root+'/review/contexts/row_196.png'","root+'/visual-review/heldout_floor_1297_frames_698_725_752.png'")
s=s.replace('三个原生384像素裁片，每格中心为保存的查询位置，落到同一飞镖盘的不同扇区。图内无新增文字。','三个原生384像素裁片，每格中心为冻结的地面HELD查询坐标。只作材料检查，模糊及重复纹理未达到亚像素认证。图内无新增文字。')
notes=[
'2026-10-29 相机多解与SMPL-X人物对齐。新视频10的局部相机诊断，不等同于旧灰T/全部人物唯一根因，不表示产生了合格新相机、人体或高斯。',
'在冻结162个FIT、45个HELD粗材料代表，33个原生诊断视角和同一条件K上，一次58对相对姿态诊断。没有新提特征、没有优化K、没有使用旧相机或几何初始化。HELD从未进入估计或根的排序。',
'57对返回可用相对姿态，其中16条边达到预登记局部支持/视差保护，可靠图分成21组。41对少于12FIT，另有一对返回0内点/0平移失败值，3对有平面替代解释。拒绝原因可能重叠。没有通过降低12点门槛发布全局相机。',
'32个三视角旋转闭环，中位0.689度，P90为31.406度，最大67.396度；18个超过预登记0.5度筛查标记。该闭环不是独立真值，不能代替材料身份和面部像素误差。',
'第698/752帧只有5FIT，RANSAC给出几乎零Sampson距离的解，但HELD244/1297分别1.358/86.079px。这个事后案例只用于解释，不作为独立性能测试，不按误差删点。',
'对完全相同5点和K执行一次确定性PoseLib五点根枚举，得到3个全正深度解，最大FIT距离均低于1.3e-8px。相对原选解的旋转差约38.54度、41.60度、0度。说明两视图最小拟合和正深度都不能唯一确定相机。',
'只按既有FIT的698→725与725→752旋转乘积比较，三根闭环差为8.465度、0.270度、41.630度，较一致根的两HELD误差为0.237/1.223px。排序完全不看HELD。仍有10<12的支持边，局部案例且未完全<1px，未获得全局或训练准入。',
'本页照片为HELD1297在698、725、752帧的384px原生整数裁片，未缩放和二维扭曲。三帧均实际审阅，中心仍在地面局部，第二第三张左侧有人体。原图片无新增文字。所有5FIT和2HELD的原生两视角裁片均实际查看，玻璃高光、帽子软材料、模糊和重复瓷砖保持不确定，没有按残差改分类。',
'4项合成全局初始化控制通过：旋转平均、全局位置/点恢复、纯旋转拒绝、断图拒绝。最终状态序列化pid重复出错，已从保存结果恢复，不重复求解。真实两视图接口的只读平移/零平移处理错误也已修正，保留原失败记录；数据/阈值/seed未改变。',
'固定K=fx1120.8319106618264,fy1119.820634890859,cx1352.8626255714278,cy763.7770175739016。来源第一YAML4000×3000，经0.676缩放、y254裁剪、半像素约定。已去畸变MP4不二次去畸变，D=0仅是当前条件基线；实际导出K未独立认证。',
'下一步检查多视图共享轨迹对分支与尺度的约束，保持冻结材料、固定K和独立保护。当前0个完整新相机、0个新SMPL-X拟合、0步新高斯训练。物理可信面部HELD仍为0，860帧尚未完整注册。不能把极线距离当三维或面部重投影。',
'证据：C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer/decision.json；relative-graph/repair-invalid-pose/result.json；minimal-root-diagnosis/result.json和cycle-result.json。历史第52页的全部材料分析仍保存在工作日志，其他51页不变。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123视频10全局初始化核验.pptx','02_有限视角重建优化路线_123视频10相机多解核验.pptx')
(H/'update_deck_video10_global_ambiguity123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_video10_global_preflight123_deck.py').read_text(encoding='utf-8').replace('video10-global-preflight123','video10-global-ambiguity123')
(H/'commit_video10_global_ambiguity123_deck.py').write_text(c,encoding='utf-8')
print('Prepared native artifact edit for same52page deck, slide52 only.')
