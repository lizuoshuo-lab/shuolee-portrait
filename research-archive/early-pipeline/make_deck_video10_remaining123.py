from pathlib import Path
import json
H=Path(__file__).parent
s=(H/'update_deck_video10_matcher123.mjs').read_text(encoding='utf-8-sig')
s=s.replace("tag='video10-matcher123'","tag='video10-remaining123'")
s=s.replace("/video10-correspondence-contrast'","/video10-remaining-material-review'")
s=s.replace("b+'/video10-support123-deck-receipt.json'","b+'/video10-matcher123-deck-receipt.json'")
s=s.replace("!review.reviewed_all72||review.rows.length!==72","review.reviewed_rows!==222||review.rows.length!==222")
start=s.index("replace('视频10的材料对应'")
end=s.index('const candidatePath=',start)
notes=[
    '2026-10-29 视频10材料与相机初始化核验。现有SMPL-X对齐问题尚未解决。本页不表示产生了合格新相机、人体或高斯模型。',
    '剩余222个代表族的696个原生观测、19页裁片和16组扩大上下文全部实际审阅。120个为可辨认粗固定局部，94个仍有歧义，6个非刚性材料静止性未认证，2个物理身份矛盾。原像素、FIT/HELD和旧轨迹不变。',
    '合并此前审阅共162个FIT、45个HELD粗背景族。其中非地面刚性候选91个FIT和29个HELD。每个33诊断视角均有至少8个粗FIT，但这不等于共享三维相机路径或亚像素真值。原视频860帧尚未完整注册。',
    '图为第196行、FIT族1338、第725/752/779帧；每格中心为保存的查询位置，分别落到同一飞镖盘的不同扇区。另一矛盾为第172行、HELD族1275。两者从后续使用中隔离，未按重投影误差删点，未把HELD转FIT。不能把同一物体当成同一物理点。图内没有新增文字和形变。',
    '沿用初始共享FIT>=12、扩展已有三维FIT>=8的顺序计数，已审阅轨迹最多覆盖6/33视角。即使把所有可用未认证FIT和同族分支理想化合并，共527族，上限也为14/33。分支并集不是合法物理轨迹，禁止直接用于真实相机求解。',
    '首版计数错误漏掉8个已人工从失败遮罩中恢复的背景族。修正为原候选和已审阅材料的并集，162/527均包含已审阅基线；6和14的计数未变。初始文件和纠错计划保留。',
    '进一步采用原162个FIT代表的确切可见性，33相机、563观测，生成一次已知非退化合成几何，不使用真实二维位置或旧优化相机。移除首相机6自由度及全局尺度后，Jacobian为1126x677、秩677、无额外局部连续歧义。12列中心有限差分和已知尺度规范通过，最小列归一化奇异值0.00573194。',
    '合成满秩不证明真实内参、视差、物理身份、动态场景、全局唯一解或面部精度。它说明不能把顺序注册失败等同于轨迹结构必然不可解，应单独核验全局共享轨迹初始化。旧顺序计数保护不被悄悄降低。',
    '独立且物理可信的面部HELD仍为0。实际相机、输出K、人物微动和材料对应不能唯一归因。本轮没有真实相机估计、SMPL-X重拟合或高斯训练，没有替换历史查看器模型。',
    '下一步已登记一次全局初始化前置核验：从原始记录封装浮点观测、固定K与像素约定，检查本地已有求解器接口；真实候选预算和独立背景保护另行登记后才执行。不开新权重下载，不重新提同批特征，不二维扭曲图像。',
    '证据目录：C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-remaining-material-review。主要文件：decision.json、visual-review.json、ceiling-corrected-result.json、generic-observability/result.json。'
]
replacement="""replace('遮挡误配也能通过往返检查','相机初始化与重复纹理错配');
replace('冰箱背景点转到人物','新审阅222组材料；同一物体仍可能匹配到错误部位。');
replace('固定置信度门槛','相同轨迹的合成连接结构可解，实际相机仍待验证。');
replace('第644／698帧','第725／752／779帧：飞镖盘裁片中心落在不同扇区。');
const figure=root+'/review/contexts/row_196.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'三个原生384像素裁片，每格中心为保存的查询位置，落到同一飞镖盘的不同扇区。图内无新增文字。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text="""+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');\n"
s=s[:start]+replacement+s[end:]
s=s.replace('02_有限视角重建优化路线_123视频10匹配对照.pptx','02_有限视角重建优化路线_123视频10材料与初始化.pptx')
s=s.replace("'/preserve_deck_video10_matcher123.py'","'/preserve_deck_video10_remaining123.py'")
(H/'update_deck_video10_remaining123.mjs').write_text(s,encoding='utf-8')
print('Prepared existing-slide52 edit; no authoring started yet.')
