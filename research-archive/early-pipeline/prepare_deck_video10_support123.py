from pathlib import Path
from PIL import Image
import json
H=Path(__file__).parent
N=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-semantic-role')
tag='video10-support123'
# Existing native pixels, first3observations only. No labels or geometric warp.
Image.open(N/'material-review/contexts/row_003_family_244.png').crop((0,0,1152,384)).save(N/'material-support-evidence.png')
s=(H/'update_deck_video10_semantic123.mjs').read_text(encoding='utf-8').replace("tag='video10-semantic123'",f"tag='{tag}'")
s=s.replace("/robust-portrait-122/video10-temporal-bridge'","/robust-portrait-122/video10-semantic-role'")
s=s.replace("b+'/video10-temporal123-deck-receipt.json'","b+'/video10-semantic123-deck-receipt.json'")
s=s.replace("root+'/visual-review.json'","root+'/material-visual-review.json'")
s=s.replace('review.reviewed_rows!==96','review.reviewed_rows!==113')
a=s.index("replace('人体框内的静止背景'");z=s.index('const notes=',a)
s=s[:a]+"""replace('人体框误排背景','视频10的材料对应与相机连接');
replace('33帧遮罩已审阅','113族已审阅，合并粗背景60FIT／27HELD。');
replace('1帧遮罩失败','连接计数最多覆盖5／33视角，尚未重算相机。');
replace('门框案例','空调局部（第27／54／80帧）。粗对应可辨认，亚像素尚未验证。');
const figure=root+'/material-support-evidence.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'第27、54和80帧同一门上空调的原生384像素裁片，图内没有新增文字。粗局部可辨认不代表亚像素定位。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
"""+s[z:]
a=s.index('notes.text=');z=s.index('const candidatePath=',a)
notes=['2026-10-29 视频10材料对应与相机连接。',
 '113个冻结族的全部原生观察和16组扩大上下文已实际审阅。96个恢复族含64FIT/32HELD，17个原背景族为失败遮罩617的独立控制，原617遮罩仍被拒绝。原坐标、描述子、99对匹配、整族划分、K均保持不变。',
 '本轮可辨认粗背景43FIT/18HELD。与先前27个粗背景族按原族去重后共60FIT/27HELD，其中非地面刚性候选45FIT/21HELD。数量是可辨认粗局部，不是认证独立物理点、逐点亚像素真值或无偏匹配准确率。',
 '反光锅面、重复鞋纹/地砖/飞镖盘等仍保留歧义。数字显示屏内容在20:14与20:15间改变，数字不能当作固定材料。不能因为同一物体被看见，就断言检测中心对应同一表面点。',
 '最大共享FIT为806/859帧的12个粗族。所有达到原12点初始化条件的起始对均已做只读计数扩展，后续要求8个当前潜在三维局部支持。乐观闭包仅752/779/806/832/859共5视角，另有18个视角单帧不足8个已审阅FIT。该运算没有估计F/E、相机、三维点或人体。成功三角化可能比计数更严格。',
 '已经确认人体外接框误排部分静止背景，但修复区域语义尚未建立整圈相机约束。不能把本轮新视频证据唯一归因于旧灰衣SMPL-X的全部错位。旧模型的相机物理误差、材料歧义、人物微动、模板表达及训练形变/静态导出仍分别保留。',
 '下一步对连接断裂只登记一次相同原生查询位置的RootSIFT/RoMa外观对应对照，复用现有模型，不再重跑SIFT、不下载权重，不以置信度或双向一致性当物理真值。新相机和从头训练仍需完整准入。',
 '图像来源：video10-semantic-role/material-review/contexts/row_003_family_244.png的前3个原生384像素裁片。没有新增文字、缩放或重采样。',
 '证据：'+str(N/'material-visual-review.json')+'；'+str(N/'material-support.json')+'；'+str(N/'graph-audit-result.json')]
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');\n"+s[z:]
s=s.replace('02_有限视角重建优化路线_123视频10语义遮罩核验.pptx','02_有限视角重建优化路线_123视频10材料连接核验.pptx')
s=s.replace("b+'/preserve_deck_video10_semantic123.py'","b+'/preserve_deck_video10_support123.py'")
(H/'update_deck_video10_support123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_video10_semantic123_deck.py').read_text(encoding='utf-8').replace('video10-semantic123',tag)
(H/'commit_video10_support123_deck.py').write_text(c,encoding='utf-8')
print('prepared only, authoring not yet run')
