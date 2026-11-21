from pathlib import Path
import json
from PIL import Image
H=Path(__file__).parent;tag='pattern-coverage123'
root=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/pattern-preflight/coverage')
figure=Image.new('RGB',(576,192))
for i,name in enumerate(['slot024_key017_native192.png','slot049_key021_native192.png','slot074_key021_native192.png']):figure.paste(Image.open(root/'visual-review'/name).convert('RGB'),(i*192,0))
figure.save(root/'visual-review/invalid_foot_locations_native.png')
s=(H/'update_deck_native_patch123.mjs').read_text(encoding='utf-8').replace('video10-native-patch123',tag)
s=s.replace("const prior=await read(b+'/video10-convergence123-deck-receipt.json'),review=await read(root+'/../video10-correspondence-contrast/head-roi-contrast/native-affine-contrast/visual-review/review.json');","const prior=await read(b+'/video10-native-patch123-deck-receipt.json'),review=await read(root+'/../../local-clothed-surface-123/pattern-preflight/coverage/visual-review/review.json');")
s=s.replace('review.rows.length!==24','!review.all99_context_and_face_tiles_reviewed')
a=s.index("replace('相机误差降低");b=s.index('const figure=',a)
s=s[:a]+"""replace('局部细化没有修复人像对应','脚部预测落在桌面和上衣');
replace('24组实拍查询','99帧核验：54帧出现出画脚点，图内点仍有错误部位对应。');
replace('鼻嘴查询虽能闭环','人体mask内的错误脚点仍可能拉偏拟合，置信度不能代替可见性。');
replace('同一鼻嘴查询','第120、245、370帧，预测点居中的原生192px裁片。图像未扭曲。');
"""+s[b:]
s=s.replace("root+'/../video10-correspondence-contrast/head-roi-contrast/native-photometric-contrast/visual-review/row19_source_baseline_refined.png'","root+'/../../local-clothed-surface-123/pattern-preflight/coverage/visual-review/invalid_foot_locations_native.png'")
a=s.index("alt:'");b=s.index("'});",a)
s=s[:a]+"alt:'video4第120、245、370帧的原生192像素裁片，查询中心为预测脚点，实际可见部位分别为桌面、上衣和袖子。无图内文字或像素形变。'"+s[b+1:]
notes=[
'2026-10-29。新人物video4.mp4（pattern花纹上衣）区域核验完成。SHA256 d9555dc04d313f4615ad4de08640f0b44efb097b4654151bc27bb946a71455d5。491原帧按既定stride5解码99帧，2704x1520，不二次去畸变。',
'仅一次现有官方YOLOv8n与RTMW-L推理。YOLO640/conf0.45/IoU0.7；RTMW288x384，RGB官方归一化、1.25padding、0.04bodymargin、现有argmax解码，无翻转或阈值扫描。99帧人物区域连续，但检测身份与精确解剖可见性不同。48.36秒完成。',
'原生逆仿射与独立算术差值0px，这是坐标映射一致性，不能当真实关键点精度。所有133点及原始SimCC响应保留。594个脚部预测有105个超出原图，涉及54帧。其余489个在图内的预测并未因此通过可见性认证。',
'实际审阅99个上下文与99个人脸原生裁片、5个512px预测脚部上下文及4个192px点位裁片。接触表在工具显示时缩小，只用于覆盖审阅，不用于亚像素判断。192px点位裁片按原生尺寸单独查看。',
'第120帧slot24/key17 left_big_toe=(1412.86955,1370.47018)，原图对应桌面。第245帧slot49/key21 right_small_toe=(1530.58454,974.92645)，原图对应上衣/头发。第370帧slot74/key21=(1218.85655,939.44572)，原图对应袖子。第490帧slot98/key17=(1390.82499,1429.59519)，原图对应桌边设备/线缆附近。四例均不是该脚部物理点。',
'现有image_support已经严格拒绝出画点，并计算离personmask距离，保留4px容差。它不验证解剖部位，落在上衣内的脚点仍可能通过该过滤。可选mesh_surface_zbuffer是模型自遮挡近似，源码明确external_occluders_modeled=False。尚未为pattern生成mask或拟合，不能声称上述105个出画点已进入训练或实际权重不为零。',
'99帧只覆盖左侧、正面和右侧，未显示背面；腿脚经常被桌子遮挡或截断，个别视角有部分粉色鞋。头发遮挡侧脸，slot36/source180与slot79/source395可见闭眼例。人体微动没有量化，重复百叶窗/反射不能自动作为固定背景真值。',
'旧灰衣SMPL-X鼻嘴真实错位，原面部中位43.99px，旧全身门槛约50至53px不足以保证面部精度。当前新训练入口已有独立精细对齐准入。此次新人物的点位错误不能唯一解释旧人物的内外参、材料身份和微动。',
'下一步预登记一次99帧原生纹理候选：全局2500、人物1500、头部1000预算，独立近重复族与整族留出，在任何几何残差前冻结。只做材料支持核验，不把RTMW编号当跨视角物理点。新相机、SMPL-X和Gaussian训练均未开始。独立可信面部HELD仍0。',
'证据：local-clothed-surface-123/pattern-preflight/coverage/result.json、support-audit.json、visual-review/review.json、decision.json及native-material-plan.json。前51页不变，原历史PLY不变，未验证00部署或WPS云往返。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123原生匹配细化核验.pptx','02_有限视角重建优化路线_123遮挡关键点核验.pptx')
s=s.replace('preserve_deck_native_patch123.py','preserve_deck_pattern_coverage123.py')
(H/'update_deck_pattern_coverage123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_native_patch123_deck.py').read_text(encoding='utf-8').replace('video10-native-patch123',tag)
(H/'commit_pattern_coverage123_deck.py').write_text(c,encoding='utf-8')
print('Prepared one slide52 edit; no output deck replaced.')
