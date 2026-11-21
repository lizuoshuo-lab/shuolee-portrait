from pathlib import Path
import json
H=Path(__file__).parent;tag='video10-native-patch123'
s=(H/'update_deck_video10_convergence123.mjs').read_text(encoding='utf-8').replace('video10-convergence123',tag)
s=s.replace('video10-joint-tracks123-deck-receipt.json','video10-convergence123-deck-receipt.json')
s=s.replace("read(root+'/joint-track-candidate/convergence-repair/visual-review/review.json')","read(root+'/../video10-correspondence-contrast/head-roi-contrast/native-affine-contrast/visual-review/review.json')")
s=s.replace('review.rows.length!==4','review.rows.length!==24')
a=s.index("replace('共享三维点");b=s.index('const figure=',a)
s=s[:a]+"""replace('相机误差降低，人体对齐仍未通过','局部细化没有修复人像对应');
replace('留出中位3.36降到1.29px','24组实拍查询：原匹配4组通过，两种细化均只有3组。');
replace('33个观测变差','鼻嘴查询虽能闭环，反向定位仍错2.40px。停止这条局部细化路线。');
replace('第484／510／537帧','同一鼻嘴查询：源图、原匹配、平移细化。未改相机，未启动训练。');
"""+s[b:]
s=s.replace("root+'/joint-track-candidate/convergence-repair/visual-review/heldout_834_part_1.png'","root+'/../video10-correspondence-contrast/head-roi-contrast/native-photometric-contrast/visual-review/row19_source_baseline_refined.png'")
a=s.index("alt:'");b=s.index("'});",a)
s=s[:a]+"alt:'原始第19组鼻嘴胡须查询，源图、原第二视角匹配、原生平移细化。3个384px原生裁片，未扭曲，未在图内写字。粗材料相似不能证明亚像素身份。'"+s[b+1:]
notes=[
'2026-10-29。SMPL-X对齐仍未通过。此次只完成局部光度匹配分支的合成控制和实拍对照，及另一人物video4的原生输入核验。没有改K、外参、SMPL-X或高斯。',
'24个已冻结原生查询，31px局部窗口，2D平移加亮度增益偏置，已知平移0.35/-0.40px、增益1.05、偏置2/255、噪声0.5/255、固定种子12313。24/24合成控制通过，中心误差中位0.008462px、最大0.039676px。只验证理想复制图小变换的实现。',
'实拍复用整图RoMa结果，不做新推理。相同24组的A-B-C-A和各边反向定位共144拟合，统一30评估预算。保留原置信度0.9、双向与循环小于1px和参数边界限制。原4/24通过变成3/24，通过集为行12、16、20，无新增通过，行19失败。118/144成功，37次触及边界。',
'第19行是FIT族163鼻嘴/胡须附近。原循环0.174px，平移细化循环0.585px但第一段FB误差2.403px且有边界/求解失败。循环自洽不是物理对应精度，未删观察或转移HELD。图像为三列384px原生裁片，中心查询，图内无文字。',
'冻结局部解的只读导数投影显示，原4组通过中的3组满足预登记的缺失仿射几何支持条件。因此只追加一个不同几何模型的有界对照，无窗口/阈值扫描。该24组已经用于方法开发，不能当独立验证集。',
'仿射几何控制已知矩阵[[1.03,.02],[-.015,.98]]、同平移与亮度、噪声和30评估预算，8列导数通过差分核验。24/24通过，中心最大0.019947px、角点最大0.068512px。',
'真实仿射结果仍3/24通过，较平移无新增通过。95/144求解成功，52次触及边界。实际审阅两方案各24组120个原生192px裁片，另看鼻嘴384px上下文。耳/镜腿/胡须粗对应可辨，颈侧、头顶和背景存在遮挡与材料歧义。两方案均拒绝，局部细化分支结束。',
'失败不能唯一归因于内参、外参、残余畸变或人物微动。相机先前修复的HELD仍93/161不小于1px、最大42.07px。物理认证的面部HELD为0，33诊断视角未覆盖860训练帧。',
'旧灰衣SMPL-X面部残差中位43.99px，旧全身门槛容许约50至53px，没有面部解剖可见性保护。这是已证实的放行缺陷。独立投影算术误差小于0.0004px，不等于物理相机准确。',
'所有7个历史源视频经完整SHA256匹配已确认只改名。花纹人物pattern是video4.mp4，初次旧路径不存在时在解码前停止并保留记录，哈希确认后只修正输入路径，未更换视频。已提取99/491原生2704x1520帧并保存原帧号/时戳。',
'实际审阅video4的0、120、245、370、490帧，桌子遮住下身与脚，侧脸有头发遮挡。未见背面只描述这5个样本，不推广全部99帧。下一步只核验区域与可见性，保持相机、SMPL-X和训练准入独立。',
'证据：robust-portrait-122/video10-correspondence-contrast/head-roi-contrast下native-patch-control、native-photometric-contrast、native-affine-contrast；local-clothed-surface-123/pattern-preflight及source-name-reconciliation.json。前51页保持原字节与渲染。没有新模型、00部署或云同步验证。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123视频10数值修复与局部匹配.pptx','02_有限视角重建优化路线_123原生匹配细化核验.pptx')
s=s.replace('preserve_deck_video10_joint_tracks123.py','preserve_deck_native_patch123.py')
(H/'update_deck_native_patch123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_video10_convergence123_deck.py').read_text(encoding='utf-8').replace('video10-convergence123',tag)
(H/'commit_native_patch123_deck.py').write_text(c,encoding='utf-8')
print('Prepared same slide52 edit only.')
