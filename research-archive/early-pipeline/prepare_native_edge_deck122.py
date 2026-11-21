from pathlib import Path
import json
H=Path(__file__).parent
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-edge-profile')
r=json.loads((O/'review.json').read_text(encoding='utf-8'));assert r['actual_review_complete']
s=(H/'update_deck_residualEdge_122.mjs').read_text(encoding='utf-8')
s=s.replace('edgeAssociation-122','nativeEdgeProfile-122').replace('residual-edge-association','native-edge-profile')
s=s.replace("root+'/edge-comparison.png'","root+'/edge-profile-comparison.png'")
s=s.replace("build+'/residualLinesVerified-122-deck-receipt.json'","build+'/edgeAssociation-122-deck-receipt.json'")
s=s.replace('review.rows_reviewed!==11','review.uniform_rows!==11')
start=s.index("replace('残余畸变补偿");end=s.index('const images=',start)
s=s[:start]+"""replace('残余畸变测量中的取边错误','原生像素定位与模型偏差');
replace('窗框与横向建筑边的原生等间隔裁片。','两条建筑轮廓的原生等间隔裁片。');
replace('最强梯度会跳邻边，干扰补偿参数的估计。','部分轮廓抖动减小，真实定位精度仍未确认。');
replace('连续关联仍缺可靠覆盖，尚未应用补偿或新训练。','明暗响应与邻边混合会产生系统偏差，尚未训练。');
"""+s[end:]
s=s.replace('上行为A3窗框，下行为B2建筑横边，原生96px裁片，保持比例，图内无文字','上行为C1，下行为D1，原生96px裁片，保持比例，图内无文字')
start=s.index('p.resolve(slide.id).speakerNotes.text=');end=s.index(".concat([previous]).join('\\n\\n');",start)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 D4原生整数像素局部边缘成像剖面定位。',*r['findings_zh'],
 '本次是测量模型控制，不是新镜头补偿。3个局部几何参数为边偏移、局部方向和模糊宽度，各颜色通道使用仿射亮度平面和阶跃幅值，最小二乘消去线性光度参数。积分覆盖原生像素面积，未将原图二次去畸变或插值检测。',
 '1320处全部结果保留。统计直线度时沿用D1被冻结的旧有效点，不按D4残差或边界命中筛掉样本。1306数值收敛与943触界不能代表材料真实性。',
 '36个独立面积积分已知单边控制的最大定位误差约1.02e-8px，属于与模型接近的合成实现检查，不是实拍精度。更关键的模型失配控制：gamma-after-blur产生0.446784px偏差、形式标准差0.001663px；双边误拟合成单边产生1.545932px偏差、形式标准差0.008724px。真实视频未确定采用该gamma，不能将控制数值当成真实相机的偏差。',
 '图片上C1、下D1，每行7个96px原生裁片，取索引0/20/40/60/80/100/119。外部裁片展示不改变训练原图，图内无文字。',
 '本组建筑边D1至D4到此有界结束，未使用任何新补偿相机或提取器用于人物训练。后续审计完整项目证据与剩余可辨识性，不扫阈值、扩大sigma边界或重复旧试验。',
 '证据：robust-portrait-122/native-edge-profile/{plan,controls,result,summary,review}.json及原生裁片。其他51页保持不变，以下保留历史记录。']
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[end:]
s=s.replace('preserve_deck_residualEdge_122.py','preserve_deck_nativeEdge_122.py').replace('02_有限视角重建优化路线_122取边关联核验.pptx','02_有限视角重建优化路线_122原生边缘定位.pptx')
(H/'update_deck_nativeEdge_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_residualEdge_122.py').read_text(encoding='utf-8').replace('edgeAssociation-122','nativeEdgeProfile-122')
(H/'preserve_deck_nativeEdge_122.py').write_text(p,encoding='utf-8')
print('NATIVE_EDGE_DECK_READY')
