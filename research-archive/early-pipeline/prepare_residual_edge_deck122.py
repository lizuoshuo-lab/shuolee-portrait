from pathlib import Path
import json
from PIL import Image
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O=R/'residual-edge-association';A=R/'residual-edge-identity'
r=json.loads((O/'review.json').read_text(encoding='utf-8'));assert r['actual_review_complete']
canvas=Image.new('RGB',(672,192))
canvas.paste(Image.open(A/'crops/02_A3_uniform.png'),(0,0));canvas.paste(Image.open(A/'crops/09_B2_uniform.png'),(0,96));canvas.save(O/'edge-comparison.png')
s=(H/'update_deck_residualLines_122.mjs').read_text(encoding='utf-8')
s=s.replace('residualLinesVerified-122','edgeAssociation-122').replace('residual-distortion-direction','residual-edge-association')
s=s.replace("root+'/structure-comparison.png'","root+'/edge-comparison.png'")
s=s.replace("build+'/nativeSupervision-122-deck-receipt.json'","build+'/residualLinesVerified-122-deck-receipt.json'")
s=s.replace('review.native_line_crops_reviewed!==13','review.rows_reviewed!==11')
start=s.index("replace('白衣");end=s.index('const images=',start)
s=s[:start]+"""replace('残余畸变补偿的独立检验','残余畸变测量中的取边错误');
replace('三帧建筑原生裁片：第0、232、463帧。','窗框与横向建筑边的原生等间隔裁片。');
replace('6条拟合、5条留出；留出误差未下降。','最强梯度会跳邻边，干扰补偿参数的估计。');
replace('候选未采用。对应错配仍在，尚未启动新训练。','连续关联仍缺可靠覆盖，尚未应用补偿或新训练。');
"""+s[end:]
s=s.replace('height:width*1100/4560','height:width*192/672')
s=s.replace('第0、232、463帧原生建筑裁片，保持比例，图内无文字','上行为A3窗框，下行为B2建筑横边，原生96px裁片，保持比例，图内无文字')
start=s.index('p.resolve(slide.id).speakerNotes.text=');end=s.index(".concat([previous]).join('\\n\\n');",start)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 D2原始取边审阅与D3只改变关联的测量试验。',*r['findings_zh'],
 'D2实际核对全部13条等间隔原生裁片及26组相邻跳变。A3最大法向偏移突变8.947px，B3为5.979px。解析区间覆盖D1全系数范围，A3/A4/B3部分三点偏离邻点弦线下界3.019/3.676/4.188px。此结论仅针对这些已观测三点和该光滑逆映射族，不是面部误差或任意镜头模型结论。',
 'D3沿相同D1灰度梯度剖面枚举全部原门槛局部峰，种子及固定2px邻接范围预登记。前向与反向都唯一才继续，歧义/缺失即停止。未填空、重播种、拟合直线、平滑曲线或扫参数，原始最强峰偏移逐值一致。',
 '图上行为A3、下行为B2，每行7个原生96px裁片，采样索引0/20/40/60/80/100/119。图中无标注，显示为整图等比例放大，不是新补偿图。',
 '旧119预览/原K/外参/输入及划分全部保留，未训练新人物。下一步解决物理边两侧及成像剖面歧义后，才可登记新的独立补偿验证。',
 '证据：robust-portrait-122/residual-edge-identity/{plan,result,interval-bound,review}.json 和 residual-edge-association/{plan,controls,result,review}.json。以下保留原阶段记录。']
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[end:]
s=s.replace('preserve_deck_residualLines_122.py','preserve_deck_residualEdge_122.py').replace('02_有限视角重建优化路线_122残余补偿_核验完成.pptx','02_有限视角重建优化路线_122取边关联核验.pptx')
(H/'update_deck_residualEdge_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_residualLines_122.py').read_text(encoding='utf-8').replace('residualLinesVerified-122','edgeAssociation-122')
(H/'preserve_deck_residualEdge_122.py').write_text(p,encoding='utf-8')
print('EDGE_ASSOCIATION_BUILDER_READY')
