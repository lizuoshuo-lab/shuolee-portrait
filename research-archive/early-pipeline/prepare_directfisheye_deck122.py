from pathlib import Path
import json
H=Path(__file__).parent;R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
research=json.loads((R/'directfisheye-research/review.json').read_text(encoding='utf-8'))
t7=json.loads((R/'native-capacity/review.json').read_text(encoding='utf-8'))
assert t7['actual_review_complete'] and research['source_files']==21
s=(H/'update_deck_nativeEdge_122.mjs').read_text(encoding='utf-8')
s=s.replace('nativeEdgeProfile-122','directFisheyeAudit-122').replace('/native-edge-profile','/directfisheye-research')
s=s.replace("const figure=root+'/edge-profile-comparison.png';","const figure=root+'/../native-capacity/capacity-control.png';")
s=s.replace("build+'/edgeAssociation-122-deck-receipt.json'","build+'/nativeEdgeProfile-122-deck-receipt.json'")
s=s.replace("if(review.actual_review_complete!==true||review.uniform_rows!==11)throw Error('Review incomplete');","if(review.source_files!==21||review.training_executed!==false)throw Error('Review incomplete');")
start=s.index("replace('残余畸变测量");end=s.index('const images=',start)
s=s[:start]+"""replace('原生像素定位与模型偏差','DirectFisheye-GS训练适配');
replace('两条建筑轮廓的原生等间隔裁片。','本机合成控制：20万点、1520×2704的一次更新。');
replace('部分轮廓抖动减小，真实定位精度仍未确认。','CVO可借鉴，主点与全视角增密需要适配。');
replace('明暗响应与邻边混合会产生系统偏差，尚未训练。','原生回读仍未通过，尚未验证人像重影改善。');
"""+s[end:]
s=s.replace('上行为C1，下行为D1，原生96px裁片，保持比例，图内无文字','本机T7合成控制的RGB与alpha整体及原生裁片，非DirectFisheye训练结果，图内无文字')
start=s.index('p.resolve(slide.id).speakerNotes.text=');end=s.index(".concat([previous]).join('\\n\\n');",start)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 DirectFisheye-GS官方源码调研与本机T7原生容量控制。',*research['findings_zh'],
 'CVO对照应固定相机、同一初值和渲染器、每次两个视图、曝光预算及更新预算，单独改变关联与随机配对。关闭额外EMA、颜色映射与相机优化，归一化损失并汇总所有视角的增密统计。这是设计，尚未执行。',
 '官方论文：https://arxiv.org/html/2604.00648v1 ，项目页：https://yzxqh.github.io/DirectFisheye-GS/ ，官方源码：https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS/tree/db9b3a25bff79a59f02ce90f5bd235cb59c03aaa 。文件、哈希与具体行位置记录在robust-portrait-122/directfisheye-research。',
 '本页图片来自本机T7已知合成几何测试，不是DirectFisheye-GS或真实人物训练结果。图左两个缩略图依次为RGB和alpha，右侧为同区域456×384原生裁片，无变形、无图内文字。',
 *t7['findings_zh'],
 '研究结论与T7/T7b完整证据已写入原进展页及原工作日志。以下保留历史技术路线记录。']
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[end:]
s=s.replace('preserve_deck_nativeEdge_122.py','preserve_deck_directFisheye_122.py').replace('02_有限视角重建优化路线_122原生边缘定位.pptx','02_有限视角重建优化路线_122DirectFisheye核验.pptx')
(H/'update_deck_directFisheye_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_nativeEdge_122.py').read_text(encoding='utf-8').replace('nativeEdgeProfile-122','directFisheyeAudit-122')
(H/'preserve_deck_directFisheye_122.py').write_text(p,encoding='utf-8')
print('DIRECTFISHEYE_DECK_READY')
