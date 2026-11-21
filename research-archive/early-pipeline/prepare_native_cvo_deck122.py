from pathlib import Path
import json
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
r=json.loads((R/'native-cvo-control/review.json').read_text(encoding='utf-8'))
s=(H/'update_deck_directFisheye_122.mjs').read_text(encoding='utf-8')
s=s.replace('directFisheyeAudit-122','nativeCVO-122').replace('/directfisheye-research','/native-cvo-control')
s=s.replace("const figure=root+'/../native-capacity/capacity-control.png';","const figure=root+'/cvo-native-control.png';")
s=s.replace("build+'/nativeEdgeProfile-122-deck-receipt.json'","build+'/directFisheyeAudit-122-deck-receipt.json'")
s=s.replace("if(review.source_files!==21||review.training_executed!==false)throw Error('Review incomplete');",
            "if(review.checks_CPU!==22||review.checks_GPU!==15||review.person_training!==false)throw Error('Review incomplete');")
a=s.index("replace('原生像素定位");b=s.index('const images=',a)
s=s[:a]+"""replace('DirectFisheye-GS训练适配','跨视角更新与原生参数保存');
replace('本机合成控制：20万点、1520×2704的一次更新。','同一合成几何的竖屏、横屏全幅与原生裁片。');
replace('CVO可借鉴，主点与全视角增密需要适配。','双视角统计已核验，20万点无损回读与参考一致。');
replace('原生回读仍未通过，尚未验证人像重影改善。','PLY预览仍有响应差，真实人物重影尚未验收。');
"""+s[b:]
s=s.replace('本机T7合成控制的RGB与alpha整体及原生裁片，非DirectFisheye训练结果，图内无文字',
            '本机T8合成双视角控制，两幅全图等比缩略及各372×384原生裁片，图内无文字')
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index(".concat([previous]).join('\\n\\n');",a)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 T8 CVO双视角训练组件与无损原生参数包。',*r['findings_zh'],
       '数值、原图、预登记范围和源码哈希位于robust-portrait-122/native-cvo-control。官方算法思想来源：https://arxiv.org/html/2604.00648v1 ，https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS 。',
       '图片为本机T8已知几何，不是官方论文图片，也不是人物训练结果。下方继续保留历史研究记录。']
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[b:]
s=s.replace('preserve_deck_directFisheye_122.py','preserve_deck_nativeCVO_122.py').replace('02_有限视角重建优化路线_122DirectFisheye核验.pptx','02_有限视角重建优化路线_122双视角组件.pptx')
(H/'update_deck_nativeCVO_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_directFisheye_122.py').read_text(encoding='utf-8').replace('directFisheyeAudit-122','nativeCVO-122')
(H/'preserve_deck_nativeCVO_122.py').write_text(p,encoding='utf-8')
print('T8_DECK_SOURCES_PREPARED')
