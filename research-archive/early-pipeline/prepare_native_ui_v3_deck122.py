from pathlib import Path
import json
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
r=json.loads((R/'native-ui-v3/review.json').read_text(encoding='utf-8'))
s=(H/'update_deck_nativeCVO_122.mjs').read_text(encoding='utf-8')
s=s.replace('nativeCVO-122','nativeUI3-122').replace('/native-cvo-control','/native-ui-v3')
s=s.replace("const figure=root+'/cvo-native-control.png';","const figure=root+'/../native-cvo-control/cvo-native-control.png';")
s=s.replace("build+'/directFisheyeAudit-122-deck-receipt.json'","build+'/nativeCVO-122-deck-receipt.json'")
s=s.replace("if(review.checks_CPU!==22||review.checks_GPU!==15||review.person_training!==false)throw Error('Review incomplete');","if(review.state!=='passed'||!review.actual_open_file||review.person_training!==false)throw Error('Review incomplete');")
a=s.index("replace('DirectFisheye");b=s.index('const images=',a)
s=s[:a]+"""replace('跨视角更新与原生参数保存','原生参数包加载与相机回放');
replace('同一合成几何的竖屏、横屏全幅与原生裁片。','同一合成几何的竖屏、横屏全幅与原生裁片。');
replace('双视角统计已核验，20万点无损回读与参考一致。','打开包后，两种相机的原生浮点回放与参考一致。');
replace('PLY预览仍有响应差，真实人物重影尚未验收。','通用PLY保留相机，真实人物重影仍未通过验收。');
"""+s[b:]
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index(".concat([previous]).join('\\n\\n');",a)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 T9 无损原生包实际导入与记录相机回放。',*r['findings_zh'],
 '实测证据：robust-portrait-122/native-ui-v3/plan.json、review.json、ui-state.json及原始CUDA浮点输出。',
 '配图为T8已知合成几何的等比例全幅和原生裁片。T9实际导入后回放与这两幅参考的RGB/alpha逐像素最大差均为0。图片内没有文字，不是实拍人物或官方DirectFisheye训练结果。',
 '算法思想参考：https://arxiv.org/html/2604.00648v1 ，https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS 。后续保留历史技术记录。']
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[b:]
s=s.replace('preserve_deck_nativeCVO_122.py','preserve_deck_nativeUI3_122.py').replace('02_有限视角重建优化路线_122双视角组件.pptx','02_有限视角重建优化路线_122原生包回放.pptx')
(H/'update_deck_nativeUI3_122.mjs').write_text(s,encoding='utf-8')
s=(H/'preserve_deck_nativeCVO_122.py').read_text(encoding='utf-8').replace('nativeCVO-122','nativeUI3-122')
(H/'preserve_deck_nativeUI3_122.py').write_text(s,encoding='utf-8')
print('T9_DECK_SOURCES_READY')
