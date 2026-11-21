from pathlib import Path
import json
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
r=json.loads((R/'gsplat-distortion-research/review.json').read_text(encoding='utf-8'))
s=(H/'update_deck_nativeUI3_122.mjs').read_text(encoding='utf-8')
s=s.replace('nativeUI3-122','gsplatDistortion-122').replace('/native-ui-v3','/gsplat-distortion-research')
s=s.replace("const figure=root+'/../native-cvo-control/cvo-native-control.png';","const figure=root+'/distortion-control.png';")
s=s.replace("build+'/nativeCVO-122-deck-receipt.json'","build+'/nativeUI3-122-deck-receipt.json'")
s=s.replace("if(review.state!=='passed'||!review.actual_open_file||review.person_training!==false)throw Error('Review incomplete');",
            "if(review.checks_passed!==8||review.human_training_started!==false)throw Error('Review incomplete');")
a=s.index("replace('跨视角");b=s.index('const images=',a)
s=s[:a]+"""replace('原生参数包加载与相机回放','gsplat 的畸变相机与训练边界');
replace('同一合成几何的竖屏、横屏全幅与原生裁片。','无畸变 / 给定畸变 / 差值×10：六个已知高斯。');
replace('打开包后，两种相机的原生浮点回放与参考一致。','本机已验证畸变渲染与高斯梯度；标定须独立完成。');
replace('通用PLY保留相机，真实人物重影仍未通过验收。','1.5.3 此分支不提供 K、畸变系数或外参梯度。');
"""+s[b:]
s=s.replace('本机T8合成双视角控制，两幅全图等比缩略及各372×384原生裁片，图内无文字','本机gsplat六高斯合成CUDA控制：无畸变、给定Brown畸变、绝对差十倍；图内无文字')
s=s.replace('left:75,top:190,width,height:width*192/672','left:75,top:225,width,height:width*64/304')
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index(".concat([previous]).join('\\n\\n');",a)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 gsplat畸变相机源码调研及一次合成CUDA能力检查。',*r['findings_zh'],
       '图中每幅原生96×64，三个区域用8像素白色间隔，最后为绝对RGB差值放大10倍。没有人物数据，不表示标定或重影改善。',
       '官方版本说明：https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html',
       '官方3DGUT训练指导：https://github.com/nerfstudio-project/gsplat/blob/'+r['main_commit']+'/docs/3dgut.md',
       '实测：robust-portrait-122/gsplat-distortion-research/probe-result.json 与 sources.json。旧历史结果保留如下。']
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[b:]
s=s.replace('preserve_deck_nativeUI3_122.py','preserve_deck_gsplatDistortion_122.py').replace('02_有限视角重建优化路线_122原生包回放.pptx','02_有限视角重建优化路线_122畸变接口调研.pptx')
(H/'update_deck_gsplatDistortion_122.mjs').write_text(s,encoding='utf-8')
s=(H/'preserve_deck_nativeUI3_122.py').read_text(encoding='utf-8').replace('nativeUI3-122','gsplatDistortion-122')
(H/'preserve_deck_gsplatDistortion_122.py').write_text(s,encoding='utf-8')
print('GSPLAT_DECK_SOURCE_READY')
