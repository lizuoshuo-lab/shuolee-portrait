from pathlib import Path
import json
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
r=json.loads((R/'native-residual-camera/review.json').read_text(encoding='utf-8'))
s=(H/'update_deck_gsplatDistortion_122.mjs').read_text(encoding='utf-8')
s=s.replace('gsplatDistortion-122','residualCamera-122').replace('/gsplat-distortion-research','/native-residual-camera')
s=s.replace("const figure=root+'/distortion-control.png';","const figure=root+'/residual-control.png';")
s=s.replace("build+'/nativeUI3-122-deck-receipt.json'","build+'/gsplatDistortion-122-deck-receipt.json'")
s=s.replace('review.checks_passed!==8','review.checks_passed!==26')
a=s.index("replace('原生参数包");b=s.index('const images=',a)
s=s[:a]+"""replace('gsplat 的畸变相机与训练边界','残余畸变的统一相机链路');
replace('无畸变 / 给定畸变 / 差值×10：六个已知高斯。','同一 K、外参与畸变参数贯穿投影、训练与回读。');
replace('本机已验证畸变渲染与高斯梯度；标定须独立完成。','合成训练与回读一致；实拍畸变系数仍待验证。');
replace('1.5.3 此分支不提供 K、畸变系数或外参梯度。','26项集成、11项冻结检查通过；未启动人物训练。');
"""+s[b:]
s=s.replace('frame=image.frame,width=1130','frame=image.frame,width=620')
s=s.replace('left:75,top:225,width,height:width*64/304','left:330,top:190,width,height:width*160/288')
s=s.replace('本机gsplat六高斯合成CUDA控制：无畸变、给定Brown畸变、绝对差十倍；图内无文字','残余畸变合成控制：独立射线、CUDA及差值；两台训练相机。原生96×80，图内无文字')
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index(".concat([previous]).join('\\n\\n');",a)+len(".concat([previous]).join('\\n\\n');")
notes=['2026-10-28 R2 输出域残余畸变工程链路。',*r['findings_zh'],
       '图上排：独立射线、CUDA响应、绝对差×1000。下排：两台不同畸变相机20步合成训练结果；右下留白。均96×80，显示等比例。',
       '本地证据：robust-portrait-122/native-residual-camera/result.json、frozen-view-addendum.json、historical-n6-bridge.json。下面保留历史内容。']
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[b:]
s=s.replace('preserve_deck_gsplatDistortion_122.py','preserve_deck_residualCamera_122.py').replace('02_有限视角重建优化路线_122畸变接口调研.pptx','02_有限视角重建优化路线_122残余相机链路.pptx')
(H/'update_deck_residualCamera_122.mjs').write_text(s,encoding='utf-8')
s=(H/'preserve_deck_gsplatDistortion_122.py').read_text(encoding='utf-8').replace('gsplatDistortion-122','residualCamera-122')
(H/'preserve_deck_residualCamera_122.py').write_text(s,encoding='utf-8')
print('R2_DECK_READY')
